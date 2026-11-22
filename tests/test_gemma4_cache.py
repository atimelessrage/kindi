# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded host memory for the Gemma 4 prompt cache.

These tests use fake ring/global arrays and intercept the pinned-memory copy, so
they need neither a GPU nor a model.  They exercise the budget accounting,
oldest-first eviction and the telemetry counters directly.
"""
import numpy as np
import pytest

from kindi.gemma4_cache import G4Cache


class FakeShard(object):
    ring_slots = 8

    def __init__(self):
        self.local = {0}
        self.cache = {
            0: {"k": np.zeros((8, 2, 4), dtype=np.float16),
                "v": np.zeros((8, 2, 4), dtype=np.float16)},
            1: {"k": np.zeros((4096, 2, 4), dtype=np.float16),
                "v": np.zeros((4096, 2, 4), dtype=np.float16)},
        }


@pytest.fixture
def make_cache(monkeypatch):
    monkeypatch.setenv("KINDI_G4_PROMPT_CACHE", "1")
    monkeypatch.delenv("KINDI_G4_CKPT_MAX", raising=False)
    monkeypatch.delenv("KINDI_G4_PARK_MAX", raising=False)
    shard = FakeShard()
    monkeypatch.setattr(G4Cache, "_each", lambda self, fn: [fn(self.eng)])
    monkeypatch.setattr(G4Cache, "_to_host",
                        staticmethod(lambda a: np.zeros(a.shape, dtype=a.dtype)))

    def make(**kw):
        return G4Cache(shard, log=kw.pop("log", lambda *a: None), **kw)
    return make


def test_checkpoints_are_evicted_to_the_byte_budget(make_cache):
    logs = []
    g4 = make_cache(budget_bytes=1000, log=logs.append)   # one ring snapshot is 256 B
    for n in range(1024, 1041):
        g4.advanced(list(range(n)), checkpoint=True)
        assert g4.metrics()["used_bytes"] <= 1000

    m = g4.metrics()
    assert m["used_bytes"] <= m["budget_bytes"] == 1000
    assert m["evictions"] > 0
    assert m["checkpoints_dropped"] > 0
    assert m["peak_bytes"] > m["budget_bytes"]     # the cap is enforced after admission
    assert m["checkpoints"] == len(g4.ckpts)
    assert any("evicted" in line for line in logs)


def test_parked_conversations_are_bounded_and_evicted_oldest_first(make_cache):
    logs = []
    g4 = make_cache(budget_bytes=40000, log=logs.append)  # one 1024-token park is 33,024 B
    for i in range(3):
        g4.tokens = list(range(1024))
        g4.ckpts = []
        g4._park_active(None)
        assert g4.metrics()["used_bytes"] <= 40000

    m = g4.metrics()
    assert m["used_bytes"] <= m["budget_bytes"]
    assert m["parked"] == 1
    assert m["parked_dropped"] == 2
    assert m["evictions"] == 2
    assert len(g4.parked) == 1
    assert any("evicted" in line for line in logs)
    assert any("parked" in line for line in logs)


def test_parking_makes_room_by_evicting_an_older_parked_conversation(make_cache):
    logs = []
    g4 = make_cache(budget_bytes=40000, log=logs.append)
    for i in range(2):
        g4.tokens = list(range(1024))
        g4.ckpts = []
        g4._park_active(None)
    m = g4.metrics()
    assert m["parked"] == 1
    assert m["evictions"] == 1
    assert m["parked_dropped"] == 1
    assert m["used_bytes"] == 33024
    assert any("make room" in line for line in logs)


def test_an_oversized_checkpoint_or_park_is_not_created(make_cache):
    logs = []
    g4 = make_cache(budget_bytes=100, log=logs.append)
    g4.advanced(list(range(1024)), checkpoint=True)
    g4.tokens = list(range(1024))
    g4._park_active(None)

    m = g4.metrics()
    assert m["checkpoints"] == 0
    assert m["parked"] == 0
    assert m["used_bytes"] == 0
    assert any("not kept" in line for line in logs)
    assert any("not parked" in line for line in logs)


def test_cache_off_releases_host_snapshots(make_cache):
    g4 = make_cache(budget_bytes=40000)
    g4.tokens = list(range(1024))
    g4._park_active(None)
    assert g4.parked
    g4.on = False
    assert g4.prepare([1, 2, 3]) == 0
    assert not g4.parked
    assert g4.metrics()["used_bytes"] == 0


def test_checkpoint_count_limit_is_reported_as_an_eviction(make_cache):
    g4 = make_cache(budget_bytes=10 ** 9)
    g4.ckpt_max = 2
    for n in range(1024, 1027):
        g4.advanced(list(range(n)), checkpoint=True)
    m = g4.metrics()
    assert m["checkpoints"] == 2
    assert m["evictions"] == 1
    assert m["checkpoints_dropped"] == 1
    assert m["evicted_bytes"] == 256


def test_parking_does_not_double_count_the_active_checkpoint(make_cache):
    g4 = make_cache(budget_bytes=33050)   # globals 32,768 B + one 256 B ring
    g4.advanced(list(range(1024)), checkpoint=True)
    g4.tokens = list(range(1024))
    g4._park_active(None)
    m = g4.metrics()
    assert m["parked"] == 1
    assert m["evictions"] == 0
    assert m["used_bytes"] == 33024


def test_parking_and_restoring_return_free_pinned_blocks(make_cache, monkeypatch):
    """Upstream #44: the host copies must not stay in CuPy's pinned pool."""
    calls = []
    monkeypatch.setattr(G4Cache, "_give_back", staticmethod(lambda: calls.append(1)))
    monkeypatch.setattr(G4Cache, "_globals_set", lambda self, g, n: None)
    monkeypatch.setattr(G4Cache, "_rings_set", lambda self, snap: None)
    g4 = make_cache(budget_bytes=40000)
    g4.tokens = list(range(1024))
    g4._park_active(None)
    assert calls                                  # parking gives the pool back
    calls.clear()
    assert g4.prepare(list(range(1024)) + [1024]) == 1024
    assert calls                                  # restoring gives the parked globals back


def test_zero_checkpoint_limit_does_not_keep_a_ring_snapshot(make_cache):
    g4 = make_cache(budget_bytes=40000)
    g4.ckpt_max = 0
    g4.tokens = list(range(1024))
    g4._park_active(None)
    assert g4.parked[0]["ckpts"] == []
    assert g4.metrics()["used_bytes"] == 32768


def test_dropped_checkpoints_return_their_pinned_blocks(make_cache, monkeypatch):
    """Upstream #44: trimming a checkpoint list must also give the dropped ring copies back."""
    calls = []
    monkeypatch.setattr(G4Cache, "_give_back", staticmethod(lambda: calls.append(1)))
    g4 = make_cache(budget_bytes=1 << 20)
    g4.ckpt_max = 1
    g4.advanced(list(range(1024)), checkpoint=True)
    assert len(g4.ckpts) == 1
    calls.clear()
    g4.advanced(list(range(1026)), checkpoint=True)   # drops the 1024 checkpoint
    assert len(g4.ckpts) == 1
    assert calls


def test_prepare_shrink_returns_the_dropped_checkpoints(make_cache, monkeypatch):
    """Upstream #44: resuming from an earlier point drops the checkpoints past it."""
    calls = []
    monkeypatch.setattr(G4Cache, "_give_back", staticmethod(lambda: calls.append(1)))
    monkeypatch.setattr(G4Cache, "_rings_set", lambda self, snap: None)
    g4 = make_cache(budget_bytes=1 << 20)
    g4.advanced(list(range(1024)), checkpoint=True)
    g4.advanced(list(range(1026)), checkpoint=True)
    assert len(g4.ckpts) == 2
    calls.clear()
    assert g4.prepare(list(range(1024)) + [9999]) == 1024   # keeps the 1024 checkpoint, drops the 1026 one
    assert [c[0] for c in g4.ckpts] == [1024]
    assert calls


def test_budget_evictions_return_the_pinned_blocks(make_cache, monkeypatch):
    """Upstream #44: the pool must get the blocks of an evicted parked conversation."""
    calls = []
    monkeypatch.setattr(G4Cache, "_give_back", staticmethod(lambda: calls.append(1)))
    g4 = make_cache(budget_bytes=33025)          # room for one 1024-token park (33,024 B)
    g4.tokens = list(range(1024))
    g4._park_active(None)
    assert len(g4.parked) == 1
    calls.clear()
    g4.tokens = list(range(1024))
    g4._park_active(None)                        # the second park evicts the first
    assert len(g4.parked) == 1
    assert calls
