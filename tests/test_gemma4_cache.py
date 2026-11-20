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


def test_zero_checkpoint_limit_does_not_keep_a_ring_snapshot(make_cache):
    g4 = make_cache(budget_bytes=40000)
    g4.ckpt_max = 0
    g4.tokens = list(range(1024))
    g4._park_active(None)
    assert g4.parked[0]["ckpts"] == []
    assert g4.metrics()["used_bytes"] == 32768
