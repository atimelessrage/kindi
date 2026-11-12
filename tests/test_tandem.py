# SPDX-License-Identifier: AGPL-3.0-only
"""The dense tandem placement and handoff, without a real GPU or CPU pack.

The fake backends record the layer ranges and the hidden-state exchange.  The real GPU path is
covered by the code-level checks in the release gate; this test suite keeps the routing contract
(which backend owns the embedding, head and each layer) and the two placement modes honest.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from kindi import cache, tandem
from kindi.engine import Engine


class FakeEngine(object):
    D = 4

    def __init__(self, pack, ctx=4096, log=None, layers=None, embed=True, head=True):
        self.ctx = int(ctx)
        self.layers = None if layers is None else tuple(layers)
        self.embed_owned, self.head_owned = bool(embed), bool(head)
        self.calls = []
        self.h = np.zeros(self.D, dtype=np.float32)
        self.pending = 0
        self.vocab = {"tok": 0}
        self.ylog = np.arange(5, dtype=np.float32)
        self.ylog_n, self.ylog_map = 5, None
        self.k = object()
        self.aout = np.zeros(1, dtype=np.int32)
        self.dims = {"full_attention_interval": 2}
        self.NL = 4
        self._kvi = {1: 0, 3: 1}
        self.isa, self.threads = "fake", 1
        self.M, self.conv = [None] * self.NL, [None] * self.NL
        self.Kc, self.Vc = [None] * 2, [None] * 2
        if layers is None:                     # the CPU engine owns the whole model
            self.M = [np.zeros(1, dtype=np.float32) for _ in (0, 2)]
            self.conv = [np.zeros(1, dtype=np.float32) for _ in (0, 2)]
            self.Kc = [np.zeros((self.ctx, 1, 1), dtype=np.float16) for _ in (1, 3)]
            self.Vc = [np.zeros((self.ctx, 1, 1), dtype=np.float16) for _ in (1, 3)]
        else:                                  # the GPU engine owns only its range
            for il in self.layers:
                if il in self._kvi:
                    k = self._kvi[il]
                    self.Kc[k] = np.zeros((self.ctx, 1, 1), dtype=np.float16)
                    self.Vc[k] = np.zeros((self.ctx, 1, 1), dtype=np.float16)
                else:
                    self.M[il] = np.zeros(1, dtype=np.float32)
                    self.conv[il] = np.zeros(1, dtype=np.float32)

    def begin(self, tok):
        self.calls.append(("begin", int(tok)))
        self.h = np.full(self.D, float(tok), dtype=np.float32)

    def layer(self, il, pos):
        self.calls.append(("layer", int(il), int(pos)))
        self.h += 1.0

    def feed(self, tok, pos):
        self.calls.append(("feed", int(tok), int(pos)))

    def step(self, tok, pos, **kw):
        self.calls.append(("step", int(tok), int(pos)))
        return 42

    def finish(self):
        self.calls.append(("finish",))
        self.ylog = np.arange(5, dtype=np.float32) + 100
        return 77

    def head(self):
        self.calls.append(("head",))
        self.ylog = np.arange(5, dtype=np.float32) + 200
        return 88

    def get_hidden(self):
        self.calls.append(("get_hidden",))
        return self.h.copy()

    def set_hidden(self, values):
        self.calls.append(("set_hidden", list(values)))
        self.h = np.asarray(values, dtype=np.float32).copy()
        self.pending = 0

    def flush(self):
        self.calls.append(("flush",))
        self.pending = 0

    def reset(self):
        self.calls.append(("reset",))
        self.h.fill(0)
        self.pending = 0
        for a in self.M + self.conv + self.Kc + self.Vc:
            if a is not None:
                a.fill(0)


@pytest.fixture
def fake_pack(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text(json.dumps({"dimensions": {"layers": 4}}), encoding="utf-8")
    return str(pack)


@pytest.fixture
def fake_backends(monkeypatch):
    monkeypatch.setattr(tandem, "CPUEngine", FakeEngine)
    monkeypatch.setattr(tandem, "GPUEngine", FakeEngine)


def test_cpu_first_owns_the_prefix_and_the_gpu_head(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="cpu-first", log=lambda *a: None)
    assert eng.cpu.layers is None
    assert eng.gpu.layers == (2, 3)
    assert eng.gpu.embed_owned is False
    assert eng.gpu.head_owned is True
    assert eng.cpu_range == (0, 1) and eng.gpu_range == (2, 3)
    assert eng.ylog is eng.gpu.ylog

    assert eng.step(10, 0) == 77
    assert eng.cpu.calls == [("begin", 10), ("layer", 0, 0), ("layer", 1, 0), ("get_hidden",)]
    assert eng.gpu.calls[:4] == [("set_hidden", [12.0, 12.0, 12.0, 12.0]),
                                  ("layer", 2, 0), ("layer", 3, 0), ("finish",)]
    assert eng.ylog is eng.gpu.ylog


def test_gpu_first_owns_the_prefix_and_the_cpu_head(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="gpu-first", log=lambda *a: None)
    assert eng.gpu.layers == (0, 1)
    assert eng.gpu.embed_owned is True
    assert eng.gpu.head_owned is False
    assert eng.cpu_range == (2, 3) and eng.gpu_range == (0, 1)
    assert eng.ylog is eng.cpu.ylog

    assert eng.step(9, 0) == 88
    assert eng.gpu.calls[:3] == [("begin", 9), ("layer", 0, 0), ("layer", 1, 0)]
    assert eng.gpu.calls[3] == ("get_hidden",)
    assert eng.cpu.calls[:3] == [("set_hidden", [11.0, 11.0, 11.0, 11.0]),
                                 ("layer", 2, 0), ("layer", 3, 0)]
    assert eng.cpu.calls[-1] == ("head",)
    assert eng.ylog is eng.cpu.ylog


def test_feed_leaves_the_residual_without_evaluating_the_head(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=1, mode="cpu-first", log=lambda *a: None)
    eng.feed(3, 7)
    assert ("finish",) not in eng.gpu.calls
    assert eng.gpu.calls[-1] == ("flush",)


def test_reset_clears_both_backends(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=1, mode="gpu-first", log=lambda *a: None)
    eng.reset()
    assert ("reset",) in eng.cpu.calls and ("reset",) in eng.gpu.calls


def test_invalid_split_is_rejected_before_a_backend_is_built(fake_pack, fake_backends):
    with pytest.raises(SystemExit, match="between 1 and 3"):
        tandem.TandemEngine(fake_pack, cpu_layers=0, mode="cpu-first", log=lambda *a: None)
    with pytest.raises(SystemExit, match="cpu-first or gpu-first"):
        tandem.TandemEngine(fake_pack, cpu_layers=1, mode="middle", log=lambda *a: None)


def cache_cfg(tmp_path):
    return {"checkpoint_every": "16", "prefix_max": "0", "prefix_min": "0", "park_max": "0",
            "checkpoint_budget_mb": "0", "state_dir": str(tmp_path), "head": "exact"}


def test_runner_reads_the_prompt_and_steps_the_last_token(fake_pack, fake_backends, tmp_path):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=1, mode="cpu-first", log=lambda *a: None)
    runner = tandem.TandemRunner(eng, cache_cfg(tmp_path), "test", log=lambda *a: None)
    nxt, cached = runner.prepare([1, 2, 3], 3)
    assert nxt == 77 and cached == 0
    assert runner.tokens == [1, 2, 3]
    assert ("get_hidden",) in eng.cpu.calls
    assert runner.step(4) == 77
    assert runner.tokens == [1, 2, 3, 4]


def test_joint_checkpoint_contains_both_owners(fake_pack, fake_backends):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="cpu-first", log=lambda *a: None)
    view = eng.cache_view()
    assert view.M[0] is eng.cpu.M[0]
    assert view.M[1] is eng.gpu.M[2]
    assert view.Kc[0] is eng.cpu.Kc[0]
    assert view.Kc[1] is eng.gpu.Kc[1]
    assert view.conv[0] is eng.cpu.conv[0] and view.conv[1] is eng.gpu.conv[2]


def test_joint_checkpoint_restores_both_owners(fake_pack, fake_backends, tmp_path):
    eng = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="cpu-first", log=lambda *a: None)
    view = eng.cache_view()
    for i, a in enumerate(view.M):
        a.fill(10 + i)
    for i, a in enumerate(view.conv):
        a.fill(30 + i)
    runner = tandem.TandemRunner(eng, cache_cfg(tmp_path), "test", log=lambda *a: None)
    runner._save_ckpt(4)
    for a in view.M + view.conv:
        a.fill(0)
    runner._restore(runner.ckpt[4])
    assert [int(a[0]) for a in view.M] == [10, 11]
    assert [int(a[0]) for a in view.conv] == [30, 31]


def test_joint_prefix_reuse_resumes_at_a_checkpoint(fake_pack, fake_backends, tmp_path):
    eng = tandem.TandemEngine(fake_pack, ctx=512, cpu_layers=2, mode="gpu-first", log=lambda *a: None)
    runner = tandem.TandemRunner(eng, cache_cfg(tmp_path), "test", log=lambda *a: None)
    first = list(range(300))
    nxt, cached = runner.prepare(first, len(first), len(first))
    assert nxt == 88 and cached == 0
    second = first[:260] + [100, 101]
    nxt, cached = runner.prepare(second, len(second), len(second))
    assert cached == 256
    assert runner.tokens == second


def test_tandem_runner_recomputes_the_pick_after_its_checkpoint_cache_is_deleted(
        fake_pack, fake_backends, tmp_path):
    """If the prompt checkpoint cache is deleted, the runner must read the prompt again
    and compute a fresh next-token pick instead of relying on stale reuse state."""
    eng = tandem.TandemEngine(fake_pack, ctx=512, cpu_layers=2, mode="gpu-first", log=lambda *a: None)
    runner = tandem.TandemRunner(eng, cache_cfg(tmp_path), "test", log=lambda *a: None)
    first = list(range(300))
    nxt, cached = runner.prepare(first, len(first), len(first))
    assert nxt == 88 and cached == 0
    assert runner.ckpt

    calls_before = len(eng.cpu.calls) + len(eng.gpu.calls)
    runner.ckpt.clear()
    runner.parked.clear()

    nxt, cached = runner.prepare(first, len(first), len(first))
    assert nxt == 88 and cached == 0
    assert runner.tokens == first
    assert len(eng.cpu.calls) + len(eng.gpu.calls) > calls_before


def test_joint_fingerprint_separates_the_two_placements(fake_pack, fake_backends, tmp_path):
    cpu_first = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="cpu-first", log=lambda *a: None)
    gpu_first = tandem.TandemEngine(fake_pack, ctx=64, cpu_layers=2, mode="gpu-first", log=lambda *a: None)
    cfg = cache_cfg(tmp_path)
    a = cache.engine_fingerprint(cpu_first.cache_view(), cfg)[0]
    b = cache.engine_fingerprint(gpu_first.cache_view(), cfg)[0]
    assert a != b


def test_layer_tensor_selection_keeps_only_owned_layers_and_optional_ends():
    bi = {
        "token_embd.weight": {"rows": 1},
        "output.weight": {"rows": 2},
        "blk.0.attn_q.weight": {"rows": 3},
        "blk.1.ffn_up.weight": {"rows": 4},
        "blk.2.attn_q.weight": {"rows": 5},
    }
    got = Engine._select_layers(bi, (1, 2), embed=False, head=True, head_name="output.weight")
    assert set(got) == {"blk.1.ffn_up.weight", "blk.2.attn_q.weight", "output.weight"}
    got = Engine._select_layers(bi, (0,), embed=True, head=False, head_name="output.weight")
    assert set(got) == {"token_embd.weight", "blk.0.attn_q.weight"}
