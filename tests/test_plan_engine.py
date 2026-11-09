# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only regression tests for the plan engine's token-count and Hadamard-block contracts.

The plan engine is CUDA-only: ``kindi/spec.py`` and ``kindi/tp.py`` import CuPy at module level and
building a plan needs a card. These tests import those modules with a stand-in ``cupy`` when CuPy
is absent, then exercise the decisions taken before any buffer, kernel or graph exists:

  * ``TMAX``, the largest T-token pass, and the draft clamp ``k -> TMAX - 1``: a 2B request may
    draft 7 tokens and verify 8, the T = 8 plan the 2B measured at 373.5-375.7 tok/s;
  * the 2B's Hadamard-block-512 split rule: one column split, so no ``k_tc_sum`` and no reordered
    sum, which is what keeps T = 1/4/8 token-identical to the plain engine;
  * the Hadamard-block guards that refuse a pack the plans cannot run: the 9B pack has HB 4096,
    so both the tensor-parallel plan builder and the installed tensor-core plan engine stop with
    a SystemExit before building anything.

No GPU, CUDA toolkit or model pack is needed, and no kernel runs here.
"""
import importlib
import os
import sys
import types

import pytest


def _install_cupy_stub():
    """The minimum ``cupy`` surface the two modules touch at import time (CI has no CuPy)."""
    cp = types.ModuleType("cupy")
    cuda = types.ModuleType("cupy.cuda")
    cuda.runtime = types.ModuleType("cupy.cuda.runtime")
    cuda.runtime.is_hip = False
    cuda.compiler = types.ModuleType("cupy.cuda.compiler")
    cuda.function = types.ModuleType("cupy.cuda.function")
    cp.cuda = cuda
    cp.ndarray = type("ndarray", (), {})
    sys.modules.update({"cupy": cp, "cupy.cuda": cuda, "cupy.cuda.runtime": cuda.runtime,
                        "cupy.cuda.compiler": cuda.compiler, "cupy.cuda.function": cuda.function})


@pytest.fixture(scope="module")
def spec_tp():
    """kindi.spec and kindi.tp, imported without a GPU (real CuPy when it is installed)."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        _install_cupy_stub()
    return importlib.import_module("kindi.spec"), importlib.import_module("kindi.tp")


@pytest.fixture()
def tmax(spec_tp, monkeypatch):
    """Set KINDI_TC_TMAX and reload kindi.spec; the module goes back to the ambient value after."""
    spec, _ = spec_tp
    ambient = os.environ.get("KINDI_TC_TMAX")

    def load(value):
        if value is None:
            monkeypatch.delenv("KINDI_TC_TMAX", raising=False)
        else:
            monkeypatch.setenv("KINDI_TC_TMAX", value)
        return importlib.reload(spec).TMAX

    yield load
    # the monkeypatch teardown runs after this finalizer: restore and reload by hand
    if ambient is None:
        os.environ.pop("KINDI_TC_TMAX", None)
    else:
        os.environ["KINDI_TC_TMAX"] = ambient
    importlib.reload(spec)


# ------------------------------------------------------------------ TMAX: the largest T-token pass
def test_tmax_default_plans_the_eight_token_verify(tmax):
    # the 2B plan engine's measured best runs the verify pass at T = 8 (k = 7 drafts + 1)
    assert tmax(None) == 8


@pytest.mark.parametrize("env, want", [("8", 8), ("3", 3), ("2", 2), ("1", 2), ("0", 2), ("-4", 2),
                                       ("64", 8), ("999", 8)])
def test_tmax_is_clamped_to_the_supported_range(tmax, env, want):
    # no plan exists above T = 8, and a verify pass needs at least T = 2 (k >= 1)
    assert tmax(env) == want


class _Eng(object):
    """the part of the engine generate_spec drives; step always picks token 0"""

    def __init__(self):
        self.steps = 0

    def reset(self):
        pass

    def feed_batch(self, ids, pos=0):
        pass

    def step(self, tok, pos):
        self.steps += 1
        return 0


class _Draft(object):
    """a drafter that proposes nothing, so generate_spec records the k it was asked for"""

    def __init__(self):
        self.calls = []

    def reset(self, ids):
        pass

    def propose(self, k):
        self.calls.append(k)
        return []

    def push(self, tok):
        pass


def _first_draft(spec, k):
    dr = _Draft()
    out = spec.generate_spec(_Eng(), [5, 6], 20, k=k, drafter=dr)
    assert len(out) == 20                 # plain steps, all token 0; no verify pass is run
    return dr.calls[0] if dr.calls else None


@pytest.mark.parametrize("k, want", [(4, 4), (7, 7), (8, 7), (99, 7)])
def test_generate_spec_drafts_at_most_tmax_minus_one(spec_tp, k, want):
    # k drafts + the current token = one T = k + 1 verify pass, so k must stay below TMAX
    spec, _ = spec_tp
    assert _first_draft(spec, k) == want


def test_generate_spec_off_drafts_nothing(spec_tp):
    spec, _ = spec_tp
    assert _first_draft(spec, 0) is None


# ------------------------------------------------------------- the 2B's one-split (token-exact) rule
@pytest.mark.parametrize("rows, cols, tied", [(2048, 8192, True), (5120, 40960, False),
                                              (248320, 2048, False)])
def test_2b_hadamard_512_keeps_one_column_split(spec_tp, rows, cols, tied):
    """HB 512: a column split's k_tc_sum changes the summation order; one split keeps the 2B's
    plan token-identical to the plain engine at every T (the 27B, HB 1024, keeps its splits)."""
    spec, _ = spec_tp
    shard = types.SimpleNamespace(HB=512, D=2048)
    warps, splits = spec.tune(shard, rows, cols)
    assert splits == 1
    assert warps > 0


# ------------------------------------------------------------------- the Hadamard-block guards
class _ShardProbe(object):
    """A shard without __init__ (no CUDA) and without buffers: both guards run before the first
    buffer is read, so reaching an AttributeError means the guard let the block size through.
    The guards are called unbound so this object needs no Shard base class."""

    def __getattr__(self, name):
        raise AttributeError("probe shard has no %r: the Hadamard-block guard let it through" % name)


def _probe(hb):
    shard = _ShardProbe()
    shard.D, shard.HB, shard.HK, shard.HD = 4096, hb, 128, 128
    shard.NK, shard.NV, shard.NH, shard.NHKV, shard.NFF = 8, 8, 8, 8, 16384
    shard.tp = shard.k = None
    return shard


def test_tensor_parallel_plans_refuse_hadamard_4096(spec_tp):
    # the 9B pack's Hadamard block: the two-card plans only store K/V for HB 1024
    _, tp = spec_tp
    with pytest.raises(SystemExit, match=r"needs Hadamard blocks of 1024"):
        tp.Shard.build_plans(_probe(4096))


def test_tensor_parallel_plans_accept_hadamard_1024(spec_tp):
    _, tp = spec_tp
    with pytest.raises(AttributeError):     # the guard passed; the next statement reads s.hA
        tp.Shard.build_plans(_probe(1024))


def test_kv_mode_defaults_to_q8_0(spec_tp, monkeypatch):
    _, tp = spec_tp
    monkeypatch.delenv("KINDI_KV", raising=False)
    assert tp.kv_mode() == "q8_0"


@pytest.mark.parametrize("value, want", [("q8", "q8_0"), ("Q8_0", "q8_0"), ("k8v4", "k8v4"),
                                         ("K8_V4", "k8v4")])
def test_kv_mode_accepts_the_cache_names(spec_tp, monkeypatch, value, want):
    _, tp = spec_tp
    monkeypatch.setenv("KINDI_KV", value)
    assert tp.kv_mode() == want


def test_kv_mode_rejects_an_unknown_cache(spec_tp, monkeypatch):
    _, tp = spec_tp
    monkeypatch.setenv("KINDI_KV", "q4_0")
    with pytest.raises(SystemExit, match="KINDI_KV must be q8_0 or k8v4"):
        tp.kv_mode()


def test_k8v4_saves_eight_kib_per_position(spec_tp):
    """27B geometry: q8_0 K+V is 34,816 bytes/token; q8_0 K + 4-bit V is 26,624."""
    _, tp = spec_tp
    shard = types.SimpleNamespace(NATT=16, NHKV=4, HD=256, NH=24, dims={"rope_dim": 64},
                                  tp=types.SimpleNamespace(kv="q8_0"))
    q8 = tp.Shard.per_pos_bytes(shard)
    shard.tp.kv = "k8v4"
    v4 = tp.Shard.per_pos_bytes(shard)
    assert q8 - v4 == 16 * 4 * (256 - 128)       # 8,192 bytes per token
    assert q8 == 16 * 2 * 4 * (256 + 16) + 24 * 4 + (24 * 256 * 4) // 256 + 64 * 4 + 64


def test_plan_engine_refuses_the_9b_before_building_plans(spec_tp, monkeypatch):
    """KINDI_TC=1 installs a build_plans wrapper; the 9B (HB 4096) must stop there, not in a kernel."""
    spec, tp = spec_tp
    shard_cls = tp.Shard
    saved = (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
             spec._chunk.reserve, spec._chunk.build)
    monkeypatch.setenv("KINDI_TC", "1")
    try:
        spec.install(log=lambda *a, **k: None)
        with pytest.raises(SystemExit,
                           match=r"tensor-core plan engine supports Hadamard blocks of 512 and 1024"):
            tp.Shard.build_plans(_probe(4096))
    finally:
        (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
         spec._chunk.reserve, spec._chunk.build) = saved
        spec._installed[0] = False
