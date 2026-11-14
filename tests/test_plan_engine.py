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
import json
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


# ------------------------------------- optional accelerators must switch off, never stop the engine
def test_a_missing_mtp_head_switches_speculation_off(spec_tp, monkeypatch, tmp_path):
    # before 2026-11-14 a stale KINDI_MTP_HEAD raised FileNotFoundError from os.path.getsize inside
    # the engine's __init__ wrapper, so `kindi serve` never started
    spec, _ = spec_tp
    said = []
    monkeypatch.setenv("KINDI_MTP_HEAD", str(tmp_path / "gone.npz"))
    assert spec.head_paths("mtp", said.append) == []
    assert "KINDI_MTP_HEAD" not in os.environ           # cleared: nothing downstream retries it
    assert said and "gone.npz" in said[0] and "not a file" in said[0]


def test_an_existing_mtp_head_is_kept_and_not_logged(spec_tp, monkeypatch, tmp_path):
    spec, _ = spec_tp
    head = tmp_path / "head.npz"
    head.write_bytes(b"x" * 64)
    said = []
    monkeypatch.setenv("KINDI_MTP_HEAD", str(head))
    assert spec.head_paths("mtp", said.append) == [str(head)]
    assert os.environ["KINDI_MTP_HEAD"] == str(head)
    assert said == []


def test_the_draft_head_is_only_checked_in_the_modes_that_use_it(spec_tp, monkeypatch, tmp_path):
    spec, _ = spec_tp
    monkeypatch.setenv("KINDI_SPEC_DRAFT_HEAD", str(tmp_path / "gone.npz"))
    monkeypatch.delenv("KINDI_MTP_HEAD", raising=False)
    assert spec.head_paths("mtp", lambda *a: None) == []        # mtp never reads the drafter's head
    assert os.environ["KINDI_SPEC_DRAFT_HEAD"]                  # ... so it stays for the next mode
    assert spec.head_paths("draft2b", lambda *a: None) == []    # this one does, and drops it
    assert "KINDI_SPEC_DRAFT_HEAD" not in os.environ


def test_an_unset_head_is_silent(spec_tp):
    spec, _ = spec_tp
    said = []
    assert spec.head_paths("auto", said.append) == []           # the normal case: nothing to say
    assert said == []


def test_optional_returns_the_built_accelerator(spec_tp):
    spec, _ = spec_tp
    said = []
    assert spec._optional(said.append, "the DSpine drafter", lambda: "built") == "built"
    assert said == []


def test_optional_turns_a_broken_accelerator_off_with_one_line(spec_tp):
    spec, _ = spec_tp
    said = []

    def broken():
        raise FileNotFoundError("no config.json")

    assert spec._optional(said.append, "the DSpine drafter /x", broken) is None
    assert len(said) == 1 and "speculation off" in said[0] and "no config.json" in said[0]


def test_a_garbage_mtp_head_switches_speculation_off_and_the_engine_still_generates(spec_tp, monkeypatch, tmp_path):
    """The corrupt-head path, with the real MTPDraft: the file exists, so it passes the
    head_paths existence check and the reserve's getsize, then MTPDraft raises (np.load on
    garbage) inside _build. The engine must come out with mtp=None and still generate."""
    spec, tp = spec_tp
    shard_cls = tp.Shard
    saved = (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
             spec._chunk.reserve, spec._chunk.build, spec.build)
    garbage = tmp_path / "head.npz"
    garbage.write_bytes(b"this is not an npz file" * 64)
    monkeypatch.setenv("KINDI_TC", "1")
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    monkeypatch.setenv("KINDI_MTP_HEAD", str(garbage))
    monkeypatch.delenv("KINDI_SPEC_DRAFT_HEAD", raising=False)
    said = []
    try:
        spec._chunk.build = lambda eng: None                    # o_build, captured by install
        spec.build = lambda eng, log: None                      # the tc plan build: nothing to do here
        spec.install(log=said.append)
        eng = types.SimpleNamespace(
            cards=[types.SimpleNamespace(tc_layout=True, tcb={})],
            log=said.append, drafter=None)
        with pytest.raises(Exception):                          # the real constructor fails on garbage
            spec._mtp.MTPDraft(eng, str(garbage), log=said.append)
        spec._chunk.build(eng)                                  # must not raise
        assert eng.mtp is None
        assert any("speculation off" in s and "head.npz" in s for s in said)
        eng.reset = lambda: None
        eng.feed_batch = lambda ids, pos=0: None
        eng.step = lambda tok, pos: 0
        out = spec.generate_spec(eng, [5, 6], 20, k=4, drafter=_Draft())
        assert len(out) == 20                                   # plain stepping, no verify pass
    finally:
        (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
         spec._chunk.reserve, spec._chunk.build, spec.build) = saved
        spec._installed[0] = False


def test_a_broken_dspine_drafter_does_not_abort_the_engine_build(spec_tp, monkeypatch):
    """DSpine was the one accelerator site without the _optional guard: a stale KINDI_DSPINE
    directory (or a corrupt config.json inside it) raised out of _build, so `kindi serve`
    never started - the same failure class as the MTP head guard above."""
    spec, tp = spec_tp
    shard_cls = tp.Shard
    saved = (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
             spec._chunk.reserve, spec._chunk.build, spec.build)
    monkeypatch.setenv("KINDI_TC", "1")
    monkeypatch.setenv("KINDI_SPEC", "dspine")
    monkeypatch.setenv("KINDI_DSPINE", "/no/such/directory")
    monkeypatch.delenv("KINDI_SPEC_DRAFT_HEAD", raising=False)
    said = []
    try:
        spec._chunk.build = lambda eng: None                    # o_build, captured by install
        spec.build = lambda eng, log: None                      # the tc plan build: nothing to do here
        spec.install(log=said.append)
        monkeypatch.setattr(spec._dsp, "fits", lambda s: True)

        def broken(eng, d, log=print):
            raise FileNotFoundError("no config.json")

        monkeypatch.setattr(spec._dsp, "DSpine", broken)
        eng = types.SimpleNamespace(
            cards=[types.SimpleNamespace(tc_layout=True, tcb={"FEAT": object()})],
            _dsp_want=True, log=said.append, drafter=None)
        spec._chunk.build(eng)                                  # must not raise
        assert eng.dspine is None
        assert any("speculation off" in s and "/no/such/directory" in s for s in said)
    finally:
        (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
         spec._chunk.reserve, spec._chunk.build, spec.build) = saved
        spec._installed[0] = False


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


def test_hb_4096_packs_get_the_exact_activations_by_default(monkeypatch):
    # 4/4 with the high-precision activations against 3/4 without (measured on the 4060 Ti), so for an
    # HB-4096 pack the exact setting is the default instead of a flag every user has to find
    from kindi import engine as _eng
    monkeypatch.delenv("KINDI_TC_HP", raising=False)
    assert _eng.tc_hp_for({"hadamard_block": 4096}) == "1"
    assert _eng.tc_hp_for({"hadamard_block": "4096"}) == "1"
    assert _eng.tc_hp_for({"hadamard_block": 1024}) is None
    assert _eng.tc_hp_for({}) is None                 # a pack without the key folds 512
    monkeypatch.setenv("KINDI_TC_HP", "0")           # an explicit choice always wins (it then refuses)
    assert _eng.tc_hp_for({"hadamard_block": 4096}) is None


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
    monkeypatch.setenv("KINDI_KV_EXPERIMENTAL", "1")      # k8v4 is quarantined; its gate has its own suite
    assert tp.kv_mode() == want


def test_kv_mode_rejects_an_unknown_cache(spec_tp, monkeypatch):
    _, tp = spec_tp
    monkeypatch.setenv("KINDI_KV", "q4_0")
    with pytest.raises(SystemExit, match="KINDI_KV must be q8_0 or k8v4"):
        tp.kv_mode()


def test_k8v4_saves_eight_kib_per_position(spec_tp):
    """27B geometry: q8_0 K+V is 34,816 bytes/token; q8_0 K + 4-bit V is 26,624."""
    _, tp = spec_tp
    shard = types.SimpleNamespace(NATT=16, NATT_own=16, NHKV=4, HD=256, NH=24, dims={"rope_dim": 64},
                                  tp=types.SimpleNamespace(kv="q8_0"))
    q8 = tp.Shard.per_pos_bytes(shard)
    shard.tp.kv = "k8v4"
    v4 = tp.Shard.per_pos_bytes(shard)
    assert q8 - v4 == 16 * 4 * (256 - 128)       # 8,192 bytes per token
    assert q8 == 16 * 2 * 4 * (256 + 16) + 24 * 4 + (24 * 256 * 4) // 256 + 64 * 4 + 64


# ------------------------------------------- the K/V cache formats: q8_0 unchanged, k8v4 halves V
def test_kv_plan_defaults_to_the_q8_kernels(spec_tp):
    """q8_0 (the default) keeps the store and value kernels it has always run."""
    spec, _ = spec_tp
    chunk = importlib.import_module("kindi.chunk")
    q8 = types.SimpleNamespace(kv="q8_0")
    assert spec.kv_plan(q8) == ("k_store_q8_T", "k_attn_vq8w", "k_attn_vq8w_m", 1)
    assert chunk.kv_plan(q8) == ("k_store_q8_T", "k_att_flash_T", "k_att_pv_T", 1)


def test_kv_plan_selects_the_v4_kernels(spec_tp):
    """k8v4 must take the V4 store and value kernels on the chunk path and in the tensor-core
    plans; a q8 value kernel against V4 storage reads the wrong bytes."""
    spec, _ = spec_tp
    chunk = importlib.import_module("kindi.chunk")
    v4 = types.SimpleNamespace(kv="k8v4")
    assert spec.kv_plan(v4) == ("k_store_k8v4_T", "k_attn_vq4w", "k_attn_vq4w_m", 2)
    assert chunk.kv_plan(v4) == ("k_store_k8v4_T", "k_att_flash4_T", "k_att_pv4_T", 2)


def test_kv_plan_follows_the_engine_not_a_model(spec_tp):
    """The choice is the engine's kv mode, so a new pack of any geometry selects the same kernels
    (tp.py refuses k8v4 when its head size is not what the value kernels cover)."""
    spec, _ = spec_tp
    chunk = importlib.import_module("kindi.chunk")
    for shard in (types.SimpleNamespace(kv="k8v4", HD=256, NATT=16),
                  types.SimpleNamespace(kv="k8v4", HD=256, NATT=32)):
        assert spec.kv_plan(shard)[1] == "k_attn_vq4w"
        assert chunk.kv_plan(shard)[1] == "k_att_flash4_T"


def test_the_v4_kernels_are_compiled_and_wrapped(spec_tp):
    """A selection is only as good as its registration: a name missing from WRAP / OWN / FUSED
    fails at get_function on a card, so pin the lists here (no GPU needed)."""
    spec, _ = spec_tp
    chunk = importlib.import_module("kindi.chunk")
    assert "k_store_k8v4" in chunk.WRAP             # -> k_store_k8v4_T, used by the chunk and tc plans
    assert {"k_att_flash4_T", "k_att_pv4_T"} <= set(chunk.OWN)
    assert "k_attn_vq4w" in spec.WRAP               # -> d_k_attn_vq4w for the _m wrapper
    assert "k_attn_vq4w_m" in spec.FUSED


def test_plan_engine_refuses_the_9b_unless_hp(spec_tp, monkeypatch):
    """KINDI_TC=1 installs a build_plans wrapper; the 9B (HB 4096) is refused unless the
    high-precision activations are on (measured: 4/4 with KINDI_TC_HP=1, 3/4 without)."""
    spec, tp = spec_tp
    shard_cls = tp.Shard
    saved = (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
             spec._chunk.reserve, spec._chunk.build)
    monkeypatch.setenv("KINDI_TC", "1")
    try:
        spec.install(log=lambda *a, **k: None)
        with pytest.raises(SystemExit, match=r"refuses Hadamard blocks of 4096 unless KINDI_TC_HP=1"):
            tp.Shard.build_plans(_probe(4096))
    finally:
        (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
         spec._chunk.reserve, spec._chunk.build) = saved
        spec._installed[0] = False


def test_plan_engine_accepts_the_9b_with_hp(spec_tp, monkeypatch):
    """With KINDI_TC_HP=1 the HB 4096 guard passes (the tcf.cu folds dispatch
    ff_rmsfold<8>/ff_smfold<8>/ff_fold<8> for 4096) and the build proceeds."""
    spec, tp = spec_tp
    shard_cls = tp.Shard
    saved = (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
             spec._chunk.reserve, spec._chunk.build)
    monkeypatch.setenv("KINDI_TC", "1")
    monkeypatch.setattr(spec, "HP", True)          # HP is read at import time
    probe = _probe(4096)
    probe.tp = types.SimpleNamespace(W=1)          # one card: the tp.py guard accepts HB 4096 with W == 1
    try:
        spec.install(log=lambda *a, **k: None)
        with pytest.raises(AttributeError):   # the guard passed; the next statement reads s.hA
            tp.Shard.build_plans(probe)
    finally:
        (shard_cls._slices, shard_cls.build_plans, shard_cls._load,
         spec._chunk.reserve, spec._chunk.build) = saved
        spec._installed[0] = False


# ------------------------------------------------------- proportional two-card layout
def test_tp_split_mode_defaults_to_equal_and_accepts_auto(spec_tp, monkeypatch):
    _, tp = spec_tp
    monkeypatch.delenv("KINDI_TP_SPLIT", raising=False)
    assert tp.tp_split_mode() == "equal"
    for raw in ("auto", "vram", "1", "on"):
        monkeypatch.setenv("KINDI_TP_SPLIT", raw)
        assert tp.tp_split_mode() == "auto"
    for raw in ("equal", "off", "0"):
        monkeypatch.setenv("KINDI_TP_SPLIT", raw)
        assert tp.tp_split_mode() == "equal"
    monkeypatch.setenv("KINDI_TP_SPLIT", "total-vram")
    with pytest.raises(SystemExit, match="KINDI_TP_SPLIT must be auto or equal"):
        tp.tp_split_mode()


def test_vram_share_uses_usable_free_bytes(spec_tp):
    _, tp = spec_tp
    reserve = 512 << 20
    assert tp.vram_share([16 << 30, 8 << 30], reserve) == pytest.approx((15.5 / 23.0), abs=1e-6)
    assert tp.vram_share([8 << 30, 8 << 30], reserve) == 0.5
    assert tp.vram_share([0, 8 << 30], reserve) == 0.5       # an unusable card never gets a bigger share
    assert tp.vram_share([0, 0], reserve) == 0.5


def _b27_dims():
    # 27B: 16 GDN key groups, 48 value heads of width 128 (rep 3), 24 q / 4 KV heads, 64 layers, HB 1024
    return {"groups": 16, "state_size": 128, "gdn_inner": 6144, "heads": 24, "kv_heads": 4,
            "head_dim": 256, "ffn": 17408}


def test_proportional_layout_splits_where_alignment_permits(spec_tp):
    _, tp = spec_tp
    L = tp.plan_split(_b27_dims(), 1024, 248320, 2 / 3)
    # 16 GDN key groups with 3 v heads each: a 1024-point fold needs 8 key groups, so this
    # pack keeps 8/8 and 24/24; the 4 KV heads need 2 each for a 1024-point attention fold,
    # so q stays 12/12. FFN and the output head can still take the free-VRAM share.
    assert L["gdn_groups"] == [(0, 8), (8, 16)]
    assert L["gdn_v"] == [(0, 24), (24, 48)]
    assert L["attn_kv"] == [(0, 2), (2, 4)]
    assert L["attn_q"] == [(0, 12), (12, 24)]
    assert L["ffn_rows"] == L["ffn_cols"] == [(0, 11264), (11264, 17408)]
    assert L["head_split"] == [0, 165552, 248320]
    for rows in (L["gdn_groups"], L["gdn_v"], L["attn_q"], L["attn_kv"], L["ffn_rows"]):
        for a, b in rows:
            assert b > a
    for a, b in L["gdn_v"]:
        assert ((b - a) * 128) % 1024 == 0
    for a, b in L["attn_q"]:
        assert ((b - a) * 256) % 1024 == 0
    for a, b in L["ffn_rows"]:
        assert (b - a) % 1024 == 0
    assert L["head_split"][1] % 16 == 0
    assert L["head_split"][2] - L["head_split"][1] > 0
    assert tp.layout_tag(L) != tp.layout_tag(tp.plan_split(_b27_dims(), 1024, 248320, 0.5))


def test_proportional_gdn_split_for_a_two_to_one_model(spec_tp):
    # A geometry with two v heads per key group (gdn_inner 4096) can split GDN 12/4 while
    # keeping every local NV * HK width on a 1024-point boundary. The 27B (rep 3) cannot:
    # 12 key groups give 36 * 128 = 4608 features, not a whole 1024-point block.
    _, tp = spec_tp
    dims = dict(_b27_dims(), gdn_inner=4096)
    L = tp.plan_split(dims, 1024, 248320, 2 / 3)
    assert L["gdn_groups"] == [(0, 12), (12, 16)]
    assert L["gdn_v"] == [(0, 24), (24, 32)]
    assert L["attn_kv"] == [(0, 2), (2, 4)]
    assert L["attn_q"] == [(0, 12), (12, 24)]
    assert L["ffn_rows"] == [(0, 11264), (11264, 17408)]


def test_equal_fraction_keeps_the_even_blocks(spec_tp):
    _, tp = spec_tp
    L = tp.plan_split(_b27_dims(), 1024, 248320, 0.5)
    assert L["gdn_groups"] == [(0, 8), (8, 16)]
    assert L["gdn_v"] == [(0, 24), (24, 48)]
    assert L["attn_kv"] == [(0, 2), (2, 4)]     # 4 KV heads, 2-head alignment: the only valid split
    assert L["ffn_rows"] == [(0, 8192), (8192, 17408)]


def test_proportional_layout_refuses_an_unaligned_pack(spec_tp):
    _, tp = spec_tp
    dims = _b27_dims()
    dims["ffn"] += 1
    with pytest.raises(SystemExit, match="not a multiple of Hadamard block"):
        tp.plan_split(dims, 1024, 248320, 2 / 3)
    with pytest.raises(SystemExit, match="share must be between 0 and 1"):
        tp.plan_split(_b27_dims(), 1024, 248320, 1.0)


def test_shard_slices_follow_the_proportional_offsets(spec_tp):
    """Card 1 gets GDN key groups [8,16) with v heads 24..47 tiled, q heads [12,24), 2 KV heads and FFN [11264,17408)."""
    _, tp = spec_tp
    L = tp.plan_split(_b27_dims(), 1024, 248320, 2 / 3)
    sh = object.__new__(tp.Shard)
    # layers= (all four, the full-engine case) and need_head= are the fields a partial shard narrows
    sh.tp = types.SimpleNamespace(NKg=16, NVg=48, NFFg=17408, head_split=L["head_split"],
                                  layers=tuple(range(4)), need_head=True)
    sh.card = 1
    sh.k0, sh.k1 = L["gdn_groups"][1]
    sh.v0, sh.v1 = L["gdn_v"][1]
    sh.q0, sh.q1 = L["attn_q"][1]
    sh.kv0, sh.kv1 = L["attn_kv"][1]
    sh.NK, sh.NV = sh.k1 - sh.k0, sh.v1 - sh.v0
    sh.NH, sh.NHKV = sh.q1 - sh.q0, sh.kv1 - sh.kv0
    sh.g0, sh.g1 = L["ffn_rows"][1]
    sh.d0, sh.d1 = L["ffn_cols"][1]
    sh.NFF = sh.g1 - sh.g0
    sh.HK, sh.HD, sh.NL, sh.D = 128, 256, 4, 5120
    sh.dims = {"full_attention_interval": 4}
    tern, flt = tp.Shard._slices(sh)
    qkv = tern["blk.0.attn_qkv.weight"][0]
    assert qkv[:2].tolist() == [1024, 1025]          # q group 8
    assert qkv[-2:].tolist() == [10238, 10239]       # the last tiled v head
    assert flt["blk.0.ssm_alpha.weight"][1].tolist() == [8, 9, 10, 11, 12, 13, 14, 15,
                                                         24, 25, 26, 27, 28, 29, 30, 31,
                                                         40, 41, 42, 43, 44, 45, 46, 47]
    assert tern["blk.0.ssm_out.weight"][1] == (3072, 6144)
    assert tern["blk.3.attn_q.weight"][0][:2].tolist() == [6144, 6145]
    assert tern["blk.3.attn_q.weight"][0][-2:].tolist() == [12286, 12287]
    assert tern["blk.3.attn_k.weight"][0][:2].tolist() == [512, 513]
    assert tern["blk.3.attn_k.weight"][0][-1] == 1023
    assert tern["blk.3.attn_output.weight"][1] == (3072, 6144)
    assert tern["blk.0.ffn_gate.weight"][0][0] == 11264
    assert tern["blk.0.ffn_gate.weight"][0][-1] == 17407
    assert tern["blk.0.ffn_down.weight"][1] == (11264, 17408)
    assert tern["output.weight"][0][0] == 165552
    assert tern["output.weight"][0][-1] == 248319
    assert sh.cols == {"sV": (3072, 6144), "sA": (3072, 6144), "sF": (11264, 17408)}


def test_engine_fingerprint_includes_the_tp_layout(spec_tp):
    from kindi import cache
    _, tp = spec_tp
    a = types.SimpleNamespace(W=2, split_layout={"gdn_groups": [(0, 8), (8, 16)]})
    b = types.SimpleNamespace(W=2, split_layout={"gdn_groups": [(0, 12), (12, 16)]})
    fa, da = cache.engine_fingerprint(a, {"head": "exact"})
    fb, db = cache.engine_fingerprint(b, {"head": "exact"})
    assert fa != fb
    assert da["tp_split"] == a.split_layout
    assert db["tp_split"] == b.split_layout


def test_split_flash_decode_is_disabled_for_the_padded_4b(spec_tp):
    """KINDI_FD is not safe for the 4B's 2560-wide products; the other Bonsai widths stay on the split path."""
    _, tp = spec_tp
    assert tp.fd_supported({"width": 2560}) is False
    assert tp.fd_supported({"width": 2048}) is True
    assert tp.fd_supported({"width": 4096}) is True
    assert tp.fd_supported({"width": 5120}) is True


def test_streamed_g5_cache_is_rejected_when_a_tied_head_key_is_missing(spec_tp, tmp_path):
    """A g5 cache written before the tied-head alias has output.weight, not token_embd.weight.
    The streamed path must refuse it so Shard._load rebuilds the cache instead of building a plan
    that reads a missing tensor."""
    spec, _ = spec_tp
    (tmp_path / "g5_aaaaaaaaaaaa-w1-c0.json").write_text(json.dumps({"output.weight": [0, 1, 1]}))
    (tmp_path / "g5n_aaaaaaaaaaaa-w1-c0.bin").write_bytes(b"\0\0\0\0")
    shard = types.SimpleNamespace(
        card=0,
        tp=types.SimpleNamespace(pack=str(tmp_path), man={"source_sha256": "a" * 64}, W=1, split_tag=""),
        _slices=lambda: ({"token_embd.weight": (None, None)}, {}))
    assert spec.load_stream(shard) is False
