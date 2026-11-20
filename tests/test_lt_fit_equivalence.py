"""The report and the engine must plan the same Lightning placement from the same spec.

`kindi plan`'s slots line once drifted from the engine because the fit arithmetic existed twice
(`plan._lt_fit` and `lightning._plan`). It lives in `kindi/ltfit.py` now; these tests pin that both
callers produce the same number, on synthetic specs small enough to compute by hand, and that neither
can grow its own copy again without failing here.
"""
import types

import pytest

from kindi import lightning, ltfit, plan

def _expert_bytes(spec):
    """One expert, by the engine's own rule (up + down, Q2_0): 2 * n_ff * n_embd // 64 * 18."""
    return 2 * spec.n_ff * spec.n_embd // 64 * 18


def _spec(n_moe=8, n_exp=32, n_used=4, n_attn=2, n_kv=2, hd=8):
    """The fields `lightning._plan` and `plan._lt_fit` read, and nothing else."""
    return types.SimpleNamespace(
        n_layer=n_attn + n_moe, n_exp=n_exp, n_used=n_used, n_ff=256, n_embd=128,
        n_kv=n_kv, hd=hd, n_sh=4, d_state=16, d_xbc=32,
        kind=["attn"] * n_attn + ["moe"] * n_moe, ctx_train=131072)


def _kv_bytes(spec, ctx):
    """`_plan` computes K/V as attn layers * 2 * n_kv * hd * 2 bytes per position."""
    return spec.kind.count("attn") * 2 * spec.n_kv * spec.hd * 2 * ctx


def _engine(spec, free, ctx):
    """A Lightning instance without a device: `_plan`'s arithmetic, none of its allocations."""
    eng = lightning.Engine.__new__(lightning.Engine)
    eng.s, eng.ctx, eng.cp = spec, ctx, None
    eng.dev = types.SimpleNamespace(id=0)
    eng.tinfo = {}                          # dense bytes = 0
    eng._buf_bytes = lambda: 0
    eng._reserve = lambda: 0
    eng._plan_weighted = lambda *a, **k: None
    eng.log = lambda *a, **k: None
    return eng


def _plan_room(spec, room_layers, ctx, dcache, monkeypatch):
    """Run both sides on the same room, measured in expert-layers (room_layers * one expert).

    Returns (report_n_card, engine_n_card, report_room)."""
    eb = _expert_bytes(spec)
    free = room_layers * eb + _kv_bytes(spec, ctx)
    eng = _engine(spec, free, ctx)
    monkeypatch.setattr(lightning, "_free_bytes", lambda cp, devs: [free])
    moe_layers = [il for il, k in enumerate(spec.kind) if k == "moe"]
    report_n, report_room = plan._lt_fit(spec, free, 0, _kv_bytes(spec, ctx), 0, 0, 0, eb,
                                        moe_layers, dcache=dcache)
    eng._plan(None)
    return report_n, eng.n_card, report_room


@pytest.mark.parametrize("room_layers,dcache,expected", [
    (300, 10, 32),      # every expert of every layer fits: 300 // 8 = 37, capped at n_exp
    (180, 10, 6),       # staging (2 * 32) + slots (8 * 10) off the top, then (180 - 144) // 6
    (180, 4, 14),       # n_used is the engine's slot floor: (180 - 64 - 8 * 4) // 6
    (8, 10, 4),         # a two-layer MoE model: no staging split, 8 // 2, no division by zero
])
def test_report_and_engine_plan_the_same_expert_count(room_layers, dcache, expected, monkeypatch):
    n_moe = 2 if room_layers == 8 else 8
    spec = _spec(n_moe=n_moe)
    monkeypatch.setenv("KINDI_LT_DCACHE", str(dcache))          # the engine reads it at plan time
    report_n, engine_n, room = _plan_room(spec, room_layers, ctx=512, dcache=dcache,
                                          monkeypatch=monkeypatch)
    assert room == room_layers * _expert_bytes(spec)
    assert report_n == expected, report_n
    assert engine_n == report_n            # exact, not approximate: same arithmetic


def test_both_refuse_the_same_room(monkeypatch):
    """4 MiB cannot hold the staging buffers, let alone 4 experts per layer: report says None, the
    engine raises. The two failure boundaries are the same line in the same helper."""
    spec = _spec()
    eb = _expert_bytes(spec)
    free = 4 * eb + _kv_bytes(spec, 512)
    eng = _engine(spec, free, 512)
    monkeypatch.setattr(lightning, "_free_bytes", lambda cp, devs: [free])
    moe_layers = [il for il, k in enumerate(spec.kind) if k == "moe"]
    report_n, _room = plan._lt_fit(spec, free, 0, _kv_bytes(spec, 512), 0, 0, 0, eb, moe_layers)
    assert report_n is None
    with pytest.raises(MemoryError):
        eng._plan(None)


def test_the_slots_default_is_one_number(monkeypatch):
    """`KINDI_LT_DCACHE` is read at call time by one function, so the engine and the report agree
    even when the environment changes after import (the old module constant did not)."""
    spec = _spec(n_used=4)
    monkeypatch.setenv("KINDI_LT_DCACHE", "6")
    assert ltfit.dcache_default(4) == 6            # env above n_used
    assert ltfit.dcache_default(12) == 12          # n_used is a floor
    rept, engine_n, _room = _plan_room(spec, 300, ctx=512, dcache=None, monkeypatch=monkeypatch)
    assert rept == engine_n == 32
    free = 300 * _expert_bytes(spec) + _kv_bytes(spec, 512)
    eng = _engine(spec, free, 512)
    monkeypatch.setattr(lightning, "_free_bytes", lambda cp, devs: [free])
    eng._plan(None)
    assert eng.dcache == 6


def test_neither_caller_has_its_own_formula(monkeypatch):
    """One implementation: point the helper at a sentinel and both sides must return it, which a
    second copy of the arithmetic (the bug this replaces) could not do."""
    calls = []

    def sentinel(room, n_moe, n_exp, n_used, eb, dcache):
        calls.append((room, n_moe, n_exp, n_used, eb, dcache))
        return 7

    monkeypatch.setattr(ltfit, "fit_experts", sentinel)
    spec = _spec()
    rept, engine_n, _room = _plan_room(spec, 300, ctx=512, dcache=10, monkeypatch=monkeypatch)
    assert rept == engine_n == 7
    assert len(calls) == 2, calls
    assert all(c[3] == spec.n_used and c[4] == _expert_bytes(spec) for c in calls)


def test_rate_slots_prices_the_refill_path_from_measured_rates():
    """One token's `n_used` experts plus one lookahead per measured tier-time multiple: with
    13,456.7 MiB/s RAM, 2,509.8 MiB/s disk and 6.0 GiB/s PCIe, a 1 MiB expert needs four working
    sets (4 slots) and four used experts give 16 slots."""
    rates = {"ram_bytes_s": 13456.7 * 2 ** 20, "disk_bytes_s": 2509.8 * 2 ** 20,
             "pcie_bytes_s": 6.0 * 2 ** 30}
    assert ltfit.rate_slots(4, 256, 2 ** 20, rates, room=1 << 40, n_moe=8) == 16
    assert ltfit.rate_slots(4, 256, 2 ** 20, {"ram_bytes_s": rates["ram_bytes_s"],
                                              "pcie_bytes_s": rates["pcie_bytes_s"]},
                            room=1 << 40, n_moe=8) == 8
    assert ltfit.rate_slots(4, 256, 2 ** 20, {"pcie_bytes_s": rates["pcie_bytes_s"]},
                            room=1 << 40, n_moe=8) == 4
    assert ltfit.rate_slots(4, 256, 2 ** 20, {}, room=1 << 40, n_moe=8) is None


def test_rate_slots_never_exceeds_what_the_room_can_hold():
    """The prediction is capped by the same fit that the report and engine use, and a room that
    cannot hold `n_used` residents has no slot answer."""
    rates = {"ram_bytes_s": 13456.7 * 2 ** 20, "disk_bytes_s": 2509.8 * 2 ** 20,
             "pcie_bytes_s": 6.0 * 2 ** 30}
    assert ltfit.rate_slots(4, 32, 1000, rates, room=98000, n_moe=4) == 6
    assert ltfit.rate_slots(4, 32, 1000, rates, room=4 * 1000, n_moe=4) is None


def test_the_slots_cost_is_the_report_counterfactual(monkeypatch):
    """The number the slots line prints is `_lt_fit(dcache=0)` against `_lt_fit(dcache=engine)`:
    on this spec 10 slots per layer cost 13 experts per layer (19 -> 6), not one."""
    monkeypatch.setenv("KINDI_LT_DCACHE", "10")
    spec = _spec()
    eb = _expert_bytes(spec)
    free = 180 * eb + _kv_bytes(spec, 512)
    moe_layers = [il for il, k in enumerate(spec.kind) if k == "moe"]
    with_slots, _ = plan._lt_fit(spec, free, 0, _kv_bytes(spec, 512), 0, 0, 0, eb, moe_layers)
    without, _ = plan._lt_fit(spec, free, 0, _kv_bytes(spec, 512), 0, 0, 0, eb, moe_layers, dcache=0)
    assert (with_slots, without) == (6, 19)
