# SPDX-License-Identifier: AGPL-3.0-only
"""Conformance for the experimental FP8 E4M3 K/V cache (`KINDI_KV=fp8`).

The FP8 cache is a lossy candidate whose quality is decided by the KLD gate
(REQ-0014): it must prove KLD no worse than q8_0 at 2K/8K, 4/4 exactness against
the fp16 reference at verify length on Ada, and a measured peak against q8_0
before it can be called distribution-lossless. Until then it is quarantined:

  * `kv_mode()` refuses `KINDI_KV=fp8` unless `KINDI_KV_EXPERIMENTAL=1`;
  * the decode kernels are distinct (`k_store_fp8`, `k_attn_sf8t`, `k_attn_vf8w`),
    and the chunk/tensor-core plan builders refuse fp8 rather than reading its
    bytes as q8_0;
  * the fp8 variant is registered with the KLD gate;
  * the K/V format is part of the prompt-cache fingerprint, so an fp8 opening
    is never resumed as q8_0.

No GPU, CUDA toolkit or model pack is needed: these are the decisions taken
before any buffer exists. The GPU-side gate (KLD + verify + sampled peak on
the 2B) is run on the rig and recorded in REQ-0014.
"""
import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_cupy_stub():
    """The minimum `cupy` surface kindi.tp/spec import at module level (CI has no CuPy)."""
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
def modules():
    """kindi.tp, spec, chunk, cache, support_matrix and config_reference, without a GPU."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        _install_cupy_stub()
    return tuple(importlib.import_module("kindi." + name)
                 for name in ("tp", "spec", "chunk", "cache", "support_matrix", "config_reference"))


def test_fp8_is_refused_without_the_experimental_flag(modules, monkeypatch):
    tp = modules[0]
    monkeypatch.setenv("KINDI_KV", "fp8")
    monkeypatch.delenv("KINDI_KV_EXPERIMENTAL", raising=False)
    with pytest.raises(SystemExit) as err:
        tp.kv_mode()
    msg = str(err.value)
    assert "KINDI_KV_EXPERIMENTAL=1" in msg
    assert "experimental" in msg and "KLD" in msg and "q8_0" in msg


@pytest.mark.parametrize("flag", ["0", "", "true", "yes", "2"])
def test_only_the_exact_opt_in_enables_fp8(modules, monkeypatch, flag):
    tp = modules[0]
    monkeypatch.setenv("KINDI_KV", "fp8")
    monkeypatch.setenv("KINDI_KV_EXPERIMENTAL", flag)
    with pytest.raises(SystemExit, match="KINDI_KV_EXPERIMENTAL=1"):
        tp.kv_mode()


def test_the_opt_in_enables_the_fp8_aliases(modules, monkeypatch):
    tp = modules[0]
    monkeypatch.setenv("KINDI_KV_EXPERIMENTAL", "1")
    for value in ("fp8", "FP8", " f8_e4m3 ", "f8_e4m3fn"):
        monkeypatch.setenv("KINDI_KV", value)
        assert tp.kv_mode() == "fp8"


def test_q8_0_needs_no_opt_in(modules, monkeypatch):
    tp = modules[0]
    monkeypatch.delenv("KINDI_KV_EXPERIMENTAL", raising=False)
    monkeypatch.setenv("KINDI_KV", "q8_0")
    assert tp.kv_mode() == "q8_0"


def test_fp8_is_decode_only_in_the_plan_builders(modules):
    tp, spec, chunk = modules[:3]
    eng = types.SimpleNamespace(kv="fp8")
    with pytest.raises(SystemExit, match="decode attention only"):
        chunk.kv_plan(eng)
    with pytest.raises(SystemExit, match="decode attention only"):
        spec.kv_plan(eng)
    # q8_0 keeps its kernels unchanged
    eng.kv = "q8_0"
    assert spec.kv_plan(eng) == ("k_store_q8_T", "k_attn_vq8w", "k_attn_vq8w_m", 1)
    assert chunk.kv_plan(eng) == ("k_store_q8_T", "k_att_flash_T", "k_att_pv_T", 1)


def test_fp8_is_registered_with_the_kld_gate(modules):
    kld = importlib.import_module("kindi.kld")
    assert "tp-fp8" in kld.VARIANTS
    assert "tp-fp8" in kld.__doc__


def test_fp8_modes_come_from_tp(modules):
    sm = modules[4]
    kv = sm.kv_modes()
    assert kv["default"] == "q8_0"
    assert "fp8" in kv["modes"] and kv["gated"]


def test_an_fp8_state_can_never_be_loaded_into_a_q8_engine(modules):
    cache = modules[3]
    card = types.SimpleNamespace(chunk={}, g5l=False, g5=False)
    fp, dfp = cache.engine_fingerprint(types.SimpleNamespace(cards=[card], kv="fp8"), {"head": "exact"})
    fq, dq = cache.engine_fingerprint(types.SimpleNamespace(cards=[card], kv="q8_0"), {"head": "exact"})
    assert dfp["kv"] == "fp8" and dq["kv"] == "q8_0"
    assert fp != fq


def test_the_gate_is_documented(modules):
    ref = Path(modules[5].REFERENCE).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "| `KINDI_KV` |" in ref
    assert "| `KINDI_KV_EXPERIMENTAL` |" in ref
