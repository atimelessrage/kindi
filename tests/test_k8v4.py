# SPDX-License-Identifier: AGPL-3.0-only
"""Conformance for the experimental K8/V4 K/V cache (`KINDI_KV=k8v4`).

The 4-bit V cache is lossy: on the 2B, `kindi verify` matches llama.cpp's fp16 K/V reference on 2 of
4 prompts (first difference at tokens 35-36), while the q8_0 default is 4/4. A cache that changes the
model's output must not be one environment variable away, so k8v4 is quarantined:

  * `kv_mode()` refuses `KINDI_KV=k8v4` unless `KINDI_KV_EXPERIMENTAL=1`, and the message names the
    flag and the measured loss;
  * the opt-in still selects the V4 store and value kernels and the halved V row stride, so the flag
    is not a no-op, and q8_0 keeps its kernels;
  * a state saved with k8v4 can never be loaded into a q8_0 engine: the cache fingerprint carries the
    K/V format;
  * the flag is listed in the generated configuration reference.

No GPU, CUDA toolkit or model pack is needed: these are the decisions taken before any buffer exists.
The GPU-side conformance (V4 store vs NumPy, chunk V4 vs per-token V4, save/restart/restore) lives in
`bench-logs/check_k8v4_chunk.py`, `bench-logs/test_k8v4_cache.py` and `bench-logs/test_k8v4_dist_cache.py`.
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
    """kindi.tp, spec, chunk, cache and config_reference, imported without a GPU."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        _install_cupy_stub()
    return tuple(importlib.import_module("kindi." + name)
                 for name in ("tp", "spec", "chunk", "cache", "config_reference"))


# ------------------------------------------------------------------- the quarantine (the gate itself)
def test_k8v4_is_refused_without_the_experimental_flag(modules, monkeypatch):
    tp = modules[0]
    monkeypatch.setenv("KINDI_KV", "k8v4")
    monkeypatch.delenv("KINDI_KV_EXPERIMENTAL", raising=False)
    with pytest.raises(SystemExit) as err:
        tp.kv_mode()
    msg = str(err.value)
    assert "KINDI_KV_EXPERIMENTAL=1" in msg          # names the way out
    assert "experimental" in msg and "lossy" in msg  # says what it is
    assert "token-exact" in msg and "q8_0" in msg    # says what it costs and what the default is


@pytest.mark.parametrize("flag", ["0", "", "true", "yes", "2"])
def test_only_the_exact_opt_in_enables_k8v4(modules, monkeypatch, flag):
    # an explicit flag, not a truthy-looking one: any other value keeps the quarantine
    tp = modules[0]
    monkeypatch.setenv("KINDI_KV", "k8v4")
    monkeypatch.setenv("KINDI_KV_EXPERIMENTAL", flag)
    with pytest.raises(SystemExit, match="KINDI_KV_EXPERIMENTAL=1"):
        tp.kv_mode()


def test_the_opt_in_enables_both_aliases(modules, monkeypatch):
    tp = modules[0]
    monkeypatch.setenv("KINDI_KV_EXPERIMENTAL", "1")
    for value in ("k8v4", "K8_V4", " k8v4 "):
        monkeypatch.setenv("KINDI_KV", value)
        assert tp.kv_mode() == "k8v4"


def test_q8_0_needs_no_opt_in(modules, monkeypatch):
    # the supported default must keep working with the flag absent
    tp = modules[0]
    monkeypatch.delenv("KINDI_KV_EXPERIMENTAL", raising=False)
    monkeypatch.setenv("KINDI_KV", "q8_0")
    assert tp.kv_mode() == "q8_0"
    monkeypatch.delenv("KINDI_KV", raising=False)
    assert tp.kv_mode() == "q8_0"


# ------------------------------------------------- the opt-in changes the arithmetic, not just a flag
def test_the_opt_in_selects_the_v4_kernels(modules, monkeypatch):
    tp, spec, chunk = modules[:3]
    monkeypatch.setenv("KINDI_KV_EXPERIMENTAL", "1")
    monkeypatch.setenv("KINDI_KV", "k8v4")
    eng = types.SimpleNamespace(kv=tp.kv_mode())
    assert eng.kv == "k8v4"
    # the store must write V4, the chunk and the tensor-core plans must read it, and V's row is halved
    assert spec.kv_plan(eng) == ("k_store_k8v4_T", "k_attn_vq4w", "k_attn_vq4w_m", 2)
    assert chunk.kv_plan(eng) == ("k_store_k8v4_T", "k_att_flash4_T", "k_att_pv4_T", 2)
    eng.kv = "q8_0"
    assert spec.kv_plan(eng) == ("k_store_q8_T", "k_attn_vq8w", "k_attn_vq8w_m", 1)
    assert chunk.kv_plan(eng) == ("k_store_q8_T", "k_att_flash_T", "k_att_pv_T", 1)


def test_a_v4_state_can_never_be_loaded_into_a_q8_engine(modules):
    # saved openings are keyed by the engine fingerprint; the K/V format is part of it, so a k8v4
    # opening is never resumed by a q8_0 engine (and vice versa)
    cache = modules[3]
    card = types.SimpleNamespace(chunk={}, g5l=False, g5=False)
    fp4, d4 = cache.engine_fingerprint(types.SimpleNamespace(cards=[card], kv="k8v4"), {"head": "exact"})
    fp8, d8 = cache.engine_fingerprint(types.SimpleNamespace(cards=[card], kv="q8_0"), {"head": "exact"})
    assert d4["kv"] == "k8v4" and d8["kv"] == "q8_0"
    assert fp4 != fp8


def test_the_quarantine_is_documented(modules):
    # kv_mode reads KINDI_KV_EXPERIMENTAL, so the generated reference must list it
    ref = Path(modules[4].REFERENCE).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "| `KINDI_KV` |" in ref
    assert "| `KINDI_KV_EXPERIMENTAL` |" in ref
