# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic CUDA checks for the native FP8 (E4M3/E5M2) tensor-core prompt GEMM.

No real FP8 Gemma 4 GGUF is available, so the kernels are checked against a
host reference built from the same quantized inputs.  The tests need Ada
(sm_89) or Blackwell (sm_100/sm_120) and are skipped elsewhere.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from kindi.dtypes import FP8_E4M3, FP8_E5M2, fp8_to_f32
from kindi.gemma4 import Kernels, Ops, fp8_kernels


@pytest.fixture(scope="module")
def fp8():
    try:
        cp.cuda.Device().use()
        kernels = fp8_kernels()
    except Exception as exc:                       # no driver/device on a CPU-only host
        pytest.skip("CUDA is unavailable: %s" % exc)
    if kernels is None:
        pytest.skip("native FP8 tensor cores need Ada (sm_89) or Blackwell")
    return kernels


def _finite_bytes(kind):
    values = fp8_to_f32(np.arange(256, dtype=np.uint8), kind)
    return np.flatnonzero(np.isfinite(values)).astype(np.uint8)


def _quantize(fp8, x, kind):
    nt, cols = x.shape
    q = cp.empty((nt, cols), dtype=cp.uint8)
    s = cp.empty(nt, dtype=cp.float32)
    fp8["quant"][kind]((nt,), (256,),
                       (cp.asarray(x, dtype=cp.float32), q, s,
                        np.int32(cols), np.int32(nt)))
    return q, s


def _dequant_rows(q, s, kind, cols):
    return fp8_to_f32(cp.asnumpy(q), kind).reshape(q.shape[0], cols) * cp.asnumpy(s)[:, None]


def _weights(rows, cols, kind, seed):
    rng = np.random.RandomState(seed)
    return rng.choice(_finite_bytes(kind), size=rows * cols).astype(np.uint8)


@pytest.mark.parametrize("kind", [FP8_E4M3, FP8_E5M2])
def test_quantization_fills_the_format_range(fp8, kind):
    nt, cols = 5, 64
    x = np.linspace(-1.0, 1.0, nt * cols, dtype=np.float32).reshape(nt, cols)
    q, s = _quantize(fp8, x, kind)
    got = _dequant_rows(q, s, kind, cols)
    maxv = 448.0 if kind == FP8_E4M3 else 57344.0
    assert np.allclose(np.abs(got).max(axis=1), np.abs(x).max(axis=1), rtol=0.02, atol=0.0)
    assert np.allclose(got, x, rtol=0.13, atol=1e-5 * maxv)


@pytest.mark.parametrize("kind", [FP8_E4M3, FP8_E5M2])
@pytest.mark.parametrize("rows,cols,nt", [(64, 64, 64), (37, 96, 5), (130, 128, 70)])
def test_fp8_tensor_core_gemm_matches_the_quantized_reference(fp8, kind, rows, cols, nt):
    rs = np.random.RandomState(rows * 131 + cols * 7 + nt)
    w = _weights(rows, cols, kind, seed=rows + cols + nt)
    x = (rs.standard_normal((nt, cols)) * 8.0).astype(np.float32)
    q, s = _quantize(fp8, x, kind)
    out = cp.empty((nt, rows), dtype=cp.float32)
    fp8["mm"][kind](((rows + 63) // 64, (nt + 63) // 64), (256,),
                    (cp.asarray(w), q, s, out, np.int32(rows), np.int32(cols), np.int32(nt)))
    got = cp.asnumpy(out)
    wf = fp8_to_f32(w, kind).reshape(rows, cols).astype(np.float32)
    ref = _dequant_rows(q, s, kind, cols) @ wf.T
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() <= 2e-3 * np.abs(ref).max()


@pytest.mark.parametrize("kind", [FP8_E4M3, FP8_E5M2])
def test_fp8_gemm_writes_every_tile_position(fp8, kind):
    rows, cols, nt = 130, 96, 70
    one = 0x38 if kind == FP8_E4M3 else 0x3C
    w = np.full(rows * cols, one, dtype=np.uint8)
    x = np.ones((nt, cols), dtype=np.float32)
    q, s = _quantize(fp8, x, kind)
    out = cp.empty((nt, rows), dtype=cp.float32)
    fp8["mm"][kind](((rows + 63) // 64, (nt + 63) // 64), (256,),
                    (cp.asarray(w), q, s, out, np.int32(rows), np.int32(cols), np.int32(nt)))
    assert np.allclose(cp.asnumpy(out), cols, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("kind", [FP8_E4M3, FP8_E5M2])
def test_ops_linear_batch_uses_the_native_kernel(fp8, kind):
    rows, cols, nt = 64, 64, 8
    rs = np.random.RandomState(23)
    w = _weights(rows, cols, kind, seed=5)
    x = (rs.standard_normal((nt, cols)) * 4.0).astype(np.float32)
    ops = Ops.__new__(Ops)
    ops.cp, ops.k, ops.fp8 = cp, Kernels(), fp8
    got = ops.linear_batch(cp.asarray(w), cp.asarray(x), rows, cols, kind)
    q, s = _quantize(fp8, x, kind)
    ref = _dequant_rows(q, s, kind, cols) @ fp8_to_f32(w, kind).reshape(rows, cols).T
    assert np.abs(cp.asnumpy(got) - ref).max() <= 2e-3 * np.abs(ref).max()


@pytest.mark.parametrize("kind", [FP8_E4M3, FP8_E5M2])
def test_ops_linear_batch_falls_back_without_the_native_kernel(kind):
    try:
        from cupy_backends.cuda.libs import cublas  # noqa: F401
    except Exception as exc:
        pytest.skip("cuBLAS is unavailable: %s" % exc)
    rows, cols, nt = 64, 64, 5
    w = _weights(rows, cols, kind, seed=9)
    x = np.random.RandomState(11).standard_normal((nt, cols)).astype(np.float32)
    ops = Ops.__new__(Ops)
    ops.cp, ops.k, ops.fp8 = cp, Kernels(), None
    got = cp.asnumpy(ops.linear_batch(cp.asarray(w), cp.asarray(x), rows, cols, kind))
    ref = x @ fp8_to_f32(w, kind).reshape(rows, cols).T
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max())
