# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic CUDA checks for the Gemma 4 BF16/FP8 decode and embedding kernels.

No FP8 or BF16 Gemma 4 GGUF is available, so these tests compare the CUDA
conversion with the host reference on every FP8 byte pattern and on a small
matrix. They are skipped when CuPy or a CUDA device is unavailable.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from kindi.dtypes import BF16, FP8_E4M3, FP8_E5M2, bf16_to_f32, fp8_to_f32
from kindi.gemma4 import Kernels, Ops


@pytest.fixture(scope="module")
def kernels():
    try:
        cp.cuda.Device().use()
        return Kernels()
    except Exception as exc:                       # no driver/device on a CPU-only host
        pytest.skip("CUDA is unavailable: %s" % exc)


@pytest.fixture(scope="module")
def ops(kernels):
    # Do not run Ops.__init__: it probes cuBLAS, which is absent on some hosts. The dense
    # dispatch under test only needs the CuPy module, the kernels and the feature switches.
    o = Ops.__new__(Ops)
    o.cp, o.k = cp, kernels
    o.mmq_on = o.kq_dp4a = o.q5_dp4a = o.dp4a = False
    return o


def _dequant(kernels, raw, kind):
    out = cp.empty(raw.size, dtype=cp.float32)
    kernels.k_g4_fp8_dequant(((raw.size + 255) // 256,), (256,),
                              (raw, out, np.int32(raw.size), np.int32(kind)))
    return cp.asnumpy(out)


def _same(got, want):
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.array_equal(np.isinf(got), np.isinf(want))
    assert np.allclose(got[np.isfinite(got)], want[np.isfinite(want)], rtol=0, atol=0)


@pytest.mark.parametrize("kind", [FP8_E4M3, FP8_E5M2])
def test_cuda_fp8_dequant_matches_the_host_codec(kernels, kind):
    raw = cp.arange(256, dtype=cp.uint8)
    got = _dequant(kernels, raw, kind)
    _same(got, fp8_to_f32(np.arange(256, dtype=np.uint8), kind))


@pytest.mark.parametrize("kind", [FP8_E4M3, FP8_E5M2])
def test_cuda_fp8_gemv_matches_the_host_codec(kernels, kind):
    rows, cols = 5, 64
    finite = np.array([b for b in range(256) if np.isfinite(fp8_to_f32(np.array([b], dtype=np.uint8), kind)[0])],
                      dtype=np.uint8)
    raw = cp.asarray(np.resize(finite, rows * cols).reshape(rows, cols))
    x = cp.asarray(np.random.RandomState(9).standard_normal(cols).astype(np.float32))
    out = cp.empty(rows, dtype=cp.float32)
    kernels.k_g4_fp8_mv((rows,), (256,), (raw, x, out, np.int32(rows), np.int32(cols), np.int32(kind)))
    w = fp8_to_f32(cp.asnumpy(raw), kind)
    got = cp.asnumpy(out)
    assert np.allclose(got, w @ cp.asnumpy(x), rtol=1e-6, atol=1e-5)


def test_cuda_embedding_supports_bf16_and_fp8(kernels):
    width = 32
    bf = (np.arange(2 * width, dtype=np.uint16) * 0x0101).reshape(2, width)
    fp = (np.arange(2 * width, dtype=np.uint8).reshape(2, width) | 0x38)
    for kind, raw, ref in ((BF16, bf, bf16_to_f32(bf)),
                           (FP8_E4M3, fp, fp8_to_f32(fp, FP8_E4M3)),
                           (FP8_E5M2, fp, fp8_to_f32(fp, FP8_E5M2))):
        packed = cp.asarray(raw.view(np.uint8).reshape(-1))
        out = cp.empty(width, dtype=cp.float32)
        kernels.k_g4_embed((1,), (256,), (packed, out, np.int32(1), np.int32(width), np.int32(kind)))
        _same(cp.asnumpy(out), ref[1])


def _dense_case(kind):
    rng = np.random.RandomState(12)
    rows, cols, nt = 4, 64, 2
    finite = np.arange(256, dtype=np.uint8)
    if kind != BF16:
        finite = np.array([b for b in finite if np.isfinite(fp8_to_f32(np.array([b]), kind)[0])], dtype=np.uint8)
    raw = np.resize(finite, rows * cols).reshape(rows, cols)
    if kind == BF16:
        packed_np = (raw.astype(np.uint16) << 8).view(np.uint8).reshape(rows, -1)
        w = bf16_to_f32(raw.astype(np.uint16) << 8)
    else:
        packed_np = raw
        w = fp8_to_f32(raw, kind)
    packed = cp.asarray(packed_np.reshape(-1))
    x = cp.asarray(rng.standard_normal(cols).astype(np.float32))
    xs = cp.asarray(rng.standard_normal((nt, cols)).astype(np.float32))
    return packed, w, x, xs


@pytest.mark.parametrize("kind", [BF16, FP8_E4M3, FP8_E5M2])
def test_ops_dense_gemv_and_qkv_dispatch(ops, kind):
    rows, cols = 4, 64
    packed, w, x, _ = _dense_case(kind)
    got = cp.asnumpy(ops.linear(packed, x, rows, cols, kind))
    assert np.allclose(got, w @ cp.asnumpy(x), rtol=1e-6, atol=1e-5)

    qkv = cp.asnumpy(ops.qkv(packed, packed, packed, x, rows, rows, cols, kind, kind, kind))
    want = np.concatenate((w @ cp.asnumpy(x), w @ cp.asnumpy(x), w @ cp.asnumpy(x)))
    assert np.allclose(qkv, want, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("kind", [BF16, FP8_E4M3, FP8_E5M2])
def test_ops_dense_batch_dispatch(ops, kind):
    try:
        from cupy_backends.cuda.libs import cublas  # noqa: F401
    except Exception as exc:
        pytest.skip("cuBLAS is unavailable: %s" % exc)
    rows, cols = 4, 64
    packed, w, _, xs = _dense_case(kind)
    got_b = cp.asnumpy(ops.linear_batch(packed, xs, rows, cols, kind))
    assert np.allclose(got_b, cp.asnumpy(xs) @ w.T, rtol=1e-6, atol=1e-5)

    qkv_b = cp.asnumpy(ops.qkv_batch(packed, packed, packed, xs, rows, rows, cols, kind, kind, kind))
    want_b = np.concatenate((cp.asnumpy(xs) @ w.T,) * 3, axis=1)
    assert np.allclose(qkv_b, want_b, rtol=1e-6, atol=1e-5)
