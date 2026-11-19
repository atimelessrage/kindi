"""The CPU Q2_0 expert lane reproduces the kernel's per-run FMA order (CPU only).

No GPU and no pack: a tiny synthetic matrix is checked against a pure-Python
reference built from ``math.fma`` and the same five-step XOR reduction. The
real kernels are compared in ``tools/q2_expert_lane.py``, which needs a card.
"""
import math

import numpy as np
import pytest

from kindi import moe_expert_lane as lane

pytestmark = pytest.mark.skipif(not lane.available(), reason="numba with its fmaf intrinsic is required")


def _random(experts=2, rows=32, cols=64, seed=5):
    rng = np.random.default_rng(seed)
    runs, nb = cols // 32, cols // 64
    codes = rng.integers(0, 3, size=(experts, rows, runs * 2), dtype=np.uint32)
    scales = (rng.standard_normal((experts, rows, nb)) * 0.01).astype(np.float16)
    q = rng.integers(-127, 128, size=(runs, 32), dtype=np.int8)
    xq = np.empty((runs, 8), np.int32)
    for r in range(runs):
        for word in range(8):
            g, k = word // 4, word % 4
            w = 0
            for j in range(4):
                w |= (int(q[r, g * 16 + 4 * j + k]) & 255) << (8 * j)
            xq[r, word] = np.int32(w if w < 2 ** 31 else w - 2 ** 32)
    dx = (rng.standard_normal(runs) * 0.02).astype(np.float32)
    sx = q.astype(np.int32).sum(1).astype(np.int32)
    return codes, scales, xq.reshape(-1), dx, sx, q


def _reference(codes, scales, xq, dx, sx, rows, cols, expert):
    runs, nb = cols // 32, cols // 64
    c = np.asarray(codes).reshape(-1, rows, runs * 2)[expert]
    s = np.asarray(scales, np.float32).reshape(-1, rows, nb)[expert]
    q = np.asarray(xq, np.int32).reshape(runs, 8)
    out = np.empty(rows, np.float32)
    for r in range(rows):
        acc = [np.float32(0.0)] * 32
        for h in range(runs):
            idot = 0
            for grp in range(8):
                word = int(c[r, 2 * h + (0 if grp < 4 else 1)])
                shift = 2 * (grp % 4)
                value = int(q[h, grp])
                for j in range(4):
                    code = (word >> (shift + 8 * j)) & 3
                    b = (value >> (8 * j)) & 255
                    if b >= 128:
                        b -= 256
                    idot += code * b
            x = np.float32(idot - int(sx[h]))
            t = np.float32(s[r, h // 2]) * np.float32(dx[h])
            acc[h % 32] = np.float32(math.fma(float(t), float(x), float(acc[h % 32])))
        step = 16
        while step:
            old = list(acc)
            for i in range(32):
                acc[i] = np.float32(old[i] + old[i ^ step])
            step >>= 1
        out[r] = acc[0]
    return out


def test_the_lane_matches_a_python_fma_reference():
    codes, scales, xq, dx, sx, _q = _random()
    for expert in (0, 1):
        got = lane.Q2Matrix(codes, scales, 32, 64, experts=2).matvec(expert, xq, dx, sx)
        want = _reference(codes, scales, xq, dx, sx, 32, 64, expert)
        assert np.array_equal(got.view(np.int32), want.view(np.int32)), expert


def test_expert_bytes_are_the_q2_block_bytes():
    assert lane.expert_bytes(1856, 2688) == 1856 * (2688 // 64) * 18
    with pytest.raises(ValueError):
        lane.expert_bytes(8, 96)


def test_a_short_or_misaligned_matrix_is_refused():
    codes, scales, *_ = _random(cols=64)
    with pytest.raises(ValueError):
        lane.Q2Matrix(codes, scales, 32, 96)                  # not a multiple of 64
    with pytest.raises(ValueError):
        lane.Q2Matrix(codes[:, :, :-1], scales, 32, 64)       # wrong word count
    with pytest.raises(ValueError):
        lane.Q2Matrix(codes, scales[:, :-1], 32, 64)          # wrong scale count


def test_the_activation_has_one_scale_per_run():
    codes, scales, xq, dx, sx, _q = _random()
    matrix = lane.Q2Matrix(codes, scales, 32, 64, experts=2)
    with pytest.raises(ValueError):
        matrix.matvec(0, xq[:-8], dx, sx)
    with pytest.raises(ValueError):
        matrix.matvec(0, xq, dx[:-1], sx)
    with pytest.raises(ValueError):
        matrix.matvec(2, xq, dx, sx)
