# SPDX-License-Identifier: AGPL-3.0-only
"""Reference tests for the host BF16 / FP8 codecs."""
import numpy as np

from kindi.dtypes import BF16, FP8_E4M3, FP8_E5M2, bf16_to_f32, fp8_to_f32


def _scalar_fp8(byte, kind):
    """Independent scalar reference for every possible 8-bit pattern."""
    sign = -1.0 if byte & 0x80 else 1.0
    if kind == FP8_E4M3:
        exp, mant = (byte >> 3) & 0x0f, byte & 0x07
        if exp == 15 and mant == 7:
            return float("nan")
        value = mant * 2.0 ** -9 if exp == 0 else (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
    else:
        exp, mant = (byte >> 2) & 0x1f, byte & 0x03
        if exp == 31:
            return float("nan") if mant else sign * float("inf")
        value = mant * 2.0 ** -16 if exp == 0 else (1.0 + mant / 4.0) * 2.0 ** (exp - 15)
    return sign * value


def test_every_e4m3_and_e5m2_pattern_decodes_like_the_scalar_reference():
    patterns = np.arange(256, dtype=np.uint8)
    for kind in (FP8_E4M3, FP8_E5M2):
        got = fp8_to_f32(patterns, kind)
        want = np.array([_scalar_fp8(int(b), kind) for b in patterns], dtype=np.float32)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.array_equal(np.isinf(got), np.isinf(want))
        assert np.array_equal(np.signbit(got[np.isfinite(got)]), np.signbit(want[np.isfinite(want)]))
        assert np.allclose(got[np.isfinite(got)], want[np.isfinite(want)], rtol=0, atol=0)


def test_named_fp8_values():
    e4 = fp8_to_f32(np.array([0x00, 0x01, 0x38, 0x3C, 0x7E, 0x7F, 0xB8], dtype=np.uint8), FP8_E4M3)
    assert e4[:5].tolist() == [0.0, 2.0 ** -9, 1.0, 1.5, 448.0]
    assert np.isnan(e4[5]) and e4[6] == -1.0
    e5 = fp8_to_f32(np.array([0x00, 0x01, 0x3C, 0x40, 0x7C, 0x7D], dtype=np.uint8), FP8_E5M2)
    assert e5[:4].tolist() == [0.0, 2.0 ** -16, 1.0, 2.0]
    assert np.isinf(e5[4]) and np.isnan(e5[5])


def test_bf16_is_the_top_half_of_float32():
    bits = np.array([0x0000, 0x3F80, 0xC000, 0x7F80], dtype=np.uint16)
    got = bf16_to_f32(bits)
    assert got.tolist()[:3] == [0.0, 1.0, -2.0]
    assert np.isinf(got[3])
    assert BF16 == 30
