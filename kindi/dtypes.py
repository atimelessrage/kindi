# SPDX-License-Identifier: AGPL-3.0-only
"""Host-side codecs for the unquantized GGUF float formats Kindi understands.

BF16 is the top 16 bits of an IEEE float32.  The FP8 names follow the OCP
formats used by GGML's ``GGML_TYPE_F8_E4M3`` and ``GGML_TYPE_F8_E5M2``:

* E4M3FN: one sign bit, four exponent bits (bias 7), three mantissa bits,
  finite values only.  ``0x7f`` and ``0xff`` are NaN; the largest finite value
  is 448.
* E5M2: one sign bit, five exponent bits (bias 15), two mantissa bits, IEEE
  infinities and NaNs.

The conversion is exact and vectorised.  It is used when a Bonsai pack carries
an FP8 auxiliary tensor (the ternary language weights are a separate format),
and it is the reference for the CUDA Gemma 4 codec.
"""
import numpy as np

BF16 = 30
FP8_E4M3 = 43
FP8_E5M2 = 44
FP8_TYPES = (FP8_E4M3, FP8_E5M2)


def bf16_to_f32(bits):
    """uint16 BF16 bit patterns -> float32 (the low 16 mantissa bits are zero)."""
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def fp8_to_f32(raw, kind):
    """uint8 FP8 bit patterns -> float32 for ``FP8_E4M3`` or ``FP8_E5M2``."""
    if kind not in FP8_TYPES:
        raise ValueError("unsupported FP8 type %r" % (kind,))
    b = np.asarray(raw, dtype=np.uint8).astype(np.uint32)
    sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    if kind == FP8_E4M3:
        exp = (b >> 3) & 0x0f
        mant = b & 0x07
        finite = ~((exp == 15) & (mant == 7))
        value = np.where(
            exp == 0,
            mant.astype(np.float32) * np.float32(2.0 ** -9),
            (np.float32(1.0) + mant.astype(np.float32) * np.float32(0.125))
            * np.exp2(exp.astype(np.int32) - 7).astype(np.float32),
        )
        out = sign * value
        out[~finite] = np.nan
        return out
    exp = (b >> 2) & 0x1f
    mant = b & 0x03
    value = np.where(
        exp == 0,
        mant.astype(np.float32) * np.float32(2.0 ** -16),
        (np.float32(1.0) + mant.astype(np.float32) * np.float32(0.25))
        * np.exp2(exp.astype(np.int32) - 15).astype(np.float32),
    )
    inf = exp == 31
    nan = inf & (mant != 0)
    out = sign * value
    out[inf & ~nan] = np.where(sign[inf & ~nan] < 0, -np.inf, np.inf)
    out[nan] = np.nan
    return out
