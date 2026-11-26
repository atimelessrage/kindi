# SPDX-License-Identifier: AGPL-3.0-only
"""`paircodes` is the whole pack-load cost, so it is written for speed; this pins its bytes.

Every matrix in a pack is stored as base-3 codes, four trits to a byte, column-major by row block, and
the CPU engine reads it as one uint32 word per (row block, column pair) with the eight rows' 4-bit
codes in the eight nibbles.  `kindi.cpu.paircodes` used to derive each trit with `% 3` / `// 3` and
gather a uint32 code per column pair; that was 96% of a 2.5 27B pack's 340 s load.  It now decodes
each byte through one 256-entry table (`_PAIR_LUT`) and assembles the word from four bytes, which is
6.7x faster and must be the same bits.

The reference below is the previous implementation, frozen.  It is the test's own definition of the
layout: a change to the pack format or to the fast path that moves a bit fails here first, before a
340 s pack load would.
"""
import numpy as np
import pytest

from kindi import cpu


def _reference(codes, R, C, chunk=4096):
    """The per-trit decoder `paircodes` replaced: same output, ~6.7x the time."""
    G = C // 128
    out = np.empty((R // 8, G, 64), dtype=np.uint32)
    codes = codes.reshape(C // 4, R)
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        b = np.asarray(codes[:, r0:r1]).T.astype(np.uint8)
        t = np.stack([b % 3, b // 3 % 3, b // 9 % 3, b // 27 % 3], axis=2).reshape(r1 - r0, G, 64, 2)
        pc = cpu._CODE[t[..., 0].astype(np.uint32) + 3 * t[..., 1].astype(np.uint32)]
        pc = pc.reshape((r1 - r0) // 8, 8, G, 64)
        word = np.zeros(((r1 - r0) // 8, G, 64), dtype=np.uint32)
        for lane in range(8):
            word |= pc[:, lane] << np.uint32(4 * lane)
        out[r0 // 8:r1 // 8] = word
    return out


def test_the_lookup_table_is_the_trit_arithmetic_for_every_byte():
    """All 256 byte values, not just the 81 a pack uses: a wrong entry is a wrong weight."""
    lut = cpu._PAIR_LUT
    assert lut.dtype == np.uint16 and lut.shape == (256,)
    for byte in range(256):
        t = [byte % 3, byte // 3 % 3, byte // 9 % 3, byte // 27 % 3]
        assert int(lut[byte] & 0xFF) == int(cpu._CODE[t[0] + 3 * t[1]]), byte
        assert int(lut[byte] >> 8) == int(cpu._CODE[t[2] + 3 * t[3]]), byte


@pytest.mark.parametrize("R,C", [(8, 128), (64, 256), (256, 512), (104, 3072), (1024, 128)])
def test_paircodes_matches_the_frozen_decoder(R, C):
    rng = np.random.default_rng(R * 31 + C)
    for high in (81, 256):                       # a real pack's code range, then every byte value
        codes = rng.integers(0, high, size=(C // 4, R), dtype=np.uint8)
        fast = cpu.paircodes(codes, R, C)
        assert fast.dtype == np.uint32 and fast.shape == (R // 8, C // 128, 64)
        assert np.array_equal(fast, _reference(codes, R, C)), (R, C, high)


def test_paircodes_is_the_same_across_chunk_boundaries():
    """A matrix larger than `chunk` rows takes the sliced path; it must land exactly like one pass."""
    rng = np.random.default_rng(7)
    R, C = 24 * 8, 256
    codes = rng.integers(0, 81, size=(C // 4, R), dtype=np.uint8)
    assert np.array_equal(cpu.paircodes(codes, R, C, chunk=64), _reference(codes, R, C, chunk=64))
    assert np.array_equal(cpu.paircodes(codes, R, C, chunk=4096), _reference(codes, R, C, chunk=4096))


def test_paircodes_reads_a_memory_map_the_same_way(tmp_path):
    """The engine hands it a memmap slice of `bcodes.bin`, not a plain array."""
    rng = np.random.default_rng(11)
    R, C = 64, 512
    n = (C // 4) * R
    raw = rng.integers(0, 81, size=2 * n, dtype=np.uint8)               # two matrices back to back
    path = tmp_path / "bcodes.bin"
    raw.tofile(str(path))
    mm = np.memmap(str(path), dtype=np.uint8, mode="r")
    codes = mm[n:2 * n]                                                 # what Engine.__init__ slices
    assert np.array_equal(cpu.paircodes(codes, R, C), _reference(codes, R, C))
