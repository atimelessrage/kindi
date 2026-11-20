# SPDX-License-Identifier: AGPL-3.0-only
"""Grouped-query attention validation and kernel selection without a GPU.

The 27B pack has 24 query heads and 4 K/V heads: 6:1.  The plain engine used to
accept only 4:1.  These tests pin the accepted ratios, the head-size rule of the
generic kernels, and the exact kernels/arguments each ratio launches.
"""
import numpy as np
import pytest

from kindi import engine as engine_mod
from kindi.engine import Engine


class FakeKernels(object):
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(grid, block, args=()):
            self.calls.append((name, tuple(grid), tuple(block), args))
        return record


def stub(nh, nhkv, hd=256):
    eng = Engine.__new__(Engine)
    eng.NH, eng.NHKV, eng.HD = nh, nhkv, hd
    eng.G = Engine._gqa_groups(nh, nhkv, hd)
    for name in ("sbuf", "apart", "att", "qr", "qn"):
        setattr(eng, name, np.zeros(4096, dtype=np.float32))
    eng.k = FakeKernels()
    return eng


def test_the_shipped_ratios_are_accepted():
    assert Engine._gqa_groups(8, 2, 256) == 4      # 2B / 4B / 9B fast path
    assert Engine._gqa_groups(24, 4, 256) == 6     # 27B
    assert Engine._gqa_groups(4, 4, 256) == 1
    assert Engine._gqa_groups(32, 4, 256) == 8


@pytest.mark.parametrize("nh,nhkv,hd", [
    (24, 0, 256),      # no K/V heads
    (25, 4, 256),      # not a whole number of q heads per K/V head
    (36, 4, 256),      # 9 q heads: past the generic kernels
    (12, 4, 128),      # generic kernels cover head size 256 only
])
def test_unsupported_groupings_are_rejected(nh, nhkv, hd):
    with pytest.raises(SystemExit, match="unsupported grouped head dimensions"):
        Engine._gqa_groups(nh, nhkv, hd)


def test_six_to_one_launches_the_generic_kernels():
    eng = stub(24, 4)
    eng._attn_core("K", "V", 130, 1)
    names = [c[0] for c in eng.k.calls]
    assert names == ["k_attn_sg", "k_softmax_rows", "k_attn_vgn", "k_attn_combn"]
    sg = eng.k.calls[0]
    assert sg[1] == (4, min(32, (130 + 7) // 8))
    assert int(sg[3][-1]) == 6 and int(sg[3][3]) == 130
    vgn = eng.k.calls[2]
    assert vgn[1] == (4, 1, 1) and tuple(int(x) for x in vgn[3][3:]) == (130, 24, 4, 6)
    comb = eng.k.calls[3]
    assert int(comb[3][3]) == 130 and int(comb[3][4]) == 24


def test_four_to_one_keeps_the_fast_kernels():
    eng = stub(8, 2)
    eng._attn_core("K", "V", 130, 1)
    names = [c[0] for c in eng.k.calls]
    assert names == ["k_attn_s4w", "k_softmax_rows", "k_attn_v4p", "k_attn_comb"]
    assert eng.k.calls[0][1] == (2, min(32, (130 + 31) // 32))
    assert int(eng.k.calls[2][3][3]) == 130


def test_every_kernel_the_generic_path_launches_is_compiled():
    for name in ("k_attn_sg", "k_attn_vgn", "k_attn_combn"):
        assert name in engine_mod.KERNELS
