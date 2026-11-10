# SPDX-License-Identifier: AGPL-3.0-only
"""Conformance for the PQ2_0 and Q2_0 import in `kindi/convert.py`.

PQ2_0 (ggml type 142) and Q2_0 (type 42) hold the same ternary weights as PTQ1_0, in a coarser
layout: a 2-bit slot per weight with one fp16 scale per group (128 weights for PQ2_0, 64 for Q2_0).
The slot code is the trit index itself (0 = -d, 1 = 0, 2 = +d, as in llama.cpp's
`quantize_row_q2_0_ref` / `quantize_row_pq2_0_ref`, which write `round(w/d) + 1`). The converter
re-codes the file into the pack's 4-trits-per-byte form, so the decoded trits and the scale bits
must be exactly those of the file.

The tests build blocks the way llama.cpp writes them (fp16 scale first, element j in byte j//4 at
bits 2*(j%4)) and check the decoder against that, plus the two refusal rules: a non-ternary code 3,
and a Q2_0 128-group whose two 64-weight halves disagree on the scale (the pack stores one scale per
128 weights).

No GPU or model file is needed. The real-file conformance - a converted pack bit-identical to the
PTQ1_0 one - is `bench-logs/convert_pq2_q2.py`.
"""
import numpy as np
import pytest

from kindi.convert import PQ2_0, PTQ1_0, Q2_0, phase_bytes, q2_groups, trits_from_slots


def encode_slots(trits, group, d=None):
    """llama.cpp's Q2_0 / PQ2_0 block: one fp16 scale, then element j in byte j//4 at bits 2*(j%4)."""
    trits = np.asarray(trits, dtype=np.uint8)
    nb = trits.size // group
    qs = np.zeros((nb, group // 4), dtype=np.uint8)
    for j in range(group):
        qs[:, j // 4] |= (trits.reshape(nb, group)[:, j] << (2 * (j % 4))).astype(np.uint8)
    if d is None:
        d = np.arange(1, nb + 1, dtype=np.uint16) * 0x0400       # distinct fp16 bit patterns
    return np.concatenate([np.asarray(d, dtype=np.uint16).view(np.uint8).reshape(nb, 2), qs], axis=1)


def test_the_type_ids_are_the_published_ones():
    # the ids are written into the files; a renumbering would silently read the wrong layout
    assert (PTQ1_0, PQ2_0, Q2_0) == (143, 142, 42)


@pytest.mark.parametrize("group", [128, 64])
def test_the_slot_decoder_recovers_every_trit(group):
    rng = np.random.RandomState(11)
    trits = rng.randint(0, 3, size=(7 * group,)).astype(np.uint8)
    raw = encode_slots(trits, group)
    assert raw.shape == (7, 2 + group // 4)                  # 34 B per 128, 18 B per 64
    assert np.array_equal(trits_from_slots(raw, group), trits.reshape(7, group))


def test_the_element_order_is_slot_in_byte_not_strided():
    # element j = byte j//4, bits 2*(j%4): the first four trits are four slots of one byte
    trits = np.zeros(64, dtype=np.uint8)
    trits[:8] = [0, 1, 2, 0, 2, 1, 0, 1]
    raw = encode_slots(trits, 64)
    assert np.array_equal(trits_from_slots(raw, 64)[0, :8], trits[:8])
    assert raw[0, 2] == 0 + (1 << 2) + (2 << 4) + (0 << 6)


@pytest.mark.parametrize("group", [64, 128])
def test_a_non_ternary_code_is_refused(group):
    # code 3 means +2*scale; folding it into a trit would silently change the weight
    raw = encode_slots(np.zeros(group, dtype=np.uint8), group)
    raw[0, 2] = 3                                            # element 0 -> code 3
    with pytest.raises(SystemExit, match="code 3"):
        trits_from_slots(raw, group)


def test_q2_groups_fold_two_64_weight_halves_into_one_128_weight_group():
    rng = np.random.RandomState(3)
    trits = rng.randint(0, 3, size=(5 * 128,)).astype(np.uint8)
    d = np.repeat(np.arange(1, 6, dtype=np.uint16) * 0x0400, 2)      # the same scale in both halves
    raw = encode_slots(trits, 64, d)                                # 10 blocks of 64
    tr, ds = q2_groups(trits_from_slots(raw, 64), np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.uint16), 5, 1)
    assert tr.shape == (5, 128) and np.array_equal(tr, trits.reshape(5, 128))
    assert np.array_equal(ds.reshape(-1), d[0::2])


def test_q2_groups_handle_several_128_groups_per_row():
    # ne0 384: three 128-weight groups, six 64-weight blocks, per row
    rng = np.random.RandomState(5)
    trits = rng.randint(0, 3, size=(2 * 384,)).astype(np.uint8)
    d = np.repeat(np.arange(1, 7, dtype=np.uint16) * 0x0400, 2)       # 12 blocks, the halves agree
    raw = encode_slots(trits, 64, d)
    tr, ds = q2_groups(trits_from_slots(raw, 64), np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.uint16), 2, 3)
    assert tr.shape == (2, 384) and np.array_equal(tr, trits.reshape(2, 384))
    assert ds.shape == (2, 3) and np.array_equal(ds.reshape(-1), d[0::2])


def test_q2_groups_refuse_halves_with_different_scales():
    # the pack has one scale per 128 weights; a legacy group-128 Q2_0 file read as 64 lands here
    d = np.array([0x0400, 0x3C00], dtype=np.uint16)
    with pytest.raises(SystemExit, match="one scale per 128"):
        q2_groups(np.ones((2, 64), dtype=np.uint8), d, 1, 1)


def test_phase_bytes_packs_four_trits_per_byte():
    tr = np.array([[0, 1, 2, 1, 2, 0, 1, 0]], dtype=np.uint8)
    b = phase_bytes(tr)
    assert b.shape == (1, 2)                      # one byte per four trits, rows kept
    assert b[0, 0] == 0 + 3 * 1 + 9 * 2 + 27 * 1
    assert b[0, 1] == 2 + 3 * 0 + 9 * 1 + 27 * 0
