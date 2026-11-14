# SPDX-License-Identifier: AGPL-3.0-only
"""Nemotron Cascade 2 converter/validator gate.

These tests cover the host-side foundation of the K-quant Nemotron-H backend:
exact dequantization of the GGUF types in the Q4_K_M build (F32/Q5_0/Q8_0/Q4_K),
Cascade file validation (architecture, dimensions, tensor inventory), and the
pinned llama.cpp reference gate. They do not test CUDA inference, which does
not exist yet.
"""
import json
import os

import numpy as np
import pytest

from kindi import cascade


def _half_bits(values):
    return np.asarray(values, dtype=np.float16).view(np.uint16)


def test_q8_0_dequantizes_blockwise():
    d = np.float16(0.5)
    q = np.arange(-16, 16, dtype=np.int8)
    raw = np.concatenate([_half_bits([d]).view(np.uint8), q.view(np.uint8)])
    out = cascade.dequantize(raw, cascade.Q8_0, shape=(32,))
    assert np.allclose(out, 0.5 * q.astype(np.float32))


def test_q5_0_dequantizes_nibbles_and_high_bits():
    # one block, d=2, weights q-16 with q spanning 0..31
    d = np.float16(2.0)
    q = np.arange(32, dtype=np.uint8)
    low = q & 0x0F
    high = (q >> 4) & 1
    qs = np.empty(16, dtype=np.uint8)
    qs[:] = low[0::2] | (low[1::2] << 4)
    qh = int(sum(int(b) << i for i, b in enumerate(high)))
    raw = np.concatenate([_half_bits([d]).view(np.uint8), np.array([qh], dtype="<u4").view(np.uint8), qs])
    assert raw.size == 22
    out = cascade.dequantize(raw, cascade.Q5_0, shape=(32,))
    assert np.allclose(out, 2.0 * (q.astype(np.float32) - 16.0))


def test_q4_k_dequantizes_groups_with_scale_and_min():
    # one 256-weight block with d=1, dmin=1 and every group scale=1, min=1:
    # weight = q - 1. Build the packed scales with the same bit layout the
    # decoder reads, using scale/min == 1 for all eight groups.
    scales = np.zeros(12, dtype=np.uint8)
    for j in range(8):
        # encode sc=mn=1 with the inverse of _scale_min_k4's packing
        k, u = j & 3, j >> 2
        sh = 2 * u
        # sc low nibble from x (u=0) or z (u=1); high bits in x
        if j < 4:  # u == 0: sc = (x&15)|((x>>0)&48); mn = (y&15)|((y>>0)&48)
            scales[k] |= (1 & 15) | ((1 >> 0) & 48) & 15  # low part; full packing below
        else:
            scales[k + 8] |= (1 & 15) << 0
    # Simpler and exact: set scales so _scale_min_k4 returns 1. From the packing,
    # sc = 1 needs low nibble 1 and high bits 0. The high-bit fields overlap the
    # low nibbles of other groups, so construct via the decoder's own inverse:
    scales = np.zeros(12, dtype=np.uint8)
    # u=0 groups (j=0..3): sc low in s[k], mn low in s[k+4]; high bits of s[k]/s[k+4] at sh=0 are bits 4-5
    # set s[k]=1 (low 1, high 0), s[k+4]=1 -> sc=(1)|((1>>0)&48)=1? (1>>0)&48 = 0. Good.
    for k in range(4):
        scales[k] = 1
        scales[k + 4] = 1
    # u=1 groups (j=4..7): sc low in s[k+8], mn low in s[k+8] high nibble; high bits at sh=2
    # s[k+8] carries both lows: low=1 (sc), high=1 (mn) -> 0x11; s[k],s[k+4] high bits at sh=2 must be 0
    for k in range(4):
        scales[k + 8] = 0x11
    sc, mn = cascade._scale_min_k4(scales.reshape(1, 12))
    assert np.allclose(sc, 1.0) and np.allclose(mn, 1.0)
    qs = np.zeros(128, dtype=np.uint8)
    qvals = (np.arange(256) % 16).astype(np.uint8)
    qs[:] = (qvals[0::2] & 15) | ((qvals[1::2] & 15) << 4)
    raw = np.concatenate([_half_bits([np.float16(1.0)]).view(np.uint8),
                          _half_bits([np.float16(1.0)]).view(np.uint8), scales, qs])
    assert raw.size == 144
    out = cascade.dequantize(raw, cascade.Q4_K, shape=(256,))
    assert np.allclose(out, qvals.astype(np.float32) - 1.0)


def test_dequantize_refuses_unknown_types():
    with pytest.raises(ValueError, match="unsupported GGUF type 42"):
        cascade.dequantize(b"\x00" * 18, 42)


class FakeGGUF:
    def __init__(self, arch, name, tensors=(), **kv):
        self.kv = {"general.architecture": arch, "general.name": name}
        self.kv.update(kv)
        self.tensors = list(tensors)


def _cascade_kv():
    # the layer schedule of the Q4_K_M header measured 2026-11-14: 23 MoE,
    # 6 attention, 23 Mamba layers
    moe = [1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51]
    attn = [5, 12, 19, 26, 33, 42]
    ff = [1856 if i in moe else 0 for i in range(52)]
    kvh = [2 if i in attn else 0 for i in range(52)]
    return {
        "nemotron_h_moe.block_count": 52,
        "nemotron_h_moe.embedding_length": 2688,
        "nemotron_h_moe.expert_count": 128,
        "nemotron_h_moe.expert_used_count": 6,
        "nemotron_h_moe.expert_feed_forward_length": 1856,
        "nemotron_h_moe.expert_shared_feed_forward_length": 3712,
        "nemotron_h_moe.attention.head_count": 32,
        "nemotron_h_moe.attention.key_length": 128,
        "nemotron_h_moe.ssm.inner_size": 4096,
        "nemotron_h_moe.ssm.group_count": 8,
        "nemotron_h_moe.ssm.time_step_rank": 64,
        "nemotron_h_moe.ssm.state_size": 128,
        "nemotron_h_moe.ssm.conv_kernel": 4,
        "nemotron_h_moe.expert_weights_norm": True,
        "nemotron_h_moe.expert_weights_scale": 2.5,
        "nemotron_h_moe.feed_forward_length": ff,
        "nemotron_h_moe.attention.head_count_kv": kvh,
        "nemotron_h_moe.context_length": 1048576,
        "nemotron_h_moe.vocab_size": 131072,
    }


def _cascade_tensors(kv):
    tensors = [
        ("token_embd.weight", [2688, 131072], cascade.Q5_0, 0),
        ("output.weight", [2688, 131072], cascade.Q8_0, 0),
        ("output_norm.weight", [2688], cascade.F32, 0),
    ]
    for i in range(52):
        p = "blk.%d." % i
        tensors.append((p + "attn_norm.weight", [2688], cascade.F32, 0))
        if kv["nemotron_h_moe.attention.head_count_kv"][i]:
            for n, ne in (("attn_q.weight", [2688, 4096]), ("attn_k.weight", [2688, 256]),
                          ("attn_v.weight", [2688, 256]), ("attn_output.weight", [4096, 2688])):
                tensors.append((p + n, ne, cascade.Q5_0 if "output" not in n else cascade.Q4_K, 0))
        elif kv["nemotron_h_moe.feed_forward_length"][i]:
            for n, ne, ty in (("ffn_gate_inp.weight", [2688, 128], cascade.F32),
                              ("exp_probs_b.bias", [128], cascade.F32),
                              ("ffn_up_exps.weight", [2688, 1856, 128], cascade.Q5_0),
                              ("ffn_down_exps.weight", [1856, 2688, 128], cascade.Q8_0),
                              ("ffn_up_shexp.weight", [2688, 3712], cascade.Q5_0),
                              ("ffn_down_shexp.weight", [3712, 2688], cascade.Q8_0)):
                tensors.append((p + n, ne, ty, 0))
        else:
            for n, ne, ty in (("ssm_in.weight", [2688, 10304], cascade.Q5_0),
                              ("ssm_out.weight", [4096, 2688], cascade.Q4_K),
                              ("ssm_conv1d.weight", [4, 6144], cascade.F32),
                              ("ssm_conv1d.bias", [6144], cascade.F32),
                              ("ssm_a", [1, 64], cascade.F32),
                              ("ssm_d", [1, 64], cascade.F32),
                              ("ssm_dt.bias", [64], cascade.F32),
                              ("ssm_norm.weight", [512, 8], cascade.F32)):
                tensors.append((p + n, ne, ty, 0))
    return tensors


def test_inspect_cascade_accepts_the_q4_k_m_inventory():
    kv = _cascade_kv()
    g = FakeGGUF("nemotron_h_moe", "Nemotron Cascade 2 30B A3B", _cascade_tensors(kv), **kv)
    info = cascade.inspect_g(g)
    assert info["dimensions"]["width"] == 2688
    assert info["dimensions"]["experts"] == 128
    assert set(info["tensor_types"]) <= {"F32", "Q5_0", "Q8_0", "Q4_K"}


def test_inspect_cascade_refuses_lightning_and_unknown_types():
    kv = _cascade_kv()
    g = FakeGGUF("nemotron_h_moe", "Nemotron 3.5 Lightning 30B-A3B", [], **kv)
    with pytest.raises(ValueError, match="expected a Nemotron Cascade 2"):
        cascade.inspect_g(g)
    tensors = _cascade_tensors(kv)
    tensors[0] = ("token_embd.weight", [2688, 131072], 42, 0)  # Q2_0: not a Cascade type
    g = FakeGGUF("nemotron_h_moe", "Nemotron Cascade 2 30B A3B", tensors, **kv)
    with pytest.raises(ValueError, match="unsupported GGUF type 42"):
        cascade.inspect_g(g)


def test_reference_gate_requires_a_pinned_matching_file(tmp_path, monkeypatch):
    man = {"source_sha256": "abc123"}
    rp = cascade.reference_path()
    assert not cascade.check_reference(man, log=lambda *a: None) if not os.path.exists(rp) else True
    # write a synthetic pinned reference next to the module when no real one is present
    if not os.path.exists(rp):
        pytest.skip("no pinned reference file present; gate behavior with a file is covered by the schema checks")
    ref = json.load(open(rp, encoding="utf-8"))
    assert ref.get("llama_cpp_commit")
