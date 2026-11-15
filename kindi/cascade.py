"""Nemotron Cascade 2 30B-A3B (GGUF architecture nemotron_h_moe) support path.

This module is the converter/validator for the K-quant Nemotron-H backend work.
It recognizes Nemotron Cascade 2 GGUFs, dequantizes the GGUF types used by the
Q4_K_M build (F32, Q5_0, Q8_0 and Q4_K) exactly on the host, and defines the
pinned llama.cpp reference gate that `kindi verify` will compare against once
the GPU K-quant matmul kernels land.

It is not a CUDA inference engine yet: there are no Nemotron-H K-quant matmul
kernels in this module, so `verify`/`bench`/`serve` remain gated until those
kernels exist. Do not add this model to the catalog, installer, or support
matrix until `kindi verify` passes against the pinned reference.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .gguf import GGUF

F32, Q5_0, Q8_0, Q4_K = 0, 6, 8, 12

# GGUF type -> (elements per block, bytes per block)
BLOCK = {F32: (1, 4), Q5_0: (32, 22), Q8_0: (32, 34), Q4_K: (256, 144)}
TYPE_NAMES = {F32: "F32", Q5_0: "Q5_0", Q8_0: "Q8_0", Q4_K: "Q4_K"}
SUPPORTED_TYPES = set(BLOCK)

MODEL_ID = "nemotron-cascade-2-30b-a3b"
REFERENCE_FILE = "reference-%s.json" % MODEL_ID


def _half(raw):
    """Little-endian fp16 bit patterns (uint16 array) -> float32 values."""
    bits = np.frombuffer(np.asarray(raw, dtype=np.uint8).tobytes(), dtype="<u2")
    return bits.view(np.float16).astype(np.float32)


def _q8_0_blocks(blocks):
    blocks = np.asarray(blocks, dtype=np.uint8)
    d = _half(blocks[:, 0:2])
    q = blocks[:, 2:34].view(np.int8)
    return (d[:, None] * q.astype(np.float32)).reshape(-1)


def _q5_0_blocks(blocks):
    blocks = np.asarray(blocks, dtype=np.uint8)
    d = _half(blocks[:, 0:2])
    qh = np.frombuffer(blocks[:, 2:6].tobytes(), dtype="<u4").reshape(-1)
    qs = blocks[:, 6:22]
    q = np.empty((blocks.shape[0], 32), dtype=np.uint16)
    q[:, 0::2] = qs & 0x0F
    q[:, 1::2] = qs >> 4
    high = ((qh[:, None] >> np.arange(32, dtype=np.uint32)[None, :]) & 1).astype(np.uint16) * 16
    q = (q + high).astype(np.int16) - 16
    return (d[:, None] * q.astype(np.float32)).reshape(-1)


def _scale_min_k4(scales):
    """Q4_K 6-bit scale/min pairs for one block: `scales` is (nblocks, 12) uint8.

    Returns (sc, mn), each (nblocks, 8) float32, for the eight 32-weight groups.
    The bit packing is the one used by ggml's Q4_K and by kindi/gemma4.cu
    `g4_scale_min`.
    """
    scales = np.asarray(scales, dtype=np.uint8)
    sc = np.empty((scales.shape[0], 8), dtype=np.int32)
    mn = np.empty((scales.shape[0], 8), dtype=np.int32)
    for j in range(8):
        k = j & 3
        u = j >> 2
        x = scales[:, k].astype(np.int32)
        y = scales[:, k + 4].astype(np.int32)
        z = scales[:, k + 8].astype(np.int32)
        sh = 2 * u
        if u:
            sc[:, j] = (z & 15) | ((x >> sh) & 48)
            mn[:, j] = ((z >> 4) & 15) | ((y >> sh) & 48)
        else:
            sc[:, j] = (x & 15) | ((x >> sh) & 48)
            mn[:, j] = (y & 15) | ((y >> sh) & 48)
    return sc.astype(np.float32), mn.astype(np.float32)


def _q4_k_blocks(blocks):
    blocks = np.asarray(blocks, dtype=np.uint8)
    d = _half(blocks[:, 0:2])
    dmin = _half(blocks[:, 2:4])
    sc, mn = _scale_min_k4(blocks[:, 4:16])
    qs = blocks[:, 16:144]
    q = np.empty((blocks.shape[0], 256), dtype=np.uint16)
    q[:, 0::2] = qs & 0x0F
    q[:, 1::2] = qs >> 4
    group = np.arange(256, dtype=np.int32) // 32
    w = d[:, None] * sc[:, group] * q.astype(np.float32) - dmin[:, None] * mn[:, group]
    return w.reshape(-1).astype(np.float32)


def dequantize(raw, gguf_type, shape=None):
    """Exactly dequantize one GGUF tensor (flat, row-major) to float32.

    `raw` is the tensor bytes exactly as stored in the GGUF (after the header
    and tensor offset). `shape` is optional: when given, the result is checked
    to hold prod(shape) elements.
    """
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(raw, dtype=np.uint8).copy()
    else:
        raw = np.asarray(raw, dtype=np.uint8)
    if gguf_type == F32:
        out = raw.view(np.float32)
    elif gguf_type == Q8_0:
        if raw.size % 34:
            raise ValueError("Q8_0 tensor has %d bytes, not a multiple of 34" % raw.size)
        out = _q8_0_blocks(raw.reshape(-1, 34))
    elif gguf_type == Q5_0:
        if raw.size % 22:
            raise ValueError("Q5_0 tensor has %d bytes, not a multiple of 22" % raw.size)
        out = _q5_0_blocks(raw.reshape(-1, 22))
    elif gguf_type == Q4_K:
        if raw.size % 144:
            raise ValueError("Q4_K tensor has %d bytes, not a multiple of 144" % raw.size)
        out = _q4_k_blocks(raw.reshape(-1, 144))
    else:
        raise ValueError("unsupported GGUF type %d for Cascade dequantization" % int(gguf_type))
    out = np.ascontiguousarray(out, dtype=np.float32)
    if shape is not None:
        n = int(np.prod(shape))
        if out.size != n:
            raise ValueError("dequantized %d values, expected %d" % (out.size, n))
    return out


def is_cascade(g):
    """A Nemotron-H MoE GGUF that is Nemotron Cascade 2 (not ternary Lightning)."""
    kv = getattr(g, "kv", None) or {}
    if kv.get("general.architecture") != "nemotron_h_moe":
        return False
    name = str(kv.get("general.name") or kv.get("general.basename") or "")
    return "cascade" in name.lower()


def _need(kv, key, want, what):
    got = kv.get("nemotron_h_moe." + key)
    if got != want:
        raise ValueError("kindi cascade: expected %s=%r, got %r" % (what, want, got))


def inspect_g(g):
    """Validate a loaded Nemotron Cascade 2 GGUF (header only)."""
    if not is_cascade(g):
        raise ValueError("kindi cascade: expected a Nemotron Cascade 2 nemotron_h_moe GGUF, got %r / %r"
                         % (g.kv.get("general.architecture"), g.kv.get("general.name")))
    kv = g.kv
    a = "nemotron_h_moe."
    _need(kv, "block_count", 52, "layers")
    _need(kv, "embedding_length", 2688, "width")
    _need(kv, "expert_count", 128, "experts")
    _need(kv, "expert_used_count", 6, "experts per token")
    _need(kv, "expert_feed_forward_length", 1856, "expert width")
    _need(kv, "expert_shared_feed_forward_length", 3712, "shared expert width")
    _need(kv, "attention.head_count", 32, "query heads")
    _need(kv, "attention.key_length", 128, "head dim")
    _need(kv, "ssm.inner_size", 4096, "Mamba inner size")
    _need(kv, "ssm.group_count", 8, "Mamba groups")
    _need(kv, "ssm.time_step_rank", 64, "Mamba heads")
    _need(kv, "ssm.state_size", 128, "Mamba state")
    _need(kv, "ssm.conv_kernel", 4, "Mamba conv kernel")
    if kv.get(a + "expert_weights_norm") is not True:
        raise ValueError("kindi cascade: expected expert_weights_norm=True")
    if float(kv.get(a + "expert_weights_scale", 0)) != 2.5:
        raise ValueError("kindi cascade: expected expert_weights_scale=2.5")

    ff = kv.get(a + "feed_forward_length")
    kvh = kv.get(a + "attention.head_count_kv")
    if not isinstance(ff, list) or not isinstance(kvh, list) or len(ff) != 52 or len(kvh) != 52:
        raise ValueError("kindi cascade: expected per-layer feed_forward_length and head_count_kv lists")

    counts = {}
    seen = set()
    for name, ne, ty, _off in g.tensors:
        if name in seen:
            raise ValueError("kindi cascade: duplicate tensor %s" % name)
        seen.add(name)
        if ty not in SUPPORTED_TYPES:
            raise ValueError("kindi cascade: tensor %s has unsupported GGUF type %d (%s); "
                             "the Q4_K_M build needs F32/Q5_0/Q8_0/Q4_K"
                             % (name, ty, TYPE_NAMES.get(ty, "unknown")))
        block, _size = BLOCK[ty]
        n = int(np.prod(ne))
        if n % block:
            raise ValueError("kindi cascade: tensor %s has %d elements, not aligned to block %d"
                             % (name, n, block))
        counts[TYPE_NAMES[ty]] = counts.get(TYPE_NAMES[ty], 0) + 1

    for name in ("token_embd.weight", "output.weight", "output_norm.weight"):
        if name not in seen:
            raise ValueError("kindi cascade: missing tensor %s" % name)
    for i in range(52):
        p = "blk.%d." % i
        if p + "attn_norm.weight" not in seen:
            raise ValueError("kindi cascade: missing tensor %sattn_norm.weight" % p)
        if kvh[i]:  # attention layer
            for n in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"):
                if p + n not in seen:
                    raise ValueError("kindi cascade: missing tensor %s%s" % (p, n))
        elif ff[i]:  # MoE layer
            for n in ("ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_up_exps.weight",
                      "ffn_down_exps.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"):
                if p + n not in seen:
                    raise ValueError("kindi cascade: missing tensor %s%s" % (p, n))
        else:  # Mamba layer
            for n in ("ssm_in.weight", "ssm_out.weight", "ssm_conv1d.weight",
                      "ssm_conv1d.bias", "ssm_a", "ssm_d", "ssm_dt.bias", "ssm_norm.weight"):
                if p + n not in seen:
                    raise ValueError("kindi cascade: missing tensor %s%s" % (p, n))

    return {
        "name": kv.get("general.name", "Nemotron Cascade 2 30B A3B"),
        "context_length": int(kv.get(a + "context_length", 262144)),
        "dimensions": {
            "layers": 52, "width": 2688, "heads": 32, "kv_heads": max(kvh),
            "head_dim": 128, "experts": 128, "experts_per_token": 6,
            "expert_ff": 1856, "shared_expert_ff": 3712, "vocab": int(kv.get(a + "vocab_size", 131072)),
            "mamba_inner": 4096, "mamba_heads": 64, "mamba_groups": 8,
        },
        "tensor_count": len(g.tensors),
        "tensor_types": counts,
    }


def inspect_cascade(path):
    return inspect_g(GGUF(path))


def convert(g, gguf_path, out_dir, version, pack_format, sha256, log=print):
    info = inspect_g(g)
    os.makedirs(out_dir, exist_ok=True)
    man = {
        "kindi_version": version, "pack_format": pack_format, "backend": "kindi_cascade",
        "model": info["name"], "model_id": MODEL_ID,
        "source_file": os.path.basename(gguf_path), "model_path": os.path.abspath(gguf_path),
        "source_sha256": sha256(gguf_path), "context_length": info["context_length"],
        "dimensions": info["dimensions"], "tensor_count": info["tensor_count"],
        "tensor_types": info["tensor_types"],
        "native_status": "converter/validator only; GPU K-quant Nemotron-H kernels pending",
    }
    with open(os.path.join(out_dir, "kindi_pack.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    log("recognized %s (Nemotron Cascade 2, nemotron_h_moe, %d tensors, %s): Kindi Cascade "
        "converter/validator; the weights stay in the GGUF file. GPU inference kernels are "
        "still pending, so this pack is not yet servable."
        % (info["name"], info["tensor_count"], info["tensor_types"]))
    return man


def _tensor_bytes(g, path, name, max_bytes=None):
    for n, ne, ty, off in g.tensors:
        if n == name:
            break
    else:
        raise KeyError(name)
    block, size = BLOCK[ty]
    nbytes = (int(np.prod(ne)) // block) * size
    if max_bytes is not None:
        nbytes = min(nbytes, max_bytes)
    with open(path, "rb") as f:
        f.seek(g.data_start + off)
        return f.read(nbytes)


def _dequant_gate(g, path, log=print):
    """Dequantize the head of one tensor of each supported type; all must be finite."""
    size_bytes = os.path.getsize(path)
    wanted = []
    for name, ne, ty, off in g.tensors:
        if g.data_start + off >= size_bytes:
            continue  # a header-only fetch: the tensor bytes are not on disk
        if ty in SUPPORTED_TYPES and name not in [w[0] for w in wanted]:
            block, size = BLOCK[ty]
            wanted.append((name, ne, ty, block, size))
        if len(wanted) >= len(SUPPORTED_TYPES):
            break
    for name, ne, ty, block, size in wanted:
        raw = _tensor_bytes(g, path, name, max_bytes=block * size * 8)
        got = dequantize(raw, ty)
        if not np.isfinite(got).all() or not np.any(got):
            raise ValueError("kindi cascade: dequant gate failed on %s" % name)
        log("dequant gate: %s (%s) -> %d finite values" % (name, TYPE_NAMES[ty], got.size))


def reference_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), REFERENCE_FILE)


def check_reference(man, log=print):
    """Return True when the pinned llama.cpp reference exists and matches this pack."""
    rp = reference_path()
    if not os.path.exists(rp):
        log("no pinned llama.cpp reference at %s" % os.path.basename(rp))
        log("generate it from a running llama-server for this exact GGUF (temperature 0, "
            "fixed prompts, token ids + expected tokens + top-2 gaps), then re-run "
            "kindi verify; the file must carry source_sha256 and llama_cpp_commit")
        return False
    ref = json.load(open(rp, encoding="utf-8"))
    if ref.get("source_sha256") != man.get("source_sha256"):
        log("reference %s is for a different GGUF (source_sha256 mismatch)" % os.path.basename(rp))
        return False
    if not ref.get("llama_cpp_commit"):
        log("reference %s has no llama_cpp_commit pin" % os.path.basename(rp))
        return False
    prompts = ref.get("prompts") or []
    def _ids_ok(v):
        return isinstance(v, list) and v and all(isinstance(t, int) and not isinstance(t, bool) for t in v)
    usable = (
        isinstance(prompts, list) and prompts and all(
            isinstance(p, dict) and isinstance(p.get("prompt"), str) and p["prompt"]
            and _ids_ok(p.get("ids")) and _ids_ok(p.get("expected"))
            and (p.get("gap") is None or (
                isinstance(p["gap"], list) and len(p["gap"]) == len(p["expected"])
                and all(g is None or isinstance(g, (int, float)) for g in p["gap"])))
            for p in prompts))
    if not usable:
        log("reference %s has no usable prompts (each needs prompt, token ids, expected ids, "
            "and a gap list matching expected)" % os.path.basename(rp))
        return False
    log("pinned llama.cpp reference: %s (%d prompts, llama.cpp %s)"
        % (os.path.basename(rp), len(prompts), ref.get("llama_cpp_commit")))
    return True


def verify(cfg, man, log=print):
    """Cascade gate: header + dequant + pinned reference.

    Returns 0 only when the reference is pinned and matches; token comparison
    against it is still pending the GPU kernels, so a pinned reference alone
    returns 2 (not a pass) with that stated.
    """
    path = man["model_path"]
    info = inspect_cascade(path)
    log("cascade: %s, %d tensors, %s" % (info["name"], info["tensor_count"], info["tensor_types"]))
    g = GGUF(path)
    _dequant_gate(g, path, log=log)
    pinned = check_reference(man, log=log)
    if not pinned:
        return 2
    log("pinned reference matches this GGUF; token comparison is pending the GPU "
        "K-quant Nemotron-H kernels (this is not a pass)")
    return 2
