#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the Raon-1 mini R0 tensor manifest from downloaded headers.

This does not download or load checkpoint data. It reads the seven safetensors headers
(8 MiB fetched per shard), the GGUF header (128 MiB fetched), config.json and the
safetensors index, then writes compact inventories and a validation summary.

Run from the repository root:
    python bench-logs/raon-r0/build_manifest.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import struct
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kindi.gguf import GGUF  # noqa: E402

HERE = Path(__file__).resolve().parent

# The published types used by the Q4_K_M header, plus the common neighboring types.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4",
    32: "Q4_0_4_8", 33: "Q4_0_8_8", 34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4",
    37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8", 39: "MXFP4",
}

BASE_REVISION = "8c9b802acacdbda9df16606689fcf437cf7262fb"
GGUF_REVISION = "cd61e81afcdfdf4b13cb345b3587c0ba4b918d00"
BASE_SHARDS = [
    ("model-00001-of-00007.safetensors", 4998364808,
     "c18545414b88b1b1c694c23a275141885e26168bedca23f821b9b87d316f3867"),
    ("model-00002-of-00007.safetensors", 5000376992,
     "bbf5fb70a5a6c4af680b84ed482891c7e819829b4805b7511ef6b7daf98a9af1"),
    ("model-00003-of-00007.safetensors", 5000397960,
     "7e46f3844fd2d659c833bac9d09944407011c541bcf9bc7e71900bf7ac4148f4"),
    ("model-00004-of-00007.safetensors", 4998818792,
     "6264c2de64c006fb41d2dd215c87e5132681dd2e0e3419d4a0206daa8ae5ea59"),
    ("model-00005-of-00007.safetensors", 4974921360,
     "ad0dcf3f8ae44e80f1d5171156ef86cd8633b89a826a34a93e0367cfcf0d190e"),
    ("model-00006-of-00007.safetensors", 4989223544,
     "6628b23347ac6f3a2f8b42dfb93edb7e6e1c08a3a6bae6f09acefb7340190e25"),
    ("model-00007-of-00007.safetensors", 2740041632,
     "64f717749ef89be949fa95a82118aed668257057d98a807862761b9ed2af74ec"),
]
BASE_FILES = {
    "config.json": (2028, "ccd1541ed7d4e26dd86e0b9c706f13e5dbdf968c68824c27c2eb53f23d735ee3"),
    "configuration_raon1_mini.py": (6989, "3656b59a41dec3b228852973f0e2472ed081b14bcd80b36040e0e0afccba42c0"),
    "modeling_raon1_mini.py": (56961, "ca0742570e8c9b730115f2feedbcd9f424dc47d21b132729f27c4d8ac6998fc6"),
    "model.safetensors.index.json": (1837333, "698935d27fd0a7be12be30beb0f93408e378b158cb0913390652c7c6a51713a2"),
    "generation_config.json": (167, "9480ec0f58bcbb8df2050aa8312b3fdd7248792c4b0e861582feeb75fdfe71f5"),
    "tokenizer_config.json": (281, "d5d612aeef88d4564d2010ab7f65b7e54821acfd7ccda80739dd19761840bcde"),
    "chat_template.jinja": (6735, "34a9f0219d315211631885b22d8e4973943c092ef319b57c11853127e392b0ac"),
    "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
}
GGUF_FILE = {
    "name": "raon-1-mini-preview-Q4_K_M.gguf",
    "size": 19686434240,
    "sha256": "ba56ad4bd6d15f8c4255876ff252890143074a5d3835fccfc656cee915999ee8",
    "header_sha256": "22257ab636f6e900e2e5c9384ca0a310b794c4a2f3f97e0cbac36e980a111841",
    "header_bytes": 134217728,
}
LLAMA_CPP = {
    "repo": "https://github.com/krafton-ai/llama.cpp",
    "branch": "raon-1-mini-preview",
    "commit": "fb895662d25868756a19253ca5e0688f99b18994",
    "commit_date": "2026-10-20T13:33:00Z",
    "arg_sha256": "a10b6e6af910366c5c1abb89e47a7c6711fc1f10708b5fbfd8b59ba1a11d0f4a",
}
OFFICIAL_REPO = {
    "url": "https://github.com/krafton-ai/Raon-1-mini-preview",
    "commit": "cfa219538a11a0f10a6ae40ede76bdb8efc4dc4e",
    "commit_date": "2026-10-27T08:08:21Z",
}
SOURCE_PINS = {
    "created_utc": "2026-11-11",
    "base_model": {
        "repo": "KRAFTON/Raon-1-mini-preview",
        "revision": BASE_REVISION,
        "last_modified": "2026-10-27T08:07:23.000Z",
        "license": "apache-2.0",
        "gated": False,
        "shards": [
            {"name": name, "size": size, "sha256": sha}
            for name, size, sha in BASE_SHARDS
        ],
        "files": {
            name: {"size": size, "sha256": sha}
            for name, (size, sha) in BASE_FILES.items()
        },
        "index_total_size": 32699649368,
        "safetensors_parameters": {
            "BF16": 3113981970,
            "F32": 95952,
            "F8_E4M3": 25669140480,
            "total": 28783218402,
        },
        "tokenizer_json": {
            "size": 12219054,
            "sha256_lfs_oid": "47752d33008f5dd7cc91d875f1b798d1f1ad8f6a0e84aac9e67cbb49f9da16af",
        },
    },
    "gguf": {
        "repo": "KRAFTON/Raon-1-mini-preview-gguf",
        "revision": GGUF_REVISION,
        "file": GGUF_FILE["name"],
        "size": GGUF_FILE["size"],
        "sha256": GGUF_FILE["sha256"],
        "header": {
            "bytes": GGUF_FILE["header_bytes"],
            "sha256": GGUF_FILE["header_sha256"],
        },
        "sha256sums_url": "https://huggingface.co/KRAFTON/Raon-1-mini-preview-gguf/raw/main/SHA256SUMS",
    },
    "llama_cpp": LLAMA_CPP,
    "official_repo": OFFICIAL_REPO,
    "evidence": {
        "base_config_url": "https://huggingface.co/KRAFTON/Raon-1-mini-preview/raw/main/config.json",
        "base_index_url": "https://huggingface.co/KRAFTON/Raon-1-mini-preview/raw/main/model.safetensors.index.json",
        "base_model_card_url": "https://huggingface.co/KRAFTON/Raon-1-mini-preview",
        "gguf_model_card_url": "https://huggingface.co/KRAFTON/Raon-1-mini-preview-gguf",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "notes": (
            "Only tensor/metadata headers were fetched. The base shards, tokenizer.json and "
            "the 19.7 GB GGUF payload were not downloaded or executed. The model card says "
            "access is granted on request; no Hugging Face token was used for the headers."
        ),
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> tuple[int, dict]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: too short for a safetensors header")
    header_len = struct.unpack("<Q", raw[:8])[0]
    if header_len + 8 > len(raw):
        raise ValueError(
            f"{path}: header length {header_len} exceeds downloaded {len(raw)} bytes"
        )
    header = json.loads(raw[8:8 + header_len])
    meta = header.pop("__metadata__", None)
    return header_len, {"metadata": meta, "tensors": header}


def parse_base(headers_dir: Path) -> tuple[list[dict], dict]:
    index = json.loads((HERE / "upstream" / "model.safetensors.index.json").read_text("utf-8"))
    weight_map = index["weight_map"]
    rows: list[dict] = []
    header_info: dict[str, dict] = {}
    for shard, size, sha in BASE_SHARDS:
        path = headers_dir / (shard + ".head")
        if not path.is_file():
            raise SystemExit(f"missing safetensors header: {path}")
        header_len, parsed = read_safetensors_header(path)
        header_info[shard] = {
            "header_bytes": header_len,
            "downloaded_header_bytes": path.stat().st_size,
            "tensor_count": len(parsed["tensors"]),
            "metadata": parsed["metadata"],
        }
        for name, value in parsed["tensors"].items():
            rows.append({
                "name": name,
                "dtype": value["dtype"],
                "shape": list(value["shape"]),
                "bytes": int(value["data_offsets"][1]) - int(value["data_offsets"][0]),
                "shard": shard,
            })
    names = {r["name"] for r in rows}
    if len(rows) != len(names):
        raise SystemExit("duplicate tensor name in safetensors headers")
    if set(weight_map) != names:
        missing = sorted(set(weight_map) - names)[:8]
        extra = sorted(names - set(weight_map))[:8]
        raise SystemExit(f"index/header mismatch; missing={missing} extra={extra}")
    for row in rows:
        if weight_map[row["name"]] != row["shard"]:
            raise SystemExit(f"wrong shard for {row['name']}: {row['shard']} != {weight_map[row['name']]}")
    total_bytes = sum(r["bytes"] for r in rows)
    if total_bytes != index["metadata"]["total_size"]:
        raise SystemExit(f"tensor byte sum {total_bytes} != index total {index['metadata']['total_size']}")
    return rows, {
        "index_total_size": index["metadata"]["total_size"],
        "shard_count": len(BASE_SHARDS),
        "shard_file_bytes": sum(size for _, size, _ in BASE_SHARDS),
        "headers": header_info,
    }


def validate_base(rows: list[dict], config: dict) -> dict:
    if len(rows) != 19434:
        raise SystemExit(f"expected 19434 base tensors, got {len(rows)}")
    if len(config["rda_config"]["mla_layers"]) != 7:
        raise SystemExit("expected 7 MLA layers")
    if len(config["rda_config"]["rda_layers"]) != 18:
        raise SystemExit("expected 18 RDA layers")
    for key, expected in [
        ("hidden_size", 2560), ("num_hidden_layers", 25), ("num_attention_heads", 40),
        ("head_dim", 128), ("vocab_size", 259840), ("intermediate_size", 12032),
        ("first_k_dense_replace", 1), ("moe_layer_freq", 1), ("num_experts", 128),
        ("num_experts_per_token", 6), ("moe_intermediate_size", 1088),
        ("num_shared_experts", 1), ("q_lora_rank", 576), ("kv_lora_rank", 512),
        ("qk_nope_head_dim", 128), ("qk_rope_head_dim", 64), ("v_head_dim", 128),
        ("raon_mhc_streams", 4), ("max_position_embeddings", 262144),
        ("tie_word_embeddings", False),
    ]:
        if config.get(key) != expected:
            raise SystemExit(f"config {key}={config.get(key)!r}, expected {expected!r}")
    if config["rda_config"]["conv_kernel_size"] != 4:
        raise SystemExit("expected RDA conv kernel 4")
    if config["rda_config"]["num_heads"] != 40 or config["rda_config"]["head_dim"] != 128:
        raise SystemExit("unexpected RDA head geometry")

    moe_layers = [i for i in range(25) if i >= config["first_k_dense_replace"]]
    if len(moe_layers) != 24:
        raise SystemExit("expected 24 MoE layers")
    rda_zero = [n - 1 for n in config["rda_config"]["rda_layers"]]
    mla_zero = [n - 1 for n in config["rda_config"]["mla_layers"]]
    if sorted(rda_zero + mla_zero) != list(range(25)):
        raise SystemExit("RDA/MLA schedule does not cover all 25 layers exactly once")
    if sorted(mla_zero) != [3, 7, 11, 15, 19, 23, 24]:
        raise SystemExit(f"unexpected MLA 0-based layers: {mla_zero}")

    expert_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        name = row["name"]
        if name.startswith("model.layers."):
            layer = int(name.split(".")[2])
            if layer not in range(25):
                raise SystemExit(f"bad layer index in {name}")
            if ".moe.experts." in name:
                expert = int(name.split(".experts.")[1].split(".")[0])
                expert_rows[(layer, expert)].append(row)
    if len(expert_rows) != 3072:
        raise SystemExit(f"expected 3072 expert slots, got {len(expert_rows)}")
    for layer in moe_layers:
        ids = {expert for (layer_no, expert) in expert_rows if layer_no == layer}
        if ids != set(range(128)):
            raise SystemExit(f"layer {layer}: expert ids are not 0..127")
    for (layer, expert), group in expert_rows.items():
        if len(group) != 6:
            raise SystemExit(f"layer {layer} expert {expert}: expected 6 tensors, got {len(group)}")
    per_expert_bytes = {sum(r["bytes"] for r in group) for group in expert_rows.values()}
    if per_expert_bytes != {8616960}:
        raise SystemExit(f"unexpected per-expert bytes: {sorted(per_expert_bytes)}")
    expert_bytes = sum(sum(r["bytes"] for r in group) for group in expert_rows.values())
    if expert_bytes != 26471301120:
        raise SystemExit(f"unexpected expert byte total: {expert_bytes}")

    required = {
        "lm_head.weight": ("BF16", [259840, 2560]),
        "model.embed_tokens.weight": ("BF16", [259840, 2560]),
        "model.norm.weight": ("BF16", [2560]),
        "model.layers.0.mlp.gate_proj.weight": ("BF16", [12032, 2560]),
        "model.layers.0.mlp.up_proj.weight": ("BF16", [12032, 2560]),
        "model.layers.0.mlp.down_proj.weight": ("BF16", [2560, 12032]),
        "model.layers.1.moe.gate.weight": ("BF16", [128, 2560]),
        "model.layers.1.moe.gate.e_score_correction_bias": ("F32", [128]),
        "model.layers.1.moe.shared_experts.gate_proj.weight": ("BF16", [1088, 2560]),
        "model.layers.1.moe.shared_experts.up_proj.weight": ("BF16", [1088, 2560]),
        "model.layers.1.moe.shared_experts.down_proj.weight": ("BF16", [2560, 1088]),
        "model.layers.1.moe.experts.0.gate_proj.weight": ("F8_E4M3", [1088, 2560]),
        "model.layers.1.moe.experts.0.up_proj.weight": ("F8_E4M3", [1088, 2560]),
        "model.layers.1.moe.experts.0.down_proj.weight": ("F8_E4M3", [2560, 1088]),
        "model.layers.1.moe.experts.0.gate_proj.weight_scale_inv": ("U8", [1088, 80]),
        "model.layers.1.moe.experts.0.up_proj.weight_scale_inv": ("U8", [1088, 80]),
        "model.layers.1.moe.experts.0.down_proj.weight_scale_inv": ("U8", [2560, 34]),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [5120, 2560]),
        "model.layers.0.self_attn.q_conv1d.weight": ("BF16", [5120, 4]),
        "model.layers.0.self_attn.A_log": ("F32", [40]),
        "model.layers.0.self_attn.dt_bias": ("F32", [5120]),
        "model.layers.3.self_attn.q_a_proj.weight": ("BF16", [576, 2560]),
        "model.layers.3.self_attn.q_b_proj.weight": ("BF16", [7680, 576]),
        "model.layers.3.self_attn.kv_a_proj_with_mqa.weight": ("BF16", [576, 2560]),
        "model.layers.3.self_attn.kv_b_proj.weight": ("BF16", [10240, 512]),
        "model.layers.3.self_attn.g_proj.weight": ("BF16", [5120, 2560]),
    }
    by_name = {r["name"]: r for r in rows}
    for name, (dtype, shape) in required.items():
        row = by_name.get(name)
        if row is None:
            raise SystemExit(f"missing required tensor {name}")
        if row["dtype"] != dtype or row["shape"] != shape:
            raise SystemExit(f"{name}: {row['dtype']} {row['shape']} != {dtype} {shape}")

    dtypes = Counter(r["dtype"] for r in rows)
    if dtypes != Counter({"F8_E4M3": 9216, "U8": 9216, "BF16": 892, "F32": 110}):
        raise SystemExit(f"unexpected dtype counts: {dtypes}")

    total_params = 0
    active_params = Fraction(0)
    for row in rows:
        if row["dtype"] == "U8":
            continue
        count = math.prod(row["shape"])
        total_params += count
        if ".moe.experts." in row["name"]:
            active_params += Fraction(count * 6, 128)
        else:
            active_params += count
    hf_total = 28783218402
    scalar_params = sum(math.prod(r["shape"]) for r in rows if r["shape"] == [])
    if total_params - hf_total != scalar_params:
        raise SystemExit(
            f"header total {total_params} differs from HF summary {hf_total} "
            f"by {total_params - hf_total}, not by {scalar_params} scalar tensors"
        )
    active_int = int(active_params)
    if not (4_000_000_000 < active_int < 4_500_000_000):
        raise SystemExit(f"unexpected active params: {active_int}")

    layers = []
    for layer in range(25):
        group = [r for r in rows if r["name"].startswith(f"model.layers.{layer}.")]
        is_mla = layer in mla_zero
        is_moe = layer in moe_layers
        layers.append({
            "layer": layer,
            "attention": "mla" if is_mla else "rda",
            "mlp": "moe" if is_moe else "dense",
            "tensors": len(group),
            "bytes": sum(r["bytes"] for r in group),
            "expert_slots": len({int(r["name"].split(".experts.")[1].split(".")[0])
                                 for r in group if ".moe.experts." in r["name"]}),
        })
    if sum(x["tensors"] for x in layers) + 3 != len(rows):
        raise SystemExit("per-layer tensor count does not add up to the inventory")

    core_bytes = sum(r["bytes"] for r in rows) - expert_bytes
    return {
        "tensor_count": len(rows),
        "tensor_bytes": sum(r["bytes"] for r in rows),
        "dtypes": dict(sorted(dtypes.items())),
        "parameters": {
            "header_total": total_params,
            "hf_summary_total": hf_total,
            "scalar_tensors_excluded_by_hf": scalar_params,
            "active": active_int,
            "active_fraction": float(active_params / total_params),
        },
        "layers": layers,
        "layer_types": {"rda": 18, "mla": 7, "moe": 24, "dense": 1},
        "expert_slots": len(expert_rows),
        "expert_tensors": sum(len(group) for group in expert_rows.values()),
        "expert_bytes": expert_bytes,
        "expert_bytes_per_slot": 8616960,
        "core_bytes": core_bytes,
        "attention_schedule_zero_based": {
            "rda": rda_zero,
            "mla": mla_zero,
        },
        "shard_count": len(BASE_SHARDS),
        "shard_file_bytes": sum(size for _, size, _ in BASE_SHARDS),
    }


def parse_gguf(path: Path) -> tuple[list[dict], dict]:
    g = GGUF(str(path))
    if g.version != 3:
        raise SystemExit(f"unexpected GGUF version {g.version}")
    if len(g.tensors) != 731:
        raise SystemExit(f"expected 731 GGUF tensors, got {len(g.tensors)}")
    if g.data_start != 12048512:
        raise SystemExit(f"unexpected GGUF data_start {g.data_start}")
    rows = []
    for i, (name, ne, ty, off) in enumerate(g.tensors):
        next_off = g.tensors[i + 1][3] if i + 1 < len(g.tensors) else GGUF_FILE["size"] - g.data_start
        rows.append({
            "name": name,
            "shape": list(ne),
            "type_id": int(ty),
            "type": GGML_TYPES.get(int(ty), f"UNKNOWN_{int(ty)}"),
            "offset": int(off),
            "bytes": int(next_off) - int(off),
        })
    payload = sum(r["bytes"] for r in rows)
    if payload != GGUF_FILE["size"] - g.data_start:
        raise SystemExit(f"GGUF payload {payload} != {GGUF_FILE['size'] - g.data_start}")
    meta = {}
    for key in sorted(g.kv):
        if key.startswith("tokenizer.ggml."):
            if key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type"):
                meta[key] = {"count": len(g.kv[key])}
            else:
                meta[key] = g.kv[key]
        elif key.startswith(("general.", "raon1.")):
            meta[key] = g.kv[key]
    if meta.get("general.architecture") != "raon1":
        raise SystemExit("GGUF architecture is not raon1")
    if meta.get("general.license") != "apache-2.0":
        raise SystemExit("GGUF license is not apache-2.0")
    if meta.get("general.file_type") != 15:
        raise SystemExit("GGUF file_type is not Q4_K_M (15)")
    expected_meta = {
        "raon1.block_count": 25,
        "raon1.embedding_length": 2560,
        "raon1.context_length": 262144,
        "raon1.expert_count": 128,
        "raon1.expert_used_count": 6,
        "raon1.expert_shared_count": 1,
        "raon1.expert_feed_forward_length": 1088,
        "raon1.feed_forward_length": 12032,
        "raon1.attention.head_count": 40,
        "raon1.attention.q_lora_rank": 576,
        "raon1.attention.kv_lora_rank": 512,
        "raon1.attention.key_length_mla": 192,
        "raon1.attention.value_length_mla": 128,
        "raon1.hyper_connection.count": 4,
        "raon1.leading_dense_block_count": 1,
        "raon1.vocab_size": 259840,
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise SystemExit(f"GGUF {key}={meta.get(key)!r}, expected {expected!r}")
    if meta["raon1.attention.head_count_kv"] != [
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 1, 1,
    ]:
        raise SystemExit("unexpected GGUF head_count_kv schedule")
    if meta["tokenizer.ggml.tokens"]["count"] != 259840:
        raise SystemExit("unexpected GGUF tokenizer size")

    expert_tensors = [r for r in rows if "exps" in r["name"]]
    if len(expert_tensors) != 72:
        raise SystemExit(f"expected 72 packed expert tensors, got {len(expert_tensors)}")
    for row in expert_tensors:
        if row["shape"][-1] != 128:
            raise SystemExit(f"packed expert tensor {row['name']} does not have 128 experts")
        if row["bytes"] % 128:
            raise SystemExit(f"packed expert tensor {row['name']} is not divisible by 128")
    per_layer = defaultdict(int)
    for row in expert_tensors:
        per_layer[row["name"].split(".")[1]] += row["bytes"]
    if len(per_layer) != 24:
        raise SystemExit("packed experts do not cover 24 layers")
    per_expert = sorted({row["bytes"] // 128 for row in expert_tensors})
    per_expert_by_layer = {
        layer: total // 128 for layer, total in sorted(per_layer.items(), key=lambda kv: int(kv[0]))
    }
    type_counts = Counter(r["type"] for r in rows)
    return rows, {
        "tensor_count": len(rows),
        "tensor_bytes": payload,
        "file_bytes": GGUF_FILE["size"],
        "data_start": g.data_start,
        "gguf_version": g.version,
        "type_counts": dict(sorted(type_counts.items())),
        "metadata": meta,
        "packed_expert_tensors": len(expert_tensors),
        "packed_expert_bytes_per_expert_values": per_expert,
        "packed_expert_bytes_per_expert_by_layer": per_expert_by_layer,
        "packed_expert_bytes_per_layer": dict(sorted(per_layer.items(), key=lambda kv: int(kv[0]))),
    }


def cache_numbers(config: dict) -> dict:
    mla = 7
    rda = 18
    heads = config["num_attention_heads"]
    head_dim = config["rda_config"]["head_dim"]
    proj = heads * head_dim
    conv = config["rda_config"]["conv_kernel_size"]
    qk = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    v = config["v_head_dim"]
    kv_lora = config["kv_lora_rank"]
    rope = config["qk_rope_head_dim"]
    compressed = mla * (kv_lora + rope) * 2
    expanded = mla * heads * (qk + v) * 2
    rda_state = rda * (3 * proj * (conv - 1) * 2 + heads * head_dim * head_dim * 4)
    ctx = config["max_position_embeddings"]
    return {
        "mla_layers": mla,
        "rda_layers": rda,
        "compressed_mla_bytes_per_token": compressed,
        "compressed_mla_kib_per_token": compressed / 1024,
        "expanded_hf_mla_bytes_per_token": expanded,
        "expanded_hf_mla_kib_per_token": expanded / 1024,
        "rda_state_bytes_per_sequence": rda_state,
        "rda_state_mib_per_sequence": rda_state / (1 << 20),
        "context_tokens": ctx,
        "compressed_mla_bytes_at_context": compressed * ctx,
        "compressed_mla_gib_at_context": compressed * ctx / (1 << 30),
        "expanded_hf_mla_bytes_at_context": expanded * ctx,
        "expanded_hf_mla_gib_at_context": expanded * ctx / (1 << 30),
        "total_compressed_state_gib_at_context": (compressed * ctx + rda_state) / (1 << 30),
        "total_expanded_hf_state_gib_at_context": (expanded * ctx + rda_state) / (1 << 30),
        "notes": (
            "Published model card: ~8 KiB/token BF16, i.e. 7 MLA layers x 576 BF16 values "
            "= 8,064 B/token. The HF reference DynamicCache stores expanded per-head K/V "
            "(40 x 192 + 40 x 128) x 2 B x 7 = 179,200 B/token. The GGUF metadata exposes "
            "key_length=576 and value_length=512; the exact llama.cpp cache layout must be "
            "confirmed from its source, not inferred from the model card."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headers", type=Path, default=HERE / "headers")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--gguf", type=Path, default=HERE / "headers" / "raon-q4km.head")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, (size, expected_sha) in BASE_FILES.items():
        path = HERE / "upstream" / name
        if not path.is_file():
            raise SystemExit(f"missing upstream file: {path}")
        if path.stat().st_size != size:
            raise SystemExit(f"{name}: size {path.stat().st_size} != {size}")
        actual = sha256_file(path)
        if actual != expected_sha:
            raise SystemExit(f"{name}: sha256 {actual} != {expected_sha}")
    arg_path = HERE / "upstream" / "llama-common-arg.cpp"
    if sha256_file(arg_path) != LLAMA_CPP["arg_sha256"]:
        raise SystemExit("llama.cpp arg.cpp hash mismatch")

    base_rows, index_info = parse_base(args.headers)
    config = json.loads((HERE / "upstream" / "config.json").read_text("utf-8"))
    base = validate_base(base_rows, config)
    base["index"] = index_info

    gguf_rows, gguf = parse_gguf(args.gguf)
    if sha256_file(args.gguf) != GGUF_FILE["header_sha256"]:
        raise SystemExit("GGUF header hash mismatch")

    cache = cache_numbers(config)

    base_doc = {
        "source": {
            "repo": SOURCE_PINS["base_model"]["repo"],
            "revision": BASE_REVISION,
            "license": "apache-2.0",
            "note": "Headers only; tensor data not downloaded.",
        },
        "fields": ["name", "dtype", "shape", "bytes", "shard"],
        "summary": base,
        "tensors": [
            [r["name"], r["dtype"], r["shape"], r["bytes"], r["shard"]]
            for r in base_rows
        ],
    }
    gguf_doc = {
        "source": {
            "repo": SOURCE_PINS["gguf"]["repo"],
            "revision": GGUF_REVISION,
            "file": GGUF_FILE["name"],
            "sha256": GGUF_FILE["sha256"],
            "note": "Header only; payload not downloaded.",
        },
        "fields": ["name", "shape", "type_id", "type", "offset", "bytes"],
        "summary": gguf,
        "tensors": [
            [r["name"], r["shape"], r["type_id"], r["type"], r["offset"], r["bytes"]]
            for r in gguf_rows
        ],
    }
    summary = {
        "source_pins": {
            "base_revision": BASE_REVISION,
            "gguf_revision": GGUF_REVISION,
            "llama_cpp_commit": LLAMA_CPP["commit"],
            "official_repo_commit": OFFICIAL_REPO["commit"],
        },
        "base": base,
        "gguf_q4_k_m": gguf,
        "cache": cache,
        "comparison": {
            "base_tensor_bytes": base["tensor_bytes"],
            "q4_file_bytes": GGUF_FILE["size"],
            "q4_to_base_ratio": GGUF_FILE["size"] / base["tensor_bytes"],
            "base_expert_share": base["expert_bytes"] / base["tensor_bytes"],
            "base_active_expert_bytes_6_of_128": base["expert_bytes"] * 6 // 128,
            "base_cold_6_expert_bytes": base["expert_bytes_per_slot"] * 6,
            "q4_expert_bytes_per_slot_min": min(gguf["packed_expert_bytes_per_expert_by_layer"].values()),
            "q4_expert_bytes_per_slot_max": max(gguf["packed_expert_bytes_per_expert_by_layer"].values()),
            "q4_cold_6_expert_bytes_min": min(gguf["packed_expert_bytes_per_expert_by_layer"].values()) * 6,
            "q4_cold_6_expert_bytes_max": max(gguf["packed_expert_bytes_per_expert_by_layer"].values()) * 6,
        },
        "validation": {
            "base_index_matches_headers": True,
            "all_3072_expert_slots_have_128_ids_and_6_tensors": True,
            "base_params_match_hf_safetensors_summary": True,
            "gguf_metadata_matches_config": True,
            "gguf_file_type_q4_k_m": 15,
        },
    }

    (args.out / "base-tensor-inventory.json").write_text(
        json.dumps(base_doc, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (args.out / "gguf-q4km-tensor-inventory.json").write_text(
        json.dumps(gguf_doc, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (args.out / "manifest-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (args.out / "source-pins.json").write_text(
        json.dumps(SOURCE_PINS, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"base: {len(base_rows)} tensors, {base['tensor_bytes']} bytes, "
          f"{base['parameters']['header_total']} header params, "
          f"{base['parameters']['active']} active")
    print(f"gguf: {len(gguf_rows)} tensors, {gguf['tensor_bytes']} payload, {gguf['file_bytes']} file")
    print(f"cache: compressed {cache['compressed_mla_bytes_per_token']} B/token, "
          f"HF expanded {cache['expanded_hf_mla_bytes_per_token']} B/token")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
