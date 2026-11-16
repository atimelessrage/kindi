# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi plan`: the dry-run placement report, without loading a weight tensor.

The fake pack carries only the manifest and the two tensor indexes the planners read, so if any
code path opened a weight file the test would fail with a missing-file error instead of passing.
"""
import json
from pathlib import Path

import pytest

from kindi import plan as _plan


def make_pack(tmp_path, *, layers=4, interval=2, scale=1):
    """A tiny pack: manifest + minimal indexes for `pack_bytes`/`gpu_estimate` to work."""
    pack = tmp_path / "pack"
    pack.mkdir()
    dims = {"layers": layers, "full_attention_interval": interval, "kv_heads": 2,
            "head_dim": 16, "state_size": 4, "gdn_inner": 8, "groups": 2}
    man = {"model": "fake", "backend": "kindi", "dimensions": dims,
           "source_sha256": "x", "f32_bytes": 0, "head": "token_embd.weight"}
    (pack / "kindi_pack.json").write_text(json.dumps(man))
    tensors = {}
    for il in range(layers):
        tensors["blk.%d.w" % il] = {"rows": 16, "ne0": 128}
    (pack / "bcodes_index.json").write_text(json.dumps({"tensors": tensors}))
    (pack / "index.json").write_text(json.dumps({"tensors": {"token_embd.weight":
                                                              {"nfloats": 128 * scale}}}))
    return pack


def test_describe_reports_every_choice_with_a_reason(tmp_path):
    pack = make_pack(tmp_path)
    rep = _plan.describe(str(pack), 100, "auto", assume_vram_gib=0.01, assume_ram_gib=4,
                         assume_cores=2)
    keys = [k for k, _ in rep["lines"]]
    assert {"model", "context", "vram", "host ram", "cpu threads", "gpu owner",
            "placement"} <= set(keys)
    assert all(reason for _, reason in rep["lines"])          # one reason per choice
    assert rep["owner"] == "plan"
    assert rep["command"].endswith("bench --tandem cpu-first --tandem-cpu-layers auto")


def test_describe_needs_no_card_when_assumed(tmp_path):
    pack = make_pack(tmp_path)
    rep = _plan.describe(str(pack), 512, "gdn:1", assume_vram_gib=8, assume_ram_gib=4,
                         assume_cores=4)
    assert rep["fit"] is True
    assert rep["cpu_layers"] == [0]                            # the first GDN layer (of [0, 2])
    assert rep["lines"][2][1].startswith("assumed")          # the vram reason names the assumption


def test_describe_refuses_honestly_when_nothing_fits(tmp_path):
    pack = make_pack(tmp_path)
    rep = _plan.describe(str(pack), 10 ** 6, "auto", assume_vram_gib=0.001, assume_ram_gib=4,
                         assume_cores=2)
    # 1M tokens cannot be budgeted on 1 MiB: the planner's refusal is reported, not raised,
    # and it is never rendered as a fake fit.
    assert rep["fit"] is False
    assert "cannot fit" in rep["lines"][-1][1]


def test_describe_without_a_card_reports_the_model_only(tmp_path, monkeypatch):
    pack = make_pack(tmp_path)
    monkeypatch.setattr(_plan, "_free_vram_gib", lambda: None)
    rep = _plan.describe(str(pack), 100, "auto", assume_ram_gib=4, assume_cores=2)
    assert rep["fit"] is None and rep["cpu_layers"] is None
    assert any("no CUDA device visible" in r for _, r in rep["lines"])
    assert rep["command"]                                    # the follow-up command is still printed


def test_describe_missing_pack_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        _plan.describe(str(tmp_path / "nope"), 100, "auto", assume_vram_gib=1)


def test_run_prints_and_returns_zero(tmp_path, capsys):
    pack = make_pack(tmp_path)
    args = _FakeArgs(assume_vram_gib=0.01, assume_ram_gib=4, assume_cores=2)
    rc = _plan.run({"pack": str(pack), "ctx": 100}, args)
    assert rc == 0                                            # sys.exit(int) - never the dict
    out = capsys.readouterr().out
    assert "placement:" in out and "follow up:" in out


class _FakeArgs:
    def __init__(self, **kw):
        self.tandem_cpu_layers = None
        self.tandem = None
        self.tandem_gpu = None
        self.assume_vram_gib = None
        self.assume_ram_gib = None
        self.assume_cores = None
        self.json = None
        self.__dict__.update(kw)


# ---------- the MoE (kindi_lightning) strategy ----------

import struct


def _gguf_value(buf, ty, v):
    if ty == 8:                                                 # STRING
        s = v.encode()
        buf += struct.pack("<Q", len(s)) + s
        return
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?>", 10: "<Q", 11: "<q", 12: "<d"}[ty]
    if fmt == "<?>":
        buf += struct.pack("<b", 1 if v else 0)
        return
    buf += struct.pack(fmt, v)


def write_fake_gguf(path, tensors):
    """A minimal GGUF v3 whose tensors are all type 42 (Q2_0), just names and shapes.

    The engine reads weights from a real file; `plan` needs only the header and `tinfo`, so
    data never follows."""
    a = "nemotron_h_moe"
    kv = [
        ("general.architecture", 8, a),
        (a + ".block_count", 4, 6),
        (a + ".embedding_length", 4, 256),
        (a + ".feed_forward_length", 9, None),                 # array I; patched below
        (a + ".attention.head_count_kv", 9, None),             # array I; patched below
        (a + ".attention.head_count", 4, 8),
        (a + ".attention.key_length", 4, 128),
        (a + ".attention.layer_norm_rms_epsilon", 6, 0.00001),
        (a + ".ssm.conv_kernel", 4, 4),
        (a + ".ssm.state_size", 4, 128),
        (a + ".ssm.group_count", 4, 2),
        (a + ".ssm.inner_size", 4, 64),
        (a + ".ssm.time_step_rank", 4, 1),
        (a + ".expert_count", 4, 32),
        (a + ".expert_used_count", 4, 4),
        (a + ".expert_feed_forward_length", 4, 128),
        (a + ".expert_shared_feed_forward_length", 4, 64),
        (a + ".context_length", 4, 262144),
        (a + ".expert_weights_norm", 7, True),
        ("tokenizer.ggml.tokens", 9, None),                    # array of strings; patched below
    ]
    arrays = {
        a + ".feed_forward_length": [0, 128, 128, 0, 128, 128],
        a + ".attention.head_count_kv": [2, 0, 0, 2, 0, 0],
        "tokenizer.ggml.tokens": ["a", "b"],
    }
    out = bytearray(b"GGUF") + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for k, ty, v in kv:
        out += struct.pack("<Q", len(k)) + k.encode() + struct.pack("<I", ty)
        if ty == 9 and k in arrays:
            vals = arrays[k]
            et = 8 if isinstance(vals[0], str) else 4
            out += struct.pack("<I", et) + struct.pack("<Q", len(vals))
            for x in vals:
                _gguf_value(out, et, x)
        else:
            _gguf_value(out, ty, v)
    for name, ne, ty, off in tensors:
        out += struct.pack("<Q", len(name)) + name.encode()
        out += struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", x) for x in ne)
        out += struct.pack("<I", ty) + struct.pack("<Q", off)
    Path(path).write_bytes(bytes(out))


def make_lightning_pack(tmp_path, free_gib=0.01):
    pack = tmp_path / "ltpack"
    pack.mkdir()
    gguf = tmp_path / "lt.gguf"
    tensors = [("token_embd.weight", [256, 10], 8, 0)]
    off = 0
    for il in range(6):
        tensors.append(("blk.%d.exps.w1" % il, [128, 256], 42, off)); off += 1
        tensors.append(("blk.%d.exps.w2" % il, [256, 128], 42, off)); off += 1
        tensors.append(("blk.%d.some_q2" % il, [256, 128], 42, off)); off += 1
    write_fake_gguf(gguf, tensors)
    (pack / "kindi_pack.json").write_text(json.dumps({
        "backend": "kindi_lightning", "model": "fake-lt", "model_path": str(gguf),
        "source_sha256": "x"}))
    return pack


def test_describe_moe_uses_the_expert_tiers(tmp_path):
    pack = make_lightning_pack(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0)
    assert rep["strategy"] == "moe-expert-tiers"
    choices = {c for c, _ in rep["lines"]}
    assert {"strategy", "host ram", "reserve", "experts", "slots", "placement"} <= choices
    # the decode-slot split is inspectable: it names the bytes, and that each slot costs one expert
    slots = next(r for c, r in rep["lines"] if c == "slots")
    assert "displace" in slots or "unused" in slots
    assert rep["dcache"] >= 4
    strat = next(r for c, r in rep["lines"] if c == "strategy")
    assert "expert" in strat and "host" in strat
    assert rep["n_exp"] == 32 and rep["fit"] is True
    assert 4 <= rep["n_card"] <= 32                            # at least n_used per layer
    assert rep["command"].endswith("bench")


def test_describe_moe_reports_a_refusal_when_nothing_fits(tmp_path):
    pack = make_lightning_pack(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.0002, assume_ram_gib=4,
                         assume_cores=2)
    # 0.2 MiB free cannot hold even 4 experts per layer plus the core: reported, not raised
    assert rep["fit"] is False
    assert rep["n_card"] < 4


def test_describe_moe_caps_the_card_budget(tmp_path):
    pack = make_lightning_pack(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, moe_cache_mib=2, reserve_mib=0)
    # one expert is 2*128*256//64*18 = 18432 B; 4 MoE layers -> 73728 B per expert on the card;
    # a 2 MiB cap fits 2<<20 // 73728 = 28 of 32 (the 350 MiB default reserve is neutralized here)
    assert rep["n_card"] == 28
    assert any("cache cap" in c for c, _ in rep["lines"])


def test_describe_moe_needs_a_vram_number_without_a_card(tmp_path, monkeypatch):
    pack = make_lightning_pack(tmp_path)
    monkeypatch.setattr(_plan, "_free_vram_gib", lambda: None)
    with __import__("pytest").raises(SystemExit):
        _plan.describe(str(pack), 2048, None, assume_ram_gib=4, assume_cores=2)


def test_describe_moe_without_the_gguf_is_an_error(tmp_path):
    pack = make_lightning_pack(tmp_path)
    (pack / "kindi_pack.json").write_text(json.dumps({"backend": "kindi_lightning"}))
    with __import__("pytest").raises(SystemExit):
        _plan.describe(str(pack), 2048, None, assume_vram_gib=1.0)


def test_describe_moe_reports_the_measured_share_with_usage_counts(tmp_path):
    """The slots line quotes measured picks when the usage file has counts, and the measured
    share never contradicts a --moe-cache-mib cap (the weighted scan used to spend free VRAM
    and report every expert on the card despite the cap)."""
    pack = make_lightning_pack(tmp_path)
    usage = tmp_path / "usage.json"
    counts = {str(il): [100 - e for e in range(32)] for il in range(16)}
    usage.write_text(json.dumps({"counts": counts}), encoding="utf-8")
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, usage_path=str(usage),
                         moe_cache_mib=1, dcache=4)
    # 1 MiB / (4 MoE layers x 18,432 B) = 14 experts per layer, and that is the ceiling
    assert rep["n_card"] <= 14
    slots = next(r for c, r in rep["lines"] if c == "slots")
    assert "%" in slots and "measured picks" in slots
    assert "--dcache" in slots
