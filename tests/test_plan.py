# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi plan`: the dry-run placement report, without loading a weight tensor.

The fake pack carries only the manifest and the two tensor indexes the planners read, so if any
code path opened a weight file the test would fail with a missing-file error instead of passing.
"""
import json
import shlex
from pathlib import Path

import pytest

from kindi import io_probe
from kindi import plan as _plan
from kindi.moe import MoESpec, Usage


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


def write_fake_gguf(path, tensors, n_exp=32):
    """A minimal GGUF v3 whose tensors are all type 42 (Q2_0), just names and shapes.

    The engine reads weights from a real file; `plan` needs only the header and `tinfo`, so
    data never follows. `n_exp` is the routed-expert count per MoE layer."""
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
        (a + ".expert_count", 4, n_exp),
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


def make_lightning_pack(tmp_path, free_gib=0.01, n_exp=32):
    pack = tmp_path / "ltpack"
    pack.mkdir()
    gguf = tmp_path / "lt.gguf"
    tensors = [("token_embd.weight", [256, 10], 8, 0)]
    off = 0
    for il in range(6):
        tensors.append(("blk.%d.exps.w1" % il, [128, 256], 42, off)); off += 1
        tensors.append(("blk.%d.exps.w2" % il, [256, 128], 42, off)); off += 1
        tensors.append(("blk.%d.some_q2" % il, [256, 128], 42, off)); off += 1
    write_fake_gguf(gguf, tensors, n_exp=n_exp)
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
    # the decode-slot split is inspectable: with every expert resident it says the slots are unused
    slots = next(r for c, r in rep["lines"] if c == "slots")
    assert "unused" in slots
    assert rep["dcache"] >= 4
    strat = next(r for c, r in rep["lines"] if c == "strategy")
    assert "expert" in strat and "host" in strat
    assert rep["n_exp"] == 32 and rep["fit"] is True
    assert 4 <= rep["n_card"] <= 32                            # at least n_used per layer
    assert rep["command"].endswith("--ctx 2048 bench --dcache %d" % rep["dcache"])


def test_describe_moe_reports_the_cold_host_tier(tmp_path, monkeypatch):
    """With the cold tier enabled, the dry run uses the same host/disk split as the engine."""
    monkeypatch.setenv("KINDI_LT_COLD", "1")
    monkeypatch.setenv("KINDI_LT_HOST_MB", "1")
    pack = make_lightning_pack(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, moe_cache_mib=1, dcache=4)
    tier = rep["host_tier"]
    assert tier["budget_bytes"] == 1 << 20
    assert tier["host_experts"] > 0 and tier["disk_experts"] > 0
    assert tier["host_bytes"] <= tier["budget_bytes"]
    assert tier["host_bytes"] + tier["disk_bytes"] == \
        (tier["host_experts"] + tier["disk_experts"]) * rep["expert_bytes"]
    host = next(r for c, r in rep["lines"] if c == "host ram")
    experts = next(r for c, r in rep["lines"] if c == "experts")
    assert "%d pinned in host memory" % tier["host_experts"] in host
    assert "%d file-backed" % tier["disk_experts"] in host
    assert "1 MiB host budget" in host
    assert "file-backed overflow" in experts


def test_the_fit_prices_a_slot_at_more_than_one_expert():
    """`_lt_fit` takes the decode slots off the room *before* dividing it over the remaining
    (n_moe - 2) layers, so dcache slots per layer cost dcache * n_moe / (n_moe - 2) experts per
    layer. With 4 MoE layers one slot costs two experts per layer, not the one the slots line
    used to claim."""
    spec = type("Spec", (), {"n_exp": 32, "n_used": 4})()
    eb, n_moe, dcache = 1000, 4, 10
    room = 2 * spec.n_exp * eb + 60 * eb                 # 30 experts per layer once the two full layers are paid
    fit = lambda d: _plan._lt_fit(spec, room, 0, 0, 0, 0, 0, eb, [0] * n_moe, dcache=d)[0]
    assert fit(0) == 30                                  # the same room as pure residency
    assert fit(dcache) == 10
    assert fit(0) - fit(dcache) == dcache * n_moe // (n_moe - 2) == 20


def test_the_slots_line_uses_the_fits_own_numbers(tmp_path, monkeypatch):
    """The line must report what the engine's fit actually does: with 4 MoE layers and dcache=10
    the slots cost 20 experts per layer (n_moe / (n_moe - 2) x dcache), so 162 experts per layer
    stay resident against 182 without them, and the two coverage percentages must be the counts
    for those two numbers."""
    monkeypatch.setenv("KINDI_MOE_WEIGHTED", "0")       # the uniform split, not the usage-weighted scan
    pack = make_lightning_pack(tmp_path, n_exp=256)
    usage = tmp_path / "usage.json"
    counts = [256 - e for e in range(256)]
    usage.write_text(json.dumps({"counts": {str(il): counts for il in range(16)}}), encoding="utf-8")
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.09, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, usage_path=str(usage))
    assert rep["fit"] is True and rep["n_exp"] == 256
    assert (rep["n_card"], rep["slots_resident"], rep["dcache"]) == (162, 182, 10)
    # 4 MoE layers: the fit divides what the slots leave over the remaining 2, so a slot costs two
    assert rep["slots_resident"] - rep["n_card"] == 2 * rep["dcache"]

    def coverage(n):                                    # the share of picks the top n of each layer take
        return 100.0 * sum(counts[:n]) / sum(counts)
    slots = next(r for c, r in rep["lines"] if c == "slots")
    assert "Each slot displaces one" not in slots
    assert ("fits %d experts per layer instead of %d" % (rep["slots_resident"], rep["n_card"])) in slots
    assert ("cost %d per layer" % (rep["slots_resident"] - rep["n_card"])) in slots
    assert ("%.1f%% of the measured picks against %.1f%%"
            % (coverage(rep["n_card"]), coverage(rep["slots_resident"]))) in slots
    assert "%.1f%%" % coverage(162) == "86.4%" and "%.1f%%" % coverage(182) == "91.6%"


RATES = {"source": "kindi io", "ram_mib_s": 13456.7,
         "ram_bytes_s": 13456.7 * 2 ** 20, "disk_mib_s": 2509.8,
         "disk_bytes_s": 2509.8 * 2 ** 20, "pcie_h2d_gib_s": 6.0,
         "pcie_bytes_s": 6.0 * 2 ** 30, "pcie_d2h_gib_s": 6.1}


def test_describe_moe_sizes_slots_from_the_measured_rates(tmp_path):
    """A slot count chosen from `kindi io` prints the RAM/PCIe/disk rates it used, in its own
    `rates:` line and in the slot explanation."""
    pack = make_lightning_pack(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, rates=RATES)
    assert rep["rate_sized"] is True and rep["dcache"] == 16
    assert rep["rates"] == RATES
    choices = {c for c, _ in rep["lines"]}
    assert "rates" in choices
    rates = next(r for c, r in rep["lines"] if c == "rates")
    assert "RAM 13456.7 MiB/s" in rates
    assert "PCIe H2D 6.0 GiB/s" in rates
    assert "disk 2509.8 MiB/s" in rates
    slots = next(r for c, r in rep["lines"] if c == "slots")
    assert "assumed RAM 13456.7 MiB/s" in slots
    assert "sized from the measured I/O rates" in next(r for c, r in rep["lines"] if c == "strategy")


def test_the_planned_bench_command_reproduces_the_rate_sized_slot_count(tmp_path):
    """The plan's child command must set the count the report used, not the engine's default 10.

    It is a portable --dcache flag, not a POSIX-only KINDI_LT_DCACHE=... prefix.
    """
    pack = make_lightning_pack(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, rates=RATES)
    assert rep["rate_sized"] is True and rep["dcache"] == 16
    assert rep["command"].split()[-5:] == ["--ctx", "2048", "bench", "--dcache",
                                           str(rep["dcache"])]
    assert "KINDI_LT_DCACHE=" not in rep["command"]


def test_follow_up_commands_quote_a_spaced_pack_and_interpreter(tmp_path, monkeypatch):
    """A spaced pack or interpreter path must stay one argument when the command is pasted.

    POSIX is parsed back with `shlex`; Windows must carry the cmd.exe double quotes.
    """
    pack = tmp_path / "my packs" / "bonsai 2b"
    pack.mkdir(parents=True)
    monkeypatch.setattr(_plan, "_WINDOWS", False)
    monkeypatch.setattr(_plan.sys, "executable", "/opt/my python/bin/python3")
    posix = _plan._moe_bench_command(str(pack), 2048, 16)
    assert shlex.split(posix) == [
        "/opt/my python/bin/python3", "-m", "kindi", "--pack", str(pack),
        "--ctx", "2048", "bench", "--dcache", "16"]
    tandem = _plan._command(str(pack), "cpu-first", None)
    assert shlex.split(tandem) == [
        "/opt/my python/bin/python3", "-m", "kindi", "--pack", str(pack),
        "bench", "--tandem", "cpu-first", "--tandem-cpu-layers", "auto"]

    monkeypatch.setattr(_plan, "_WINDOWS", True)
    monkeypatch.setattr(_plan.sys, "executable", r"C:\Program Files\Python 313\python.exe")
    win_pack = r"C:\my packs\bonsai 2b"
    win = _plan._moe_bench_command(win_pack, 2048, 16)
    assert win == ('"C:\\Program Files\\Python 313\\python.exe" -m kindi --pack '
                   '"C:\\my packs\\bonsai 2b" --ctx 2048 bench --dcache 16')
    win_tandem = _plan._command(win_pack, "cpu-first", None)
    assert win_tandem == ('"C:\\Program Files\\Python 313\\python.exe" -m kindi --pack '
                          '"C:\\my packs\\bonsai 2b" bench --tandem cpu-first '
                          '--tandem-cpu-layers auto')


def test_the_planned_bench_command_carries_the_planned_context(tmp_path):
    """The fit and slot count were computed at the planned ctx, so the child bench must use it.

    `--ctx` is a global option and is only accepted before the `bench` subcommand.
    """
    pack = make_lightning_pack(tmp_path)
    rep = _plan.describe(str(pack), 3072, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, rates=RATES)
    argv = rep["command"].split()
    assert argv[argv.index("--ctx") + 1] == "3072"
    assert argv.index("--ctx") < argv.index("bench")
    assert argv[-3:] == ["bench", "--dcache", str(rep["dcache"])]


def test_run_measures_rates_for_a_lightning_pack(tmp_path, monkeypatch, capsys):
    pack = make_lightning_pack(tmp_path)
    monkeypatch.setattr(_plan, "_io_rates", lambda: RATES)
    monkeypatch.setenv("KINDI_TANDEM_RESERVE_MB", "0")
    args = _FakeArgs(assume_vram_gib=0.5, assume_ram_gib=4, assume_cores=2)
    assert _plan.run({"pack": str(pack), "ctx": 2048}, args) == 0
    out = capsys.readouterr().out
    assert "rates:" in out and "PCIe H2D 6.0 GiB/s" in out
    assert "--dcache 16" in out


def test_run_reuses_a_saved_io_report_instead_of_probing(tmp_path, monkeypatch, capsys):
    pack = make_lightning_pack(tmp_path)
    saved = tmp_path / "io.json"
    io_probe.save_report({
        "ram": {"available": True, "mib_s": 13456.7},
        "disk": {"available": True, "mib_s": 2509.8, "unbuffered": True},
        "pcie": {"available": True, "h2d_gib_s": 6.0, "d2h_gib_s": 6.1,
                 "device": "test"},
    }, str(saved))
    monkeypatch.setattr(_plan, "_io_rates",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("the probe ran")))
    monkeypatch.setenv("KINDI_TANDEM_RESERVE_MB", "0")
    args = _FakeArgs(assume_vram_gib=0.5, assume_ram_gib=4, assume_cores=2,
                     io_report=str(saved))
    assert _plan.run({"pack": str(pack), "ctx": 2048}, args) == 0
    out = capsys.readouterr().out
    assert "using saved I/O report" in out
    assert "rates:" in out and "PCIe H2D 6.0 GiB/s" in out
    assert "--dcache 16" in out


def test_run_falls_back_to_one_probe_when_the_saved_report_is_missing(tmp_path, monkeypatch, capsys):
    pack = make_lightning_pack(tmp_path)
    monkeypatch.setattr(io_probe, "probe", lambda **kw: {
        "ram": {"available": True, "mib_s": 13456.7},
        "disk": {"available": True, "mib_s": 2509.8, "unbuffered": True},
        "pcie": {"available": True, "h2d_gib_s": 6.0, "d2h_gib_s": 6.1},
    })
    monkeypatch.setenv("KINDI_TANDEM_RESERVE_MB", "0")
    args = _FakeArgs(assume_vram_gib=0.5, assume_ram_gib=4, assume_cores=2,
                     io_report=str(tmp_path / "missing.json"))
    assert _plan.run({"pack": str(pack), "ctx": 2048}, args) == 0
    out = capsys.readouterr().out
    assert "no saved I/O report at" in out and "measuring RAM/disk/PCIe now" in out
    assert "rates:" in out and "PCIe H2D 6.0 GiB/s" in out


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


def _moe_pack_with_counts(tmp_path):
    pack = make_lightning_pack(tmp_path)
    usage = tmp_path / "usage.json"
    usage.write_text(json.dumps({"counts": {str(il): [100 - e for e in range(32)]
                                          for il in range(16)}}), encoding="utf-8")
    return pack, usage


def test_describe_moe_reports_the_tandem_chunk_split(tmp_path):
    """`--moe-chunk` reports what MoeTandem would do for a sample prompt chunk: the lane split,
    the deduped copy list and its bytes. It runs the scheduler's bookkeeping, not a copy."""
    pack, usage = _moe_pack_with_counts(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, usage_path=str(usage),
                         moe_cache_mib=1, dcache=4, chunk_tokens=4)
    chunk = rep["chunk"]
    assert chunk["tokens"] == 4 and chunk["lane"] == "copy"
    assert chunk["selections"] == chunk["tokens"] * chunk["layers"] * chunk["n_used"]
    assert chunk["card_pairs"] + chunk["copy_pairs"] + chunk["cpu_pairs"] == chunk["selections"]
    assert chunk["cpu_pairs"] == 0                     # the default lane is copy
    assert chunk["copy_pairs"] > 0 and 0 < chunk["copies"] <= chunk["copy_pairs"]
    assert chunk["copy_bytes"] == chunk["copies"] * 18432
    assert chunk["saved_copies"] == chunk["copy_pairs"] - chunk["copies"]
    choices = {c for c, _ in rep["lines"]}
    assert {"chunk", "copies"} <= choices
    copies = next(r for c, r in rep["lines"] if c == "copies")
    assert "no bytes are moved" in copies


def test_describe_moe_chunk_can_plan_the_cpu_lane(tmp_path):
    pack, usage = _moe_pack_with_counts(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, usage_path=str(usage),
                         moe_cache_mib=1, dcache=4, chunk_tokens=2, chunk_lane="cpu")
    chunk = rep["chunk"]
    assert chunk["lane"] == "cpu" and chunk["cpu_pairs"] > 0
    # every host expert is computed on the host, so this sample copies nothing
    assert chunk["copy_pairs"] == 0 and chunk["copies"] == 0 and chunk["copy_bytes"] == 0


def test_describe_moe_chunk_works_without_usage_counts(tmp_path):
    """No counts: the sample takes cyclic ranks of the index order, so the report still shows the
    dedupe a real chunk gets (adjacent tokens share experts)."""
    pack = make_lightning_pack(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.5, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, moe_cache_mib=1, dcache=4,
                         chunk_tokens=4)
    chunk = rep["chunk"]
    assert chunk["selections"] == chunk["tokens"] * chunk["layers"] * chunk["n_used"]
    assert chunk["copies"] < chunk["copy_pairs"]        # overlapping sample tokens share copies


def test_run_prints_the_chunk_report_when_asked(tmp_path, capsys):
    pack = make_lightning_pack(tmp_path)
    args = _FakeArgs(assume_vram_gib=0.5, assume_ram_gib=4, assume_cores=2, reserve_mib=0,
                     moe_cache_mib=1, dcache=4, moe_chunk=2, moe_lane="copy")
    rc = _plan.run({"pack": str(pack), "ctx": 2048}, args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "chunk:" in out and "copies:" in out and "lane" in out


def test_slots_cost_is_the_weighted_plans_counterfactual():
    """What the decode slots cost a weighted plan is the difference from the same room with
    `dcache=0`, not an assumed one expert per layer. Two layers, 4 experts, 1 used, 1-byte
    experts, room 10:

    - with 1 slot per layer the budget is 10 + 2m - 10; m=2 fits 4 experts and serves 90%,
      m=3 cannot fit, so the plan takes 4 experts and 90%.
    - with no slots the budget is 10 + 2m - 8; m=2 fits 6 experts (3 per layer) and serves 100%.
    """
    spec = MoESpec([0, 1], 4, 1, 1, shared=True)
    usage = Usage(spec, counts={0: [8, 1, 1, 0], 1: [8, 1, 1, 0]})
    with_slots, n_card = _plan._weighted_placement(spec, usage, 10, None, 1, 2, 1, 2)
    no_slots, _ = _plan._weighted_placement(spec, usage, 10, None, 1, 2, 0, 2)
    assert with_slots.counts("card") == {0: 2, 1: 2} and n_card == 2
    assert no_slots.counts("card") == {0: 3, 1: 3}
    cost = _plan._slots_cost(with_slots, no_slots)
    assert cost == {"served": 0.9, "served_without": 1.0, "cost_pp": 10.0,
                    "experts": 4, "experts_without": 6}
    assert _plan._slots_cost_phrase(cost) == (
        "the usage-weighted placement leaves room for them: it serves 90.0% of the measured "
        "picks against 100.0% with the same room and no slots, so they cost 10.0 points of "
        "served share and 2 card experts (4 instead of 6)")


def test_slots_cost_needs_both_side_placements():
    spec = MoESpec([0], 4, 1, 1, shared=True)
    usage = Usage(spec, counts={0: [8, 1, 1, 0]})
    placed, _ = _plan._weighted_placement(spec, usage, 8, None, 1, 1, 1, 2)
    assert _plan._slots_cost(placed, None) is None
    assert _plan._slots_cost(None, placed) is None


def test_describe_moe_reports_the_slots_cost_on_a_tight_card(tmp_path):
    """On a card where the slots genuinely compete with experts, the slots line states what the
    weighted plan gives up: 4 slots per layer cost 9 card experts and 5.8 points of served share
    against the same room with `dcache=0`, and the report dict carries the same numbers.
    """
    pack, usage = _moe_pack_with_counts(tmp_path)
    rep = _plan.describe(str(pack), 2048, None, assume_vram_gib=0.077, assume_ram_gib=4,
                         assume_cores=2, reserve_mib=0, usage_path=str(usage),
                         moe_cache_mib=None, dcache=4)
    assert rep["slots_cost"] == {"served": 0.94203, "served_without": 1.0, "cost_pp": 5.8,
                                "experts": 119, "experts_without": 128}
    slots = next(r for c, r in rep["lines"] if c == "slots")
    assert ("it serves 94.2% of the measured picks against 100.0% with the same room and no "
            "slots, so they cost 5.8 points of served share and 9 card experts (119 instead "
            "of 128)") in slots
