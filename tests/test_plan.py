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
