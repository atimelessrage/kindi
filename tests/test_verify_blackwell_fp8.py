# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for the Blackwell FP8 verification lane.

The hardware checks are exercised on a Blackwell runner by
``tools/verify_blackwell_fp8.py``; these tests pin the gating, dry-run and report contract without
needing CUDA.
"""
from tools.verify_blackwell_fp8 import (
    ARCH_BY_CC,
    arch_label,
    arch_matches,
    main,
    write_report,
)
import json

import pytest


def test_supported_blackwell_arches_are_exact():
    assert ARCH_BY_CC == {100: "sm_100", 120: "sm_120"}
    assert arch_label(100) == "sm_100"
    assert arch_label(120) == "sm_120"
    assert arch_label("120") == "sm_120"
    assert arch_label((10, 0)) == "sm_100"
    assert arch_label([12, 0]) == "sm_120"
    assert arch_label(89) is None
    assert arch_label(None) is None


def test_arch_gate_rejects_ada_and_the_other_blackwell_arch():
    assert not arch_matches(89, "auto")
    assert not arch_matches(89, "sm_100")
    assert arch_matches(100, "auto")
    assert arch_matches(100, "sm_100")
    assert not arch_matches(100, "sm_120")
    assert arch_matches(120, "auto")
    assert not arch_matches(120, "sm_100")
    with pytest.raises(ValueError):
        arch_matches(100, "sm_90")


def test_dry_run_needs_no_cuda_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "results"
    assert main(["--dry-run", "--arch", "sm_120", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "dry run" in text
    assert "compute capability 100 or 120" in text
    assert "Ops.linear_batch" in text
    assert not out.exists()


def test_report_records_unsupported_as_not_a_pass(tmp_path):
    out = tmp_path / "blackwell"
    write_report(
        out,
        {
            "status": "unsupported",
            "requested_arch": "sm_120",
            "arch": None,
            "started_utc": "2026-11-12T00:00:00Z",
            "finished_utc": "2026-11-12T00:00:01Z",
            "device": None,
            "runtime": {"python": "3.11", "numpy": "1.26"},
            "checks": [
                {
                    "name": "device-arch",
                    "ok": False,
                    "detail": "compute capability 89 (not Blackwell)",
                    "elapsed_ms": 0.1,
                }
            ],
        },
    )
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["status"] == "unsupported"
    assert result["checks"][0]["ok"] is False
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "Status: **unsupported**" in text
    assert "device-arch" in text
    assert (out / "run.log").exists()
