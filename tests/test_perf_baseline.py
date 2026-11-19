"""The perf-regression gate (tools/check_perf_baseline.py) must pass on the recorded
matrix and fail on a real regression or a disappeared baseline cell. The CPU-only and
tandem cells are the 2026-11-19 controlled same-context pairs (one run per placement for
2B/4B, two for 9B); the GPU and tensor-core cells are the 2026-11-09/13 medians. The 4B tandem verify is
3/4 with one llama.cpp tie, so its baseline is throughput evidence, not an exactness
claim. The 2026-11-10 CPU and 2026-11-13 tandem rows remain history.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_perf_baseline.py"


def load():
    spec = importlib.util.spec_from_file_location("check_perf_baseline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tolerance_rejects_non_finite_and_negative_values():
    mod = load()
    assert mod._tolerance("0") == 0.0
    assert mod._tolerance("0.2") == 0.2
    for bad in ("-0.1", "nan", "inf", "fast"):
        with pytest.raises(argparse.ArgumentTypeError,
                           match="finite non-negative number"):
            mod._tolerance(bad)


def test_recorded_baseline_passes_the_gate():
    mod = load()
    current = {"%s/%s" % k: v for k, v in mod.read_cells(mod.DEFAULT_CSV).items()}
    baseline = json.loads(mod.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    assert baseline, "baseline must record cells"
    assert mod.check(current, baseline, 0.15) == []


def test_gate_fails_on_regression_and_missing_cell():
    mod = load()
    baseline = {"gpu/2b": {"read": 100.0, "gen_24000": 50.0}}
    regressed = {"gpu/2b": {"read": 90.0, "gen_24000": 40.0}}  # -10% / -20%
    problems = mod.check(regressed, baseline, 0.15)
    assert any("gen_24000" in p for p in problems)
    assert not any("read" in p for p in problems)  # within tolerance
    assert mod.check({}, baseline, 0.15), "missing cell must fail"


def test_gate_cli_exit_codes(tmp_path):
    mod = load()
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(
        "mode,available,model,read,gen_100,gen_2000,gen_8000,gen_16000,gen_24000,runs,source\n"
        "gpu,1,2b,100,10,10,10,10,10,1,x\n", encoding="utf-8")
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"gpu/2b": {"read": 100.0, "gen_24000": 10.0}}),
                    encoding="utf-8")
    assert mod.main(["--csv", str(csv_path), "--baseline", str(base)]) == 0
    base.write_text(json.dumps({"gpu/2b": {"read": 200.0, "gen_24000": 10.0}}),
                    encoding="utf-8")
    assert mod.main(["--csv", str(csv_path), "--baseline", str(base)]) == 1
