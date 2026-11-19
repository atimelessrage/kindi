"""The long-context gate must fail on a clamped context or a missing deep cell."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_long_context.py"


def load():
    spec = importlib.util.spec_from_file_location("check_long_context", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_numeric_arguments_are_validated_at_argparse_time():
    mod = load()
    for bad in ("0", "-1", "fast"):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            mod._positive_int(bad)
    for bad in ("0", "-1", "nan", "inf", "fast"):
        with pytest.raises(argparse.ArgumentTypeError, match="finite positive number"):
            mod._positive_float(bad)


def test_required_position_tracks_the_context():
    mod = load()
    assert mod.required_position(24576) == 24000
    assert mod.required_position(8192) == 8000
    assert mod.required_position(1000) == 100
    assert mod.required_position(100) is None


def test_parse_accepts_a_deep_cell():
    mod = load()
    text = """[kindi 10:00:00] GPU memory: 7000 MiB free of 8192 with a 24576-token context
[kindi 10:00:00] prompt reading:      190.0 tokens/s
[kindi 10:00:00] generation (exact head) at position   100:   37.6 tokens/s
[kindi 10:00:00] generation (exact head) at position 24000:  160.6 tokens/s
"""
    out = mod.parse_bench_output(text, 24576)
    assert out["status"] == "pass"
    assert out["loaded_context"] == 24576
    assert out["required_position"] == 24000
    assert out["required_tps"] == 160.6
    assert out["read_tps"] == 190.0
    assert out["generation"] == {"100": 37.6, "24000": 160.6}


def test_parse_rejects_a_clamped_context():
    mod = load()
    text = """[kindi 10:00:00] GPU memory: 7000 MiB free of 8192 with a 16000-token context
[kindi 10:00:00] generation (exact head) at position 16000:  80.0 tokens/s
"""
    out = mod.parse_bench_output(text, 24576)
    assert out["status"] == "context-clamped"
    assert out["loaded_context"] == 16000
    assert out["required_tps"] is None


def test_parse_rejects_a_missing_deep_cell():
    mod = load()
    text = """[kindi 10:00:00] GPU memory: 7000 MiB free of 8192 with a 24576-token context
[kindi 10:00:00] generation (exact head) at position 16000:  80.0 tokens/s
"""
    out = mod.parse_bench_output(text, 24576)
    assert out["status"] == "missing-position"
    assert out["required_position"] == 24000
    assert out["required_tps"] is None


def test_parse_rejects_a_log_without_a_context():
    mod = load()
    out = mod.parse_bench_output("no GPU here\n", 24576)
    assert out["status"] == "no-context"
    assert out["loaded_context"] is None


def test_main_writes_the_result_artifact(tmp_path, monkeypatch):
    mod = load()
    row = {
        "pack": str(tmp_path / "pack"),
        "ctx": 24576,
        "status": "pass",
        "exit_code": 0,
        "duration_s": 1.0,
        "loaded_context": 24576,
        "required_position": 24000,
        "required_tps": 160.6,
        "read_tps": 190.0,
        "generation": {"24000": 160.6},
        "command": [],
        "output": "[kindi 10:00:00] generation (exact head) at position 24000: 160.6 tokens/s\n",
    }
    monkeypatch.setattr(mod, "run_bench", lambda *args, **kwargs: dict(row))
    rc = mod.main(["--pack", str(tmp_path / "pack"), "--ctx", "24576",
                   "--out", str(tmp_path / "out")])
    assert rc == 0
    payload = json.loads((tmp_path / "out" / "long-context.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["required_tps"] == 160.6
    assert (tmp_path / "out" / "bench.log").exists()
