# SPDX-License-Identifier: AGPL-3.0-only
"""The MTP context harness parses its positions and counts before building an engine."""
import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "mtp_context_bench.py"


def load():
    spec = importlib.util.spec_from_file_location("mtp_context_bench", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_positions_are_comma_separated_positive_integers():
    mod = load()
    assert mod._positions("8192, 24576") == [8192, 24576]
    assert mod._positions("100") == [100]
    for bad in ("8192,fast", "8192,", "", "0", "-1", "8192, 0"):
        with pytest.raises(argparse.ArgumentTypeError, match="comma-separated positive integers"):
            mod._positions(bad)


def test_positive_counts_reject_zero_negative_and_text():
    mod = load()
    assert mod._positive_int("128") == 128
    for bad in ("0", "-1", "fast", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            mod._positive_int(bad)


def test_main_rejects_bad_positions_before_the_engine_is_built(capsys):
    mod = load()
    with pytest.raises(SystemExit) as exc:
        mod.main(["--pack", "pack", "--head", "head.npz", "--positions", "8192,fast"])
    assert exc.value.code == 2
    assert "comma-separated positive integers" in capsys.readouterr().err
