"""The direct 27B A/B compares generation paths, not reference verdicts."""
import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "spec_ab.py"


def load():
    spec = importlib.util.spec_from_file_location("spec_ab", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_modes_are_validated_at_argparse_time():
    mod = load()
    assert mod._modes("NGRAM, BOTH") == ["ngram", "both"]
    for bad in ("fast", "", ","):
        with pytest.raises(argparse.ArgumentTypeError):
            mod._modes(bad)
    for bad in ("0", "-1", "fast"):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            mod._positive_int(bad)


def test_identical_tokens_and_bytes_are_reported_as_identical():
    mod = load()
    row = mod.compare_outputs([1, 2, 3], [1, 2, 3], b"abc", b"abc")
    assert row["identical"] is True
    assert row["tokens_identical"] is True and row["bytes_identical"] is True
    assert row["first_token_mismatch"] is None and row["first_byte_mismatch"] is None
    assert row["plain_byte_sha256"] == row["spec_byte_sha256"]


def test_a_token_difference_is_not_hidden_by_equal_length():
    mod = load()
    row = mod.compare_outputs([1, 9, 3], [1, 2, 3], b"abc", b"abc")
    assert row["identical"] is False
    assert row["tokens_identical"] is False and row["bytes_identical"] is True
    assert row["first_token_mismatch"] == 1
    assert row["first_byte_mismatch"] is None


def test_a_byte_difference_is_reported_separately_from_the_token_stream():
    mod = load()
    row = mod.compare_outputs([1, 2, 3], [1, 2, 3], b"abc", b"abd")
    assert row["identical"] is False
    assert row["tokens_identical"] is True and row["bytes_identical"] is False
    assert row["first_token_mismatch"] is None
    assert row["first_byte_mismatch"] == 2


def test_different_lengths_are_a_mismatch():
    mod = load()
    row = mod.compare_outputs([1, 2], [1, 2, 3], b"ab", b"abc")
    assert row["identical"] is False
    assert row["first_token_mismatch"] == 2
    assert row["first_byte_mismatch"] == 2
