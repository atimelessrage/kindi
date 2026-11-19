# SPDX-License-Identifier: AGPL-3.0-only
"""The canonical ternary-MoE census must stay one machine-checked source.

CI runs `scripts/check_ternary_moe.py` directly; these tests also prove the checker is not vacuous
by mutating a count and a slot product in memory.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = ROOT / "scripts" / "check_ternary_moe.py"
    spec = importlib.util.spec_from_file_location("check_ternary_moe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_canonical_census_is_consistent():
    problems = _checker().check_doc()
    assert not problems, "ternary-MoE census drift:\n" + "\n".join(problems)


def test_the_checker_catches_count_and_slot_drift():
    checker = _checker()
    text = checker.DEFAULT_DOC.read_text(encoding="utf-8")

    count_drift = text.replace(
        "| Qwen3.6-35B-A3B build rows | 2 |",
        "| Qwen3.6-35B-A3B build rows | 3 |",
    )
    assert count_drift != text
    problems = checker.check_text(count_drift)
    assert any("Qwen3.6-35B-A3B build rows" in problem for problem in problems)

    slot_drift = text.replace(
        "i.e. 24 expert slots (6 × 4)",
        "i.e. 25 expert slots (6 × 4)",
    )
    assert slot_drift != text
    problems = checker.check_text(slot_drift)
    assert any("micro-terse" in problem for problem in problems)
