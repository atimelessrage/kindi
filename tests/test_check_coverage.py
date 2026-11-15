"""Unit tests for the per-module coverage floor gate (audit M4).

The gate logic is exercised against synthetic coverage JSON in tmp_path, so
these tests are fast and independent of how much of the suite actually ran.
"""
import json

import pytest

from tools.check_coverage import check, read_coverage, read_floors


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _coverage(tmp_path, files):
    """A coverage.py JSON report with the given {module: percent}."""
    payload = {"files": {name: {"summary": {"percent_covered": pct}}
                         for name, pct in files.items()}}
    return _write(tmp_path / "coverage.json", payload)


def test_check_passes_when_every_module_is_at_its_floor(tmp_path):
    coverage = read_coverage(_coverage(tmp_path, {"kindi/server.py": 46.0,
                                                  "kindi/cache.py": 56.0}))
    assert check(coverage, {"kindi/server.py": 40, "kindi/cache.py": 50}) == []


def test_check_reports_a_module_below_its_floor(tmp_path):
    coverage = read_coverage(_coverage(tmp_path, {"kindi/server.py": 30.0}))
    problems = check(coverage, {"kindi/server.py": 40})
    assert len(problems) == 1 and "kindi/server.py" in problems[0] and "30.0" in problems[0]


def test_check_reports_a_missing_module(tmp_path):
    # A floored module that the run never imported is a problem, not a silent pass.
    coverage = read_coverage(_coverage(tmp_path, {"kindi/server.py": 99.0}))
    problems = check(coverage, {"kindi/server.py": 40, "kindi/cache.py": 40})
    assert len(problems) == 1 and "kindi/cache.py" in problems[0]


def test_backslash_keys_from_a_windows_run_are_normalized(tmp_path):
    coverage = read_coverage(_coverage(tmp_path, {"kindi\\server.py": 46.0}))
    assert coverage == {"kindi/server.py": 46.0}
    assert check(coverage, {"kindi/server.py": 40}) == []


def test_floors_file_is_well_formed_and_names_real_modules(tmp_path):
    # The shipped floors file: every key must be a module that exists on disk,
    # so a typo or a renamed module fails here instead of silently gating nothing.
    import pathlib
    floors_path = pathlib.Path(__file__).resolve().parents[1] / "tests" / "coverage_floors.json"
    floors = read_floors(floors_path)
    assert floors, "the floors file must not be empty"
    root = pathlib.Path(__file__).resolve().parents[1]
    for module, floor in floors.items():
        assert 0 < floor <= 100, module
        assert (root / module).is_file(), "floored module does not exist: %s" % module
        assert module.startswith("kindi/"), module

