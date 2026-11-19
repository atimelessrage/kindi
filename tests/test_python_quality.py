# SPDX-License-Identifier: AGPL-3.0-only
"""The Python lint/type ratchet: existing debt is allowed, growth is not.

The gate runs the real tools in CI; these checks pin its comparison and the
committed baseline's shape without requiring the analyzers in the unit suite.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_python_quality.py"
BASELINE = ROOT / "tools" / "python_quality_baseline.json"


def _load():
    spec = importlib.util.spec_from_file_location("check_python_quality", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_equal_or_smaller_current_count_passes():
    mod = _load()
    baseline = {"kindi/a.py": {"F401": 2}, "tools/b.py": {"B905": 1}}
    assert mod._regressions(baseline, baseline) == []
    smaller = {"kindi/a.py": {"F401": 1}, "tools/b.py": {"B905": 1}}
    assert mod._regressions(baseline, smaller) == []


def test_a_growing_count_is_a_regression():
    mod = _load()
    problems = mod._regressions({"kindi/a.py": {"F401": 2}}, {"kindi/a.py": {"F401": 3}})
    assert len(problems) == 1
    assert "kindi/a.py F401: 3 > baseline 2" in problems[0]


def test_a_new_file_or_rule_is_a_regression():
    mod = _load()
    baseline = {"kindi/a.py": {"F401": 2}}
    assert "new.py F401: 1 > baseline 0" in mod._regressions(
        baseline, {"kindi/a.py": {"F401": 2}, "new.py": {"F401": 1}})[0]
    assert "kindi/a.py B904: 1 > baseline 0" in mod._regressions(
        baseline, {"kindi/a.py": {"F401": 2, "B904": 1}})[0]


def test_pyright_parser_counts_errors_by_file_and_rule(monkeypatch):
    mod = _load()
    payload = {"generalDiagnostics": [
        {"file": str(ROOT / "kindi" / "a.py"), "severity": "error",
         "rule": "reportArgumentType"},
        {"file": str(ROOT / "kindi" / "a.py"), "severity": "error",
         "rule": "reportArgumentType"},
        {"file": str(ROOT / "kindi" / "a.py"), "severity": "warning",
         "rule": "reportUnusedImport"},
    ]}
    monkeypatch.setattr(mod, "_run", lambda command, what: SimpleNamespace(
        returncode=1, stdout=json.dumps(payload), stderr=""))
    assert mod._pyright_findings("pyright") == {
        "kindi/a.py": {"reportArgumentType": 2}}


def test_pylint_parser_counts_findings_by_file_and_code(monkeypatch):
    mod = _load()
    rows = [{"path": "tools/b.py", "message-id": "E1101"},
            {"path": "tools/b.py", "message-id": "E1101"}]
    monkeypatch.setattr(mod, "_run", lambda command, what: SimpleNamespace(
        returncode=2, stdout=json.dumps(rows), stderr=""))
    assert mod._pylint_findings("pylint") == {"tools/b.py": {"E1101": 2}}


def test_the_committed_baseline_has_consistent_totals():
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert payload["schema"] == 2
    assert payload["scope"] == {
        "ruff_rules": "F,E9,B,I",
        "mypy_targets": ["kindi", "tools", "examples", "bridge/kindi_bridge"],
        "pyright_targets": ["kindi", "tools", "examples", "bridge/kindi_bridge"],
        "pyright_mode": "basic",
        "pyright_platform": "All",
        "pylint_targets": ["kindi", "tools", "examples", "bridge/kindi_bridge"],
        "pylint_checks": "E,F (import-error disabled)",
    }
    for tool in ("ruff", "mypy", "pyright", "pylint"):
        findings = payload[tool]["findings"]
        assert findings, "baseline has no %s findings" % tool
        assert payload[tool]["version"]
        assert payload[tool]["total"] == sum(
            count for by_code in findings.values() for count in by_code.values())
