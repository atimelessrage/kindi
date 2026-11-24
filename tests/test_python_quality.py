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


def test_a_file_target_covered_by_its_directory_is_deduplicated():
    mod = _load()
    assert mod._dedupe_targets(("tools", "tools/moe_expert_lane.py")) == ("tools",)
    assert mod._dedupe_targets(("tools/moe_expert_lane.py", "tools")) == ("tools",)


def test_mypy_does_not_receive_a_deduplicated_file_target(monkeypatch):
    mod = _load()
    seen = []
    monkeypatch.setattr(mod, "MYPY_TARGETS", ("tools", "tools/moe_expert_lane.py"))
    monkeypatch.setattr(mod, "_run", lambda command, what: (
        seen.append(command) or SimpleNamespace(returncode=0, stdout="", stderr="")))
    assert mod._mypy_findings() == {}
    assert seen[0].count("tools/moe_expert_lane.py") == 0


def test_analyzer_scope_outlives_only_the_unrunnable_gpu_and_upstream_modules():
    """Pyright and Pylint skip exactly two documented families; the fork's own files stay in scope."""
    mod = _load()
    assert mod.VENDORED_UNWIRED == ("kindi/qwen4exp_", "kindi/qx_")
    assert mod.UPSTREAM_UNVERIFIED == ("kindi/inkling2",)
    assert mod._out_of_scope("kindi/inkling2.py")
    assert mod._out_of_scope("kindi/inkling2_gen.py")
    assert not mod._out_of_scope("kindi/server.py")
    assert not mod._out_of_scope("tools/check_python_quality.py")


def test_pylint_targets_the_other_tools_python_version(monkeypatch):
    mod = _load()
    seen = []
    monkeypatch.setattr(mod, "_run", lambda command, what: (
        seen.append(command) or SimpleNamespace(returncode=0, stdout="[]", stderr="")))
    assert mod._pylint_findings("pylint") == {}
    assert "--py-version=3.10" in seen[0]
    assert mod.PYLINT_PY_VERSION == "3.10"


def test_pylint_parser_counts_findings_by_file_and_code(monkeypatch):
    mod = _load()
    rows = [{"path": "tools/b.py", "message-id": "E1101"},
            {"path": "tools/b.py", "message-id": "E1101"}]
    monkeypatch.setattr(mod, "_run", lambda command, what: SimpleNamespace(
        returncode=2, stdout=json.dumps(rows), stderr=""))
    assert mod._pylint_findings("pylint") == {"tools/b.py": {"E1101": 2}}


def _quality_payload(ruff=1, mypy=1, pyright=1, pylint=1):
    return {
        "ruff": {"version": "0.16.0", "total": ruff,
                 "findings": {"kindi/a.py": {"F401": ruff}} if ruff else {}},
        "mypy": {"version": "2.4.0", "total": mypy,
                 "findings": {"kindi/a.py": {"assignment": mypy}} if mypy else {}},
        "pyright": {"version": "1.1.414", "total": pyright,
                    "findings": {"kindi/a.py": {"reportArgumentType": pyright}} if pyright else {}},
        "pylint": {"version": "4.1.3", "total": pylint,
                   "findings": {"kindi/a.py": {"E1101": pylint}} if pylint else {}},
    }


def _write_baseline(path, payload):
    path.write_text(json.dumps({"schema": 2, **payload}), encoding="utf-8")


def test_update_requires_a_reason(monkeypatch):
    mod = _load()
    assert mod.main(["--update"]) == 2
    assert mod.main(["--reason", "cleanup"]) == 2


def test_update_refuses_growth_and_leaves_the_baseline_alone(tmp_path, monkeypatch):
    mod = _load()
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, _quality_payload(ruff=1))
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline)
    monkeypatch.setattr(mod, "_counts", lambda *args: _quality_payload(ruff=2))
    monkeypatch.setattr(mod, "_head", lambda: "abc123")
    assert mod._update(None, None, None, None, None, None, None, "cleanup") == 1
    assert json.loads(baseline.read_text(encoding="utf-8"))["ruff"]["total"] == 1


def test_update_records_the_reason_and_delta_of_a_reviewed_lowering(tmp_path, monkeypatch):
    mod = _load()
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, _quality_payload(ruff=2))
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline)
    monkeypatch.setattr(mod, "_counts", lambda *args: _quality_payload(ruff=1))
    monkeypatch.setattr(mod, "_head", lambda: "abc123")
    assert mod._update(None, None, None, None, None, None, None, "removed the F401") == 0
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["review"]["reason"] == "removed the F401"
    assert payload["review"]["head"] == "abc123"
    assert payload["review"]["deltas"]["ruff"] == -1


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
        "pylint_py_version": "3.10",
    }
    for tool in ("ruff", "mypy", "pyright", "pylint"):
        findings = payload[tool]["findings"]
        assert findings, "baseline has no %s findings" % tool
        assert payload[tool]["version"]
        assert payload[tool]["total"] == sum(
            count for by_code in findings.values() for count in by_code.values())
