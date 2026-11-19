"""The dependency audit must parse pip-audit output and gate only on new pairs."""
import json

from tools import check_dependencies


def _report(path, *pairs):
    dependencies = []
    for name, advisory in pairs:
        dependencies.append({"name": name, "version": "1.0",
                             "vulns": [{"id": advisory}]})
    path.write_text(json.dumps({"dependencies": dependencies}), encoding="utf-8")
    return path


def test_report_only_lists_a_finding_without_failing(tmp_path, capsys):
    report = _report(tmp_path / "audit.json", ("fastapi", "CVE-2024-0001"))
    assert check_dependencies.main(["--report", str(report), "--report-only"]) == 0
    assert "NEW fastapi CVE-2024-0001" in capsys.readouterr().out


def test_update_records_a_baseline_and_then_passes(tmp_path):
    report = _report(tmp_path / "audit.json", ("fastapi", "CVE-2024-0001"))
    baseline = tmp_path / "baseline.json"
    args = ["--report", str(report), "--baseline", str(baseline)]
    assert check_dependencies.main(args + ["--update"]) == 0
    assert "fastapi CVE-2024-0001" in baseline.read_text(encoding="utf-8")
    assert check_dependencies.main(args) == 0


def test_a_new_advisory_fails_against_the_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    first = _report(tmp_path / "first.json", ("fastapi", "CVE-2024-0001"))
    assert check_dependencies.main(["--report", str(first), "--baseline", str(baseline),
                                    "--update"]) == 0
    second = _report(tmp_path / "second.json", ("fastapi", "CVE-2024-0001"),
                     ("uvicorn", "CVE-2025-0002"))
    assert check_dependencies.main(["--report", str(second), "--baseline", str(baseline)]) == 1


def test_missing_report_is_an_error_unless_allowed(tmp_path):
    missing = tmp_path / "missing.json"
    assert check_dependencies.main(["--report", str(missing), "--report-only"]) == 2
    assert check_dependencies.main(["--report", str(missing), "--report-only",
                                    "--allow-missing"]) == 0
