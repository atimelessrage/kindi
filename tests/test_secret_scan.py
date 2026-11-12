"""The tracked-file secret scan must stay aligned between CI, the baseline and pre-commit."""
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_secret_baseline.py"


def load():
    spec = importlib.util.spec_from_file_location("check_secret_baseline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_baseline_paths_are_tracked_and_do_not_include_the_baseline():
    baseline = json.loads((ROOT / ".secrets.baseline").read_text(encoding="utf-8"))
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"])
    tracked = {raw.decode("utf-8", "replace") for raw in tracked.split(b"\0") if raw}
    paths = set(baseline["results"])
    assert paths <= tracked
    assert ".secrets.baseline" not in paths
    assert baseline["version"] == "1.5.0"


def test_pre_commit_hook_scans_text_files_with_the_shared_baseline():
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "id: detect-secrets" in text
    assert "rev: v1.5.0" in text
    assert '"--baseline", ".secrets.baseline"' in text
    assert r"^\.secrets\.baseline$" in text
    assert ".json|.lock|.log|.md|.txt" not in text


def test_ci_scans_tracked_files_and_checks_baseline_freshness():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert '"detect-secrets==1.5.0"' in text
    assert "tools/check_secret_baseline.py --check" in text
    assert "detect-secrets-hook --baseline .secrets.baseline" in text


def test_baseline_check_reports_only_drifted_files():
    mod = load()
    baseline = {"results": {
        "docs/example.md": [{"line_number": 1, "type": "Hex High Entropy String"}],
        "old.txt": [{"line_number": 2, "type": "Secret Keyword"}],
    }}
    current = {"results": {
        "docs/example.md": [{"line_number": 3, "type": "Hex High Entropy String"}],
        "new.txt": [{"line_number": 4, "type": "Secret Keyword"}],
    }}
    problems = mod.results_differences(baseline, current)
    assert any("docs/example.md" in problem for problem in problems)
    assert any("old.txt: baseline-only" in problem for problem in problems)
    assert any("new.txt: new finding" in problem for problem in problems)
    assert not any("hashed_secret" in problem for problem in problems)


def test_baseline_normalization_uses_posix_paths():
    mod = load()
    normalized = mod._normalize_results({
        "docs\\example.md": [{"filename": "docs\\example.md", "line_number": 1}],
    })
    assert list(normalized) == ["docs/example.md"]
    assert normalized["docs/example.md"][0]["filename"] == "docs/example.md"
