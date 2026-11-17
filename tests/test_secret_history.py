"""The history scan must see what the tracked-file scan cannot: a line that no longer exists.

`tests/test_no_secrets.py` scans the present tree. A credential committed and deleted afterwards
is invisible there and permanent in history, which is what a push publishes. These tests pin that
the history scanner finds such a line, that the file scanner does not, and that this repository's
own baseline gate is current.
"""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_secret_history.py"
BASELINE = ROOT / "tools" / "secret_history_baseline.json"


def load():
    spec = importlib.util.spec_from_file_location("check_secret_history", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
                   + list(args), check=True, capture_output=True)


@pytest.fixture()
def repo_with_a_deleted_key(tmp_path):
    """Commit a key, then delete it: the working tree is clean, the history is not."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    notes = repo / "notes.md"
    notes.write_text("build notes\n", encoding="utf-8")
    _git(repo, "add", "notes.md")
    _git(repo, "commit", "-q", "-m", "notes")
    key = "AKIA" + "A" * 16
    notes.write_text("build notes\nkey: " + key + "\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "add a key by accident")
    notes.write_text("build notes\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "remove the key")
    return repo


def test_a_key_deleted_from_the_tree_is_still_found_in_history(repo_with_a_deleted_key):
    mod = load()
    hits = list(mod.scan("--all", repo_with_a_deleted_key))
    assert any(path == "notes.md" and name == "AWS access key" for _commit, path, name in hits), hits
    # and the current tree really is clean, so only history carries it
    assert "AKIA" not in (repo_with_a_deleted_key / "notes.md").read_text(encoding="utf-8")


def test_the_counting_gate_flags_a_new_row_and_passes_on_the_recorded_one(repo_with_a_deleted_key):
    mod = load()
    current = mod.counted("--all", repo_with_a_deleted_key)
    # counted once, at the commit that added it: the removal is the same content again, and a
    # cleanup must not raise the gate
    assert current[("notes.md", "AWS access key")] == 1
    assert mod.differences(current, current) == []             # recorded counts pass
    assert mod.differences({}, current) != []                  # nothing recorded: a new row
    fewer = {("notes.md", "AWS access key"): 0}
    assert mod.differences(fewer, current) != []


def test_the_baseline_is_counted_rows_and_carries_no_matched_text():
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert payload["counts"], "the baseline has no rows"
    assert len(payload["head"]) == 40
    for key, count in payload["counts"].items():
        path, detector = key.rsplit("\t", 1)
        assert path and detector and int(count) > 0
        assert "AKIA" not in key and "-----BEGIN" not in key   # keys, not evidence text


def test_this_repository_adds_no_new_history_hit():
    """Scan the recent history against the recorded baseline; CI runs the full range."""
    mod = load()
    baseline, _note = mod.read_baseline(BASELINE)
    current = mod.counted("HEAD~20..HEAD")
    assert mod.differences(baseline, current) == []
