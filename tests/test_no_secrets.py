"""Fail when a scanned file contains a credential, a machine path or a personal identifier.

This is a cheap, dependency-free guard for the common accident: a key, token, local path, cloud
folder or address is added to a source file or doc. It scans `git ls-files`, skips binary files,
and reports only the file, line and detector name, never the matched text. The patterns live in
`tools/secret_patterns.py`, shared with `tools/check_secret_history.py`.

It scans tracked files and untracked files Git does not ignore (a secret in a new file used to be
invisible). It does not replace the history scan: commit metadata and deleted lines are outside its
scope, which is what `tools/check_secret_history.py` covers. `.secrets.baseline` /
`tools/check_secret_baseline.py` are the entropy-based pass over the same tracked tree.
"""
import subprocess
import sys
from pathlib import Path

from tools.secret_patterns import ALLOW, HEX, PATTERNS, SKIP, hits_in_line

ROOT = Path(__file__).resolve().parents[1]


def _git_files(*args):
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"] + list(args), check=True,
                         stdout=subprocess.PIPE).stdout
    return [raw.decode("utf-8", "replace") for raw in out.split(b"\0") if raw]


def scanned_files():
    """Tracked files plus untracked files Git does not ignore.

    Scanning only `git ls-files` made a credential pasted into a brand-new file
    invisible until someone ran `git add`: the guard said ALL PASS while the
    secret sat in the working tree.
    """
    seen = set()
    for rel in _git_files() + _git_files("--others", "--exclude-standard"):
        if rel not in seen:
            seen.add(rel)
            yield rel


def findings():
    for rel in scanned_files():
        if Path(rel).suffix.lower() in SKIP:
            continue
        try:
            raw = (ROOT / rel).read_bytes()
        except OSError:
            continue
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16", "replace")
        elif b"\x00" in raw:
            continue
        else:
            text = raw.decode("utf-8", "replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for name in hits_in_line(line):
                yield rel, lineno, name


def test_the_local_drive_path_rule_is_not_vacuous():
    """A scanner rule that has never matched would have passed the drive-letter paths the audit
    found, so the shape is pinned here: JSON-escaped and prose forms both match."""
    drive = dict(PATTERNS)["local drive path"]
    assert drive.search('"pack": "T:' + "\\\\" * 2 + 'kindi-packs' + "\\\\" * 2 + 'bonsai-27b"')
    assert drive.search("pack is at " + "D:" + "\\" + "ollamaModels" + "\\" + "gemma4")
    assert not drive.search("pack `~/.kindi/packs/bonsai-27b`")
    assert not drive.search("the T:/kindi worktree")


def test_the_cloud_storage_rule_is_not_vacuous():
    """A synced folder is personal data even when its contents are dull: the account name and the
    sibling directories are the person's. This rule is what a synced-folder path slipped past
    before (2026-11-17), so pin both the hit and the prose that must not trip it."""
    cloud = dict(PATTERNS)["cloud storage path"]
    # assembled from pieces so this guard's own test file is not a finding
    synced = "/".join(["%USERPROFILE%", "Dropbox", "work", "notes.md"])
    assert cloud.search(synced)
    assert hits_in_line("the base at " + synced) == ("cloud storage path",)
    assert hits_in_line("OneDrive" + "/" + "notes.txt") == ("cloud storage path",)
    assert not cloud.search("the user's Dropbox folder, read-only")
    assert not cloud.search("Dropbox and OneDrive are synced by the OS")


def test_untracked_files_are_scanned(tmp_path, monkeypatch):
    """A credential pasted into a brand-new file used to be invisible until `git add`: the guard
    said ALL PASS while the secret sat in the working tree. `findings()` must list it with no
    commit and no `git add` at all."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    (repo / "committed.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "committed.py"], cwd=str(repo), check=True)
    (repo / "fresh.py").write_text('AWS = "AKIA' + "A" * 16 + '"\n', encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", repo)

    found = list(findings())
    assert ("fresh.py", 1, "AWS access key") in found, found
    assert not any(rel == "committed.py" for rel, _line, _name in found)


def test_no_credentials_or_home_paths_in_tracked_files():
    bad = list(findings())
    assert not bad, "possible secrets/personal data in scanned files:\n" + "\n".join(
        "%s:%d: %s" % item for item in bad[:40])


if __name__ == "__main__":
    test_no_credentials_or_home_paths_in_tracked_files()
    print("ALL PASS")
