#!/usr/bin/env python3
"""Scan git history for credentials and personal data, not only the current tree.

`tests/test_no_secrets.py` and `tools/check_secret_baseline.py` both look at the files that
exist now. A key that was committed and deleted afterwards survives only in history, and so
does a machine path or an address in an old revision - and history is what a push publishes. This walks every commit's message and every
commit's diff once (`git log --all -p -U0`) and reports the commit, path and detector for each hit;
like the file scan it never prints the matched text.

Commit subjects are scanned with the same rules as file content, because a subject is content a
push publishes: a message that names a synced folder, a home directory or an account is the same
personal data the file scan refuses, and nothing else in the repository ever looks at it. A subject
is one line, so it carries at most one hit per detector.

History cannot be "fixed" without a rewrite, so the gate is a **count baseline**
(`tools/secret_history_baseline.json`, one `path<TAB>detector` row with its hit count), the
same reviewed-snapshot idea as `.secrets.baseline`: existing rows are recorded with their
reason, and any *new* row or a count above the baseline fails. The file is a record of legacy
exposure, not an endorsement - the first baseline (2026-11-17) lists what an old benchmark
dump and old AGENTS/log revisions left in history, so that a push is a decision rather than
an accident.

Examples:
  python tools/check_secret_history.py                       # gate against the baseline
  python tools/check_secret_history.py --range v0.6.0..HEAD  # one release range
  python tools/check_secret_history.py --all-hits            # every hit, for a rewrite review
  python tools/check_secret_history.py --update-baseline     # after a reviewed change
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from secret_patterns import hits_in_line  # noqa: E402  (after the path setup)

DEFAULT_BASELINE = ROOT / "tools" / "secret_history_baseline.json"
RECORD = "\x01"
FIELD = "\x02"
DIFF = re.compile(r"^diff --git a/(.+?) b/(.+)$")



def _git(args, repo=ROOT):
    proc = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0:
        raise SystemExit("git %s failed: %s" % (" ".join(args), (proc.stderr or "").strip()[-500:]))
    return proc.stdout


def messages(rev_range="--all", repo=ROOT):
    """Yield (commit, "(commit message)", detector) for every hit in a commit message.

    The whole message is scanned, subject and body: `git log -p` shows only the subject in its
    header, so a body that names a synced folder was invisible to every guard in the repository.
    A path is a leak in a message for the same reason it is one in a file - a push publishes it.
    """
    args = ["log", "--no-color", "--format=%H" + FIELD + "%B" + RECORD]
    args += rev_range.split() if isinstance(rev_range, str) else list(rev_range)
    for record in _git(args, repo).split(RECORD):
        record = record.strip("\n")
        if not record.strip():
            continue
        commit, _, body = record.partition(FIELD)
        for line in body.splitlines():
            for name in hits_in_line(line):
                yield commit, "(commit message)", name


def scan(rev_range="--all", repo=ROOT):
    """Yield (commit, path, detector) for every credential or personal-data hit in history.

    Only **added** lines count. A leak enters history once, when it is added; the line that
    removes it is the same content again, so counting it would make cleaning a leak raise the
    total and fail the gate. The removed line is still in the scan for `--all-hits`, which is the
    view a rewrite review needs, but it is not a new hit.
    """
    args = ["log", "-p", "-U0", "--no-color", "--format=%s" % (RECORD + "%H" + FIELD + "%s")]
    args += rev_range.split() if isinstance(rev_range, str) else list(rev_range)
    commit, a_path, b_path = "", "", ""
    for hit in messages(rev_range, repo):
        yield hit
    for line in _git(args, repo).splitlines():
        if line.startswith(RECORD):
            commit, _, _subject = line[1:].partition(FIELD)
            a_path = b_path = ""
            continue
        match = DIFF.match(line)
        if match:
            a_path, b_path = match.group(1), match.group(2)
            continue
        if not line or line[0] != "+" or line.startswith("+++"):
            continue
        path = b_path
        if not path:
            continue
        for name in hits_in_line(line[1:]):
            yield commit, path, name


def counted(rev_range="--all", repo=ROOT):
    """{(path, detector): count} over history; the matched text is never kept."""
    counts = {}
    for _commit, path, name in scan(rev_range, repo):
        counts[(path, name)] = counts.get((path, name), 0) + 1
    return counts


def read_baseline(path):
    """{(path, detector): count} plus the note the baseline carries."""
    if not path.is_file():
        return {}, ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = {tuple(key.rsplit("\t", 1)): int(value)
              for key, value in payload.get("counts", {}).items()}
    return counts, str(payload.get("note", ""))


def write_baseline(path, counts, repo=ROOT):
    payload = {
        "note": ("Legacy exposure in history, recorded so a later commit cannot add more. "
                 "These rows are evidence dumps, old revisions and commit messages that would "
                 "need a history rewrite to remove; they are not endorsed. Review with "
                 "--all-hits."),
        "head": _git(["log", "-1", "--format=%H"], repo).strip(),
        "counts": {"%s\t%s" % key: value for key, value in sorted(counts.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def differences(baseline, current):
    """New rows, and rows whose count grew above the recorded one."""
    problems = []
    for key, count in sorted(current.items()):
        before = baseline.get(key, 0)
        if count > before:
            problems.append({"path": key[0], "detector": key[1],
                             "new": count - before, "baseline": before})
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=ROOT, help="repository to scan (default: this one)")
    ap.add_argument("--range", default="--all", help="git revision range (default: --all)")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--all-hits", action="store_true",
                    help="print every hit (commit, path, detector) instead of gating")
    ap.add_argument("--update-baseline", action="store_true",
                    help="record the current counts as the baseline (after a review)")
    a = ap.parse_args(argv)
    repo = a.repo.expanduser().resolve()

    if a.all_hits:
        hits = [{"commit": commit[:7], "path": path, "detector": name}
                for commit, path, name in scan(a.range, repo)]
        if a.json:
            print(json.dumps({"findings": hits}, indent=2))
        else:
            for hit in hits:
                print("%s %s: %s" % (hit["commit"], hit["path"], hit["detector"]))
            print("%d hit(s) over %s" % (len(hits), ", ".join(a.range.split())))
        return 0

    current = counted(a.range, repo)
    if a.update_baseline:
        write_baseline(a.baseline, current, repo)
        print("baseline updated: %d row(s) -> %s" % (len(current), a.baseline))
        return 0

    baseline, note = read_baseline(a.baseline)
    problems = differences(baseline, current)
    if a.json:
        print(json.dumps({"new": problems, "rows": len(current),
                          "baseline_rows": len(baseline)}, indent=2))
    if problems:
        if not a.json:
            print("new credential or personal data in git history (the text is not printed):",
                  file=sys.stderr)
            for problem in problems[:80]:
                print("  %s: %s +%d (baseline %d)" % (problem["path"], problem["detector"],
                                                      problem["new"], problem["baseline"]),
                      file=sys.stderr)
            print("fix the commit before it is pushed, or review and re-record with "
                  "--update-baseline; baseline note: %s" % (note or "(none)"), file=sys.stderr)
        return 1
    if not a.json:
        print("git history: no new hit over %s (%d recorded row(s) in %s)"
              % (", ".join(a.range.split()), sum(baseline.values()), a.baseline.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
