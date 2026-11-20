# SPDX-License-Identifier: AGPL-3.0-only
"""Generate the Kindi requirement history from git.

Requirements are one Markdown file per stable ID under
``docs/development/requirements/``.  This script records, for every current
requirement file, the commit that added it and the commit that last changed it.
Requirement files that were deleted are read from git history and listed in a
separate section.

Usage::

    python tools/project_memory.py
    python tools/project_memory.py --check

The generated file is ``docs/development/REQUIREMENTS-HISTORY.md``.  Do not
edit it by hand.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ_REL = "docs/development/requirements"
REQ_DIR = ROOT / REQ_REL
HISTORY = ROOT / "docs" / "development" / "REQUIREMENTS-HISTORY.md"
STATUSES = {"active", "experimental", "proposed", "retired"}
ID_RE = re.compile(r"^REQ-\d{4}$")
STATUS_RE = re.compile(r"^- Status:\s*([a-z]+)\s*$", re.MULTILINE)
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode:
        raise RuntimeError("git %s failed:\n%s" % (" ".join(args), proc.stderr.strip()))
    return proc.stdout


def _headings(text: str) -> str:
    match = HEADING_RE.search(text)
    return match.group(1).strip() if match else ""


def _clean_title(title: str) -> str:
    return re.sub(r"^REQ-\d{4}\s*[:\-\u2014]\s*", "", title).strip()


def parse_requirement(path: Path) -> dict[str, str]:
    """Read and validate one requirement file."""
    text = path.read_text(encoding="utf-8")
    status = STATUS_RE.search(text)
    if status is None:
        raise ValueError("%s: missing a '- Status: ...' line" % path)
    value = status.group(1)
    if value not in STATUSES:
        raise ValueError("%s: status %r is not one of %s" % (path, value, sorted(STATUSES)))
    title = _clean_title(_headings(text))
    if not title:
        raise ValueError("%s: missing a '# ...' title" % path)
    return {"id": path.stem, "title": title, "status": value, "path": path}


def current_requirements() -> list[dict[str, str]]:
    if not REQ_DIR.is_dir():
        raise SystemExit("missing requirement directory: %s" % REQ_DIR)
    rows = []
    for path in sorted(REQ_DIR.glob("REQ-*.md")):
        if not ID_RE.fullmatch(path.stem):
            raise SystemExit("%s: requirement file names must be REQ-NNNN.md" % path)
        rows.append(parse_requirement(path))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate requirement IDs")
    return sorted(rows, key=lambda row: row["id"])


def _commits(rel: str) -> list[str]:
    out = _git("log", "--follow", "--format=%H", "--", rel)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _short(commit: str) -> str:
    return "uncommitted" if commit == "uncommitted" else "`%s`" % commit[:12]


def _old_file(commit: str, rel: str) -> str:
    return _git("show", "%s^:%s" % (commit, rel), check=False)


def deleted_requirements() -> list[dict[str, str]]:
    """Return requirements whose files were deleted, newest deletion first."""
    current = {str(row["path"].relative_to(ROOT)).replace("\\", "/") for row in current_requirements()}
    out = _git(
        "log", "--all", "--diff-filter=D", "--name-only",
        "--format=__COMMIT__%H", "--", REQ_REL,
    )
    seen: dict[str, str] = {}
    for block in out.split("__COMMIT__")[1:]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        commit = lines[0]
        for rel in lines[1:]:
            if rel in current or not rel.endswith(".md"):
                continue
            if not ID_RE.fullmatch(Path(rel).stem):
                continue
            seen.setdefault(rel, commit)
    rows = []
    for rel, commit in sorted(seen.items(), key=lambda item: item[1], reverse=True):
        old = _old_file(commit, rel)
        status = STATUS_RE.search(old)
        rows.append({
            "id": Path(rel).stem,
            "title": _clean_title(_headings(old)) or Path(rel).stem,
            "status": status.group(1) if status and status.group(1) in STATUSES else "unknown",
            "commit": commit,
            "path": rel,
        })
    return rows


def _table(rows: list[dict[str, str]], deleted: bool = False) -> list[str]:
    if deleted:
        lines = [
            "| ID | Requirement | Status | Deleted | Last path |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                "| %s | %s | %s | `%s` | `%s` |" % (
                    row["id"], row["title"].replace("|", "\\|"), row["status"],
                    row["commit"][:12], row["path"],
                )
            )
        if not rows:
            lines.append("| - | None recorded in this repository's history | - | - | - |")
        return lines
    lines = [
        "| ID | Requirement | Status | Added | Last modified |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        commits = _commits(str(row["path"].relative_to(ROOT)).replace("\\", "/"))
        added = commits[-1] if commits else "uncommitted"
        modified = commits[0] if commits else "uncommitted"
        lines.append(
            "| [%s](requirements/%s.md) | %s | %s | %s | %s |" % (
                row["id"], row["id"], row["title"].replace("|", "\\|"), row["status"],
                _short(added), _short(modified),
            )
        )
    if not rows:
        lines.append("| - | No requirement files found | - | - | - |")
    return lines


def render() -> str:
    reqs = current_requirements()
    deleted = deleted_requirements()
    lines = [
        "# Requirement history",
        "",
        "<!-- Generated by `python tools/project_memory.py`; do not edit by hand. -->",
        "",
        "Each requirement is a stable file under [`requirements/`](requirements/).",
        "The commit columns are 12-character abbreviations; use `git show <hash>` to inspect",
        "the state at that commit.",
        "",
        "## Current requirements",
        "",
        *_table(reqs),
        "",
        "`uncommitted` means the requirement file is not in `HEAD` yet. Commit the requirement",
        "and run the generator again to record its real added/last-modified hashes.",
        "",
        "## Deleted requirements",
        "",
        *_table(deleted, deleted=True),
        "",
        "A deleted requirement is retained here so an old state can be recovered with",
        "`git show <commit>^:<path>`; the deletion commit is the last commit that had the file.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated history is stale")
    args = parser.parse_args(argv)
    expected = render()
    if args.check:
        actual = HISTORY.read_text(encoding="utf-8") if HISTORY.exists() else ""
        if actual != expected:
            print("stale: %s; run `python tools/project_memory.py`" % HISTORY.relative_to(ROOT))
            return 1
        return 0
    with HISTORY.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(expected)
    print("wrote %s" % HISTORY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
