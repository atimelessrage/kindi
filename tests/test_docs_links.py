# SPDX-License-Identifier: AGPL-3.0-only
"""Documentation links must point at files that exist.

A dead relative link is cheap to introduce and hard to notice, so this test checks every
Markdown file in the repository except raw benchmark logs. External URLs are not fetched;
the installer and release checks handle their own pinned artifacts.
"""
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".freebuff", ".pytest_cache", "bench-logs", "node_modules", "__pycache__"}
LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")


def _markdown_files():
    for path in ROOT.rglob("*.md"):
        # compare the path *inside* the repository: a checkout that lives under a skipped directory
        # name (this project is developed in ~/.freebuff/worktrees/...) would otherwise skip every
        # file and pass without checking anything, which is exactly what happened until 2026-11-14.
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def _targets(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in LINK.finditer(text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:", "data:", "#")):
            continue
        target = unquote(target.split("#", 1)[0].strip())
        if not target:
            continue
        # Raw benchmark logs are local evidence and may be absent from a clean checkout
        # because *.log is ignored; do not treat them as broken documentation links.
        if "raw" in Path(target).parts and target.endswith(".log"):
            continue
        yield target


# a link whose *visible text* is a repository path: it tells the reader where the file lives, so it
# must not disagree with where the link actually goes. This happened repeatedly (a page promoted from
# bench-logs/ into benchmarks/ kept its old label), which is why it is checked rather than trusted.
LABEL = re.compile(r"!?\[([^\]]+)\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")


def _label_paths(text):
    for match in LABEL.finditer(text):
        label, target = match.group(1).strip(), match.group(2)
        if target.startswith(("http://", "https://", "mailto:", "data:", "#")):
            continue
        label = label.strip().strip("`")
        if "/" not in label or "." not in label.rsplit("/", 1)[-1]:
            continue                      # not a path label (a sentence, an anchor, a directory)
        if label.startswith(("http://", "https://", "~")):
            continue
        yield label, target.split("#", 1)[0].strip()


def test_link_labels_point_where_the_link_goes():
    wrong = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, target in _label_paths(text):
            if not target:
                continue
            # a label is written either relative to the page or relative to the repository root;
            # both conventions are in use, and either one agreeing with the target is honest.
            shown = {(path.parent / unquote(label)).resolve(), (ROOT / unquote(label)).resolve()}
            real = (ROOT / target.lstrip("/")) if target.startswith("/") else \
                (path.parent / unquote(target)).resolve()
            if real in shown or not real.exists():
                continue                  # agrees, or the broken-link test owns the missing target
            wrong.append("%s: shows %s, links %s" % (path.relative_to(ROOT), label, target))
    assert not wrong, ("link text names a path the link does not open (a promoted file usually keeps "
                       "its old directory):\n" + "\n".join(wrong))


def test_no_markdown_link_opens_ignored_evidence():
    """Raw dumps under `bench-logs/` are gitignored, so a link into it is dead for every other
    checkout. Promote the page into `docs/benchmarks/` (that is what the benchmarks index means)."""
    bad = []
    for path in _markdown_files():
        for target in _targets(path):
            if "bench-logs" in Path(target).parts:
                bad.append("%s: %s" % (path.relative_to(ROOT), target))
    assert not bad, "links into gitignored local evidence:\n" + "\n".join(bad)


def test_local_markdown_links_resolve():
    broken = []
    for path in _markdown_files():
        for target in _targets(path):
            candidate = (ROOT / target.lstrip("/")) if target.startswith("/") else \
                (path.parent / target).resolve()
            if not candidate.exists():
                broken.append("%s: %s" % (path.relative_to(ROOT), target))
    assert not broken, "broken local Markdown links:\n" + "\n".join(broken)
