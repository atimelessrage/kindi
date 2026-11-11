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
        if any(part in SKIP_DIRS for part in path.parts):
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


def test_local_markdown_links_resolve():
    broken = []
    for path in _markdown_files():
        for target in _targets(path):
            candidate = (ROOT / target.lstrip("/")) if target.startswith("/") else \
                (path.parent / target).resolve()
            if not candidate.exists():
                broken.append("%s: %s" % (path.relative_to(ROOT), target))
    assert not broken, "broken local Markdown links:\n" + "\n".join(broken)
