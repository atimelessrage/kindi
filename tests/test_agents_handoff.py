"""The AGENTS.md worktree table is the handoff source of truth: its HEAD cells must
match the actual HEAD (or its parent, when the table bump is the HEAD commit itself),
or the next agent starts from a stale snapshot (this bit twice already: 8c1ac16,
then 1f2c682). Skips when git history is unavailable.

The second test keeps the 2026-11-14 split in place: AGENTS.md stays a live handoff of a few
hundred lines and keeps pointing at `docs/development/AGENTS-HISTORY.md`.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rev(spec):
    try:
        out = subprocess.run(
            ["git", "rev-parse", spec], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except OSError:
        return ""


def test_agents_is_a_live_handoff_and_links_its_history():
    """AGENTS.md is the handoff the next agent reads first; the dated record belongs in
    docs/development/AGENTS-HISTORY.md. It had grown to 1,057 lines of history before the
    2026-11-14 split, so keep the live file short and keep the pointer to the history intact.
    """
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
    lines = text.count("\n") + 1
    assert lines < 450, (
        "AGENTS.md is %d lines; move finished lease rows and dated notes to "
        "docs/development/AGENTS-HISTORY.md" % lines)
    assert "docs/development/AGENTS-HISTORY.md" in text, (
        "AGENTS.md no longer points at the dated history file")
    assert (ROOT / "docs" / "development" / "AGENTS-HISTORY.md").is_file(), (
        "AGENTS.md links docs/development/AGENTS-HISTORY.md but the file is missing")


def test_agents_worktree_table_names_current_head():
    head = _rev("HEAD")
    if not head:
        return
    parent = _rev("HEAD^")
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
    rows = [ln for ln in text.splitlines()
            if ln.startswith("| `T:/kindi`") or ln.startswith("| `<worktree>`")]
    assert len(rows) == 2, "worktree table rows not found in AGENTS.md"
    for row in rows:
        m = re.search(r"`([0-9a-f]{7,40})`", row.split("|")[3])
        assert m, "HEAD cell not found in row: %s" % row[:60]
        h = m.group(1)
        ok = head.startswith(h) or (bool(parent) and parent.startswith(h))
        assert ok, ("AGENTS.md worktree table says %s but HEAD is %s; bump the "
                    "table in the same commit" % (h, head))
