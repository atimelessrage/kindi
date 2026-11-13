"""The AGENTS.md worktree table is the handoff source of truth: its HEAD cells must
match the actual HEAD (or its parent, when the table bump is the HEAD commit itself),
or the next agent starts from a stale snapshot (this bit twice already: 8c1ac16,
then 1f2c682). Skips when git history is unavailable.
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
