"""The AGENTS.md worktree table is the handoff source of truth: its HEAD cells must
name the actual HEAD (or its parent, when the table bump is the HEAD commit itself),
or the next agent starts from a stale snapshot (this bit twice already: 8c1ac16,
then 1f2c682). Skips when git history is unavailable.

A commit that touches AGENTS.md must also refresh the hash cells (added lines naming
HEAD or its parent), otherwise the parent rule would excuse a content commit that
leaves the table stale - the third-audit finding of 2026-11-14 (be6e3a9 edited the
file and kept its parent's hash). The fix is the protocol rule 5a in AGENTS.md: bump
in the same commit, naming the commit's own parent.

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


def test_a_rewritten_table_row_must_name_head_or_its_parent():
    """The rule the HEAD check applies to a commit's added table rows, as a pure function:
    a stale hash in a rewrite is rejected, the parent (the only hash a bump can know) passes."""
    head, parent = "be6e3a9" + "0" * 33, "df851a3" + "0" * 33

    def ok(row_hash):
        return head.startswith(row_hash) or parent.startswith(row_hash)

    assert not ok("56bc351")               # an older hash: the stale-table bug
    assert ok(parent[:7]) and ok(head[:7])


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


def _changed_agents_lines():
    """Added lines in the HEAD commit's AGENTS.md diff, or None if HEAD did not touch it."""
    out = subprocess.run(
        ["git", "show", "--format=", "--unified=0", "HEAD", "--", "AGENTS.md"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=False)
    if out.returncode != 0:
        return None
    adds = [ln[1:] for ln in out.stdout.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    return adds if adds else None


def test_agents_current_state_head_line_names_current_head():
    """The '## Current state' HEAD line is what the next agent reads first; it rotted to `07e0640` while
    HEAD had moved on (seen 2026-11-14 during the §25 round, when the line said e4267e6 at HEAD 07e0640).
    It must name HEAD or its parent (the parent is the only hash a bump commit can know)."""
    head = _rev("HEAD")
    if not head:
        return
    parent = _rev("HEAD^")
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^- \*\*HEAD:\*\* `([0-9a-f]{7,40})`", text, re.M)
    assert m, "Current-state HEAD line not found in AGENTS.md"
    h = m.group(1)
    ok = head.startswith(h) or (bool(parent) and parent.startswith(h))
    assert ok, ("AGENTS.md Current-state HEAD line says %s but HEAD is %s; bump the line in the "
                "same commit" % (h, head))


def test_agents_worktree_table_names_current_head():
    head = _rev("HEAD")
    if not head:
        return
    parent = _rev("HEAD^")
    main_head = _rev("main")  # the `main` branch's own HEAD, from the same object database
    main_parent = _rev("main^")  # a bump commit on `main` can only name its parent (rule 5a)
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
    rows = [ln for ln in text.splitlines()
            if ln.startswith("| `T:/kindi`") or ln.startswith("| `<worktree>`")]
    assert len(rows) == 2, "worktree table rows not found in AGENTS.md"
    for row in rows:
        m = re.search(r"`([0-9a-f]{7,40})`", row.split("|")[3])
        assert m, "HEAD cell not found in row: %s" % row[:60]
        h = m.group(1)
        if row.startswith("| `T:/kindi`"):
            # `main` may legitimately lag (or lead) this session branch, but its row must name
            # `main`'s own HEAD -- or the parent of a bump commit on `main`, which by rule 5a
            # can only name the commit it bumps from (a commit cannot name itself). A row naming
            # anything older is still rejected, which is what the check exists for.
            ok = bool(main_head) and (main_head.startswith(h)
                                      or (bool(main_parent) and main_parent.startswith(h)))
        else:
            ok = head.startswith(h) or (bool(parent) and parent.startswith(h))
        assert ok, ("AGENTS.md worktree table says %s but %s; bump the "
                    "table in the same commit" % (h, ("main is %s" % main_head[:7]) if row.startswith("| `T:/kindi`")
                    else ("HEAD is %s" % head[:7])))
    adds = _changed_agents_lines() or []
    # A commit that touches AGENTS.md must also carry the hash refresh: its added lines have to
    # name HEAD or its parent (a bump cannot name itself, so the parent is the intended value).
    # Without this, a content commit that forgets the bump leaves the table stale and the parent
    # rule of the check above silently excuses it - which is exactly how this bit before.
    named = re.findall(r"`([0-9a-f]{7,40})`", "\n".join(adds))
    if adds:
        ok = any(head.startswith(h) or (bool(parent) and parent.startswith(h)) for h in named)
        assert ok, ("the HEAD commit touched AGENTS.md but did not refresh the worktree table "
                    "(no hash of HEAD %s or its parent %s in the added lines)"
                    % (head[:7], parent[:7]))
