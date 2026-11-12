# SPDX-License-Identifier: AGPL-3.0-only
"""The project memory must stay structured and its generated history current."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "project_memory.py"
SKILL = ROOT / ".agents" / "skills" / "project-memory" / "SKILL.md"


def load():
    spec = importlib.util.spec_from_file_location("project_memory", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_requirement_history_is_current():
    mod = load()
    assert mod.HISTORY.read_text(encoding="utf-8") == mod.render()


def test_requirements_have_stable_ids_and_required_sections():
    mod = load()
    rows = mod.current_requirements()
    assert rows
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        text = row["path"].read_text(encoding="utf-8")
        assert "- Status: " in text, row["path"]
        assert "## Requirement" in text, row["path"]
        assert "## Acceptance" in text, row["path"]
        assert "## Evidence" in text, row["path"]
        assert row["id"] in row["path"].name


def test_deleted_requirements_have_a_separate_section():
    text = load().render()
    assert "## Current requirements" in text
    assert "## Deleted requirements" in text
    assert "A deleted requirement is retained here" in text


def test_project_skill_is_discoverable():
    assert SKILL.is_file()
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\nname: project-memory\n")
    assert "description:" in text
    assert "docs/development/README.md" in text


def test_commit_hashes_and_deleted_requirements_are_recorded(tmp_path):
    """Run the generator in a temporary repository: the added hash must be real, and a
    deleted requirement must survive in the separate section with its deletion commit."""
    tool = tmp_path / "tools"
    req = tmp_path / "docs" / "development" / "requirements" / "REQ-0001.md"
    tool.mkdir(parents=True)
    req.parent.mkdir(parents=True)
    shutil.copy(SCRIPT, tool / "project_memory.py")
    req.write_text("""# REQ-0001: Temporary requirement

- Status: active

## Requirement

Temporary.

## Acceptance

Temporary.

## Evidence

Temporary.
""", encoding="utf-8")

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              text=True, capture_output=True).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("add", "docs/development/requirements/REQ-0001.md")
    git("commit", "-q", "-m", "add requirement")
    added = git("rev-parse", "HEAD")
    subprocess.run([sys.executable, str(tool / "project_memory.py")], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    current = (tmp_path / "docs" / "development" / "REQUIREMENTS-HISTORY.md").read_text(
        encoding="utf-8").split("## Deleted requirements")[0]
    assert added[:12] in current
    assert "| uncommitted |" not in current

    req.unlink()
    git("add", "-A")
    git("commit", "-q", "-m", "remove requirement")
    removed = git("rev-parse", "HEAD")
    subprocess.run([sys.executable, str(tool / "project_memory.py")], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    text = (tmp_path / "docs" / "development" / "REQUIREMENTS-HISTORY.md").read_text(
        encoding="utf-8")
    deleted = text.split("## Deleted requirements", 1)[1]
    assert "REQ-0001" in deleted
    assert removed[:12] in deleted
    assert "Temporary requirement" in deleted
