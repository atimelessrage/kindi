"""CI configuration guards: pinned test deps and the static-analysis job stay wired.

These are requirement-level checks on the delivered harness, not implementation tests:
the test dependencies must stay bounded (an open spec lets CI break with no commit) and
the CUDA static-analysis harness must stay invoked from ci.yml (D5).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _reqs():
    lines = []
    for raw in (ROOT / "tests" / "requirements-test.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def test_test_requirements_are_bounded():
    reqs = _reqs()
    assert reqs, "requirements file must list dependencies"
    for line in reqs:
        assert ">=" in line and "<" in line, "unbounded dependency spec: %s" % line
    by_name = {line.split(">=")[0].strip() for line in reqs}
    assert {"pytest", "pytest-cov", "fastapi", "pydantic", "httpx", "numpy"} <= by_name


def test_static_analysis_job_invokes_the_harness():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "static-analysis:" in ci
    assert "python tests/static/run_static_analysis.py" in ci
    assert (ROOT / "tests" / "static" / "run_static_analysis.py").exists()


def test_gpu_smoke_offers_the_optional_mtp_head_job():
    """The MTP smoke is dispatches only with a head path; without one it must stay skipped."""
    gpu = (ROOT / ".github" / "workflows" / "gpu-smoke.yml").read_text(encoding="utf-8")
    assert "mtp_head:" in gpu
    assert "mtp-head:" in gpu
    assert "${{ inputs.mtp_head != '' }}" in gpu
    assert "python tools/mtp_smoke.py" in gpu
    assert '--head "$KINDI_MTP_HEAD"' in gpu


def test_cpu_job_runs_the_documentation_and_generated_reference_gates():
    """The handoff names these as gates; before this check they could only fail by hand."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for command in ("python tools/check_claims.py",
                    "python tools/project_memory.py --check",
                    "python -m kindi.config_reference --check",
                    "python -m kindi.support_matrix --check"):
        assert command in ci, command
    assert "Documentation, claims and generated-reference gates" in ci


def test_workflow_actions_are_pinned_to_shas():
    """Every workflow's actions are immutable: CPU CI alone is not the attack surface a push adds."""
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows, "no workflow files found"
    for workflow in workflows:
        for line in workflow.read_text(encoding="utf-8").splitlines():
            if "uses:" in line:
                ref = line.split("uses:")[-1].strip().split()[0]
                assert "@" in ref and len(ref.rsplit("@", 1)[-1]) >= 40, (
                    "unpinned action in %s: %s" % (workflow.name, ref))
