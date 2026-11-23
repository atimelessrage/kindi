# SPDX-License-Identifier: AGPL-3.0-only
"""Fixture tests for tools/container_smoke.sh.

The smoke runs `kindi selftest` twice in a release image: once in a one-shot container (pack,
tokenizer, engine and both routes in process) and once with `--url` in the serving container
(the real Uvicorn route). No Docker daemon, GPU or image is needed here: a stub `docker` on PATH
records the calls and returns the exit codes each case needs.

    python tests/test_container_smoke.py
    python -m pytest tests/test_container_smoke.py -q
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "container_smoke.sh"

STUB = """#!/usr/bin/env bash
echo "docker $*" >> "$CALLS"
case "$*" in
    *"selftest --url"*) exit "${URL_RC:-0}" ;;
    *selftest*) exit "${ONESHOT_RC:-0}" ;;
    *"python3 -c"*) exit "${PROBE_RC:-0}" ;;
esac
exit 0
"""


def _bash():
    return shutil.which("bash")


def run_smoke(*args, oneshot_rc=0, url_rc=0, probe_rc=0):
    """Run the script against a stub docker; returns (rc, output, docker calls) or None without bash."""
    bash = _bash()
    if bash is None:
        return None
    d = tempfile.mkdtemp(prefix="kindi-container-smoke-")
    try:
        stub = os.path.join(d, "docker")
        with open(stub, "w", encoding="utf-8", newline="\n") as f:
            f.write(STUB)
        os.chmod(stub, 0o755)
        calls = os.path.join(d, "calls.txt")
        env = dict(os.environ, CALLS=calls, ONESHOT_RC=str(oneshot_rc), URL_RC=str(url_rc),
                   PROBE_RC=str(probe_rc), PATH=d + os.pathsep + os.environ.get("PATH", ""))
        proc = subprocess.run([bash, str(SCRIPT)] + list(args), cwd=str(ROOT), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
        recorded = Path(calls).read_text(encoding="utf-8") if os.path.exists(calls) else ""
        return proc.returncode, out, recorded
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_script_parses():
    bash = _bash()
    if bash is None:
        return
    proc = subprocess.run([bash, "-n", str(SCRIPT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")


def test_dry_run_prints_both_proofs_without_touching_docker():
    case = run_smoke("--dry-run")
    if case is None:
        return
    rc, out, calls = case
    assert rc == 0 and calls == ""
    assert "selftest" in out and "--entrypoint python3" in out
    assert "selftest --url http://127.0.0.1:8093" in out
    assert "run -d --name kindi-smoke" in out and "-e KINDI_HOST=0.0.0.0" in out


def test_both_checks_run_in_the_image_and_the_container_is_removed():
    case = run_smoke()
    if case is None:
        return
    rc, out, calls = case
    assert rc == 0, out
    assert "1/2 one-shot selftest" in out and "2/2 the serving container" in out
    assert "PASS: pack, tokenizer, engine, /health and the chat route" in out
    assert "--entrypoint python3" in calls and "-m kindi selftest" in calls
    assert "docker run -d --name kindi-smoke" in calls
    assert "selftest --url http://127.0.0.1:8093" in calls
    assert "KINDI_API_KEY=kindi-smoke" in calls          # docker exec carries it, so --url authenticates
    assert "docker rm -f kindi-smoke" in calls


def test_a_failed_one_shot_stops_before_the_server_starts():
    case = run_smoke(oneshot_rc=1)
    if case is None:
        return
    rc, out, calls = case
    assert rc == 1 and "the one-shot selftest failed" in out
    assert "run -d --name" not in calls


def test_a_failed_serving_selftest_fails_the_smoke_and_cleans_up():
    case = run_smoke(url_rc=1)
    if case is None:
        return
    rc, out, calls = case
    assert rc == 1 and "the serving selftest failed" in out
    assert "docker rm -f kindi-smoke" in calls


def test_no_health_answer_is_a_failure_with_the_logs():
    case = run_smoke("--wait", "0", probe_rc=1)
    if case is None:
        return
    rc, out, calls = case
    assert rc == 1 and "no /health answer" in out
    assert "docker logs --tail 40 kindi-smoke" in calls
    assert "selftest --url" not in calls
    assert "docker rm -f kindi-smoke" in calls


def test_keep_leaves_the_container_in_place():
    case = run_smoke("--keep")
    if case is None:
        return
    rc, out, calls = case
    assert rc == 0 and "kept kindi-smoke" in out
    assert "docker rm -f" not in calls
