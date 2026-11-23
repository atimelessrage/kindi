# SPDX-License-Identifier: AGPL-3.0-only
"""Fixture tests for install.sh's post-install `kindi selftest` step.

The installer runs `kindi selftest --url` against each serving container once it answers /health,
and `--no-selftest` skips it. Both paths are exercised here against a stubbed `docker`, so no
Docker daemon, GPU, model pack or running server is needed.

    python tests/test_installer_selftest.py
    python -m pytest tests/test_installer_selftest.py -q

KINDI_INSTALLER points the test at another copy of the installer.
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = Path(os.environ.get("KINDI_INSTALLER", ROOT / "install.sh"))


def installer_text():
    return INSTALLER.read_text(encoding="utf-8").replace("\r\n", "\n")


def function_block(name):
    """The installer's own `name() { ... }`, by name."""
    lines = installer_text().splitlines()
    pat = re.compile(r"^%s\(\)\s*\{" % re.escape(name))
    start = next((i for i, line in enumerate(lines) if pat.match(line)), None)
    assert start is not None, "no %s() in %s" % (name, INSTALLER)
    end = next((i for i in range(start, len(lines)) if lines[i] == "}"), None)
    assert end is not None, "unterminated %s() in %s" % (name, INSTALLER)
    return "\n".join(lines[start:end + 1])


def option_loop_block():
    lines = installer_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("while [ $# -gt 0 ]; do")), None)
    assert start is not None, "no option loop in %s" % INSTALLER
    end = next((i for i in range(start, len(lines)) if lines[i] == "done"), None)
    assert end is not None, "unterminated option loop in %s" % INSTALLER
    return "\n".join(lines[start:end + 1])


def _bash():
    return shutil.which("bash")


def run_function(no_selftest, dry, fake_rc=0):
    """Run the installer's selftest_server with a stub docker; returns (rc, stdout, docker calls)."""
    bash = _bash()
    if bash is None:
        return None
    d = tempfile.mkdtemp(prefix="kindi-installer-selftest-")
    try:
        script = "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'calls="calls.txt"',
            'info() { echo "info $*"; }',
            'ok() { echo "ok $*"; }',
            'warn() { echo "warn $*"; }',
            'fake_docker() { echo "docker $*" >> "$calls"; return "${FAKE_RC:-0}"; }',
            "DOCKER=fake_docker",
            'DIR="/tmp/kindi-test"',
            'NO_SELFTEST="%s"' % no_selftest,
            function_block("selftest_server"),
            'if selftest_server kindi 1 "%s"; then echo "rc=0"; else echo "rc=$?"; fi' % dry,
        ]) + "\n"
        path = os.path.join(d, "case.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)
        proc = subprocess.run([bash, path], cwd=d, env=dict(os.environ, FAKE_RC=str(fake_rc)),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        calls_path = os.path.join(d, "calls.txt")
        calls = Path(calls_path).read_text(encoding="utf-8") if os.path.exists(calls_path) else ""
        return proc.returncode, proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace"), calls
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_selftest_is_skipped_with_no_selftest():
    case = run_function(no_selftest=1, dry=0)
    if case is None:
        return
    rc, out, calls = case
    assert rc == 0 and "rc=0" in out
    assert "selftest skipped (--no-selftest)" in out
    assert calls == ""                    # nothing ran against the server


def test_selftest_prints_its_command_in_a_dry_run():
    case = run_function(no_selftest=0, dry=1)
    if case is None:
        return
    rc, out, calls = case
    assert rc == 0 and "rc=0" in out
    assert "selftest --url http://127.0.0.1:8093" in out
    assert "docker exec kindi python3 -m kindi selftest" in out
    assert calls == ""                    # --dry-run never touches the daemon


def test_selftest_runs_against_the_container():
    case = run_function(no_selftest=0, dry=0)
    if case is None:
        return
    rc, out, calls = case
    assert rc == 0 and "rc=0" in out
    assert calls.strip().startswith("docker exec kindi python3 -m kindi selftest --url http://127.0.0.1:8093")
    assert "--json /data/selftest-1.json" in calls
    assert "selftest passes the pack, the route and one generation" in out


def test_a_failed_selftest_stops_the_installer():
    case = run_function(no_selftest=0, dry=0, fake_rc=1)
    if case is None:
        return
    rc, out, calls = case
    assert rc == 0                            # the harness itself succeeds; the function reported the failure
    assert "rc=1" in out and "selftest failed" in out
    assert "selftest passes" not in out


def test_the_installer_calls_the_step_and_offers_the_opt_out():
    text = installer_text()
    assert 'selftest_server "$nm" "$((i + 1))" "$DRY"' in text, "the start loop does not run the selftest"
    assert "NO_SELFTEST=0" in text
    assert "--no-selftest) NO_SELFTEST=1 ;;" in text
    bash = _bash()
    if bash is None:
        return
    proc = subprocess.run([bash, str(INSTALLER), "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=60)
    assert proc.returncode == 0, "install.sh --help failed"
    assert b"--no-selftest" in proc.stdout


def test_the_option_loop_sets_no_selftest():
    bash = _bash()
    if bash is None:
        return
    prelude = ("DRY=0; YES=0; AUTO=0; CHOICE=''; MODE=install; PURGE=0; MODEL=auto; CTX=auto; GPU=auto; "
               "PORT=8093; DIR=/tmp; NAME=kindi; SOURCE=''; THREADS=''; MEMORY=''; LAN=0; APIKEY=''; "
               "BRIDGE=ask; BRIDGE_SOURCE=''; ROUTE_PORT=8090; NO_DSPINE=0; NO_MTP=0; NO_SELFTEST=0; "
               "PLACE=(); PLAN_IN=''; PLAN_MODE=0; usage() { :; }")
    script = "#!/usr/bin/env bash\nset -euo pipefail\n" + prelude + "\nset -- --no-selftest\n" \
             + option_loop_block() + '\necho "NO_SELFTEST=$NO_SELFTEST"\n'
    d = tempfile.mkdtemp(prefix="kindi-installer-options-")
    try:
        path = os.path.join(d, "options.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)
        proc = subprocess.run([bash, path], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        assert b"NO_SELFTEST=1" in proc.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
