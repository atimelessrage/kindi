# SPDX-License-Identifier: AGPL-3.0-only
"""The installer must recognize the CPU engine's AVX2/FMA/F16C requirement.

The helper is extracted from install.sh so the test cannot drift from the real function. A fake
`grep` makes the result independent of the host CPU; no /proc/cpuinfo or compiler is needed.
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = Path(os.environ.get("KINDI_INSTALLER", ROOT / "install.sh"))


def function_block(name):
    lines = INSTALLER.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
    pat = re.compile(r"^%s\(\)\s*\{" % re.escape(name))
    start = next((i for i, line in enumerate(lines) if pat.match(line)), None)
    assert start is not None, "no %s() in %s" % (name, INSTALLER)
    end = next((i for i in range(start, len(lines)) if lines[i] == "}"), None)
    assert end is not None, "unterminated %s() in %s" % (name, INSTALLER)
    return "\n".join(lines[start:end + 1])


def run_case(present):
    bash = shutil.which("bash")
    if bash is None:
        return None
    pattern = "|".join("*%s*" % flag for flag in present) or "!*"
    fake = 'grep() { case "$*" in %s) return 0 ;; *) return 1 ;; esac; }' % pattern
    script = ("#!/usr/bin/env bash\nset -euo pipefail\n" + fake + "\n"
              + function_block("cpu_isa_missing") + "\nprintf '%s' \"$(cpu_isa_missing)\"\n")
    d = tempfile.mkdtemp(prefix="kindi-cpu-")
    try:
        path = os.path.join(d, "case.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)
        proc = subprocess.run([bash, path], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=30)
        return proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cpu_preflight_accepts_all_three_requirements():
    got = run_case(("avx2", "fma", "f16c"))
    if got is None:
        return
    code, out, err = got
    assert code == 0, err or out
    assert out == ""


def test_cpu_preflight_reports_every_missing_flag():
    got = run_case(("avx2",))
    if got is None:
        return
    code, out, err = got
    assert code == 0, err or out
    assert out == "fma f16c "
