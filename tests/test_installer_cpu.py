# SPDX-License-Identifier: AGPL-3.0-only
"""Fixture tests for install.sh's AVX CPU preflight.

The installer's `cpu_has_avx` function is extracted from install.sh and run against fake
/proc/cpuinfo files. No Linux kernel, GPU, model pack or Docker daemon is needed.

    python tests/test_installer_cpu.py
    python -m pytest tests/test_installer_cpu.py -q

KINDI_INSTALLER points the test at another copy, which is how the old-vs-fixed check is run.
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


def run_case(cpuinfo):
    """Run the extracted function with KINDI_CPUINFO=cpuinfo in a temp dir, or None without bash."""
    bash = shutil.which("bash")
    if bash is None:
        return None
    d = tempfile.mkdtemp(prefix="kindi-cpu-preflight-")
    try:
        path = os.path.join(d, "cpuinfo")
        if cpuinfo is not None:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(cpuinfo)
        script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                  + function_block("cpu_flag") + "\n"
                  + function_block("cpu_has_avx") + "\n"
                  + "KINDI_CPUINFO=cpuinfo\n"
                  + "if cpu_has_avx; then echo yes; else echo no; fi\n")
        case = os.path.join(d, "case.sh")
        with open(case, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)
        proc = subprocess.run([bash, case], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=60)
        return proc.returncode, proc.stdout.decode("utf-8", "replace").strip(), \
            proc.stderr.decode("utf-8", "replace")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_tier(cpuinfo):
    """Run the installer's own tier classifier against a fake /proc/cpuinfo."""
    bash = shutil.which("bash")
    if bash is None:
        return None
    d = tempfile.mkdtemp(prefix="kindi-cpu-tier-")
    try:
        path = os.path.join(d, "cpuinfo")
        if cpuinfo is not None:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(cpuinfo)
        script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                  + function_block("cpu_flag") + "\n"
                  + function_block("cpu_has_avx") + "\n"
                  + function_block("cpu_isa_tier") + "\n"
                  + "KINDI_CPUINFO=cpuinfo\n"
                  + "cpu_isa_tier\n")
        case = os.path.join(d, "tier.sh")
        with open(case, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)
        proc = subprocess.run([bash, case], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=60)
        return proc.returncode, proc.stdout.decode("utf-8", "replace").strip(), \
            proc.stderr.decode("utf-8", "replace")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_avx_is_accepted():
    got = run_case("processor\t: 0\nflags\t\t: sse avx avx2 fma f16c\n")
    if got is None:
        return
    code, out, err = got
    assert code == 0, err
    assert out == "yes"


def test_avx2_without_avx_is_rejected():
    got = run_case("processor\t: 0\nflags\t\t: sse avx2 fma f16c\n")
    if got is None:
        return
    code, out, err = got
    assert code == 0, err
    assert out == "no"


def test_missing_cpuinfo_is_rejected():
    got = run_case(None)
    if got is None:
        return
    code, out, err = got
    assert code == 0, err
    assert out == "no"


def test_avx2_f16c_without_fma_is_supported():
    got = run_tier("processor\t: 0\nflags\t\t: sse avx avx2 f16c\n")
    if got is None:
        return
    code, out, err = got
    assert code == 0, err
    assert out == "avx2-f16c-no-fma"


def test_full_isa_tier_is_reported():
    got = run_tier("processor\t: 0\nflags\t\t: sse avx avx2 fma f16c\n")
    if got is None:
        return
    code, out, err = got
    assert code == 0, err
    assert out == "avx2-fma-f16c"


def test_avx_only_tier_is_reported():
    got = run_tier("processor\t: 0\nflags\t\t: sse avx\n")
    if got is None:
        return
    code, out, err = got
    assert code == 0, err
    assert out == "avx-only"


def test_missing_cpuinfo_has_no_tier():
    got = run_tier(None)
    if got is None:
        return
    code, out, err = got
    assert code == 0, err
    assert out == "none"


def test_installer_uses_the_preflight_for_cpu_setups():
    text = installer_text()
    assert "CPU_AVX=0" in text and "CPU_AVX=1" in text
    assert "CPU_TIER" in text and "cpu_isa_tier" in text
    assert 'if [ "$CPU_AVX" = 1 ]' in text
    assert "the CPU engine needs AVX" in text
