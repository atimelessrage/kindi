# SPDX-License-Identifier: AGPL-3.0-only
"""Prove the AVX-only and AVX2/F16C no-FMA CPU fallbacks are token-exact on a real Bonsai pack.

The normal CPU suite compiles and loads both paths without a pack.  These tests build the engine
with the requested ISA, load that exact library through the production loader, and run `kindi
verify` against the packs named in KINDI_CPU_AVX_PACKS or KINDI_CPU_NO_FMA_PACKS.  They are opt-in
because the model packs are not part of CI; when a variable is unset, that test is skipped.

On Windows the builds use the dynamic UCRT: the static libcmt expf contains FMA even with -mno-fma,
which would invalidate a no-FMA run.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kindi import cpu

ROOT = Path(__file__).resolve().parents[1]
PACKS = tuple(p for p in os.environ.get("KINDI_CPU_AVX_PACKS", "").split(os.pathsep) if p)
NO_FMA_PACKS = tuple(p for p in os.environ.get("KINDI_CPU_NO_FMA_PACKS", "").split(os.pathsep) if p)


def _require_toolchain():
    """Skip when the C/OpenMP toolchain this test needs is not installed."""
    if os.name == "nt":
        if not (os.environ.get("KINDI_CC") or shutil.which("clang")):
            pytest.skip("clang is not installed; set KINDI_CC to build the AVX-only fallback")
    elif not shutil.which("gcc"):
        pytest.skip("gcc is not installed; it is needed to build the AVX-only fallback")


def _build_no_fma(src, so):
    """Build the engine with no AVX2/FMA/F16C and return the compiler output."""
    cmd = cpu._build_cmd(str(src), str(so), win=os.name == "nt", native_fma=False)
    native = cmd.index("-march=native")
    cmd[native:native + 1] = [
        "-mavx", "-mno-avx2", "-mno-fma", "-mno-f16c", "-ffp-contract=off",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0, "AVX-only build failed:\n%s\n%s" % (" ".join(cmd), proc.stdout)
    assert so.is_file(), "AVX-only build did not produce %s" % so


def _build_avx2_no_fma(src, so):
    """Build the AVX2/F16C path with FMA disabled."""
    cmd = cpu._build_cmd(str(src), str(so), win=os.name == "nt", native_fma=False)
    native = cmd.index("-march=native")
    cmd[native:native + 1] = [
        "-mavx2", "-mno-fma", "-mf16c", "-ffp-contract=off",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0, "AVX2/F16C no-FMA build failed:\n%s\n%s" % (" ".join(cmd), proc.stdout)
    assert so.is_file(), "AVX2/F16C no-FMA build did not produce %s" % so


@pytest.mark.parametrize("pack", PACKS or (None,))
def test_no_fma_verify_is_token_exact(pack, tmp_path):
    if pack is None:
        pytest.skip("set KINDI_CPU_AVX_PACKS to one or more Bonsai pack directories")
    _require_toolchain()

    pack = Path(pack).resolve()
    assert (pack / "kindi_pack.json").is_file(), "missing pack manifest: %s" % pack
    src = ROOT / "kindi" / "kcpu.c"
    so = Path(cpu._so_path(str(tmp_path)))
    _build_no_fma(src, so)

    env = os.environ.copy()
    env["KINDI_DEVICE"] = "cpu"
    env["KINDI_CPU_CACHE"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "kindi", "--pack", str(pack), "--ctx", "1024", "verify"],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert "ALL PASS: token-exact with llama.cpp" in proc.stdout, proc.stdout[-4000:]


@pytest.mark.parametrize("pack", NO_FMA_PACKS or (None,))
def test_avx2_no_fma_verify_is_token_exact(pack, tmp_path):
    if pack is None:
        pytest.skip("set KINDI_CPU_NO_FMA_PACKS to one or more Bonsai pack directories")
    _require_toolchain()

    pack = Path(pack).resolve()
    assert (pack / "kindi_pack.json").is_file(), "missing pack manifest: %s" % pack
    src = ROOT / "kindi" / "kcpu.c"
    so = Path(cpu._so_path(str(tmp_path)))
    _build_avx2_no_fma(src, so)

    env = os.environ.copy()
    env["KINDI_DEVICE"] = "cpu"
    env["KINDI_CPU_CACHE"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "kindi", "--pack", str(pack), "--ctx", "1024", "verify"],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert "ALL PASS: token-exact with llama.cpp" in proc.stdout, proc.stdout[-4000:]
