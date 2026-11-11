"""The CPU engine must call the configured compiler with an argument vector, never a shell string."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kindi import cpu

SRC = Path(__file__).resolve().parents[1] / "kindi" / "kcpu.c"


def _cc():
    """A compiler for the preprocessor checks, preferring KINDI_CC."""
    for candidate in (os.environ.get("KINDI_CC"), "gcc", "clang"):
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    pytest.skip("neither gcc nor clang is installed")


def test_build_passes_paths_as_arguments(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cpu, "_native_has_fma", lambda: True)

    def fake_run(cmd):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)
    src, so = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.so")
    cpu._build(src, so)
    assert seen["cmd"] == cpu._build_cmd(src, so)


def test_linux_build_uses_gcc_with_the_documented_flags(tmp_path):
    src, so = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.so")
    assert cpu._build_cmd(src, so, win=False) == [
        "gcc", "-O3", "-march=native", "-fopenmp", "-fno-math-errno",
        "-shared", "-fPIC", "-o", so, src, "-lm"]


def test_windows_build_uses_clang_libomp_and_no_posix_flags(monkeypatch, tmp_path):
    llvm = tmp_path / "llvm"
    (llvm / "lib").mkdir(parents=True)
    monkeypatch.setenv("KINDI_CC", "clang")
    monkeypatch.setenv("KINDI_LLVM", str(llvm))
    src, dll = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.dll")
    cmd = cpu._build_cmd(src, dll, win=True, native_fma=False)
    assert cmd[0] == "clang" and cmd[1] == "-O3"
    assert "-fopenmp" in cmd and "-ffp-contract=off" in cmd
    assert "-fPIC" not in cmd and "-lm" not in cmd
    assert cmd[cmd.index("-o") + 1] == dll
    assert src in cmd
    assert cmd[-7:-4] == ["-L", str(llvm / "lib"), "-llibomp"]
    assert cmd[-4:] == ["-Xlinker", "/nodefaultlib:libcmt",
                        "-Xlinker", "/defaultlib:msvcrt"]


def test_windows_build_keeps_static_crt_when_fma_is_available(monkeypatch, tmp_path):
    llvm = tmp_path / "llvm"
    (llvm / "lib").mkdir(parents=True)
    monkeypatch.setenv("KINDI_CC", "clang")
    monkeypatch.setenv("KINDI_LLVM", str(llvm))
    src, dll = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.dll")
    cmd = cpu._build_cmd(src, dll, win=True, native_fma=True)
    assert "-Xlinker" not in cmd
    assert cmd[-3:] == ["-L", str(llvm / "lib"), "-llibomp"]


def test_cpu_cache_path_is_keyed_by_toolchain(monkeypatch, tmp_path):
    monkeypatch.setenv("KINDI_CC", "gcc")
    first = cpu._so_path(str(tmp_path))
    monkeypatch.setenv("KINDI_CC", "clang")
    second = cpu._so_path(str(tmp_path))
    assert first != second
    assert os.path.basename(first).startswith("libkcpu-")
    assert os.path.dirname(first) == str(tmp_path)


def test_build_failure_is_a_clear_exit(monkeypatch):
    monkeypatch.setattr(cpu, "_native_has_fma", lambda: True)
    monkeypatch.setattr(cpu.subprocess, "run",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(SystemExit, match="could not build the CPU engine"):
        cpu._build("kcpu.c", "libkcpu.so")


def test_source_has_the_isa_guard():
    assert "requires AVX" in SRC.read_text(encoding="utf-8")


def test_avx_only_build_preprocesses():
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", "-mavx", "-mno-avx2", "-mno-fma", "-mno-f16c", "-E", str(SRC)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")


def test_non_avx_build_is_rejected_with_the_required_isa():
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", "-mno-avx", "-mno-avx2", "-mno-fma", "-mno-f16c", "-E", str(SRC)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode != 0
    assert "requires AVX" in proc.stderr.decode("utf-8", "replace")


def test_supported_isa_source_preprocesses():
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", "-mavx2", "-mfma", "-mf16c", "-E", str(SRC)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
