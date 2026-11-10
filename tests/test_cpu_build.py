"""The CPU engine must call gcc with an argument vector, never a shell string."""
import subprocess

import pytest

from kindi import cpu


def test_build_passes_paths_as_arguments(monkeypatch, tmp_path):
    seen = {}

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
    cmd = cpu._build_cmd(src, dll, win=True)
    assert cmd[0] == "clang" and cmd[1] == "-O3"
    assert "-fopenmp" in cmd and "-ffp-contract=off" in cmd
    assert "-fPIC" not in cmd and "-lm" not in cmd
    assert cmd[cmd.index("-o") + 1] == dll
    assert cmd[-4] == src
    assert cmd[-3:] == ["-L", str(llvm / "lib"), "-llibomp"]


def test_build_failure_is_a_clear_exit(monkeypatch):
    monkeypatch.setattr(cpu.subprocess, "run",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(SystemExit, match="could not build the CPU engine"):
        cpu._build("kcpu.c", "libkcpu.so")
