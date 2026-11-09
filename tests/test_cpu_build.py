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
    assert seen["cmd"] == ["gcc", "-O3", "-march=native", "-fopenmp", "-fno-math-errno",
                           "-shared", "-fPIC", "-o", so, src, "-lm"]


def test_build_failure_is_a_clear_exit(monkeypatch):
    monkeypatch.setattr(cpu.subprocess, "run",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(SystemExit, match="could not build the CPU engine"):
        cpu._build("kcpu.c", "libkcpu.so")
