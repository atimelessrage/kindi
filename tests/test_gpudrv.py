"""The NVIDIA driver library follows the platform: nvcuda.dll on Windows, libcuda.so.1 on Linux."""
import ast
import ctypes
from pathlib import Path
import types

import pytest

from kindi import gpudrv


def test_driver_name_windows(monkeypatch):
    monkeypatch.setattr(gpudrv, "os", types.SimpleNamespace(name="nt"))
    assert gpudrv.driver_name() == "nvcuda.dll"


def test_driver_name_linux(monkeypatch):
    monkeypatch.setattr(gpudrv, "os", types.SimpleNamespace(name="posix"))
    assert gpudrv.driver_name() == "libcuda.so.1"


def test_driver_loads_the_platform_library_once(monkeypatch):
    monkeypatch.setattr(gpudrv, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(gpudrv, "HIP", False)
    monkeypatch.setattr(gpudrv, "_drv", [None])
    seen = []
    fake = object()

    def fake_cdll(name):
        seen.append(name)
        return fake

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    assert gpudrv.driver() is fake
    assert gpudrv.driver() is fake
    assert seen == ["nvcuda.dll"]


@pytest.mark.parametrize("filename", ("sched.py", "chunk1.py"))
def test_nvidia_driver_is_loaded_through_gpudrv(filename):
    """A hardcoded libcuda.so.1 bypasses the platform selector and breaks native Windows."""
    path = Path(gpudrv.__file__).resolve().parent / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hardcoded = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and "libcuda.so.1" in node.value
    ]
    assert not hardcoded, (
        "%s hardcodes libcuda.so.1 at line(s) %s; use gpudrv.driver() instead"
        % (filename, ", ".join(str(line) for line in hardcoded))
    )
    uses_gpudrv = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "driver"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gpudrv"
        for node in ast.walk(tree)
    )
    assert uses_gpudrv, "%s does not call gpudrv.driver()" % filename
