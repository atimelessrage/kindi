"""The compiled kcpu library must load through the production loader and match the Python ctypes ABI.

The CI `cpu-engine-smoke` job compiles `kindi/kcpu.c` with gcc for the supported AVX2/FMA/F16C tier,
then sets `KINDI_CPU_CACHE` and `KINDI_CPU_SMOKE=1`. The default suite skips this: it needs a compiled
library and a C toolchain, and it deliberately needs no model pack.
"""
import ctypes
import os

import pytest

from kindi import cpu


@pytest.fixture(scope="module")
def library():
    if os.environ.get("KINDI_CPU_SMOKE") != "1":
        pytest.skip("set KINDI_CPU_SMOKE=1 with a prebuilt kcpu library")
    cache = os.environ.get("KINDI_CPU_CACHE")
    name = "libkcpu.dll" if os.name == "nt" else "libkcpu.so"
    if not cache or not os.path.isfile(os.path.join(cache, name)):
        pytest.fail("KINDI_CPU_CACHE does not contain %s" % name)
    return cpu._lib()


def test_compiled_library_loads_with_the_python_abi(library):
    assert library.kc_sizeof_model() == ctypes.sizeof(cpu.Model)
    assert library.kc_sizeof_layer() == ctypes.sizeof(cpu.Layer)
    assert callable(library.kc_step)


def test_compiled_library_initializes(library):
    library.kc_init()
    assert library.kc_sizeof_model() > 0
