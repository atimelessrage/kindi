"""The CPU engine must call the configured compiler with an argument vector, never a shell string."""
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kindi import cpu

SRC = Path(__file__).resolve().parents[1] / "kindi" / "kcpu.c"


def _cc():
    """A compiler for the preprocessor checks, resolved the way the engine resolves it.

    `kindi.cpu._find_cc` also finds an LLVM install that is not on PATH (2026-11-24), so these
    checks stop skipping on a host that has clang installed but not exported.
    """
    for candidate in (os.environ.get("KINDI_CC"), "gcc", "clang"):
        if not candidate:
            continue
        found = cpu._find_cc(candidate)
        if os.path.isabs(found) or shutil.which(found):
            return found
    pytest.skip("neither gcc nor clang is installed")


def test_build_passes_paths_as_arguments(monkeypatch, tmp_path, capsys):
    seen = {}
    monkeypatch.setattr(cpu, "_native_has_fma", lambda: True)

    def fake_run(cmd):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)
    src, so = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.so")
    cpu._build(src, so)
    assert seen["cmd"] == cpu._build_cmd(src, so)
    assert "building CPU library with %s" % " ".join(seen["cmd"]) in capsys.readouterr().err


def test_linux_build_uses_gcc_with_the_documented_flags(tmp_path):
    src, so = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.so")
    assert cpu._build_cmd(src, so, win=False) == [
        "gcc", "-O3", "-march=native", "-fopenmp", "-fno-math-errno",
        "-shared", "-fPIC", "-o", so, src, "-lm"]


@pytest.mark.parametrize("isa", cpu.ISA_ORDER)
def test_each_isa_build_uses_explicit_flags(isa, tmp_path):
    src, so = str(tmp_path / "kcpu.c"), str(tmp_path / "libkcpu.so")
    cmd = cpu._build_cmd(src, so, win=False, isa=isa)
    assert "-march=native" not in cmd
    assert all(flag in cmd for flag in cpu.ISA_BUILDS[isa])


def test_each_isa_has_its_own_cache_path(tmp_path):
    paths = {isa: cpu._so_path(str(tmp_path), isa) for isa in cpu.ISA_ORDER}
    assert len(set(paths.values())) == len(paths)
    assert cpu._so_path(str(tmp_path)) not in paths.values()


def test_windows_build_uses_clang_libomp_and_no_posix_flags(monkeypatch, tmp_path):
    llvm = tmp_path / "llvm"
    (llvm / "lib").mkdir(parents=True)
    monkeypatch.setenv("KINDI_CC", "clang")
    monkeypatch.setenv("KINDI_LLVM", str(llvm))
    monkeypatch.setattr(cpu, "_find_cc", lambda name=None: "clang")   # pin the name; detection has its own tests
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
    monkeypatch.setattr(cpu, "_find_cc", lambda name=None: "clang")   # pin the name; detection has its own tests
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


def test_avx_only_library_builds_and_loads(monkeypatch, tmp_path):
    """Compile and load the fallback, not just preprocess it.

    The model-free test suite otherwise cannot catch a link/runtime break in the AVX-only
    branch.  The resulting library is placed at the keyed path the production loader uses, so
    this also checks the loader and the Python ctypes ABI without a pack.
    """
    if os.name == "nt":
        cc = cpu._find_cc()
        if not os.path.isabs(cc) and not shutil.which(cc):
            pytest.skip("clang is not installed; set KINDI_CC to build the AVX-only fallback")
        monkeypatch.setenv("KINDI_CC", cc)
    else:
        if not shutil.which("gcc"):
            pytest.skip("gcc is not installed; it is needed to build the AVX-only fallback")

    monkeypatch.setenv("KINDI_CPU_CACHE", str(tmp_path))
    monkeypatch.setenv("KINDI_CPU_ISA", "avx-only")
    so = Path(cpu._so_path(str(tmp_path), "avx-only"))
    cpu._build(str(SRC), str(so), isa="avx-only")

    library = cpu._lib()
    assert library.kc_isa_name() == b"avx-only"
    assert library.kc_sizeof_model() == ctypes.sizeof(cpu.Model)
    assert library.kc_sizeof_layer() == ctypes.sizeof(cpu.Layer)
    library.kc_init()


def test_avx2_f16c_without_fma_preprocesses():
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", "-mavx2", "-mno-fma", "-mf16c", "-E", str(SRC)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")


def test_avx2_f16c_no_fma_library_builds_and_loads(monkeypatch, tmp_path):
    """The AVX2/F16C/no-FMA branch must compile and load, not just preprocess."""
    if os.name == "nt":
        cc = cpu._find_cc()
        if not os.path.isabs(cc) and not shutil.which(cc):
            pytest.skip("clang is not installed; set KINDI_CC to build the no-FMA AVX2 fallback")
        monkeypatch.setenv("KINDI_CC", cc)
    else:
        if not shutil.which("gcc"):
            pytest.skip("gcc is not installed; it is needed to build the no-FMA AVX2 fallback")

    monkeypatch.setenv("KINDI_CPU_CACHE", str(tmp_path))
    monkeypatch.setenv("KINDI_CPU_ISA", "avx2-f16c-no-fma")
    so = Path(cpu._so_path(str(tmp_path), "avx2-f16c-no-fma"))
    cpu._build(str(SRC), str(so), isa="avx2-f16c-no-fma")

    library = cpu._lib()
    assert library.kc_isa_name() == b"avx2-f16c-no-fma"
    assert library.kc_sizeof_model() == ctypes.sizeof(cpu.Model)
    assert library.kc_sizeof_layer() == ctypes.sizeof(cpu.Layer)
    library.kc_init()


def test_full_isa_library_builds_and_loads(monkeypatch, tmp_path):
    """The native fast path compiles, loads and reports the tier the selector chose."""
    if cpu.cpu_tier() not in ("avx2-fma-f16c", "unknown"):
        pytest.skip("this host does not expose AVX2/FMA/F16C")
    if os.name == "nt":
        cc = cpu._find_cc()
        if not os.path.isabs(cc) and not shutil.which(cc):
            pytest.skip("clang is not installed; set KINDI_CC to build the native path")
        monkeypatch.setenv("KINDI_CC", cc)
    elif not shutil.which("gcc"):
        pytest.skip("gcc is not installed; it is needed to build the native path")

    monkeypatch.setenv("KINDI_CPU_CACHE", str(tmp_path))
    monkeypatch.setenv("KINDI_CPU_ISA", "avx2-fma-f16c")
    so = Path(cpu._so_path(str(tmp_path), "avx2-fma-f16c"))
    cpu._build(str(SRC), str(so), isa="avx2-fma-f16c")

    library = cpu._lib()
    assert library.kc_isa_name() == b"avx2-fma-f16c"
    assert library.kc_isa() == 4
    assert library.kc_sizeof_model() == ctypes.sizeof(cpu.Model)
    assert library.kc_sizeof_layer() == ctypes.sizeof(cpu.Layer)


def test_batch_and_state_entry_points_exist_and_are_empty_model_safe(monkeypatch, tmp_path):
    """The 2026-11-26 audit added three exports; a library built from this source must have them, and
    they must be safe to call before any batch exists (a zeroed Model has no rows and no state)."""
    cc = cpu._find_cc()
    if not os.path.isabs(cc) and not shutil.which(cc):
        pytest.skip("no compiler to build the CPU library")
    if os.name == "nt":
        monkeypatch.setenv("KINDI_CC", cc)
    monkeypatch.setenv("KINDI_CPU_CACHE", str(tmp_path))
    isa = "avx2-fma-f16c" if cpu.cpu_tier() in ("avx2-fma-f16c", "unknown") else "avx2-f16c-no-fma"
    monkeypatch.setenv("KINDI_CPU_ISA", isa)
    so = Path(cpu._so_path(str(tmp_path), isa))
    cpu._build(str(SRC), str(so), isa=isa)
    library = cpu._lib()

    m = cpu.Model()                              # zeroed: no layers, no state, no batch
    for name in ("kc_batch_release", "kc_state_load_len", "kc_batch_rollback", "kc_step_batch"):
        assert hasattr(library, name), "%s is missing from the built library" % name
    assert library.kc_state_len(ctypes.byref(m)) == 0
    buf = (ctypes.c_float * 1)()
    assert library.kc_state_load_len(ctypes.byref(m), buf, 1) == -1   # a wrong length is refused
    assert library.kc_state_load_len(ctypes.byref(m), buf, 0) == 0    # an empty state loads
    assert library.kc_batch_release() == 0
    assert library.kc_batch_release() == 0                            # idempotent
    assert library.kc_batch_rollback(ctypes.byref(m), 0) == -1        # nothing to roll back


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


@pytest.mark.parametrize("flags", (["-mavx2", "-mno-fma", "-mno-f16c"],
                                   ["-mavx2", "-mfma", "-mno-f16c"]))
def test_other_avx2_variants_preprocess(flags):
    cc = _cc()
    proc = subprocess.run([cc, "-fopenmp", *flags, "-E", str(SRC)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")


def test_find_cc_uses_kindi_cc_verbatim(monkeypatch, tmp_path):
    exe = tmp_path / "clang.exe"
    exe.write_text("")
    monkeypatch.setenv("KINDI_CC", str(exe))
    assert cpu._find_cc() == str(exe)


def test_find_cc_keeps_a_bare_name_it_cannot_find(monkeypatch):
    """A missing compiler must still name the command the build would have run."""
    monkeypatch.delenv("KINDI_CC", raising=False)
    monkeypatch.setattr(cpu.shutil, "which", lambda name: None)
    monkeypatch.setattr(cpu, "_windows_clang", lambda: None)
    assert cpu._find_cc("clang") == "clang"
    assert cpu._build_cmd("kcpu.c", "libkcpu.dll", win=True, native_fma=True)[0] == "clang"


def test_windows_clang_finds_an_llvm_install_off_path(monkeypatch, tmp_path):
    """2026-11-24: a full clang+LLVM 18.1.8 under Program Files, with `clang` not on PATH.

    The engine reported "could not run clang" and the CPU path looked unavailable.
    """
    for var in ("ProgramW6432", "ProgramFiles(x86)"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "Program Files"
    exe = root / "clang+llvm-18.1.8-x86_64-pc-windows-msvc" / "bin" / "clang.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setenv("ProgramFiles", str(root))
    assert cpu._windows_clang() == str(exe)


def test_windows_clang_prefers_the_newest_install(monkeypatch, tmp_path):
    for var in ("ProgramW6432", "ProgramFiles(x86)"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "Program Files"
    older = root / "clang+llvm-17.0.6" / "bin" / "clang.exe"
    newer = root / "clang+llvm-18.1.8" / "bin" / "clang.exe"
    for exe in (older, newer):
        exe.parent.mkdir(parents=True)
        exe.write_text("")
    monkeypatch.setenv("ProgramFiles", str(root))
    assert cpu._windows_clang() == str(newer)
    assert cpu._windows_clang() != str(older)


def test_find_cc_reports_the_install_as_the_cache_key(monkeypatch, tmp_path):
    """The cache key must name the compiler that will run, so two installs never share a library."""
    exe = tmp_path / "clang.exe"
    exe.write_text("")
    monkeypatch.setenv("KINDI_CC", str(exe))
    first = cpu._so_path(str(tmp_path), "avx2-fma-f16c")
    monkeypatch.setenv("KINDI_CC", "clang")
    assert cpu._so_path(str(tmp_path), "avx2-fma-f16c") != first


def test_rebuilt_library_is_picked_up(monkeypatch, tmp_path):
    """Editing kcpu.c must not keep running the previously loaded DLL.

    Before the source stamp, `_lib()` rebuilt the library when the source was newer and then
    returned the old cached handle for the same path - an edited kernel silently kept running the
    old code (and dlopen on the same path would return the mapped module anyway). The stamp makes a
    rebuild a new path, so `_LIB` cannot hand back the old handle.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "kcpu.c"
    src.write_text("/* v1 */")
    cache = tmp_path / "cache"
    monkeypatch.setenv("KINDI_CPU_CACHE", str(cache))
    monkeypatch.setattr(cpu, "HERE", str(src_dir))
    monkeypatch.setattr(cpu, "select_isa", lambda: None)
    monkeypatch.setattr(cpu, "check_features", lambda: None)
    built, loads = [], []

    def fake_build(s, so, isa=None):
        built.append(so)
        Path(so).write_bytes(b"")

    monkeypatch.setattr(cpu, "_build", fake_build)

    class _Fn(object):
        restype = None
        argtypes = None

        def __init__(self, value=0):
            self.value = value

        def __call__(self, *a):
            return self.value

    class FakeLib(object):
        def __init__(self, path):
            self.path = path
            loads.append(path)
            self.kc_isa_name = _Fn(b"native")

        def __getattr__(self, name):
            return _Fn()

        def kc_init(self):
            pass

        def kc_sizeof_model(self):
            return ctypes.sizeof(cpu.Model)

        def kc_sizeof_layer(self):
            return ctypes.sizeof(cpu.Layer)

    monkeypatch.setattr(ctypes, "CDLL", lambda path, *a, **k: FakeLib(path))
    saved = dict(cpu._LIB)
    cpu._LIB.clear()
    try:
        first = cpu._lib()
        assert cpu._lib() is first                       # same source, same handle
        assert len(loads) == len(built) == 1
        src.write_text("/* v2: an edited kernel */")
        second = cpu._lib()
        assert second is not first                       # the rebuild was picked up
        assert loads[1] != loads[0] and built[1] != built[0]
        assert second.path == cpu._so_path(str(cache))
    finally:
        cpu._LIB.clear()
        cpu._LIB.update(saved)
