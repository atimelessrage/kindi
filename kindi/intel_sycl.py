# SPDX-License-Identifier: AGPL-3.0-only
"""SYCL vector-add smoke for an Intel GPU (REQ-0020 M2 evidence; no backend).

The OpenCL probes (`intel-cl`, `intel-ternary`, `intel-gemm`) prove exact device compute
through OpenCL C. The M2 gate names something else: the **SYCL runtime** and SYCL device
selection. This module builds `tools/intel_sycl_vecadd.cpp` with the oneAPI DPC++ compiler,
runs it under `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, and validates what the child reports.

Three rules keep a pass meaningful:

* **Level Zero only.** A child that says `pass` while naming the OpenCL backend or the host
  device is a failure here, and the child itself refuses to pass on any other backend.
* **Exact, not sampled.** The inputs are integer-valued float32, so every sum is exact and
  the comparison is bit-for-bit over all elements, including a size that is not a multiple
  of the work-group size (the kernel's bounds guard).
* **The toolchain is reported, never assumed.** The compiler, the Visual Studio environment
  and the oneAPI root the build used are recorded, so a saved report can be re-read later.

Nothing here is imported by the engine and no support claim is made: Windows is not the M2
target, this is not a Kindi kernel, and one exact vector-add is not a model, engine or GEMM
result. Timings are observations.
"""
import hashlib
import json
import os
import re
import subprocess
import tempfile

PROBE = "intel-sycl-smoke"
SCHEMA = 1
SELECTOR = "level_zero:gpu"
# The sizes the child is built to run (it asserts them itself; this is the driver's copy).
SIZES = (1048576, 1, 1000003)
REPEATS = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "tools", "intel_sycl_vecadd.cpp")
COMPILERS = ("icpx", "dpcpp", "icx-cl")
BUILD_TIMEOUT = 600
RUN_TIMEOUT = 600
_VCVARS = ("VC", "Auxiliary", "Build", "vcvars64.bat")


def _env(name):
    return os.environ.get(name) or ""


def _oneapi_roots():
    """Candidate oneAPI roots, most explicit first: KINDI_ONEAPI_ROOT, ONEAPI_ROOT, defaults."""
    roots = []
    for value in (_env("KINDI_ONEAPI_ROOT"), _env("ONEAPI_ROOT")):
        if value:
            roots.append(value)
    if os.name == "nt":
        for base in (_env("ProgramFiles(x86)"), _env("ProgramFiles")):
            if base:
                roots.append(os.path.join(base, "Intel", "oneAPI"))
    else:
        roots.append("/opt/intel/oneapi")
    return roots


def _compiler_under(root, name):
    """The compiler executable under `<root>/compiler/latest/bin`, or an empty string."""
    for middle in ("compiler/latest/bin", "compiler/2026.2/bin"):
        path = os.path.join(root, *middle.split("/"), name + (".exe" if os.name == "nt" else ""))
        if os.path.isfile(path):
            return path
    return ""


def _version_dirs(root):
    """The versioned install directories (`2026.2`, ...): the SYCL runtime DLLs live there."""
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    return [os.path.join(root, name) for name in names if re.match(r"^\d{4}\.\d", name)]


def _bin_dirs(root):
    """Directories that must be on PATH: the compiler, and the runtime + UR adapters."""
    dirs = []

    def add(path):
        if path and os.path.isdir(path) and path not in dirs:
            dirs.append(path)

    add(os.path.join(root, "compiler", "latest", "bin"))
    add(os.path.join(root, "bin"))
    for version in _version_dirs(root):
        add(os.path.join(version, "bin"))
    add(os.path.join(root, "compiler", "latest", "lib"))
    return dirs


def _lib_dirs(root):
    """oneAPI's library directories: the DPC++ link needs `libmmd.lib` from one of these."""
    dirs = []

    def add(path):
        if path and os.path.isdir(path) and path not in dirs:
            dirs.append(path)

    add(os.path.join(root, "compiler", "latest", "lib"))
    add(os.path.join(root, "compiler", "latest", "lib", "intel64"))
    for version in _version_dirs(root):
        add(os.path.join(version, "lib"))
    return dirs


def _vcvars():
    """`vcvars64.bat` (KINDI_VCVARS wins), found via vswhere and the usual install roots."""
    explicit = _env("KINDI_VCVARS")
    if explicit:
        return explicit if os.path.isfile(explicit) else ""
    if os.name != "nt":
        return ""
    vswhere = os.path.join(_env("ProgramFiles(x86)") or r"C:\Program Files (x86)",
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.isfile(vswhere):
        try:
            proc = subprocess.run([vswhere, "-latest", "-property", "installationPath",
                                   "-products", "*"], capture_output=True, text=True, timeout=30)
            for line in (proc.stdout or "").splitlines():
                candidate = os.path.join(line.strip(), *_VCVARS)
                if line.strip() and os.path.isfile(candidate):
                    return candidate
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def toolchain(root=None, compiler=None, vcvars=None):
    """Resolve the compiler, the oneAPI root and the MSVC environment.

    ``root``/``compiler``/``vcvars`` are explicit overrides (the CLI flags); otherwise
    `KINDI_ONEAPI_ROOT`, `ONEAPI_ROOT` and the default install roots are tried, and a root
    only counts when a compiler actually exists under it.
    """
    searched = []
    found_root = ""
    found_compiler = compiler or ""
    if not found_compiler:
        for candidate in ([root] if root else []) + _oneapi_roots():
            if not candidate:
                continue
            searched.append(candidate)
            for name in COMPILERS:
                path = _compiler_under(candidate, name)
                if path:
                    found_root, found_compiler = candidate, path
                    break
            if found_compiler:
                break
        if not found_compiler:
            for name in COMPILERS:                       # a compiler already on PATH
                on_path = _which(name)
                if on_path:
                    found_compiler = on_path
                    break
    if found_compiler and not found_root:
        # `...\compiler\<ver>\bin\icpx.exe` -> the oneAPI root is three levels up from bin
        found_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(found_compiler))))
    svl = os.path.join(found_root, *"compiler/latest/bin/sycl-ls.exe".split("/")) if found_root else ""
    return {
        "root": found_root,
        "compiler": found_compiler,
        "sycl_ls": svl if svl and os.path.isfile(svl) else "",
        "vcvars": _vcvars() if vcvars is None else (vcvars if vcvars and os.path.isfile(vcvars) else ""),
        "bin_dirs": _bin_dirs(found_root) if found_root else [],
        "lib_dirs": _lib_dirs(found_root) if found_root else [],
        "searched": searched,
    }


def _which(name):
    from shutil import which
    return which(name) or ""


def build_env(tc, base=None):
    """The environment a build or a run needs: oneAPI bins (the L0 UR adapter) and libs."""
    env = dict(os.environ if base is None else base)
    if tc.get("bin_dirs"):
        env["PATH"] = os.pathsep.join(list(tc["bin_dirs"]) + [env.get("PATH", "")]).strip(os.pathsep)
    if tc.get("lib_dirs"):
        env["LIB"] = os.pathsep.join(list(tc["lib_dirs"]) + [env.get("LIB", "")]).strip(os.pathsep)
    return env


def _bat(tc, binary, source):
    """The build script: vcvars first (MSVC headers/libs), then the DPC++ compiler."""
    lines = ["@echo off"]
    if tc.get("vcvars"):
        lines.append('call "%s" >nul 2>&1' % tc["vcvars"])
    if tc.get("bin_dirs"):
        lines.append('set "PATH=%s;%%PATH%%"' % ";".join(tc["bin_dirs"]))
    if tc.get("lib_dirs"):
        lines.append('set "LIB=%s;%%LIB%%"' % ";".join(tc["lib_dirs"]))
    lines.append('"%s" -fsycl -O2 -o "%s" "%s"' % (tc["compiler"], binary, source))
    lines.append("exit /b %ERRORLEVEL%")
    return "\r\n".join(lines) + "\r\n"


def build(tc, workdir=None, builder=None):
    """Compile the smoke; returns ok/log/hashes. ``builder(bat_path)`` is injectable."""
    if not tc.get("compiler"):
        return {"ok": False, "log": "", "error": "no oneAPI DPC++ compiler found",
                "binary": "", "sha256": None, "source_sha256": None, "command": None}
    if not os.path.isfile(SOURCE):
        return {"ok": False, "log": "", "error": "missing %s" % SOURCE, "binary": "",
                "sha256": None, "source_sha256": None, "command": None}
    workdir = workdir or tempfile.mkdtemp(prefix="kindi-sycl-")
    binary = os.path.join(workdir, "intel_sycl_vecadd.exe" if os.name == "nt" else "intel_sycl_vecadd")
    bat = os.path.join(workdir, "build.bat")
    with open(bat, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(_bat(tc, binary, SOURCE))
    command = "%s -fsycl -O2 -o <binary> tools/intel_sycl_vecadd.cpp" % os.path.basename(
        tc["compiler"])
    if builder is None:
        proc = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True,
                              env=build_env(tc), timeout=BUILD_TIMEOUT)
        code, log = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        # A real build that exits 0 without producing the binary is still a failure.
        ok = int(code) == 0 and os.path.isfile(binary)
    else:
        code, log = builder(bat)          # an injected builder owns its own output
        ok = int(code) == 0
    with open(SOURCE, "rb") as f:
        source_sha = hashlib.sha256(f.read()).hexdigest()
    if ok:
        failure = None
    elif int(code) != 0:
        failure = "the build failed (exit %s)" % code
    else:
        failure = "the build exited 0 but produced no binary"
    return {"ok": ok, "log": log[-4000:], "error": failure,
            "binary": binary if ok else "",
            "sha256": _sha256(binary) if ok else None,
            "source_sha256": source_sha, "command": command}


def _sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def run(tc, binary, selector=SELECTOR, runner=None):
    """Run the child and parse its JSON; ``runner(binary, env)`` is injectable."""
    if not binary:
        return {"ok": False, "report": None, "stdout": "", "stderr": "",
                "error": "nothing to run (no binary)", "returncode": None}
    env = build_env(tc)
    env["ONEAPI_DEVICE_SELECTOR"] = selector
    argv = [binary, "--selector", selector]
    if tc.get("compiler"):
        argv += ["--compiler-version", os.path.basename(tc["compiler"])]
    if runner is None:
        proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=RUN_TIMEOUT)
        code, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    else:
        code, out, err = runner(argv, env)
    # The child prints one indented object whose rows are themselves one-line objects, so
    # parse the outermost braces, not the last line that happens to look like JSON.
    report = None
    start, end = out.find("{"), out.rfind("}")
    if start != -1 and end > start:
        try:
            report = json.loads(out[start:end + 1])
        except ValueError:
            report = None
    return {"ok": report is not None, "report": report, "stdout": out[-4000:],
            "stderr": err[-4000:], "returncode": code,
            "error": None if report is not None else "the child printed no JSON report"}


def validate(report, device_name="A770"):
    """The checks the driver enforces itself, so a lying child cannot pass.

    Returns a list of problems; empty means the child's own arithmetic *and* its device
    identity are acceptable.
    """
    problems = []
    if not isinstance(report, dict):
        return ["no child report"]
    if report.get("probe") != PROBE:
        problems.append("probe is %r" % report.get("probe"))
    if report.get("schema") != SCHEMA:
        problems.append("schema is %r" % report.get("schema"))
    if report.get("support_claim") is not False:
        problems.append("the child claims support")
    device = report.get("device") or {}
    if "level_zero" not in str(device.get("backend", "")):
        problems.append("backend is %r, not Level Zero" % device.get("backend"))
    if device_name and device_name.lower() not in str(device.get("name", "")).lower():
        problems.append("device %r does not name %s" % (device.get("name"), device_name))
    if not device.get("is_gpu"):
        problems.append("the child did not select a GPU")
    rows = ((report.get("vector_add") or {}).get("sizes")) or []
    seen = {}
    for row in rows:
        seen[row.get("n")] = row
    for n in SIZES:
        row = seen.get(n)
        if row is None:
            problems.append("size %d is missing" % n)
            continue
        if row.get("mismatches") != 0:
            problems.append("size %d has %r mismatches" % (n, row.get("mismatches")))
        if not row.get("max_abs_error") == 0:
            problems.append("size %d max_abs_error is %r" % (n, row.get("max_abs_error")))
        if row.get("error"):
            problems.append("size %d reported %r" % (n, row.get("error")))
    if report.get("result") != "pass":
        problems.append("the child's result is %r" % report.get("result"))
    if report.get("error"):
        problems.append("the child's error is %r" % report.get("error"))
    return problems


def smoke(root=None, compiler=None, vcvars=None, selector=SELECTOR, device_name="A770",
          workdir=None, builder=None, runner=None, toolchain_report=None):
    """Build and run the smoke; never raises for an expected failure."""
    tc = toolchain_report or toolchain(root=root, compiler=compiler, vcvars=vcvars)
    report = {
        "probe": PROBE,
        "schema": SCHEMA,
        "support_claim": False,
        "scope": "SYCL runtime and device selection on this host; not a model, engine or Kindi kernel",
        "toolchain": {"compiler": tc.get("compiler") or None, "oneapi_root": tc.get("root") or None,
                      "vcvars": tc.get("vcvars") or None, "device_selector": selector,
                      "bin_dirs": tc.get("bin_dirs") or [], "searched": tc.get("searched") or []},
        "build": None,
        "device": None,
        "vector_add": {"sizes": [], "expected_sizes": list(SIZES), "repeats": REPEATS},
        "result": "fail",
        "stage": "toolchain",
        "error": None,
        "problems": [],
    }
    if not tc.get("compiler"):
        report["error"] = ("no oneAPI DPC++ compiler found; set KINDI_ONEAPI_ROOT or --compiler "
                           "(searched: %s)" % (", ".join(tc.get("searched") or []) or "nothing"))
        return report
    built = build(tc, workdir=workdir, builder=builder)
    report["build"] = built
    if not built["ok"]:
        report["stage"] = "build"
        report["error"] = built["error"]
        return report
    ran = run(tc, built["binary"], selector=selector, runner=runner)
    report["stage"] = "run"
    if not ran["ok"]:
        report["error"] = ran["error"]
        report["raw"] = {"returncode": ran["returncode"], "stdout": ran["stdout"], "stderr": ran["stderr"]}
        return report
    child = ran["report"]
    report["device"] = child.get("device")
    report["vector_add"]["sizes"] = ((child.get("vector_add") or {}).get("sizes")) or []
    report["vector_add"]["local_size"] = (child.get("vector_add") or {}).get("local_size")
    report["child_result"] = child.get("result")
    report["child_stage"] = child.get("stage")
    report["child_error"] = child.get("error")
    report["problems"] = validate(child, device_name=device_name)
    if report["problems"]:
        report["stage"] = "validate"
        report["error"] = "; ".join(report["problems"])
        return report
    report["result"] = "pass"
    report["stage"] = "done"
    return report


def render(report):
    """Human lines, in the style of the other Intel probes."""
    tc = report["toolchain"]
    lines = ["SYCL vector-add smoke (oneAPI DPC++, Level Zero; no backend or support claim)"]
    lines.append("  compiler: %s" % (tc.get("compiler") or "not found"))
    lines.append("  oneAPI root: %s" % (tc.get("oneapi_root") or "not found"))
    if tc.get("vcvars"):
        lines.append("  MSVC environment: %s" % os.path.basename(tc["vcvars"]))
    if report["build"] and report["build"].get("ok"):
        lines.append("  build: ok (sha256 %s)" % (report["build"]["sha256"] or "")[:16])
    device = report.get("device") or {}
    if device:
        lines.append("  device: %s; backend %s; driver %s; %s MiB, %s KiB local" % (
            device.get("name"), device.get("backend"), device.get("driver"),
            device.get("global_mem_mib"), device.get("local_mem_kb")))
    else:
        lines.append("  device: not reached")
    for row in report["vector_add"]["sizes"]:
        lines.append("  n=%-8d mismatches %d  max_abs_error %.1f  %d launches %.2f ms" % (
            row.get("n", 0), row.get("mismatches", -1), row.get("max_abs_error", -1.0),
            row.get("repeats", 0), row.get("elapsed_ms", 0.0)))
    lines.append("  selector: %s" % tc.get("device_selector"))
    lines.append("  result: %s" % report["result"])
    if report.get("error"):
        lines.append("  error (%s): %s" % (report.get("stage"), report["error"]))
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
