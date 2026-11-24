# SPDX-License-Identifier: AGPL-3.0-only
"""The SYCL smoke driver, tested without oneAPI, a compiler or a GPU.

`kindi.intel_sycl` takes an injectable builder and runner precisely so this file can
prove the interesting half — that the driver refuses a child which passes on the wrong
backend, miscounts, or drops a size — on any machine.
"""
import json
import os

from kindi import intel_sycl


def _child(backend="ext_oneapi_level_zero", name="Intel(R) Arc(TM) A770 Graphics",
           sizes=None, result="pass", error="", is_gpu=True):
    sizes = sizes if sizes is not None else [
        {"n": n, "repeats": 5, "mismatches": 0, "max_abs_error": 0.0, "elapsed_ms": 2.0, "error": ""}
        for n in intel_sycl.SIZES]
    return {
        "probe": intel_sycl.PROBE, "schema": 1, "support_claim": False,
        "scope": "test",
        "toolchain": {"compiler": "icpx", "device_selector": intel_sycl.SELECTOR},
        "device": {"backend": backend, "name": name, "vendor": "Intel", "driver": "1.17",
                   "global_mem_mib": 15932, "local_mem_kb": 64, "is_gpu": is_gpu},
        "vector_add": {"local_size": 64, "sizes": sizes},
        "result": result, "stage": "done" if result == "pass" else "exactness", "error": error,
    }


def _tc(tmp_path):
    compiler = tmp_path / "icpx.exe"
    compiler.write_bytes(b"MZ")
    root = tmp_path / "oneAPI"
    (root / "compiler" / "latest" / "bin").mkdir(parents=True)
    (root / "2026.2" / "bin").mkdir(parents=True)
    return {"root": str(root), "compiler": str(compiler), "sycl_ls": "", "vcvars": "",
            "bin_dirs": [str(root / "compiler" / "latest" / "bin"), str(root / "2026.2" / "bin")],
            "lib_dirs": [str(root / "compiler" / "latest" / "lib")], "searched": [str(root)]}


def _builder(code=0, log=""):
    def builder(_bat):
        return code, log
    return builder


def _runner(payload, code=0, err=""):
    """A runner that prints `payload` the way the child does: one indented object."""
    text = json.dumps(payload, indent=2)

    def runner(_argv, _env):
        return code, text, err
    return runner


def test_toolchain_derives_the_root_from_an_explicit_compiler(tmp_path):
    root = tmp_path / "oneAPI"
    compiler = root / "compiler" / "2026.2" / "bin" / "icpx.exe"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"MZ")
    (root / "2026.2" / "bin").mkdir(parents=True)
    tc = intel_sycl.toolchain(compiler=str(compiler))
    assert tc["compiler"] == str(compiler)
    assert os.path.normcase(tc["root"]) == os.path.normcase(str(root))
    assert any(os.path.normcase(p).endswith(os.path.normcase(os.path.join("2026.2", "bin")))
               for p in tc["bin_dirs"]), tc["bin_dirs"]


def test_no_compiler_fails_at_the_toolchain_stage():
    tc = {"root": "", "compiler": "", "sycl_ls": "", "vcvars": "", "bin_dirs": [], "lib_dirs": [],
          "searched": ["a", "b"]}
    report = intel_sycl.smoke(toolchain_report=tc, builder=_builder(), runner=_runner(_child()))
    assert report["result"] == "fail" and report["stage"] == "toolchain"
    assert "KINDI_ONEAPI_ROOT" in report["error"] and "a, b" in report["error"]


def test_a_build_failure_is_reported_with_its_log(tmp_path):
    report = intel_sycl.smoke(toolchain_report=_tc(tmp_path), builder=_builder(1, "linker exploded"),
                              runner=_runner(_child()))
    assert report["result"] == "fail" and report["stage"] == "build"
    assert report["build"]["ok"] is False
    assert "linker exploded" in report["build"]["log"]
    assert "build failed" in report["error"]


def test_the_selector_reaches_the_child_and_a_pass_is_recorded(tmp_path):
    seen = {}

    def runner(argv, env):
        seen["argv"], seen["env"] = argv, env
        return 0, json.dumps(_child(), indent=2), ""

    report = intel_sycl.smoke(toolchain_report=_tc(tmp_path), builder=_builder(), runner=runner)
    assert report["result"] == "pass" and report["stage"] == "done"
    assert seen["env"]["ONEAPI_DEVICE_SELECTOR"] == "level_zero:gpu"
    assert "--selector" in seen["argv"] and "level_zero:gpu" in seen["argv"]
    assert report["device"]["backend"] == "ext_oneapi_level_zero"
    assert [row["n"] for row in report["vector_add"]["sizes"]] == list(intel_sycl.SIZES)


def test_a_child_on_another_backend_cannot_pass(tmp_path):
    """The child's own `pass` is not trusted: the driver checks the backend and the name."""
    report = intel_sycl.smoke(toolchain_report=_tc(tmp_path), builder=_builder(),
                              runner=_runner(_child(backend="opencl")))
    assert report["result"] == "fail" and report["stage"] == "validate"
    assert "not Level Zero" in report["error"]


def test_a_wrong_device_name_cannot_pass(tmp_path):
    report = intel_sycl.smoke(toolchain_report=_tc(tmp_path), builder=_builder(),
                              runner=_runner(_child(name="Intel(R) UHD Graphics 630")))
    assert report["result"] == "fail" and "does not name A770" in report["error"]


def test_a_mismatch_cannot_pass(tmp_path):
    sizes = [{"n": n, "repeats": 5, "mismatches": 0, "max_abs_error": 0.0, "elapsed_ms": 1.0,
              "error": ""} for n in intel_sycl.SIZES]
    sizes[1]["mismatches"] = 1
    sizes[1]["max_abs_error"] = 1.0
    report = intel_sycl.smoke(toolchain_report=_tc(tmp_path), builder=_builder(),
                              runner=_runner(_child(sizes=sizes)))
    assert report["result"] == "fail"
    assert "size 1 has 1 mismatches" in report["error"]


def test_a_missing_size_cannot_pass(tmp_path):
    sizes = [{"n": intel_sycl.SIZES[0], "repeats": 5, "mismatches": 0, "max_abs_error": 0.0,
              "elapsed_ms": 1.0, "error": ""}]
    report = intel_sycl.smoke(toolchain_report=_tc(tmp_path), builder=_builder(),
                              runner=_runner(_child(sizes=sizes)))
    assert report["result"] == "fail"
    assert "size %d is missing" % intel_sycl.SIZES[2] in report["error"]


def test_no_json_from_the_child_is_a_run_failure(tmp_path):
    report = intel_sycl.smoke(toolchain_report=_tc(tmp_path), builder=_builder(),
                              runner=lambda _argv, _env: (1, "Access violation", "boom"))
    assert report["result"] == "fail" and report["stage"] == "run"
    assert report["raw"]["stderr"] == "boom"


def test_validate_accepts_the_real_shape_and_render_names_the_result(tmp_path):
    report = intel_sycl.smoke(toolchain_report=_tc(tmp_path), builder=_builder(),
                              runner=_runner(_child()))
    assert intel_sycl.validate(_child()) == []
    text = "\n".join(intel_sycl.render(report))
    assert "result: pass" in text and "ext_oneapi_level_zero" in text
    assert "n=1048576" in text


def test_sizes_are_the_documented_ones():
    """The guard size must not be a multiple of the work-group size, or it proves nothing."""
    assert intel_sycl.SIZES[2] % 64 != 0
    assert intel_sycl.REPEATS == 5
