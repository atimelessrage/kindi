# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for the packed ternary GEMV probe (REQ-0020 M2 prototype)."""
import numpy as np

from kindi import intel_ternary
from kindi.convert import phase_bytes


def _digits(bcodes, rows, cols):
    codes = np.asarray(bcodes, dtype=np.int32).reshape(cols // 128, 32, rows)
    return np.stack([codes % 3, (codes // 3) % 3, (codes // 9) % 3, (codes // 27) % 3],
                    axis=2).transpose(3, 0, 1, 2).reshape(rows, cols)


def test_the_pack_is_the_converter_layout():
    trits, bcodes, dpm, x = intel_ternary.matrix(16, 512, seed=7)
    assert np.array_equal(bcodes, np.ascontiguousarray(phase_bytes(trits).T))
    assert bcodes.shape == (512 // 4, 16)          # [slot][row], 4 trits per byte
    assert dpm.shape == (512 // 128, 16)           # [phase][row], one scale per 128 weights
    assert dpm.dtype == np.float16
    assert x.shape == (512,)
    assert np.array_equal(np.unique(x), np.arange(-3, 4))


def test_the_packed_bytes_decode_back_to_the_trits():
    trits, bcodes, dpm, x = intel_ternary.matrix(16, 512, seed=7)
    assert np.array_equal(_digits(bcodes, 16, 512), trits)


def test_the_reference_matches_an_independent_float64_dot():
    trits, bcodes, dpm, x = intel_ternary.matrix(16, 512, seed=7)
    got = intel_ternary.reference(bcodes, dpm, x, 16, 512)
    scales = dpm.astype(np.float64).T.repeat(128, axis=1)
    want = ((trits.astype(np.float64) - 1.0) * x[None, :] * scales).sum(axis=1)
    assert np.array_equal(got.astype(np.float64), want)


def test_the_probe_is_flagged_as_a_probe():
    report = intel_ternary.run_probe(loader="missing-opencl-loader-probe.dll")
    assert report["result"] == "fail"
    assert report["loader"]["available"] is False
    assert report["support_claim"] is False
    assert report["probe"] == "intel-ternary-gemv"
    assert "no sycl" in report["backend"]
    assert "OpenCL loader unavailable" in report["error"] or "could not load" in report["error"]


def test_the_shape_is_validated_before_any_device_work():
    assert "multiple of 128" in intel_ternary.run_probe(cols=100)["error"]
    assert "--rows" in intel_ternary.run_probe(rows=0)["error"]
    assert "--repeats" in intel_ternary.run_probe(repeats=0)["error"]
    assert "--device" in intel_ternary.run_probe(device=-1)["error"]
