# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for the q8-shaped ternary GEMM probe (REQ-0020 M2)."""
import numpy as np

from kindi import intel_gemm
from kindi.convert import phase_bytes


def test_the_weights_are_the_converters_pack_layout():
    digits, codes, scales = intel_gemm.matrix(16, 512, seed=7)
    assert np.array_equal(codes, np.ascontiguousarray(phase_bytes(digits).T))
    assert codes.shape == (512 // 4, 16)                        # [slot][row], 4 trits per byte
    assert scales.shape == (512 // 128, 16)                     # [phase][row], one scale per group
    assert scales.dtype == np.float16
    values = set(np.unique(scales.astype(np.float32)).tolist())
    assert values <= {1.0, 0.5, 2.0, 1.5}


def test_the_activation_group_maximum_is_forced_to_127():
    x = intel_gemm.activations(128, 512, seed=7)
    assert x.shape == (128, 512)
    amax = np.max(np.abs(x.reshape(128, 4, 128)), axis=2)
    assert np.array_equal(amax, np.full((128, 4), 127.0, dtype=np.float32))


def test_the_quantizer_is_a_pass_through_on_this_matrix():
    """On the exact matrix every xs is 1.0 and every XQ byte is the activation itself."""
    x = intel_gemm.activations(128, 256, seed=7)
    words, xs, signed = intel_gemm.quantize(x)
    assert np.array_equal(xs, np.ones_like(xs))
    assert np.array_equal(signed.astype(np.float32), x)
    assert signed.shape == (128, 256)
    assert words.shape == (2 * 32, 128)
    # byte order inside one word matches the engine's packing (4 consecutive columns)
    raw = np.frombuffer(words.tobytes(), dtype=np.uint8).reshape(words.shape[0], 128, 4)
    for g in range(2):
        expect = (x[:, g * 128:(g + 1) * 128].astype(np.int32) & 255).astype(np.uint8)
        assert np.array_equal(raw[g * 32:(g + 1) * 32], expect.reshape(128, 32, 4).transpose(1, 0, 2))


def test_the_quantizer_handles_a_zero_group():
    x = np.zeros((8, 128), dtype=np.float32)
    words, xs, signed = intel_gemm.quantize(x)
    assert np.array_equal(xs, np.zeros_like(xs))
    assert np.array_equal(signed, np.zeros_like(signed))
    assert np.array_equal(words, np.zeros_like(words))


def test_the_reference_matches_an_independent_float64_dot():
    """The fp32-stepped reference is the exact value, proved without trusting its own steps."""
    digits, _codes, scales = intel_gemm.matrix(128, 512, seed=7)
    x = intel_gemm.activations(128, 512, seed=7)
    _words, xs, signed = intel_gemm.quantize(x)
    trits = digits.astype(np.int32) - 1                                   # (rows, cols)
    scales_f32 = np.ascontiguousarray(scales.astype(np.float32).T)
    got = intel_gemm.reference(trits, signed, xs, scales_f32)
    want = intel_gemm.reference_f64(trits, signed, xs, scales_f32)
    assert np.array_equal(got.astype(np.float64), want)


def test_the_reference_uses_the_group_scale_of_its_own_rows():
    """A scale belonging to another row must change the result: the scale index is not constant."""
    digits, _codes, _scales = intel_gemm.matrix(128, 512, seed=7)
    x = intel_gemm.activations(128, 512, seed=7)
    _words, xs, signed = intel_gemm.quantize(x)
    trits = digits.astype(np.int32) - 1                                   # (rows, cols)
    pattern = np.asarray(intel_gemm.SCALES, dtype=np.float32)
    scales_f32 = np.ascontiguousarray(                               # (rows, groups)
        np.asarray([pattern[r % 4] for r in range(128)], dtype=np.float32)[:, None]
        * pattern[None, :])
    got = intel_gemm.reference(trits, signed, xs, scales_f32)
    permuted = np.ascontiguousarray(np.roll(scales_f32, 1, axis=0))
    other = intel_gemm.reference(trits, signed, xs, permuted)
    assert not np.array_equal(got, other)


def test_the_probe_is_flagged_as_a_probe():
    report = intel_gemm.run_probe(loader="missing-opencl-loader-probe.dll")
    assert report["result"] == "fail"
    assert report["loader"]["available"] is False
    assert report["support_claim"] is False
    assert report["probe"] == "intel-ternary-gemm"
    assert "no sycl" in report["backend"]
    assert "could not load" in report["error"] or "unavailable" in report["error"]


def test_the_shape_is_validated_before_any_device_work():
    assert "--rows" in intel_gemm.run_probe(rows=64)["error"]
    assert "--rows" in intel_gemm.run_probe(rows=129)["error"]
    assert "--tokens" in intel_gemm.run_probe(tokens=64)["error"]
    assert "--cols" in intel_gemm.run_probe(cols=100)["error"]
    assert "--repeats" in intel_gemm.run_probe(repeats=0)["error"]
    assert "--device" in intel_gemm.run_probe(device=-1)["error"]
    assert "--staging" in intel_gemm.run_probe(staging="tensor")["error"]


def test_both_stagings_are_offered_and_reported():
    """The un-tiled shape stays reachable: it is the page's first measured baseline."""
    assert intel_gemm.STAGINGS == ("local", "global")
    global_report = intel_gemm.run_probe(staging="global",
                                         loader="missing-opencl-loader-probe.dll")
    local_report = intel_gemm.run_probe(loader="missing-opencl-loader-probe.dll")
    assert global_report["shape"]["staging"] == "global"
    assert global_report["shape"]["local_bytes"] == 0
    assert local_report["shape"]["staging"] == "local"
    assert local_report["shape"]["local_bytes"] == intel_gemm.LOCAL_BYTES
    assert 0 < intel_gemm.LOCAL_BYTES <= 64 * 1024        # fits the measured card's 64 KiB


def test_a_non_arc_device_is_named_in_the_warning():
    """Device order is driver-defined; a run that lands on the iGPU must say so, not pass silently."""
    source = intel_gemm.SOURCE
    assert "device_name" in intel_gemm.run_probe.__code__.co_varnames
    rendered = intel_gemm.render(
        dict(intel_gemm._base_report("OpenCL.dll", 128, 128, 512, 5, "local"),
             warnings=["the selected device is Intel(R) UHD Graphics 630, not an Arc; "
                       "pass --device-name Arc"], result="pass", error=None))
    assert any("warning:" in line and "--device-name Arc" in line for line in rendered)
    assert "gemm_tile_local" in source             # the staged kernel exists to be selected


def test_the_kernel_stages_the_tile_it_claims():
    source = intel_gemm.SOURCE
    assert "__kernel void q8x" in source and "__kernel void gemm_tile" in source
    assert "__kernel void gemm_tile_local" in source
    assert "rint" in source                       # the CUDA quantizer rounds half to even
    assert "127.0f" in source                     # the threshold pair
    assert "decode_byte" in source                # the pack's 4-trit layout, not a requantization
    assert "dp4a_manual" in source                # OpenCL C 1.2 has no dot-product instruction
    assert "barrier(CLK_LOCAL_MEM_FENCE)" in source


def test_the_two_staged_kernels_accumulate_identically():
    """Staging must not change the arithmetic: both kernels keep the same group order and scale."""
    source = intel_gemm.SOURCE
    global_body = source.split("__kernel void gemm_tile(")[1].split("__kernel void gemm_tile_local(")[0]
    local_body = source.split("__kernel void gemm_tile_local(")[1]
    for body in (global_body, local_body):
        assert "is[i][j] = dp4a_manual(a[i], b[j], is[i][j])" in body   # same int32 dot
        assert "(float)is[i][j] * (xx * dj)" in body                    # same scale, same order
        assert "barrier" in body or body is global_body
    assert local_body.count("barrier(CLK_LOCAL_MEM_FENCE)") == 3
