# SPDX-License-Identifier: AGPL-3.0-only
"""Packed ternary GEMV probe on an Intel GPU through OpenCL (REQ-0020 M2 prototype).

This is a probe, not a backend: it builds a small matrix in the *pack file's* ternary layout
(``bcodes.bin`` / ``dpm.bin``), uploads it as-is, launches one OpenCL C GEMV kernel, and checks
every output element against a host reference that decodes the same packed bytes.  It does not
quantize activations, load a model, run the Kindi engine or justify an Intel support claim.

The layout is the one ``kindi convert`` writes (``convert.phase_bytes``):

* weights: 4 trits per byte, byte = t0 + 3*t1 + 9*t2 + 27*t3, weight = (trit - 1) * scale, stored
  slot-major as ``[column/4][row]``;
* scales: one fp16 per 128 weights, stored phase-major as ``[column/128][row]``; the probe widens
  them to float32 on the host (exact) because OpenCL C 1.2 has no portable ``half`` arithmetic.

The probe's inputs are small integers and its scales are powers of two, so the dot products are
exactly representable and the device result must match the host bit for bit; a mismatch is a
failure, never a tolerance.
"""
import ctypes
import hashlib
import json
import time

import numpy as np

from .convert import phase_bytes
from .intel_cl import (
    CL_CONTEXT_PLATFORM,
    CL_MEM_COPY_HOST_PTR,
    CL_MEM_READ_ONLY,
    CL_MEM_WRITE_ONLY,
    CL_SUCCESS,
    DEFAULT_LOADER,
    _bind,
    _build_log,
    _device_report,
    _enumerate,
    _finish,
    _hex,
    _launch,
    _open,
    _read_back,
    _set_arg,
)

PROBE = "intel-ternary-gemv"
DEFAULT_ROWS = 64
DEFAULT_COLS = 1024
MIN_ROWS = 1
MAX_ROWS = 1024
MIN_COLS = 128
MAX_COLS = 8192
DEFAULT_REPEATS = 5
MAX_REPEATS = 50
PHASE = 128
SCALES = (1.0, 0.5, 2.0, 1.5)          # fp16-exact powers of two / halves: every product is exact
SEED = 7
SOURCE = """
__kernel void ternary_gemv(__global const uchar *codes,
                           __global const float *scales,
                           __global const float *x,
                           __global float *y,
                           uint rows, uint cols) {
    uint row = get_global_id(0);
    if (row >= rows) {
        return;
    }
    uint phases = cols >> 7;
    float total = 0.0f;
    for (uint ph = 0; ph < phases; ++ph) {
        int dot = 0;
        uint base = ((ph << 5) * rows) + row;        /* slot (ph*32 + s), stored [slot][row] */
        uint col = ph << 7;
        for (uint s = 0; s < 32; ++s) {
            uint b = (uint)codes[base + s * rows];
            uint c = col + (s << 2);
            dot += ((int)(b % 3u) - 1) * (int)x[c];
            dot += ((int)((b / 3u) % 3u) - 1) * (int)x[c + 1];
            dot += ((int)((b / 9u) % 3u) - 1) * (int)x[c + 2];
            dot += ((int)((b / 27u) % 3u) - 1) * (int)x[c + 3];
        }
        total += (float)dot * scales[(ph * rows) + row];
    }
    y[row] = total;
}
"""


def _fail(report, message):
    report["error"] = message
    report["result"] = "fail"
    return report


def _base_report(loader, rows, cols, repeats, seed):
    return {
        "probe": PROBE,
        "schema": 1,
        "read_only": False,
        "support_claim": False,
        "backend": "opencl packed-ternary gemv probe; no sycl, model, engine or backend",
        "scope": "packed ternary GEMV (4 trits/byte, one fp16 scale per 128 weights), "
                 "checked against an exact host reference",
        "loader": {"name": loader, "available": False, "error": None},
        "platforms": [],
        "platform": None,
        "device_count": None,
        "device_index": None,
        "device": None,
        "layout": {
            "weights": "4 trits per byte, byte = t0 + 3*t1 + 9*t2 + 27*t3; [column/4][row]",
            "scales": "fp16 per 128 weights, [column/128][row]; widened to float32 on the host",
            "weight": "trit - 1, scaled by the 128-weight group's scale",
        },
        "shape": {"rows": rows, "cols": cols, "phases": cols // 128 if isinstance(cols, int) else None,
                  "packed_bytes": None, "scale_bytes": None},
        "pack": {"seed": seed, "bcodes_sha256": None, "dpm_sha256": None},
        "build": {"ok": False, "source_bytes": len(SOURCE.encode("utf-8")), "log": None},
        "gemv": {"rows": rows, "cols": cols, "repeats": repeats, "warmup_ms": None,
                 "mismatches": None, "max_abs_error": None, "first_mismatch": None,
                 "elapsed_ms": None, "gflops": None},
        "returncodes": {},
        "error": None,
        "result": "fail",
    }


def matrix(rows, cols, seed=SEED):
    """Build (trits, bcodes, dpm, x) in the pack's layout; bcodes/dpm are exactly what is uploaded."""
    row_ids = np.arange(rows, dtype=np.int64)[:, None]
    col_ids = np.arange(cols, dtype=np.int64)[None, :]
    trits = ((row_ids * 131 + col_ids * 17 + seed * 29) % 3).astype(np.uint8)
    codes = phase_bytes(trits)                                  # (rows, cols/4), the converter's bytes
    bcodes = np.ascontiguousarray(codes.T)                      # (cols/4, rows): [slot][row]
    phases = cols // PHASE
    pattern = np.asarray(SCALES, dtype=np.float16)
    scales = np.empty((rows, phases), dtype=np.float16)
    scales[:] = pattern[np.arange(phases) % pattern.size]
    dpm = np.ascontiguousarray(scales.T)                        # (phases, rows): [phase][row]
    x = ((np.arange(cols, dtype=np.int64) * 5 + 3) % 7 - 3).astype(np.float32)
    return trits, bcodes, dpm, x


def reference(bcodes, dpm, x, rows, cols):
    """Host GEMV over the packed bytes, accumulating each 128-weight group in int32 like the kernel."""
    phases = cols // PHASE
    codes = np.asarray(bcodes, dtype=np.uint8).reshape(phases, 32, rows)
    scales = np.asarray(dpm, dtype=np.float16).astype(np.float32)
    out = np.zeros(rows, dtype=np.float32)
    for ph in range(phases):
        b = codes[ph].astype(np.int32)                          # (32, rows)
        xv = np.asarray(x, dtype=np.int32)[ph * PHASE:(ph + 1) * PHASE].reshape(32, 4)
        dot = ((b % 3 - 1) * xv[:, 0][:, None]
               + (b // 3 % 3 - 1) * xv[:, 1][:, None]
               + (b // 9 % 3 - 1) * xv[:, 2][:, None]
               + (b // 27 % 3 - 1) * xv[:, 3][:, None]).sum(axis=0)
        out = (out + dot.astype(np.float32) * scales[ph]).astype(np.float32)
    return out


def _buffer(lib, context, flags, data, report, key):
    array = np.ascontiguousarray(data)
    pointer = (ctypes.c_void_p(int(array.ctypes.data))
               if flags & CL_MEM_COPY_HOST_PTR and array.nbytes else None)
    error = ctypes.c_int(0)
    handle = lib.clCreateBuffer(context, flags, ctypes.c_size_t(int(array.nbytes)), pointer,
                                ctypes.byref(error))
    report["returncodes"][key] = _hex(error.value)
    if not handle or error.value != CL_SUCCESS:
        raise ValueError("%s returned %s" % (key, _hex(error.value)))
    return handle


def run_probe(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, repeats=DEFAULT_REPEATS, loader=None,
              opener=None, device=0, seed=SEED):
    """Launch the packed ternary GEMV on the selected Intel GPU; it never runs on the CPU."""
    loader = loader or DEFAULT_LOADER
    report = _base_report(loader, rows, cols, repeats, seed)
    if not isinstance(rows, int) or not MIN_ROWS <= rows <= MAX_ROWS:
        return _fail(report, "--rows must be an integer in [%d, %d]" % (MIN_ROWS, MAX_ROWS))
    if not isinstance(cols, int) or not MIN_COLS <= cols <= MAX_COLS or cols % PHASE:
        return _fail(report, "--cols must be a multiple of %d in [%d, %d]"
                     % (PHASE, MIN_COLS, MAX_COLS))
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        return _fail(report, "--repeats must be an integer in [1, %d]" % MAX_REPEATS)
    if not isinstance(device, int) or device < 0:
        return _fail(report, "--device must be a non-negative integer")

    _trits, bcodes, dpm, x = matrix(rows, cols, seed)
    expected = reference(bcodes, dpm, x, rows, cols)
    scales = np.ascontiguousarray(dpm.astype(np.float32))
    report["shape"]["packed_bytes"] = int(bcodes.nbytes)
    report["shape"]["scale_bytes"] = int(dpm.nbytes)
    report["pack"]["bcodes_sha256"] = hashlib.sha256(bcodes.tobytes()).hexdigest()
    report["pack"]["dpm_sha256"] = hashlib.sha256(dpm.tobytes()).hexdigest()

    lib, error = _open(loader, opener)
    if lib is None:
        return _fail(report, error or "OpenCL loader unavailable")
    report["loader"]["available"] = True
    try:
        _bind(lib)
    except (AttributeError, TypeError, ValueError) as exc:
        return _fail(report, "OpenCL loader is missing an API: %s" % exc)

    context = None
    queue = None
    program = None
    kernel = None
    buffers = []
    try:
        devices, error = _enumerate(lib, report)
        if devices is None:
            return _fail(report, error or "OpenCL device enumeration failed")
        report["device_count"] = len(devices)
        if not devices:
            return _fail(report, "no Intel OpenCL GPU device is visible")
        if device >= len(devices):
            return _fail(report, "--device %d is outside the %d visible Intel GPU(s)"
                         % (device, len(devices)))
        platform, device_handle, platform_entry = devices[device]
        report["device_index"] = device
        report["platform"] = platform_entry["name"]
        info = _device_report(lib, device_handle)
        if info is None:
            return _fail(report, "could not read the selected device's properties")
        report["device"] = info

        properties = (ctypes.c_void_p * 3)(CL_CONTEXT_PLATFORM, platform, None)
        handles = (ctypes.c_void_p * 1)(device_handle)
        error_code = ctypes.c_int(0)
        context = lib.clCreateContext(properties, 1, handles, None, None,
                                      ctypes.byref(error_code))
        report["returncodes"]["clCreateContext"] = _hex(error_code.value)
        if not context or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateContext returned %s" % _hex(error_code.value))

        error_code = ctypes.c_int(0)
        queue = lib.clCreateCommandQueue(context, device_handle, 0, ctypes.byref(error_code))
        report["returncodes"]["clCreateCommandQueue"] = _hex(error_code.value)
        if not queue or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateCommandQueue returned %s" % _hex(error_code.value))

        source = SOURCE.encode("utf-8")
        lengths = ctypes.c_size_t(len(source))
        strings = (ctypes.c_char_p * 1)(source)
        error_code = ctypes.c_int(0)
        program = lib.clCreateProgramWithSource(context, 1, strings, ctypes.byref(lengths),
                                                ctypes.byref(error_code))
        report["returncodes"]["clCreateProgramWithSource"] = _hex(error_code.value)
        if not program or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateProgramWithSource returned %s"
                         % _hex(error_code.value))

        devices_array = (ctypes.c_void_p * 1)(device_handle)
        code = int(lib.clBuildProgram(program, 1, devices_array, None, None, None))
        report["returncodes"]["clBuildProgram"] = _hex(code)
        report["build"]["log"] = _build_log(lib, program, device_handle)
        if code != CL_SUCCESS:
            first = (report["build"]["log"] or "").strip().splitlines()
            return _fail(report, "clBuildProgram returned %s%s" % (
                _hex(code), "" if not first else ": " + first[0][:200]))
        report["build"]["ok"] = True

        error_code = ctypes.c_int(0)
        kernel = lib.clCreateKernel(program, b"ternary_gemv", ctypes.byref(error_code))
        report["returncodes"]["clCreateKernel"] = _hex(error_code.value)
        if not kernel or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateKernel returned %s" % _hex(error_code.value))

        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bcodes,
                               report, "clCreateBufferCodes"))
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales,
                               report, "clCreateBufferScales"))
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x,
                               report, "clCreateBufferX"))
        buffers.append(_buffer(lib, context, CL_MEM_WRITE_ONLY,
                               np.zeros(rows, dtype=np.float32), report, "clCreateBufferY"))
        for index, handle in enumerate(buffers):
            _set_arg(lib, kernel, index, ctypes.c_void_p(int(handle)))
        _set_arg(lib, kernel, 4, ctypes.c_uint32(rows))
        _set_arg(lib, kernel, 5, ctypes.c_uint32(cols))

        started = time.perf_counter()
        _launch(lib, queue, kernel, rows, report, "clEnqueueNDRangeKernelWarmup")
        _finish(lib, queue, report, "clFinishWarmup")
        report["gemv"]["warmup_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

        started = time.perf_counter()
        for _ in range(repeats):
            _launch(lib, queue, kernel, rows, report, "clEnqueueNDRangeKernel")
        _finish(lib, queue, report, "clFinish")
        seconds = time.perf_counter() - started
        report["gemv"]["elapsed_ms"] = round(seconds * 1000.0, 3)
        if seconds > 0:
            report["gemv"]["gflops"] = round(2.0 * rows * cols * repeats / (seconds * 1e9), 3)

        raw = _read_back(lib, queue, buffers[3], rows, report, "clEnqueueReadBuffer")
        actual = np.frombuffer(raw, dtype=np.float32)
        if actual.size != rows:
            return _fail(report, "read back %d of %d outputs" % (actual.size, rows))
        differs = actual != expected
        mismatches = int(np.count_nonzero(differs))
        report["gemv"]["mismatches"] = mismatches
        if mismatches:
            first = int(np.flatnonzero(differs)[0])
            report["gemv"]["first_mismatch"] = first
            report["gemv"]["max_abs_error"] = float(np.max(np.abs(actual - expected)))
            return _fail(report, "GEMV mismatch: %d of %d rows differ (first row %d: %r != %r)"
                         % (mismatches, rows, first, float(actual[first]), float(expected[first])))
        report["gemv"]["max_abs_error"] = 0.0
        report["result"] = "pass"
        return report
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        return _fail(report, "OpenCL packed-ternary path failed: %s" % exc)
    finally:
        if kernel:
            report["returncodes"]["clReleaseKernel"] = _hex(lib.clReleaseKernel(kernel))
        if program:
            report["returncodes"]["clReleaseProgram"] = _hex(lib.clReleaseProgram(program))
        for index, handle in enumerate(buffers):
            report["returncodes"]["clReleaseMemObject%d" % index] = _hex(
                lib.clReleaseMemObject(handle))
        if queue:
            report["returncodes"]["clReleaseCommandQueue"] = _hex(
                lib.clReleaseCommandQueue(queue))
        if context:
            report["returncodes"]["clReleaseContext"] = _hex(lib.clReleaseContext(context))


def render(report):
    """Human-readable form of the packed ternary GEMV probe."""
    loader = report["loader"]
    shape = report["shape"]
    gemv = report["gemv"]
    lines = ["OpenCL packed-ternary GEMV probe (4 trits/byte, no backend, model or support claim)",
             "  loader: %s%s" % (loader["name"],
                                "" if loader["available"] else " (unavailable)")]
    if report["device"]:
        device = report["device"]
        lines.append("  device: %s; %s CU(s), %s MiB, OpenCL C %s" % (
            device["name"], device["compute_units"], device["global_mem_mib"],
            device["opencl_c_version"]))
    if report["build"]["ok"]:
        lines.append("  build: ok (%d source bytes)" % report["build"]["source_bytes"])
    lines.append("  shape: %d rows x %d cols (%d phases, %s packed bytes, %s scale bytes)" % (
        shape["rows"], shape["cols"], shape["phases"], shape["packed_bytes"],
        shape["scale_bytes"]))
    if gemv["mismatches"] is not None:
        lines.append("  gemv: mismatches: %s; warmup %.3f ms; %d launches %.3f ms; %s GFLOP/s" % (
            gemv["mismatches"], gemv["warmup_ms"], gemv["repeats"], gemv["elapsed_ms"],
            gemv["gflops"]))
    lines.append("  result: %s%s" % (
        report["result"], "" if not report["error"] else " (%s)" % report["error"]))
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
