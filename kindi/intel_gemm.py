# SPDX-License-Identifier: AGPL-3.0-only
"""Ternary GEMM probe in the engine's Q8 shape, on an Intel GPU through OpenCL (REQ-0020 M2).

This is the follow-up to ``kindi intel-ternary``. That probe proved the packed layout decodes
exactly with an unquantized activation and one work-item per row. This one runs the arithmetic the
engine's prompt path actually does, at the tile the CUDA kernels use, and reports it as an
observation with the same bit-exact discipline:

* ``q8x`` ports ``k_q8x_T2`` (``kindi/chunk.cu``: one 256-thread group per 128-column group and
  64-token band; one threshold pair per token, ``rint`` with round-half-even, four int8 per
  word). Its output feeds the GEMM directly, so a mismatch anywhere in the chain fails the run.
* ``gemm_tile`` mirrors ``k_mm_g5q``'s arithmetic: the block covers 128 tokens x 128 rows with 256
  work-items (8 x 8 outputs each); per 128-column group the dot products accumulate in int32 and
  are scaled once (``f * (xs * dj)``) before an fp32 add, in group order, with the group's scale
  read per row. Two deliberate differences, both load-time choices rather than arithmetic: the
  weight bytes are the pack's on-disk 4-trit layout ``intel-ternary`` uploads (the engine
  repacks to 5-trit lanes at load; the decode here is the 4-trit one), and there is no local-memory
  staging - every load comes from global memory, so this is the un-tiled shape to measure before
  tiled variants, not the fast one.

The inputs are constructed so every float operation is exact (activations are small integers with a
per-group maximum of exactly 127, so the quantizer is a pass-through at xs = 1.0; scales are
fp16-exact), and the host reference steps through the same fp32 operations in the same order; the
device result must match bit for bit. A mismatch is a failure, never a tolerance. OpenCL C 1.2 has
no dot-product instruction, so the int32 dot is computed byte by byte; the CUDA ``dp4a`` is an
instruction-level shortcut over exactly these integers, not a different arithmetic.

This is a probe, not a backend: no model, no engine path, no SYCL, no support claim.
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
    _open,
    _read_back,
    _set_arg,
)
from .intel_ternary import _buffer, _launch_2d

PROBE = "intel-ternary-gemm"
DEFAULT_ROWS = 128
DEFAULT_TOKENS = 128
DEFAULT_COLS = 512
MIN_TOKENS = 128
MAX_TOKENS = 1024
MIN_ROWS = 128
MAX_ROWS = 1024
MIN_COLS = 128
MAX_COLS = 8192
DEFAULT_REPEATS = 5
MAX_REPEATS = 50
GROUP = 128
SCALES = (1.0, 0.5, 2.0, 1.5)          # fp16-exact: every scaled product below stays exact in fp32
SEED = 7
SOURCE = """
/* The CUDA kernels stage tiles in shared memory and use dp4a; OpenCL C 1.2 has neither a shuffle
   nor a dot-product instruction, so the dot below is byte by byte (exactly dp4a's integers) and
   every load is a global load. tq(d) maps a digit 0/1/2 to the int8 trit -1/0/1 as a byte. */
static inline uint tq(uint d) { return (d + 255u) & 255u; }

/* Decodes one packed byte to four int8 trits in a word, in the engine's digit order
   (t0 + 3 t1 + 9 t2 + 27 t3). */
static inline uint decode_byte(uint c) {
    uint c3 = (c * 171u) >> 9;
    uint lo = tq(c - 3u * c3);
    c = c3;
    c3 = (c * 171u) >> 9;
    uint mid = tq(c - 3u * c3);
    c = c3;
    c3 = (c * 171u) >> 9;
    return lo | (mid << 8) | (tq(c - 3u * c3) << 16) | (tq(c3) << 24);
}

static inline int sgn(uint byte) { int v = (int)(byte & 255u); return v > 127 ? v - 256 : v; }

static inline int dp4a_manual(uint a, uint b, int acc) {
    return acc + sgn(a) * sgn(b) + sgn(a >> 8) * sgn(b >> 8)
               + sgn(a >> 16) * sgn(b >> 16) + sgn(a >> 24) * sgn(b >> 24);
}

/* k_q8x_T2 without shuffles: one 256-wide group per (128-column group, 64-token band). */
__kernel void q8x(__global const float *x, int ldx,
                  __global uint *xq, __global float *xs, int ldt, int t_max) {
    __local uint q[32][65];
    __local float red[8][32];
    __local float amaxv[64];
    const int g = get_global_id(1) % (ldx >> 7);
    const int t0 = (get_global_id(1) / (ldx >> 7)) * 64;
    const int lane = get_global_id(0) & 31;
    const int w = get_global_id(0) >> 5;
    for (int i = 0; i < 8; ++i) {
        int tt = w * 8 + i;
        int t = t0 + tt;
        float a = 0.0f;
        if (t < t_max) {
            float4 v = ((__global float4 *)(x + (long long)t * ldx + g * 128 + lane * 4))[0];
            a = fmax(fabs(v.x), fmax(fabs(v.y), fmax(fabs(v.z), fabs(v.w))));
        }
        red[w][lane] = a;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0) {
            float m = 0.0f;
            for (int l = 0; l < 32; ++l) m = fmax(m, red[w][l]);
            amaxv[tt] = m;
            if (t < t_max) xs[g * ldt + t] = m / 127.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (t < t_max) {
            float m = amaxv[tt];
            float id = m != 0.0f ? 127.0f / m : 0.0f;
            float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
            if (t < t_max) {
                float4 v = ((__global float4 *)(x + (long long)t * ldx + g * 128 + lane * 4))[0];
                q0 = v.x; q1 = v.y; q2 = v.z; q3 = v.w;
            }
            q[lane][tt] = ((uint)((int)rint(q0 * id) & 255))
                        | ((uint)((int)rint(q1 * id) & 255) << 8)
                        | ((uint)((int)rint(q2 * id) & 255) << 16)
                        | ((uint)((int)rint(q3 * id) & 255) << 24);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = get_global_id(0); i < 32 * 64; i += 256) {
        int k4 = i >> 6, tt = i & 63, t = t0 + tt;
        if (t < t_max) xq[((long long)g * 32 + k4) * ldt + t] = q[k4][tt];
    }
}

/* One 256-wide group computes one 128-token x 128-row tile, mirroring mm_core's 8 x 8 register
   tile; the encoding of a row's byte is the pack's 4 trits per byte. */
__kernel void gemm_tile(__global const uchar *codes, __global const float *scales,
                        __global const uint *xq, __global const float *xs,
                        __global float *y, int rows, int tokens, int ng) {
    const int gid = get_global_id(0);
    const int tile = get_global_id(1);
    const int nrt = rows >> 7;
    const int r0 = (tile % nrt) * 128;
    const int t0 = (tile / nrt) * 128;
    const int tx = gid & 15, ty = gid >> 4;
    float acc[8][8];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    for (int g = 0; g < ng; ++g) {
        int is[8][8];
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j) is[i][j] = 0;
        for (int k4 = 0; k4 < 32; ++k4) {
            uint a[8], b[8];
            for (int i = 0; i < 4; ++i) {
                a[i] = xq[((long long)g * 32 + k4) * tokens + t0 + tx * 4 + i];
                a[4 + i] = xq[((long long)g * 32 + k4) * tokens + t0 + 64 + tx * 4 + i];
            }
            for (int j = 0; j < 4; ++j)
                b[j] = decode_byte((uint)codes[((long long)g * 32 + k4) * rows + r0 + ty * 4 + j]);
            for (int j = 0; j < 4; ++j)
                b[4 + j] = decode_byte((uint)codes[((long long)g * 32 + k4) * rows + r0 + 64 + ty * 4 + j]);
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j) is[i][j] = dp4a_manual(a[i], b[j], is[i][j]);
        }
        for (int i = 0; i < 8; ++i) {
            float xx = xs[g * tokens + t0 + (i < 4 ? tx * 4 + i : 64 + tx * 4 + i - 4)];
            for (int j = 0; j < 8; ++j) {
                float dj = scales[g * rows + r0 + (j < 4 ? ty * 4 + j : 64 + ty * 4 + j - 4)];
                float f = (float)is[i][j];
                acc[i][j] += f * (xx * dj);
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        int tok = t0 + (i < 4 ? tx * 4 + i : 64 + tx * 4 + i - 4);
        for (int j = 0; j < 8; ++j) {
            int row = r0 + (j < 4 ? ty * 4 + j : 64 + ty * 4 + j - 4);
            y[(long long)tok * rows + row] = acc[i][j];
        }
    }
}
"""


def _fail(report, message):
    report["error"] = message
    report["result"] = "fail"
    return report


def _base_report(loader, rows, tokens, cols, repeats):
    return {
        "probe": PROBE,
        "schema": 1,
        "read_only": False,
        "support_claim": False,
        "backend": "opencl ternary gemm in the q8 shape; no sycl, model, engine or backend",
        "scope": "k_q8x_T2-style quantization feeding an int32-dot ternary gemm over "
                 "the pack's 4-trit layout, checked against an exact host reference",
        "loader": {"name": loader, "available": False, "error": None},
        "platforms": [],
        "platform": None,
        "device_count": None,
        "device_index": None,
        "device": None,
        "layout": {
            "quantizer": "k_q8x_T2: one 256-wide group per 128-column group and 64-token band; "
                         "one threshold pair per token; rint, round half even; four int8 per word",
            "weights": "4 trits per byte, byte = t0 + 3*t1 + 9*t2 + 27*t3; [column/4][row]; "
                       "the pack's on-disk layout, not the engine's load-time 5-trit lane repack",
            "arithmetic": "int32 dot per 128-column group, one float (xs * scale) per row, "
                          "fp32 accumulate in group order; no local-memory staging",
        },
        "shape": {"rows": rows, "tokens": tokens, "cols": cols,
                  "groups": cols // GROUP if isinstance(cols, int) else None,
                  "tile": {"rows": 128, "tokens": 128, "work_items": 256},
                  "x_bytes": None, "codes_bytes": None, "scale_bytes": None},
        "pack": {"seed": SEED, "bcodes_sha256": None, "dpm_sha256": None, "x_sha256": None},
        "build": {"ok": False, "source_bytes": len(SOURCE.encode("utf-8")), "log": None},
        "quantizer": {"xq_mismatches": None, "xs_mismatches": None},
        "gemm": {"rows": rows, "tokens": tokens, "cols": cols, "repeats": repeats,
                 "mismatches": None, "max_abs_error": None, "first_mismatch": None,
                 "elapsed_ms": None, "gflops": None},
        "returncodes": {},
        "error": None,
        "result": "fail",
    }


def matrix(rows, cols, seed=SEED):
    """Build (ternary digits, packed codes, fp16 scales) in the pack layout; codes are uploaded as-is."""
    row_ids = np.arange(rows, dtype=np.int64)[:, None]
    col_ids = np.arange(cols, dtype=np.int64)[None, :]
    digits = ((row_ids * 131 + col_ids * 17 + seed * 29) % 3).astype(np.uint8)
    codes = np.ascontiguousarray(phase_bytes(digits).T)                 # (cols/4, rows)
    phases = cols // GROUP
    pattern = np.asarray(SCALES, dtype=np.float16)
    scales = np.empty((phases, rows), dtype=np.float16)
    scales[:] = pattern[np.arange(phases) % pattern.size][:, None]
    return digits, codes, np.ascontiguousarray(scales)


def activations(tokens, cols, seed=SEED):
    """Exact-scaled activations: integers bounded by 127, with the group maximum forced to 127.

    Every group of every token contains a +-127 at its first column, so the quantizer computes
    amax = 127, xs = 1.0, id = 1.0 and passes the values through unchanged; the device quantizer
    is still executed and its XQ/XS output is compared bit for bit.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(-127, 128, size=(tokens, cols)).astype(np.float32)
    for group in range(cols // GROUP):
        x[:, group * GROUP] = 127.0
        x[:, group * GROUP + 1] = -127.0
    return np.ascontiguousarray(x)


def quantize(x):
    """Host model of q8x: (xq words [ng*32][tokens] uint32, xs [ng][tokens] f32, xq signed [t][k]).

    Every stage mirrors the kernel in fp32: the maximum is exact, the threshold and the products
    round once, and rint rounds half to even. On the exact matrix (amax = 127 everywhere) this is
    the identity; the run refuses to launch if the host model of the split differs from an
    independent float64 dot, so a model bug becomes a launch refusal, not a device mismatch.
    """
    tokens, cols = x.shape
    ng = cols // GROUP
    xq_signed = np.zeros((tokens, cols), dtype=np.int32)
    xq_words = np.zeros((ng * 32, tokens), dtype=np.uint32)
    xs = np.zeros((ng, tokens), dtype=np.float32)
    xf = np.asarray(x, dtype=np.float32)
    for group in range(ng):
        block = xf[:, group * GROUP:(group + 1) * GROUP]                 # (tokens, 128)
        amax = np.max(np.abs(block), axis=1)                              # exact
        nonzero = amax != 0.0
        scale = np.zeros_like(amax)
        np.divide(np.float32(127.0), amax, out=scale, where=nonzero)
        np.divide(amax, np.float32(127.0), out=xs[group], where=nonzero)
        packed = np.rint(block * scale[:, None]).astype(np.int32) & 255
        xq_signed[:, group * GROUP:(group + 1) * GROUP] = np.where(packed > 127, packed - 256, packed)
        words = (packed[:, 0::4] | (packed[:, 1::4] << 8)
                 | (packed[:, 2::4] << 16) | (packed[:, 3::4] << 24)).astype(np.uint32)
        xq_words[group * 32:(group + 1) * 32, :] = words.T
    return xq_words, np.ascontiguousarray(xs), xq_signed


def reference(trits_signed, xq_signed, xs, scales_f32):
    """Host GEMV: exact int64 dots per group, one fp32 multiply chain per group, in group order.

    The reference is computed with the same fp32 rounding steps the kernel performs (a float32
    accumulator, ``f * (xs * dj)`` per group), so an exact-input run compares bit for bit. The
    run checks this against a float64 independent dot before launching.
    """
    tokens, _cols = xq_signed.shape
    rows, ng = scales_f32.shape
    acc = np.zeros((tokens, rows), dtype=np.float32)
    for group in range(ng):
        block_x = np.asarray(xq_signed[:, group * GROUP:(group + 1) * GROUP], dtype=np.int64)
        block_w = np.asarray(trits_signed[:, group * GROUP:(group + 1) * GROUP], dtype=np.int64)
        dot = block_x @ block_w.T                                       # exact in int64
        s = (xs[group][:, None] * scales_f32[:, group][None, :]).astype(np.float32)
        acc = (acc + (dot.astype(np.float32) * s).astype(np.float32)).astype(np.float32)
    return acc


def reference_f64(trits_signed, xq_signed, xs, scales_f32):
    """Independent float64 form, used only to check the fp32 stepping above before launching."""
    xs_f64 = np.asarray(xs, dtype=np.float64)
    acc = np.zeros((xq_signed.shape[0], trits_signed.shape[0]), dtype=np.float64)
    ng = scales_f32.shape[1]
    for group in range(ng):
        block_x = np.asarray(xq_signed[:, group * GROUP:(group + 1) * GROUP], dtype=np.float64)
        block_w = np.asarray(trits_signed[:, group * GROUP:(group + 1) * GROUP], dtype=np.float64)
        acc += (block_x @ block_w.T) * (xs_f64[group][:, None] * scales_f32[:, group][None, :])
    return acc


def _launch_q8x(lib, queue, kernel, ng, bands, report, key):
    global_size = (ctypes.c_size_t * 2)(256, ng * bands)
    local_size = (ctypes.c_size_t * 2)(256, 1)
    code = int(lib.clEnqueueNDRangeKernel(queue, kernel, 2, None, global_size, local_size,
                                          0, None, None))
    report["returncodes"][key] = _hex(code)
    if code != CL_SUCCESS:
        raise ValueError("%s returned %s" % (key, _hex(code)))


def run_probe(rows=DEFAULT_ROWS, tokens=DEFAULT_TOKENS, cols=DEFAULT_COLS,
              repeats=DEFAULT_REPEATS, loader=None, opener=None, device=0, seed=SEED):
    """Launch the GEMM probe shapes on the selected Intel GPU; it never runs on the CPU."""
    loader = loader or DEFAULT_LOADER
    report = _base_report(loader, rows, tokens, cols, repeats)
    if not isinstance(rows, int) or not MIN_ROWS <= rows <= MAX_ROWS or rows % 128:
        return _fail(report, "--rows must be a multiple of 128 in [%d, %d]" % (MIN_ROWS, MAX_ROWS))
    if not isinstance(tokens, int) or not MIN_TOKENS <= tokens <= MAX_TOKENS or tokens % 128:
        return _fail(report, "--tokens must be a multiple of 128 in [%d, %d]"
                     % (MIN_TOKENS, MAX_TOKENS))
    if not isinstance(cols, int) or not MIN_COLS <= cols <= MAX_COLS or cols % GROUP:
        return _fail(report, "--cols must be a multiple of %d in [%d, %d]"
                     % (GROUP, MIN_COLS, MAX_COLS))
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        return _fail(report, "--repeats must be an integer in [1, %d]" % MAX_REPEATS)
    if not isinstance(device, int) or device < 0:
        return _fail(report, "--device must be a non-negative integer")

    digits, codes, scales_f16 = matrix(rows, cols, seed)
    x = activations(tokens, cols, seed)
    xq_words, xs, xq_signed = quantize(x)
    trits_signed = digits.astype(np.int32) - 1                            # (rows, cols)
    scales_f32 = np.ascontiguousarray(scales_f16.astype(np.float32).T)   # (rows, ng), host widened
    expected = reference(trits_signed, xq_signed, xs, scales_f32)
    wide = reference_f64(trits_signed, xq_signed, xs, scales_f32)
    if not np.array_equal(expected.astype(np.float64), wide):
        return _fail(report, "the fp32 stepping of the host reference disagrees with the "
                             "float64 dot on this shape; refusing to launch")
    scales_flat = np.ascontiguousarray(scales_f16.astype(np.float32))     # [phase][row] for upload
    report["shape"]["x_bytes"] = int(x.nbytes)
    report["shape"]["codes_bytes"] = int(codes.nbytes)
    report["shape"]["scale_bytes"] = int(scales_flat.nbytes)
    report["pack"]["bcodes_sha256"] = hashlib.sha256(codes.tobytes()).hexdigest()
    report["pack"]["dpm_sha256"] = hashlib.sha256(scales_f16.tobytes()).hexdigest()
    report["pack"]["x_sha256"] = hashlib.sha256(x.tobytes()).hexdigest()

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
    kernels = []
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

        for name in (b"q8x", b"gemm_tile"):
            error_code = ctypes.c_int(0)
            kernel = lib.clCreateKernel(program, name, ctypes.byref(error_code))
            report["returncodes"]["clCreateKernel%s" % name.decode().title()] = _hex(
                error_code.value)
            if not kernel or error_code.value != CL_SUCCESS:
                return _fail(report, "clCreateKernel(%s) returned %s"
                             % (name.decode(), _hex(error_code.value)))
            kernels.append(kernel)

        ng = cols // GROUP
        bands = tokens // 64
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x,
                               report, "clCreateBufferX"))
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, codes,
                               report, "clCreateBufferCodes"))
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales_flat,
                               report, "clCreateBufferScales"))
        buffers.append(_buffer(lib, context, CL_MEM_WRITE_ONLY,
                               np.zeros_like(xq_words), report, "clCreateBufferXq"))
        buffers.append(_buffer(lib, context, CL_MEM_WRITE_ONLY,
                               np.zeros_like(xs), report, "clCreateBufferXs"))
        buffers.append(_buffer(lib, context, CL_MEM_WRITE_ONLY,
                               np.zeros((tokens, rows), dtype=np.float32), report,
                               "clCreateBufferY"))
        _set_arg(lib, kernels[0], 0, ctypes.c_void_p(int(buffers[0])))
        _set_arg(lib, kernels[0], 1, ctypes.c_int32(cols))
        _set_arg(lib, kernels[0], 2, ctypes.c_void_p(int(buffers[3])))
        _set_arg(lib, kernels[0], 3, ctypes.c_void_p(int(buffers[4])))
        _set_arg(lib, kernels[0], 4, ctypes.c_int32(tokens))
        _set_arg(lib, kernels[0], 5, ctypes.c_int32(tokens))
        _set_arg(lib, kernels[1], 0, ctypes.c_void_p(int(buffers[1])))
        _set_arg(lib, kernels[1], 1, ctypes.c_void_p(int(buffers[2])))
        _set_arg(lib, kernels[1], 2, ctypes.c_void_p(int(buffers[3])))
        _set_arg(lib, kernels[1], 3, ctypes.c_void_p(int(buffers[4])))
        _set_arg(lib, kernels[1], 4, ctypes.c_void_p(int(buffers[5])))
        _set_arg(lib, kernels[1], 5, ctypes.c_int32(rows))
        _set_arg(lib, kernels[1], 6, ctypes.c_int32(tokens))
        _set_arg(lib, kernels[1], 7, ctypes.c_int32(ng))

        started = time.perf_counter()
        _launch_q8x(lib, queue, kernels[0], ng, bands, report, "clEnqueueNDRangeKernelQ8x")
        _finish(lib, queue, report, "clFinishQ8x")
        q8x_seconds = time.perf_counter() - started
        raw = _read_back(lib, queue, buffers[3], int(xq_words.size), report,
                         "clEnqueueReadBufferXq")
        got_words = np.frombuffer(raw, dtype=np.uint32).reshape(xq_words.shape)
        if got_words.size != xq_words.size or not np.array_equal(got_words, xq_words):
            report["quantizer"]["xq_mismatches"] = int(np.count_nonzero(
                got_words.size == xq_words.size and got_words != xq_words))
            return _fail(report, "the device quantizer's XQ differs from the host model; "
                                 "the GEMM input is wrong and the run is a failure")
        raw = _read_back(lib, queue, buffers[4], int(xs.size), report, "clEnqueueReadBufferXs")
        got_xs = np.frombuffer(raw, dtype=np.float32).reshape(xs.shape)
        if got_xs.size != xs.size or not np.array_equal(got_xs, xs):
            report["quantizer"]["xs_mismatches"] = int(np.count_nonzero(
                got_xs.size == xs.size and got_xs != xs))
            return _fail(report, "the device quantizer's XS differs from the host model; "
                                 "the run is a failure")
        report["quantizer"]["xq_mismatches"] = 0
        report["quantizer"]["xs_mismatches"] = 0

        started = time.perf_counter()
        for _ in range(repeats):
            _launch_2d(lib, queue, kernels[1], 256, (rows // 128) * (tokens // 128), report,
                       "clEnqueueNDRangeKernelGemm")
        _finish(lib, queue, report, "clFinishGemm")
        seconds = time.perf_counter() - started
        report["gemm"]["elapsed_ms"] = round(seconds * 1000.0, 3)
        if seconds > 0:
            report["gemm"]["gflops"] = round(2.0 * rows * tokens * cols * repeats
                                             / (seconds * 1e9), 3)
        raw = _read_back(lib, queue, buffers[5], rows * tokens, report, "clEnqueueReadBufferY")
        actual = np.frombuffer(raw, dtype=np.float32)
        if actual.size != expected.size:
            return _fail(report, "read back %d of %d outputs" % (actual.size, expected.size))
        actual = actual.reshape(expected.shape)
        differs = actual != expected
        mismatches = int(np.count_nonzero(differs))
        report["gemm"]["mismatches"] = mismatches
        report["gemm"]["q8x_elapsed_ms"] = round(q8x_seconds * 1000.0, 3)
        if mismatches:
            first = int(np.flatnonzero(differs)[0])
            report["gemm"]["first_mismatch"] = first
            report["gemm"]["max_abs_error"] = float(np.max(np.abs(actual - expected)))
            return _fail(report, "GEMM mismatch: %d of %d outputs differ (first %d: %r != %r)"
                         % (mismatches, expected.size, first, float(actual.flat[first]),
                            float(expected.flat[first])))
        report["gemm"]["max_abs_error"] = 0.0
        report["result"] = "pass"
        return report
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        return _fail(report, "OpenCL ternary-GEMM path failed: %s" % exc)
    finally:
        for index, kernel in enumerate(kernels):
            report["returncodes"]["clReleaseKernel%d" % index] = _hex(
                lib.clReleaseKernel(kernel))
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
    """Human-readable form of the ternary-GEMM probe."""
    loader = report["loader"]
    lines = ["OpenCL ternary GEMM probe (q8 shape; no backend, model or support claim)",
             "  loader: %s%s" % (loader["name"], "" if loader["available"] else " (unavailable)")]
    if report["device"]:
        device = report["device"]
        lines.append("  device: %s; %s CU(s), %s MiB, OpenCL C %s" % (
            device["name"], device["compute_units"], device["global_mem_mib"],
            device["opencl_c_version"]))
    if report["build"]["ok"]:
        lines.append("  build: ok (%d source bytes)" % report["build"]["source_bytes"])
    shape = report["shape"]
    lines.append("  shape: %d rows x %d tokens x %d cols (%d groups)" % (
        shape["rows"], shape["tokens"], shape["cols"], shape["groups"]))
    if report["quantizer"]["xq_mismatches"] is not None:
        lines.append("  q8x: xq mismatches: %s; xs mismatches: %s; %.3f ms" % (
            report["quantizer"]["xq_mismatches"], report["quantizer"]["xs_mismatches"],
            report["gemm"].get("q8x_elapsed_ms") or 0.0))
    gemm = report["gemm"]
    if gemm["mismatches"] is not None:
        lines.append("  gemm: mismatches: %s; %d launches %.3f ms; %s GFLOP/s" % (
            gemm["mismatches"], gemm["repeats"], gemm["elapsed_ms"], gemm["gflops"]))
    lines.append("  result: %s%s" % (
        report["result"], "" if not report["error"] else " (%s)" % report["error"]))
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
