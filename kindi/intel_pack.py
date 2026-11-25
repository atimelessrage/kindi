# SPDX-License-Identifier: AGPL-3.0-only
"""Run every ternary matmul of a real pack on an Intel GPU through the pack's own bytes.

The OpenCL probes (`kindi intel-smoke`, `intel-cl`, `intel-ternary`) prove the device, a dense
GEMM and the 4-trit/byte layout on synthetic matrices. This probe answers the next question a
backend has to answer first: how fast can the card stream a **converted model's** ternary weights,
at the shapes that model actually has?

It reads `bcodes.bin`, `dpm.bin` and `index.json` exactly as `kindi convert` wrote them - there is no
Intel-specific pack and no conversion step - uploads the two buffers unchanged, and runs the
phase-parallel four-row GEMV at every distinct `(rows, cols)` among the pack's ternary tensors. The
pack's scales are fp16; they are widened to float32 on the host (exact) because OpenCL C 1.2 has no
portable half arithmetic, as in `intel-ternary`.

What it reports is a **projection**: the sum of one pass over every ternary tensor, which is the
memory floor of a decode token (attention, norms, the GDN state update and the output head's
non-ternary work are excluded). It is not a model run, not an engine result and not a support
claim. It also validates one real tensor bit for bit against the host model, so "the bytes are read
correctly" is checked and not assumed.

    python -m kindi intel-pack --pack ~/.kindi/packs/bonsai2.5-27b --device-name Arc --json out.json
"""
import ctypes
import json
import os
import time

import numpy as np

from .intel_cl import (
    CL_CONTEXT_PLATFORM,
    CL_MEM_COPY_HOST_PTR,
    CL_MEM_READ_ONLY,
    CL_MEM_WRITE_ONLY,
    CL_SUCCESS,
    DEFAULT_LOADER,
    _bind,
    _build_log,
    _enumerate,
    _finish,
    _hex,
    _open,
    _read_back,
    _set_arg,
    pick_device,
)
from .intel_ternary import MAX_PARTS, PHASE, _buffer, _launch_1d, _launch_2d, default_parts, partials

PROBE = "intel-pack-gemv"
TERNARY_TYPES = {143, 142, 42}          # PTQ1_0, PQ2_0, Q2_0: every converter output is phase-grouped
MIN_PHASES = 1
MAX_REPEATS = 20
DEFAULT_REPEATS = 5
VALIDATE_MAX_BYTES = 32 << 20           # validate the smallest tensor below this many packed bytes

SOURCE = """
/* Four rows per work-item, at a byte offset into the pack's single bcodes buffer.  The layout is
   [slot][row] inside a tensor, so four consecutive bytes of a slot are four rows of the same four
   columns: one vload4 per slot, and the same per-phase int32 accumulation as the synthetic probe,
   which keeps every partial bit-identical to the scalar kernel. */
#define TRI_DOT(dst, byte, c) do {                                          \\
    int bb = (int)(byte);                                                   \\
    int g1 = (bb * 171) >> 9;                                               \\
    int g2 = (g1 * 171) >> 9;                                               \\
    int g3 = (g2 * 171) >> 9;                                               \\
    (dst) += ((bb - 3 * g1) - 1) * (int)x[(c)]                              \\
           + ((g1 - 3 * g2) - 1) * (int)x[(c) + 1]                          \\
           + ((g2 - 3 * g3) - 1) * (int)x[(c) + 2]                          \\
           + ((g3 - 3 * ((g3 * 171) >> 9)) - 1) * (int)x[(c) + 3];           \\
} while (0)

__kernel void pack_gemv4(__global const uchar *codes,
                         __global const float *scales,
                         __global const float *x,
                         __global float *partial,
                         uint rows, uint cols, uint parts, ulong coff, ulong soff) {
    uint r0 = get_global_id(0) << 2;
    uint part = get_global_id(1);
    uint phases = cols >> 7;
    if (r0 >= rows || part >= parts) {
        return;
    }
    int full = (r0 + 3u) < rows;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint ph = part; ph < phases; ph += parts) {
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        ulong base = coff + ((ulong)(ph << 5) * rows) + r0;
        uint col = ph << 7;
        for (uint s = 0; s < 32; ++s) {
            ulong off = base + (ulong)s * rows;
            uint c = col + (s << 2);
            if (full) {
                uchar4 q = vload4(0, codes + off);
                TRI_DOT(d0, q.s0, c);
                TRI_DOT(d1, q.s1, c);
                TRI_DOT(d2, q.s2, c);
                TRI_DOT(d3, q.s3, c);
            } else {
                TRI_DOT(d0, codes[off], c);
                if (r0 + 1u < rows) { TRI_DOT(d1, codes[off + 1u], c); }
                if (r0 + 2u < rows) { TRI_DOT(d2, codes[off + 2u], c); }
                if (r0 + 3u < rows) { TRI_DOT(d3, codes[off + 3u], c); }
            }
        }
        ulong sb = soff + (ulong)ph * rows + r0;
        t0 += (float)d0 * scales[sb];
        if (r0 + 1u < rows) { t1 += (float)d1 * scales[sb + 1u]; }
        if (r0 + 2u < rows) { t2 += (float)d2 * scales[sb + 2u]; }
        if (r0 + 3u < rows) { t3 += (float)d3 * scales[sb + 3u]; }
    }
    ulong pb = (ulong)part * rows + r0;
    partial[pb] = t0;
    if (r0 + 1u < rows) { partial[pb + 1u] = t1; }
    if (r0 + 2u < rows) { partial[pb + 2u] = t2; }
    if (r0 + 3u < rows) { partial[pb + 3u] = t3; }
}

__kernel void pack_gemv_reduce(__global const float *partial,
                               __global float *y,
                               uint rows, uint parts) {
    uint row = get_global_id(0);
    if (row >= rows) {
        return;
    }
    float total = 0.0f;
    for (uint part = 0; part < parts; ++part) {
        total += partial[(ulong)part * rows + row];
    }
    y[row] = total;
}
"""
KERNEL = "pack_gemv4"
REDUCE = "pack_gemv_reduce"


def _fail(report, message):
    report["error"] = message
    report["result"] = "fail"
    return report


def pack_tensors(pack_dir):
    """(tensors, manifest): every ternary tensor in the pack, in file order, with its byte offsets."""
    with open(os.path.join(pack_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    with open(os.path.join(pack_dir, "kindi_pack.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    tensors = []
    for name, rec in index["tensors"].items():
        if int(rec.get("type", -1)) not in TERNARY_TYPES:
            continue
        rows, cols = int(rec["rows"]), int(rec["ne0"])
        if cols % PHASE or "bc_off" not in rec or "dpm_off" not in rec:
            continue
        tensors.append({
            "name": name, "rows": rows, "cols": cols, "phases": cols // PHASE,
            "bc_off": int(rec["bc_off"]), "dpm_off": int(rec["dpm_off"]),
            "codes_bytes": rows * cols // 4, "scale_bytes": rows * (cols // PHASE) * 2,
        })
    return tensors, manifest, index


def shapes_of(tensors):
    """The distinct (rows, cols) a run has to time, largest first, with how many tensors use each."""
    seen = {}
    for t in tensors:
        key = (t["rows"], t["cols"])
        entry = seen.setdefault(key, {"rows": t["rows"], "cols": t["cols"], "tensors": 0,
                                      "bytes": t["codes_bytes"] + t["scale_bytes"],
                                      "first": t["name"]})
        entry["tensors"] += 1
    return sorted(seen.values(), key=lambda s: s["bytes"], reverse=True)


def project(tensors, seconds_by_shape):
    """The decode floor from one pass over every ternary tensor: bytes, seconds and tokens/s."""
    total_bytes = 0
    total_seconds = 0.0
    for t in tensors:
        total_bytes += t["codes_bytes"] + t["scale_bytes"]
        total_seconds += seconds_by_shape[(t["rows"], t["cols"])]
    return {
        "weight_bytes": int(total_bytes),
        "seconds": round(total_seconds, 6),
        "tokens_per_second": round(1.0 / total_seconds, 3) if total_seconds > 0 else None,
        "effective_gbps": round(total_bytes / total_seconds / 1e9, 3) if total_seconds > 0 else None,
    }


def _base_report(loader, pack_dir):
    return {
        "probe": PROBE,
        "schema": 1,
        "read_only": True,
        "support_claim": False,
        "backend": "opencl pack-reading gemv probe; no sycl, model run, engine or backend",
        "scope": "every ternary matmul of a converted pack at its own shape, from the pack's "
                 "on-disk 4-trit/byte bytes; a projection of the decode memory floor, not a model run",
        "pack": {"dir": os.path.basename(os.path.abspath(pack_dir)), "model": None, "source_file": None,
                 "bcodes_bytes": None, "dpm_bytes": None, "tensors": None, "ternary": None,
                 "shapes": None},
        "loader": {"name": loader, "available": False, "error": None},
        "platforms": [],
        "platform": None,
        "device_count": None,
        "device_index": None,
        "device": None,
        "kernel": {"name": KERNEL, "reduce": REDUCE, "split": "phase ph -> part ph % parts",
                   "exactness": "one real tensor is compared bit for bit at parts=1; the timed runs "
                                "use the default parts"},
        "shapes": [],
        "validation": {"tensor": None, "rows": None, "cols": None, "mismatches": None},
        "projection": {"weight_bytes": None, "seconds": None, "tokens_per_second": None,
                       "effective_gbps": None, "excluded": "attention, norms, GDN state update, "
                       "output-head f32 work, tokenizer and sampling"},
        "repeats": None,
        "warnings": [],
        "returncodes": {},
        "error": None,
        "result": "fail",
    }


def run_pack(pack_dir, repeats=DEFAULT_REPEATS, loader=None, opener=None, device=0,
             device_name=None, parts=None):
    """Time every distinct ternary shape of a converted pack on the Intel GPU; never the CPU."""
    loader = loader or DEFAULT_LOADER
    report = _base_report(loader, pack_dir)
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        return _fail(report, "--repeats must be an integer in [1, %d]" % MAX_REPEATS)
    report["repeats"] = repeats
    try:
        tensors, manifest, index = pack_tensors(pack_dir)
    except (OSError, ValueError, KeyError) as exc:
        return _fail(report, "cannot read the pack: %s" % exc)
    if not tensors:
        return _fail(report, "the pack has no ternary tensors (is it a Kindi pack?)")
    shapes = shapes_of(tensors)
    report["pack"].update({
        "model": manifest.get("model"), "source_file": manifest.get("source_file"),
        "bcodes_bytes": os.path.getsize(os.path.join(pack_dir, "bcodes.bin")),
        "dpm_bytes": os.path.getsize(os.path.join(pack_dir, "dpm.bin")),
        "tensors": len(index["tensors"]), "ternary": len(tensors), "shapes": len(shapes),
    })
    if parts is None:
        parts = default_parts(max(s["cols"] for s in shapes))
    elif not isinstance(parts, int) or not 1 <= parts <= MAX_PARTS:
        return _fail(report, "--parts must be an integer in [1, %d]" % MAX_PARTS)
    if not isinstance(device, int) or device < 0:
        return _fail(report, "--device must be a non-negative integer")

    bcodes = np.fromfile(os.path.join(pack_dir, "bcodes.bin"), dtype=np.uint8)
    dpm16 = np.fromfile(os.path.join(pack_dir, "dpm.bin"), dtype=np.float16)
    report["pack"]["bcodes_bytes"] = int(bcodes.nbytes)
    report["pack"]["dpm_bytes"] = int(dpm16.nbytes * 2)   # widened to float32 for the device

    lib, error = _open(loader, opener)
    if lib is None:
        return _fail(report, error or "OpenCL loader unavailable")
    report["loader"]["available"] = True
    try:
        _bind(lib)
    except (AttributeError, TypeError, ValueError) as exc:
        return _fail(report, "OpenCL loader is missing an API: %s" % exc)

    context = queue = program = kernel = reduce_kernel = None
    buffers = []
    try:
        devices, error = _enumerate(lib, report)
        if devices is None:
            return _fail(report, error or "OpenCL device enumeration failed")
        index_, info, error = pick_device(lib, devices, device, device_name)
        if error or index_ is None or info is None:
            return _fail(report, ("--device-name: " if device_name else "--device: ")
                         + (error or "the selected device could not be read"))
        platform, device_handle, platform_entry = devices[index_]
        report["platform"] = platform_entry["name"]
        report["device"] = info
        if "arc" not in (info.get("name") or "").lower():
            report["warnings"].append(
                "the selected device is %r, not an Arc: pass --device-name Arc for a run whose "
                "numbers may be quoted as Arc" % (info.get("name"),))

        properties = (ctypes.c_void_p * 3)(CL_CONTEXT_PLATFORM, platform, None)
        handles = (ctypes.c_void_p * 1)(device_handle)
        error_code = ctypes.c_int(0)
        context = lib.clCreateContext(properties, 1, handles, None, None, ctypes.byref(error_code))
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
            return _fail(report, "clCreateProgramWithSource returned %s" % _hex(error_code.value))
        devices_array = (ctypes.c_void_p * 1)(device_handle)
        code = int(lib.clBuildProgram(program, 1, devices_array, None, None, None))
        report["returncodes"]["clBuildProgram"] = _hex(code)
        log = _build_log(lib, program, device_handle)
        if code != CL_SUCCESS:
            first = (log or "").strip().splitlines()
            return _fail(report, "clBuildProgram returned %s%s" % (
                _hex(code), "" if not first else ": " + first[0][:200]))
        error_code = ctypes.c_int(0)
        kernel = lib.clCreateKernel(program, KERNEL.encode("ascii"), ctypes.byref(error_code))
        report["returncodes"]["clCreateKernel"] = _hex(error_code.value)
        if not kernel or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateKernel(%s) returned %s" % (KERNEL, _hex(error_code.value)))
        error_code = ctypes.c_int(0)
        reduce_kernel = lib.clCreateKernel(program, REDUCE.encode("ascii"), ctypes.byref(error_code))
        report["returncodes"]["clCreateKernelReduce"] = _hex(error_code.value)
        if not reduce_kernel or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateKernel(%s) returned %s" % (REDUCE, _hex(error_code.value)))

        x_max = max(s["cols"] for s in shapes)
        x = ((np.arange(x_max, dtype=np.int64) * 5 + 3) % 7 - 3).astype(np.float32)
        x_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x,
                        report, "clCreateBufferX")
        buffers.append(x_buf)

        # one small tensor, parts=1: the real bytes against the host model, bit for bit
        small = min(tensors, key=lambda t: t["codes_bytes"])
        validation = report["validation"]
        if small["codes_bytes"] <= VALIDATE_MAX_BYTES:
            rows, cols = small["rows"], small["cols"]
            codes = bcodes[small["bc_off"]:small["bc_off"] + small["codes_bytes"]]
            expected = partials(codes, dpm_view(dpm16, small), x[:cols], rows, cols, 1)[0]
            codes_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, codes,
                                report, "clCreateBufferValidationCodes")
            scales_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 widened(dpm16, small), report, "clCreateBufferValidationScales")
            y_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY, np.zeros(rows, dtype=np.float32),
                            report, "clCreateBufferValidation")
            part_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY, np.zeros(rows, dtype=np.float32),
                               report, "clCreateBufferValidationPartial")
            _set_args(lib, kernel, codes_buf, scales_buf, x_buf, part_buf, rows, cols, 1, 0, 0)
            _set_reduce_args(lib, reduce_kernel, part_buf, y_buf, rows, 1)
            _launch_2d(lib, queue, kernel, (rows + 3) // 4, 1, report, "clEnqueueValidation")
            _launch_1d(lib, queue, reduce_kernel, rows, report, "clEnqueueValidationReduce")
            _finish(lib, queue, report, "clFinishValidation")
            raw = _read_back(lib, queue, y_buf, rows, report, "clReadValidation")
            actual = np.frombuffer(raw, dtype=np.float32)
            differs = actual != expected
            validation.update({"tensor": small["name"], "rows": rows, "cols": cols,
                               "mismatches": int(np.count_nonzero(differs))})
            for handle in (codes_buf, scales_buf, y_buf, part_buf):
                lib.clReleaseMemObject(handle)
            bad = int(validation["mismatches"] or 0)
            if bad:
                return _fail(report, "the pack's bytes do not decode exactly: %d of %d rows of %r "
                                     "differ from the host model" % (bad, rows, small["name"]))
        else:
            report["warnings"].append("the smallest ternary tensor is larger than %d MiB; the "
                                      "device result was not checked against the host"
                                      % (VALIDATE_MAX_BYTES >> 20))

        # A single bcodes buffer is over the driver's per-allocation limit on this card (the 27B's
        # tensor bytes are ~5.8 GB), so each shape uploads its own tensor's slice instead.
        seconds_by_shape = {}
        for shape in shapes:
            rows, cols = shape["rows"], shape["cols"]
            tensor = next(t for t in tensors if t["name"] == shape["first"])
            codes_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bcodes[tensor["bc_off"]:tensor["bc_off"] + tensor["codes_bytes"]],
                                report, "clCreateBufferCodes_%dx%d" % (rows, cols))
            scales_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 widened(dpm16, tensor), report,
                                 "clCreateBufferScales_%dx%d" % (rows, cols))
            y_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY, np.zeros(rows, dtype=np.float32),
                            report, "clCreateBufferY_%dx%d" % (rows, cols))
            part_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY,
                               np.zeros(rows * parts, dtype=np.float32), report,
                               "clCreateBufferPartial_%dx%d" % (rows, cols))
            _set_args(lib, kernel, codes_buf, scales_buf, x_buf, part_buf, rows, cols, parts, 0, 0)
            _set_reduce_args(lib, reduce_kernel, part_buf, y_buf, rows, parts)
            _launch_2d(lib, queue, kernel, (rows + 3) // 4, parts, report, "clEnqueueWarmup")
            _launch_1d(lib, queue, reduce_kernel, rows, report, "clEnqueueWarmupReduce")
            _finish(lib, queue, report, "clFinishWarmup")
            started = time.perf_counter()
            for _ in range(repeats):
                _launch_2d(lib, queue, kernel, (rows + 3) // 4, parts, report, "clEnqueueKernel")
                _launch_1d(lib, queue, reduce_kernel, rows, report, "clEnqueueReduce")
            _finish(lib, queue, report, "clFinish")
            seconds = (time.perf_counter() - started) / repeats
            seconds_by_shape[(rows, cols)] = seconds
            report["shapes"].append({
                "rows": rows, "cols": cols, "tensors": shape["tensors"],
                "median_ms": round(seconds * 1000.0, 4),
                "gbps": round(shape["bytes"] / seconds / 1e9, 3) if seconds > 0 else None,
            })
            for handle in (codes_buf, scales_buf, y_buf, part_buf):
                lib.clReleaseMemObject(handle)

        report["projection"] = project(tensors, seconds_by_shape)
        report["projection"]["excluded"] = ("attention, norms, GDN state update, output-head f32 "
                                            "work, tokenizer and sampling")
        report["result"] = "pass"
        return report
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        return _fail(report, "OpenCL pack path failed: %s" % exc)
    finally:
        if reduce_kernel:
            report["returncodes"]["clReleaseKernelReduce"] = _hex(
                lib.clReleaseKernel(reduce_kernel))
        if kernel:
            report["returncodes"]["clReleaseKernel"] = _hex(lib.clReleaseKernel(kernel))
        if program:
            report["returncodes"]["clReleaseProgram"] = _hex(lib.clReleaseProgram(program))
        for index_, handle in enumerate(buffers):
            report["returncodes"]["clReleaseMemObject%d" % index_] = _hex(
                lib.clReleaseMemObject(handle))
        if queue:
            report["returncodes"]["clReleaseCommandQueue"] = _hex(lib.clReleaseCommandQueue(queue))
        if context:
            report["returncodes"]["clReleaseContext"] = _hex(lib.clReleaseContext(context))


def widened(dpm16, tensor):
    """The tensor's scales as float32 for the device: OpenCL C 1.2 has no portable half arithmetic."""
    return np.ascontiguousarray(dpm_view(dpm16, tensor).astype(np.float32).reshape(-1))


def dpm_view(dpm16, tensor):
    """The tensor's scales as the (phases, rows) fp16 array `partials` expects."""
    count = tensor["rows"] * tensor["phases"]
    begin = tensor["dpm_off"] // 2
    return np.asarray(dpm16[begin:begin + count], dtype=np.float16).reshape(
        tensor["phases"], tensor["rows"])


def _set_reduce_args(lib, kernel, partial, y, rows, parts):
    _set_arg(lib, kernel, 0, ctypes.c_void_p(int(partial)))
    _set_arg(lib, kernel, 1, ctypes.c_void_p(int(y)))
    _set_arg(lib, kernel, 2, ctypes.c_uint32(rows))
    _set_arg(lib, kernel, 3, ctypes.c_uint32(parts))


def _set_args(lib, kernel, codes, scales, x, partial, rows, cols, parts, coff, soff):
    _set_arg(lib, kernel, 0, ctypes.c_void_p(int(codes)))
    _set_arg(lib, kernel, 1, ctypes.c_void_p(int(scales)))
    _set_arg(lib, kernel, 2, ctypes.c_void_p(int(x)))
    _set_arg(lib, kernel, 3, ctypes.c_void_p(int(partial)))
    _set_arg(lib, kernel, 4, ctypes.c_uint32(rows))
    _set_arg(lib, kernel, 5, ctypes.c_uint32(cols))
    _set_arg(lib, kernel, 6, ctypes.c_uint32(parts))
    _set_arg(lib, kernel, 7, ctypes.c_uint64(coff))
    _set_arg(lib, kernel, 8, ctypes.c_uint64(soff))


def render(report):
    lines = ["OpenCL pack-reading GEMV probe (a converted pack's ternary shapes; no model run, "
             "engine or support claim)"]
    pack = report["pack"]
    lines.append("  pack: %s (%s); %s ternary of %s tensors, %s shapes, %s bcodes bytes" % (
        pack["model"], pack["source_file"], pack["ternary"], pack["tensors"], pack["shapes"],
        pack["bcodes_bytes"] or 0))
    if report["device"]:
        device = report["device"]
        lines.append("  device: %s; %s CU(s), %s MiB" % (
            device["name"], device["compute_units"], device["global_mem_mib"]))
    for shape in report["shapes"]:
        lines.append("  %5d x %5d x %2d tensor(s): %.4f ms each, %s GB/s" % (
            shape["rows"], shape["cols"], shape["tensors"], shape["median_ms"], shape["gbps"]))
    projection = report["projection"]
    if projection["tokens_per_second"] is not None:
        lines.append("  projection: %s GB moved -> %.3f ms per pass, %s GB/s effective, "
                     "%s tokens/s ceiling" % (
                         projection["weight_bytes"] / 1e9, projection["seconds"] * 1000.0,
                         projection["effective_gbps"], projection["tokens_per_second"]))
    validation = report["validation"]
    if validation["mismatches"] is not None:
        lines.append("  validation: %s (%d x %d), mismatches %s" % (
            validation["tensor"], validation["rows"], validation["cols"],
            validation["mismatches"]))
    for warning in report["warnings"]:
        lines.append("  warning: %s" % warning)
    lines.append("  result: %s%s" % (
        report["result"], "" if not report["error"] else " (%s)" % report["error"]))
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
