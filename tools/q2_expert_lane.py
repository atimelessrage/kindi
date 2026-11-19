"""Gate and measure the CPU-compute Q2_0 expert lane against the real Lightning kernels.

    python tools/q2_expert_lane.py --json docs/benchmarks/<run>/results.json

The check is the M2 gate of docs/MOE-TANDEM.md for the Lightning quant format:

* ``lt_mv_q2`` (up) and ``lt_mv_q2_down`` (down, one expert) run on synthetic
  Q2_0 weights of the real geometry (hidden 2688, expert width 1856);
* the same experts run through ``kindi.moe_expert_lane`` from the GPU's own
  ``lt_quant`` output, so only the matvec arithmetic is compared;
* every output element must be bit-identical, or the lane stays disabled.

Then it measures one expert: CPU up + down, the H2D copy of the expert's bytes,
the GPU kernels with device-resident weights, and the small transfers a lane
needs (up result to the card for ``lt_quant``, quantized activation back, output
forward). The copy lane is ``copy + gpu``. This is a microbenchmark, not a model
row: the Lightning pack is absent on this host, which is reported as the blocker.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kindi import moe_expert_lane as lane  # noqa: E402

HIDDEN = 2688                                               # Spec.n_embd of the 30B-A3B
FF = 1856                                                   # Spec.n_ff
EXPERTS = 2


def _random_q2(rng, experts, rows, cols):
    runs, nb = cols // 32, cols // 64
    codes = rng.integers(0, 3, size=(experts, rows, runs * 2), dtype=np.uint32)
    scales = (rng.standard_normal((experts, rows, nb)) * 0.01).astype(np.float16)
    return codes, scales


def _quantized(rng, cols):
    runs = cols // 32
    q = rng.integers(-127, 128, size=(runs, 32), dtype=np.int8)
    xq = np.empty((runs, 8), np.int32)
    for r in range(runs):
        for word in range(8):
            g, k = word // 4, word % 4
            w = 0
            for j in range(4):
                w |= (int(q[r, g * 16 + 4 * j + k]) & 255) << (8 * j)
            xq[r, word] = np.int32(w if w < 2 ** 31 else w - 2 ** 32)
    dx = (rng.standard_normal(runs) * 0.02).astype(np.float32)
    sx = q.astype(np.int32).sum(1).astype(np.int32)
    return xq.reshape(-1), dx, sx


def _ptrs(cp, arrays):
    return cp.asarray(np.array([a.data.ptr for a in arrays], dtype=np.uint64))


def _timed(fn, passes):
    out = []
    for _ in range(passes):
        t0 = time.perf_counter()
        fn()
        out.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(out))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python tools/q2_expert_lane.py")
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--ff", type=int, default=FF)
    ap.add_argument("--experts", type=int, default=EXPERTS)
    ap.add_argument("--passes", type=int, default=7)
    ap.add_argument("--json", default=None, help="write the result object here")
    a = ap.parse_args(argv)
    if not lane.available():
        print("q2 lane: unavailable - numba with its fmaf intrinsic is required")
        return 2
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:                                # pragma: no cover - host without CUDA
        print("q2 lane: unavailable - no CUDA device (%s)" % exc)
        return 2

    rng = np.random.default_rng(19)
    up_codes, up_scales = _random_q2(rng, a.experts, a.ff, a.hidden)
    down_codes, down_scales = _random_q2(rng, a.experts, a.hidden, a.ff)
    xq, dx, sx = _quantized(rng, a.hidden)

    src = open(os.path.join(ROOT, "kindi", "lightning.cu"), encoding="utf-8").read()
    mod = cp.RawModule(code=src, options=("--use_fast_math", "-std=c++14"))
    mv_up, mv_down, quant = mod.get_function("lt_mv_q2"), mod.get_function("lt_mv_q2_down"), mod.get_function("lt_quant")

    up = lane.Q2Matrix(up_codes, up_scales, a.ff, a.hidden, experts=a.experts)
    down = lane.Q2Matrix(down_codes, down_scales, a.hidden, a.ff, experts=a.experts)
    dev_up = [cp.asarray(up_codes[e].reshape(-1)) for e in range(a.experts)]
    dev_up_s = [cp.asarray(up_scales[e].reshape(-1)) for e in range(a.experts)]
    dev_dn = [cp.asarray(down_codes[e].reshape(-1)) for e in range(a.experts)]
    dev_dn_s = [cp.asarray(down_scales[e].reshape(-1)) for e in range(a.experts)]
    ptr_up, ptr_up_s = _ptrs(cp, dev_up), _ptrs(cp, dev_up_s)
    ptr_dn, ptr_dn_s = _ptrs(cp, dev_dn), _ptrs(cp, dev_dn_s)

    dx_dev, sx_dev, xq_dev = cp.asarray(dx), cp.asarray(sx), cp.asarray(xq)
    gpu_up = cp.zeros(a.ff, np.float32)
    mv_up(((a.ff + 7) // 8,), (128,), (ptr_up, ptr_up_s, None, np.int32(a.ff), np.int32(a.hidden),
                                      xq_dev, dx_dev, sx_dev, np.int32(0), gpu_up, np.int32(a.ff),
                                      np.int32(0), np.int32(0)))

    hq = cp.zeros((a.ff // 32, 8), np.int32)
    hd = cp.zeros(a.ff // 32, np.float32)
    hs = cp.zeros(a.ff // 32, np.int32)
    quant((((a.ff // 32) + 7) // 8,), (256,), (gpu_up, np.int32(1), np.int32(a.ff), np.int32(a.ff),
                                              np.int32(1), hq, hd, hs))
    cpu_up = up.matvec(0, xq, dx, sx)
    cpu_down = down.matvec(0, cp.asnumpy(hq).reshape(-1), cp.asnumpy(hd), cp.asnumpy(hs))

    ids = cp.asarray(np.array([0], np.int32))
    gw = cp.asarray(np.array([1.0], np.float32))
    colsh = 2 * a.ff                                        # shared width is twice the expert width
    sh_codes = cp.zeros(a.hidden * (colsh // 16), np.uint32)
    sh_scales = cp.zeros(a.hidden * (colsh // 64), np.float16)
    gq, gd, gs = cp.zeros(colsh // 4, np.int32), cp.zeros(colsh // 32, np.float32), cp.zeros(colsh // 32, np.int32)
    gpu_down = cp.zeros(a.hidden, np.float32)
    mv_down(((a.hidden + 3) // 4,), (128,), (ptr_dn, ptr_dn_s, ids, gw, np.int32(1), np.int32(a.hidden),
                                           np.int32(a.ff), hq, hd, hs, sh_codes, sh_scales, np.int32(colsh),
                                           gq, gd, gs, gpu_down))
    up_gpu, down_gpu = cp.asnumpy(gpu_up), cp.asnumpy(gpu_down)
    parity_up = bool(np.array_equal(up_gpu.view(np.int32), cpu_up.view(np.int32)))
    parity_down = bool(np.array_equal(down_gpu.view(np.int32), cpu_down.view(np.int32)))

    # measurement: CPU expert, H2D copy of its bytes, GPU kernels with device weights
    cpu_up_ms = _timed(lambda: up.matvec(0, xq, dx, sx), a.passes)
    cpu_down_ms = _timed(lambda: down.matvec(0, cp.asnumpy(hq).reshape(-1), cp.asnumpy(hd), cp.asnumpy(hs)), a.passes)
    nbytes = up.bytes // a.experts + down.bytes // a.experts
    host = np.zeros(nbytes, np.uint8)
    devbuf = cp.empty(nbytes, np.uint8)
    copy_ms = _timed(lambda: (cp.cuda.runtime.memcpy(devbuf.data.ptr, host.ctypes.data, nbytes,
                                                cp.cuda.runtime.memcpyHostToDevice),
                              cp.cuda.Stream.null.synchronize()), a.passes)

    def gpu_expert():
        cp.cuda.Stream.null.synchronize()
        mv_up(((a.ff + 7) // 8,), (128,), (ptr_up, ptr_up_s, None, np.int32(a.ff), np.int32(a.hidden),
                                          xq_dev, dx_dev, sx_dev, np.int32(0), gpu_up, np.int32(a.ff),
                                          np.int32(0), np.int32(0)))
        quant((((a.ff // 32) + 7) // 8,), (256,), (gpu_up, np.int32(1), np.int32(a.ff), np.int32(a.ff),
                                                  np.int32(1), hq, hd, hs))
        mv_down(((a.hidden + 3) // 4,), (128,), (ptr_dn, ptr_dn_s, ids, gw, np.int32(1), np.int32(a.hidden),
                                               np.int32(a.ff), hq, hd, hs, sh_codes, sh_scales, np.int32(colsh),
                                               gq, gd, gs, gpu_down))
        cp.cuda.Stream.null.synchronize()
    gpu_ms = _timed(gpu_expert, a.passes)

    def lane_xfer():
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        cp.asarray(cpu_up)
        cp.asnumpy(hq); cp.asnumpy(hd); cp.asnumpy(hs)
        cp.asarray(cpu_down)
        cp.cuda.Stream.null.synchronize()
        return 1e3 * (time.perf_counter() - t0)
    xfer_ms = float(np.median([lane_xfer() for _ in range(a.passes)]))

    result = {
        "measured": time.strftime("%Y-%m-%d"),
        "host": {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                 if isinstance(cp.cuda.runtime.getDeviceProperties(0)["name"], bytes)
                 else str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
                 "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                 "python": platform.python_version(), "cupy": cp.__version__,
                 "numba": __import__("numba").__version__},
        "geometry": {"hidden": a.hidden, "ff": a.ff, "experts": a.experts, "expert_bytes": int(nbytes)},
        "parity": {"up": parity_up, "down": parity_down,
                   "up_differing": int(np.sum(up_gpu != cpu_up)), "down_differing": int(np.sum(down_gpu != cpu_down))},
        "timing_ms": {"cpu_up": round(cpu_up_ms, 3), "cpu_down": round(cpu_down_ms, 3),
                      "cpu_expert": round(cpu_up_ms + cpu_down_ms, 3), "copy": round(copy_ms, 3),
                      "gpu_expert": round(gpu_ms, 3), "copy_lane": round(copy_ms + gpu_ms, 3),
                      "lane_xfer": round(xfer_ms, 3), "lane_total": round(cpu_up_ms + cpu_down_ms + xfer_ms, 3)},
        "rates": {"h2d_gib_s": round(nbytes / (copy_ms * 1e-3) / 2 ** 30, 3)},
        "ratio_cpu_to_copy": round((cpu_up_ms + cpu_down_ms) / (copy_ms + gpu_ms), 2),
        "passes": a.passes,
        "pack": None,
        "blocker": None,
    }
    packs = os.path.join(os.path.expanduser("~"), ".kindi", "packs")
    for name in sorted(os.listdir(packs)) if os.path.isdir(packs) else []:
        p = os.path.join(packs, name, "kindi_pack.json")
        try:
            manifest = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("backend") == "kindi_lightning":
            result["pack"] = p
            break
    if result["pack"] is None:
        result["blocker"] = ("no kindi_lightning pack on this host: the expert-level gate ran on synthetic "
                             "Q2_0 weights, but the capacity-bound model row (verify 4/4 and tok/s) needs the "
                             "Lightning GGUF/pack, which is absent")

    print("parity: up %s (%d differing), down %s (%d differing)"
          % ("IDENTICAL" if parity_up else "DIFFERS", result["parity"]["up_differing"],
             "IDENTICAL" if parity_down else "DIFFERS", result["parity"]["down_differing"]))
    t = result["timing_ms"]
    print("per expert (%d bytes): CPU %.2f ms (up %.2f + down %.2f), copy %.2f ms (%.1f GiB/s), "
          "GPU %.2f ms, copy lane %.2f ms, lane %.2f ms (%.2f ms transfers)"
          % (nbytes, t["cpu_expert"], t["cpu_up"], t["cpu_down"], t["copy"], result["rates"]["h2d_gib_s"],
             t["gpu_expert"], t["copy_lane"], t["lane_total"], t["lane_xfer"]))
    if result["blocker"]:
        print("model row: BLOCKED - " + result["blocker"])
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    return 0 if parity_up and parity_down else 1


if __name__ == "__main__":
    sys.exit(main())
