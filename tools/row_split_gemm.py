#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Probe one CPU/GPU row split of a ternary FFN-down GEMV.

The GPU owns the upper rows of one ``ffn_down.weight`` matrix and the CPU owns the lower
rows.  The two products start from the same folded FFN input, run independently, and are
merged in the GPU residual buffer before the layer's residual add.  This is the smallest
honest experiment for the row-split proposal: it measures a real layer, including the
attention/GDN block and the FFN gate/up/fold, and reports the full-GPU layer against the
same layer with one down matrix split.

It is not a full decode benchmark.  The split layer must be loaded by both engines; the
CPU engine loads one complete layer, and the GPU engine loads the same layer before its
down rows are replaced by the upper slice.  The tool therefore measures compute and
transfer, not the VRAM saved by not loading the full matrix.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _cuda(cp):
    """CuPy's CUDA submodule; the module dict keeps this opaque to pylint."""
    return cp.__dict__["cuda"]


def _rng(seed):
    """A seeded NumPy generator; the module dict avoids pylint's incomplete stubs."""
    return np.random.__dict__["RandomState"](seed)


def _run_matrix(args, cpu, gpu, cp, pack, name, total, rows):
    """CPU-owned matrix: full CPU GEMV against CPU lower rows + GPU upper rows."""
    C = int(gpu._all_bi[name]["ne0"])
    rng = _rng(args.seed)
    x = rng.normal(0.0, 0.5, size=C).astype(np.float32)
    gpu.xp[:C].set(x)

    def cpu_full():
        return cpu.gemv_rows(name, (0, total), x)

    for _ in range(args.warmup):
        cpu_full()
    gpu.load_partial_tensor(pack, name, (rows, total))

    def split():
        gpu.gemv([name], gpu.xp, [gpu.y[rows:total]])
        cpu_part = cpu.gemv_rows(name, (0, rows), x)
        _cuda(cp).Stream.null.synchronize()
        out = np.empty(total, dtype=np.float32)
        out[:rows] = cpu_part
        out[rows:] = cp.asnumpy(gpu.y[rows:total])
        return out

    for _ in range(args.warmup):
        split()
    full_times, split_times, full_out, split_out = [], [], None, None
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        full_out = cpu_full()
        full_times.append(1000.0 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        split_out = split()
        split_times.append(1000.0 * (time.perf_counter() - t0))

    assert full_out is not None and split_out is not None
    diff = np.abs(full_out - split_out)
    return {
        "mode": "cpu-matrix",
        "pack": pack,
        "layer": args.layer,
        "tensor": name,
        "rows_total": total,
        "rows_cpu": rows,
        "rows_gpu": total - rows,
        "iterations": args.iterations,
        "full_ms": {"p50": statistics.median(full_times), "p10": _pct(full_times, 10),
                     "p90": _pct(full_times, 90)},
        "split_ms": {"p50": statistics.median(split_times), "p10": _pct(split_times, 10),
                      "p90": _pct(split_times, 90)},
        "ratio_full_over_split": statistics.median(full_times) / statistics.median(split_times),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "split_forward": bool(np.all(np.isfinite(split_out))),
        "gpu": str(_cuda(cp).runtime.getDeviceProperties(_cuda(cp).Device().id)["name"]),
    }


def _run_cpu_layer(args, cpu, gpu, cp, pack, name, total, rows):
    """A CPU-owned layer: full CPU layer against CPU lower rows + GPU upper rows."""
    rng = _rng(args.seed)
    x = rng.normal(0.0, 0.5, size=int(cpu.D)).astype(np.float32)

    def full():
        cpu.reset()
        cpu.set_hidden(x)
        cpu.layer(args.layer, 0)
        return cpu.get_hidden()

    for _ in range(args.warmup):
        full()
    gpu.load_partial_tensor(pack, name, (rows, total))

    def split():
        cpu.reset()
        cpu.set_hidden(x)
        cpu.layer_pre_ffn(args.layer, 0)
        hin = cpu.ffn_input_host()
        gpu.xp[:len(hin)].set(hin)
        gpu.gemv([name], gpu.xp, [gpu.y[rows:total]])
        cpu_part = cpu.gemv_rows(name, (0, rows), hin)
        _cuda(cp).Stream.null.synchronize()
        y = cpu.output_y()
        y[:rows] = cpu_part
        y[rows:total] = cp.asnumpy(gpu.y[rows:total])
        cpu.layer_finish()
        return cpu.get_hidden()

    for _ in range(args.warmup):
        split()
    full_times, split_times, full_out, split_out = [], [], None, None
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        full_out = full()
        full_times.append(1000.0 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        split_out = split()
        split_times.append(1000.0 * (time.perf_counter() - t0))

    assert full_out is not None and split_out is not None
    diff = np.abs(full_out - split_out)
    return {
        "mode": "cpu-layer",
        "pack": pack,
        "layer": args.layer,
        "tensor": name,
        "rows_total": total,
        "rows_cpu": rows,
        "rows_gpu": total - rows,
        "iterations": args.iterations,
        "full_ms": {"p50": statistics.median(full_times), "p10": _pct(full_times, 10),
                     "p90": _pct(full_times, 90)},
        "split_ms": {"p50": statistics.median(split_times), "p10": _pct(split_times, 10),
                      "p90": _pct(split_times, 90)},
        "ratio_full_over_split": statistics.median(full_times) / statistics.median(split_times),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "split_forward": bool(np.all(np.isfinite(split_out))),
        "gpu": str(_cuda(cp).runtime.getDeviceProperties(_cuda(cp).Device().id)["name"]),
    }


def run(args):
    from kindi.cpu import Engine as CPUEngine
    from kindi.engine import Engine as GPUEngine
    from kindi.engine import cp

    pack = os.path.abspath(os.path.expanduser(args.pack))
    layer = int(args.layer)
    name = "blk.%d.ffn_down.weight" % layer
    rows = int(args.rows)
    if rows <= 0:
        raise SystemExit("kindi: --rows must be positive")

    log = lambda *a: print("kindi-row-split:", *a, file=sys.stderr, flush=True)  # noqa: E731
    cpu = CPUEngine(pack, ctx=1024, layers=(layer,), embed=False, head=False, log=log)
    gpu = GPUEngine(pack, ctx=1024, layers=(layer,), embed=False, head=False, log=log)
    rec = gpu._all_bi[name]
    total = int(rec["rows"])
    if rows >= total:
        raise SystemExit("kindi: --rows %d leaves no GPU rows in %s (%d rows)" % (rows, name, total))
    if rows % 16:
        raise SystemExit("kindi: --rows must be a multiple of 16")
    rows_cpu = (0, rows)
    rows_gpu = (rows, total)
    if args.mode == "cpu-matrix":
        result = _run_matrix(args, cpu, gpu, cp, pack, name, total, rows)
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    if args.mode == "cpu-layer":
        result = _run_cpu_layer(args, cpu, gpu, cp, pack, name, total, rows)
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    rng = _rng(args.seed)
    x = rng.normal(0.0, 0.5, size=int(gpu.D)).astype(np.float32)

    def prepare(split):
        gpu._down_split = (cpu, rows_cpu) if split else None
        gpu.reset()
        gpu.h[:] = cp.asarray(x)
        gpu.pending = 0

    def once(split):
        prepare(split)
        gpu.layer(layer, 0)
        gpu.flush()
        _cuda(cp).Stream.null.synchronize()
        return cp.asnumpy(gpu.h).copy()

    # Full-GPU layer first; replacing the down tensor would otherwise hide the baseline.
    for _ in range(args.warmup):
        once(False)
    full_times, full_out = [], None
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        full_out = once(False)
        full_times.append(1000.0 * (time.perf_counter() - t0))

    gpu.load_partial_tensor(pack, name, rows_gpu)
    gpu.set_down_split(cpu, rows_cpu)
    for _ in range(args.warmup):
        once(True)
    split_times, split_out = [], None
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        split_out = once(True)
        split_times.append(1000.0 * (time.perf_counter() - t0))

    assert full_out is not None and split_out is not None
    diff = np.abs(full_out - split_out)
    result = {
        "mode": "gpu-layer",
        "pack": pack,
        "layer": layer,
        "tensor": name,
        "rows_total": total,
        "rows_cpu": rows,
        "rows_gpu": total - rows,
        "iterations": args.iterations,
        "full_ms": {"p50": statistics.median(full_times), "p10": _pct(full_times, 10),
                     "p90": _pct(full_times, 90)},
        "split_ms": {"p50": statistics.median(split_times), "p10": _pct(split_times, 10),
                      "p90": _pct(split_times, 90)},
        "ratio_full_over_split": statistics.median(full_times) / statistics.median(split_times),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "split_forward": bool(np.all(np.isfinite(split_out))),
        "gpu": str(_cuda(cp).runtime.getDeviceProperties(_cuda(cp).Device().id)["name"]),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, help="Kindi pack directory")
    parser.add_argument("--mode", choices=("gpu-layer", "cpu-matrix", "cpu-layer"), default="gpu-layer",
                        help="gpu-layer: a GPU-owned layer with CPU lower rows; cpu-matrix/cpu-layer: a CPU-owned GEMV/layer")
    parser.add_argument("--layer", type=int, default=0, help="layer whose ffn_down is split")
    parser.add_argument("--rows", type=int, default=2048, help="rows on the CPU (lower slice)")
    parser.add_argument("--iterations", type=int, default=15, help="timed layer runs per side")
    parser.add_argument("--warmup", type=int, default=3, help="warmup layer runs per side")
    parser.add_argument("--seed", type=int, default=0, help="random input seed")
    parser.add_argument("--json", help="write the result JSON here")
    args = parser.parse_args(argv)
    return 0 if run(args)["split_forward"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
