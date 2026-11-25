#!/usr/bin/env python3
"""A/B the packed-ternary GEMV kernels on the installed Arc A770 through OpenCL.

Runs the probe's own exactness check on every arm; a mismatch aborts the benchmark instead of
recording a time. Shapes are the ones a Bonsai layer needs (rows = output channels, cols = inputs)
plus a deliberately tiny one, which measures the launch floor: two enqueued kernels with almost no
data move.

  python docs/benchmarks/2026-11-25-i7-8086k-arc-a770-gemv-rows4/bench.py --out bench.json

No model, no backend, no support claim.
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from kindi import intel_ternary  # noqa: E402

SHAPES = ((1024, 2048), (2048, 4096), (4096, 8192), (4096, 128))
VARIANTS = ("scalar", "rows4")


def one(rows, cols, variant, repeats, device_name):
    report = intel_ternary.run_probe(rows=rows, cols=cols, repeats=repeats, kernel=variant,
                                     device_name=device_name)
    if report["result"] != "pass":
        raise SystemExit("%dx%d %s failed: %s" % (rows, cols, variant,
                                                  report["error"] or report["result"]))
    gemv = report["gemv"]
    packed = report["shape"]["packed_bytes"] or 0
    scales = report["shape"]["scale_bytes"] or 0
    traffic = packed + scales + 4 * cols + 4 * rows          # weights, scales, x, y
    seconds = gemv["elapsed_ms"] / 1000.0
    return {
        "variant": variant,
        "kernel": report["kernel"]["names"][0],
        "rows": rows,
        "cols": cols,
        "parts": report["shape"]["parts"],
        "repeats": repeats,
        "packed_bytes": packed,
        "traffic_bytes": traffic,
        "elapsed_ms": gemv["elapsed_ms"],
        "per_call_ms": round(gemv["elapsed_ms"] / repeats, 4),
        "gflops": gemv["gflops"],
        "effective_gb_s": round(traffic * repeats / (seconds * 1e9), 2) if seconds > 0 else None,
        "gflops_from_gb_s": None,
        "mismatches": gemv["mismatches"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="bench.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device-name", default="Arc")
    args = parser.parse_args()

    arms = {}
    for rows, cols in SHAPES:
        key = "%dx%d" % (rows, cols)
        arms[key] = {variant: [] for variant in VARIANTS}
        for variant in VARIANTS:
            for _ in range(max(1, args.runs)):
                arms[key][variant].append(one(rows, cols, variant, args.repeats, args.device_name))
                last = arms[key][variant][-1]
                print("%-10s %-7s per_call %7.3f ms  %8.2f GB/s  %8.1f GFLOP/s"
                      % (key, variant, last["per_call_ms"], last["effective_gb_s"],
                         last["gflops"]), flush=True)

    summary = {"host": "i7-8086K, Windows, Arc A770 16 GB, driver's OpenCL C 1.2 runtime",
               "probe": intel_ternary.PROBE, "support_claim": False,
               "runs_per_arm": max(1, args.runs), "repeats": args.repeats,
               "shapes": {}, "speedup": {}}
    for key, per_variant in arms.items():
        rows = {}
        for variant, runs in per_variant.items():
            rows[variant] = {
                "per_call_ms_median": round(statistics.median(r["per_call_ms"] for r in runs), 4),
                "effective_gb_s_median": round(statistics.median(
                    r["effective_gb_s"] for r in runs), 2),
                "gflops_median": round(statistics.median(r["gflops"] for r in runs), 2),
                "mismatches": sum(r["mismatches"] or 0 for r in runs),
                "runs": runs,
            }
        summary["shapes"][key] = rows
        if "scalar" in rows and "rows4" in rows:
            summary["speedup"][key] = round(
                rows["scalar"]["per_call_ms_median"] / rows["rows4"]["per_call_ms_median"], 3)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    for key in summary["shapes"]:
        print("%-10s speedup %s" % (key, summary["speedup"].get(key)))


if __name__ == "__main__":
    main()
