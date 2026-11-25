#!/usr/bin/env python3
"""Perf-regression gate against the recorded CROSS-MODE baseline (M7).

Compares the current generated `docs/benchmarks/CROSS-MODE.csv` with the recorded
`docs/benchmarks/perf-baseline.json` and fails when a previously measured cell
disappears or regresses by more than the tolerance (default 15%). This is a
CPU-only check: it gates the *recorded numbers*, so CI can run it without a GPU.
The GPU lane (`gpu-smoke.yml`) should regenerate CROSS-MODE.csv on its runner
and then run this gate; a failure means a measured throughput regression.

Usage (from the repository root):
    python tools/check_perf_baseline.py              # gate, exit 1 on regression
    python tools/check_perf_baseline.py --update     # deliberately re-record the baseline
    python tools/check_perf_baseline.py --tolerance 0.20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "docs" / "benchmarks" / "CROSS-MODE.csv"
DEFAULT_BASELINE = ROOT / "docs" / "benchmarks" / "perf-baseline.json"
METRICS = ("read", "gen_100", "gen_2000", "gen_8000", "gen_16000", "gen_24000")


def read_cells(path: Path) -> dict:
    """{(mode, model): {metric: float}} for available rows of a CROSS-MODE.csv."""
    cells = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("available") != "1":
                continue
            key = (row["mode"], row["model"])
            cells[key] = {m: float(row[m]) for m in METRICS if row.get(m)}
    return cells


def check(current: dict, baseline: dict, tolerance: float) -> list:
    """Return a list of regression descriptions (empty when the gate passes)."""
    problems = []
    for key, base in baseline.items():
        cur = current.get(key)
        if cur is None:
            problems.append("%s: baseline cell missing from current matrix" % key)
            continue
        for metric, base_v in base.items():
            if metric not in cur:
                problems.append("%s %s: baseline %.1f missing" % (key, metric, base_v))
                continue
            if cur[metric] < base_v * (1.0 - tolerance):
                problems.append("%s %s: %.1f < %.1f (-%.0f%%, tol %.0f%%)"
                                % (key, metric, cur[metric], base_v,
                                   (1 - cur[metric] / base_v) * 100, tolerance * 100))
    return problems


def _tolerance(text):
    """A finite non-negative relative tolerance, rejected at argparse time."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            "must be a finite non-negative number (got %r)" % (text,))
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--tolerance", type=_tolerance, default=0.15,
                    help="allowed relative regression, a finite non-negative number (default 0.15)")
    ap.add_argument("--update", action="store_true",
                    help="re-record the baseline from the current CSV instead of gating")
    a = ap.parse_args(argv)

    current = {"%s/%s" % k: v for k, v in read_cells(a.csv).items()}
    if a.update:
        a.baseline.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        print("baseline updated: %d cells -> %s" % (len(current), a.baseline))
        return 0

    baseline = json.loads(a.baseline.read_text(encoding="utf-8"))
    problems = check(current, baseline, a.tolerance)
    if problems:
        print("perf regression vs %s (tolerance %.0f%%):" % (a.baseline, a.tolerance * 100))
        for p in problems:
            print("  " + p)
        return 1
    print("perf gate ok: %d baseline cells within %.0f%%" % (len(baseline), a.tolerance * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
