#!/usr/bin/env python3
"""Aggregate the served A/B runs (`ab-old*.json`, `ab-new*.json`) into `ab-summary.json`.

Each arm file is one `measure.py` run against a fresh CPU server: the `old` arm has the modules
restored from the previous commit, the `new` arm the working tree. Medians are taken over the run
rows; `*_gain_percent` is `(old - new) / old`, so a positive gain is faster.

  python docs/benchmarks/2026-11-25-i7-8086k-cpu-key-chars/ab_summary.py
"""
import glob
import json
import os
import statistics


def rows(pattern, directory):
    out = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        with open(path, encoding="utf-8") as handle:
            server = json.load(handle)["server"]
        out.append({
            "file": os.path.basename(path),
            "structured": server["structured"]["ms_per_token"],
            "unconstrained": server["unconstrained"]["ms_per_token"],
            "gap": server["gap_ms_per_token"],
        })
    return out


def summary(arm_rows):
    if not arm_rows:
        raise SystemExit("no arm rows")
    return {
        "runs": arm_rows,
        "structured_median": round(statistics.median(r["structured"] for r in arm_rows), 2),
        "unconstrained_median": round(statistics.median(r["unconstrained"] for r in arm_rows), 2),
        "gap_median": round(statistics.median(r["gap"] for r in arm_rows), 2),
    }


def gain(old, new):
    return round((old - new) / old * 100.0, 1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    old = summary(rows("ab-old*.json", here))
    new = summary(rows("ab-new*.json", here))
    data = {
        "host": "i7-8086K, Windows, Python 3.13, CPU engine, no GPU",
        "pack": "bonsai-2b",
        "shape": "10 runs per shape, 8-token replies, temperature 0, measure.py --skip-filter",
        "arms": {"old": old, "new": new},
        "structured_gain_percent": gain(old["structured_median"], new["structured_median"]),
        "gap_gain_percent": gain(old["gap_median"], new["gap_median"]),
        "unconstrained_change_percent": -gain(old["unconstrained_median"], new["unconstrained_median"]),
    }
    path = os.path.join(here, "ab-summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    print("old %.2f / %.2f, new %.2f / %.2f" % (old["structured_median"], old["gap_median"],
                                                 new["structured_median"], new["gap_median"]))


if __name__ == "__main__":
    main()
