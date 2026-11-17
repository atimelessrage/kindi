"""Summarize the same-settings A/B logs into `ab.json`.

The logs are the evidence; this script only parses them. Rerun it after adding a pass:

    python docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-ab-0.6.1/summarize.py

Stock `kindi bench` prints one line per cell; the fork prints the same lines (its
`--json` file carries the same cells). Both are parsed from the logs so the two
sides are read the same way.
"""
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("2b", "9b")
ENGINES = ("stock", "fork")
POSITIONS = (100, 2000, 8000)


def cells_of(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    read = float(re.search(r"prompt reading:\s+([\d.]+)", text).group(1))
    gen = {}
    for pos in POSITIONS:
        m = re.search(r"position\s+%d:\s+([\d.]+)" % pos, text)
        if m:
            gen["exact:%d" % pos] = float(m.group(1))
    return read, gen


def stat(values):
    return {"runs": values, "median": round(statistics.median(values), 1),
            "min": min(values), "max": max(values)}


def main():
    out = {"measured": "2026-11-17", "passes": 5,
           "settings": {"ctx": 8192, "engine": "plain engine, exact head, no speculation",
                        "stock": "v0.6.1 (7fececb) + native-Windows libcuda shim",
                        "fork": "plain engine via KINDI_TP=0; --split-at off"},
           "cells": {}}
    for model in MODELS:
        out["cells"][model] = {}
        for engine in ENGINES:
            paths = sorted(glob.glob(os.path.join(HERE, "runs", "%s-%s-8192-r*.log"
                                                  % (engine, model))))
            if len(paths) != 5:
                raise SystemExit("%s-%s: expected 5 logs, found %d" % (engine, model, len(paths)))
            reads, gens = [], {("exact:%d" % p): [] for p in POSITIONS}
            for path in paths:
                read, gen = cells_of(path)
                reads.append(read)
                for key, value in gen.items():
                    gens[key].append(value)
            out["cells"][model][engine] = {
                "read_tok_s": stat(reads),
                "gen_tok_s": {key: stat(values) for key, values in gens.items()},
            }
    path = os.path.join(HERE, "ab.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
