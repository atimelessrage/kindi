#!/usr/bin/env python3
"""Build the i7-8086K CPU benchmark tables from the raw logs."""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

# model, threads, binding, log
CASES = [
    ("2b", 6, "close/cores", "bench-2b-6t.log"),
    ("2b", 12, "default", "bench-2b-12t-default.log"),
    ("4b", 6, "close/cores", "bench-4b-6t.log"),
    ("4b", 12, "default", "bench-4b-12t-default.log"),
    ("9b", 6, "close/cores", "bench-9b-6t.log"),
    ("9b", 12, "default", "bench-9b-12t-default.log"),
]
POSITIONS = [100, 2000, 8000, 16000, 24000]
VERIFY = [
    ("2b", "verify-2b.log"),
    ("4b", "verify-4b.log"),
    ("9b", "verify-9b.log"),
]
READ_RE = re.compile(r"prompt reading:\s+([0-9.]+) tokens/s")
GEN_RE = re.compile(r"generation \(exact head\) at position\s+(\d+):\s+([0-9.]+) tokens/s")
PASS_RE = re.compile(r"ALL PASS: token-exact with llama.cpp - ([0-9.]+) tokens/s")


def parse_bench(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    read = READ_RE.search(text)
    gens = {int(pos): float(rate) for pos, rate in GEN_RE.findall(text)}
    if read is None or any(p not in gens for p in POSITIONS):
        return None
    return {
        "prompt": float(read.group(1)),
        "gen": {p: gens[p] for p in POSITIONS},
    }


def main():
    rows = []
    for model, threads, binding, name in CASES:
        path = os.path.join(RAW, name)
        if not os.path.isfile(path):
            print("missing", path)
            continue
        parsed = parse_bench(path)
        if parsed is None:
            print("incomplete", path)
            continue
        row = {
            "model": model,
            "threads": threads,
            "binding": binding,
            "run": 1,
            "prompt": parsed["prompt"],
        }
        for p in POSITIONS:
            row["gen_%d" % p] = parsed["gen"][p]
        rows.append(row)

    with open(os.path.join(HERE, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "threads", "binding", "run", "prompt"] +
                    ["gen_%d" % p for p in POSITIONS])
        for r in rows:
            w.writerow([r["model"], r["threads"], r["binding"], r["run"], "%.1f" % r["prompt"]] +
                       ["%.1f" % r["gen_%d" % p] for p in POSITIONS])

    verification = {}
    for model, name in VERIFY:
        path = os.path.join(RAW, name)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        m = PASS_RE.search(text)
        verification[model] = {
            "result": "PASS" if m else "FAIL",
            "prompts": text.count(" PASS  "),
            "tokens_per_s": float(m.group(1)) if m else None,
        }

    summary = {
        "schema": 1,
        "title": "Kindi CPU engine on Intel Core i7-8086K",
        "date": "2026-11-09",
        "rig": {
            "cpu": "Intel Core i7-8086K @ 4.00 GHz",
            "cores": 6,
            "threads": 12,
            "memory_gib": 15,
            "os": "Microsoft Windows 10 Pro 10.0.19045; WSL2 Ubuntu 24.04",
            "kernel": "6.18.54.1-microsoft-standard-WSL2",
            "gcc": "13.3.0",
            "python": "3.12.3",
            "numpy": "2.5.4",
            "engine": "kindi/cpu.py + kindi/kcpu.c, built with gcc -O3 -march=native -fopenmp -shared -fPIC",
        },
        "method": {
            "command": "KINDI_DEVICE=cpu python3 -m kindi --pack <PACK> --ctx 24576 bench",
            "context": 24576,
            "head": "exact",
            "speculation": "off",
            "runs_per_config": 1,
            "threads": [6, 12],
            "binding": {
                "6": "OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores",
                "12": "OMP_NUM_THREADS=12; no explicit OMP_PROC_BIND/OMP_PLACES",
            },
            "note": "One run per configuration; bench warms each timed operation and uses a fresh process. 24,576-token context.",
        },
        "verification": verification,
        "runs": rows,
        "pinned_12_thread_observation": {
            "2b": {
                "prompt": 1.8,
                "gen": {"100": 1.8, "2000": 1.8, "8000": 1.6, "16000": 1.5, "24000": 1.4},
                "log": "raw/bench-2b-12t-pinned.log",
                "note": "OMP_PROC_BIND=close OMP_PLACES=cores with 12 threads on 6 cores. Not used as the headline result.",
            },
            "4b": {
                "log": "raw/bench-4b-12t-pinned-incomplete.log",
                "note": "Interrupted at position 8000; not a complete benchmark.",
            },
        },
        "full_cpu_suite": {
            "command": "KINDI_DEVICE=cpu python -m pytest tests/ -q",
            "result": "127 passed, 1 warning in 5.98s",
            "python": "3.13.6 on Windows",
            "log": "raw/pytest.log",
        },
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("wrote runs.csv (%d rows) and summary.json" % len(rows))
    for model, v in verification.items():
        print(model, v)


if __name__ == "__main__":
    main()
