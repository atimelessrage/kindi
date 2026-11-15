#!/usr/bin/env python3
"""Per-module coverage floor gate (audit M4).

The CPU CI job keeps a *total* floor (``--cov-fail-under=27``), which lets a
module the CPU suite fully exercises quietly rot while the total holds. This
gate checks the modules the CPU suite can actually reach against
``tests/coverage_floors.json`` and fails when any of them drops below its
floor.

The floors file deliberately lists only CPU-verifiable modules (server, cache,
bench, tandem, vram, expert_cache, tokenizer, config, hostram, catalog,
sched, sampling and the small pure-Python modules). The heavy CUDA modules
(``engine.py``, ``spec.py``, ``tp.py``) are verified by ``kindi verify`` on
hardware, not by pytest; gating them here would need GPU coverage first
(requires the self-hosted GPU lane, audit M1). Do not add a GPU module to the
floors file until the GPU lane measures it.

Usage (from the repository root), after a pytest run with
``--cov=kindi --cov-report=json``::

    python tools/check_coverage.py                 # gate, exit 1 below a floor
    python tools/check_coverage.py --update        # re-record floors from coverage.json
    python tools/check_coverage.py --coverage coverage.json --floors tests/coverage_floors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE = ROOT / "coverage.json"
DEFAULT_FLOORS = ROOT / "tests" / "coverage_floors.json"


def _norm(path: str) -> str:
    """Coverage JSON keys use the host separator (``kindi\\server.py`` on Windows)."""
    return Path(path.replace("\\", "/")).as_posix()


def read_coverage(path: Path) -> dict:
    """{module: percent_covered} from a coverage.py JSON report."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {_norm(name): float(info["summary"]["percent_covered"])
            for name, info in data.get("files", {}).items()}


def read_floors(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {_norm(name): float(floor) for name, floor in data.items()}


def check(coverage: dict, floors: dict) -> list:
    """Return a list of below-floor descriptions (empty when the gate passes)."""
    problems = []
    for module, floor in sorted(floors.items()):
        seen = coverage.get(module)
        if seen is None:
            problems.append("%s: module not in the coverage report (was it imported?)" % module)
        elif seen < floor:
            problems.append("%s: %.1f%% < floor %.0f%%" % (module, seen, floor))
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    ap.add_argument("--floors", type=Path, default=DEFAULT_FLOORS)
    ap.add_argument("--update", action="store_true",
                    help="re-record the floors from the coverage report instead of gating")
    a = ap.parse_args(argv)

    coverage = read_coverage(a.coverage)
    if a.update:
        floors = read_floors(a.floors)
        updated = {module: int(round(coverage[module]))
                   for module in floors if module in coverage}
        a.floors.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        print("floors updated: %d modules -> %s" % (len(updated), a.floors))
        return 0

    floors = read_floors(a.floors)
    problems = check(coverage, floors)
    if problems:
        print("per-module coverage below floor (%s):" % a.floors)
        for p in problems:
            print("  " + p)
        return 1
    print("coverage floors ok: %d modules at or above their floor" % len(floors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
