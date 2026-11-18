#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Ratchet Python lint and type checking against a measured baseline.

The tree predates a lint/type gate, so making either tool fail on all current
findings would block every change. Instead, this gate records the current
(file, rule/code) counts in `tools/python_quality_baseline.json` and fails only
when a count grows or a new file/rule pair appears. Fixing debt is always allowed;
`--update` lowers the baseline deliberately after a reviewed change.

Ruff covers the whole Python tree with correctness rules F, E9, B and I (see
`ruff.toml`). Mypy covers first-party production code only (`kindi`, `tools`,
`examples` and `bridge/kindi_bridge`) with third-party imports ignored (see
`mypy.ini`); the test tree is not type-checked because its deliberately dynamic
fixtures and duplicate `conftest.py` module names are not a useful signal.

Usage:
    python tools/check_python_quality.py
    python tools/check_python_quality.py --update
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tools" / "python_quality_baseline.json"
RUFF_CONFIG = ROOT / "ruff.toml"
MYPY_CONFIG = ROOT / "mypy.ini"
MYPY_TARGETS = ("kindi", "tools", "examples", "bridge/kindi_bridge")


class QualityError(Exception):
    """The gate could not run or its inputs are invalid."""


def _run(command, what):
    try:
        proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualityError("%s failed to start: %s" % (what, exc)) from exc
    return proc


def _tool_version(command, what):
    proc = _run(command, what)
    if proc.returncode != 0:
        raise QualityError("%s failed: %s" % (what, (proc.stderr or proc.stdout).strip()))
    first = (proc.stdout or "").strip().splitlines()[0].split()
    if len(first) < 2:
        raise QualityError("cannot parse %s version: %r" % (what, proc.stdout))
    return first[1]


def _relative(path):
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(ROOT)
        except ValueError:
            pass
    return candidate.as_posix()


def _add(groups, path, code):
    by_file = groups.setdefault(_relative(path), {})
    by_file[code] = by_file.get(code, 0) + 1


def _ruff_findings(ruff):
    proc = _run([ruff, "check", str(ROOT), "--config", str(RUFF_CONFIG),
                 "--output-format=json", "--no-cache"], "ruff check")
    if proc.returncode not in (0, 1):
        raise QualityError("ruff check failed: %s" % (proc.stderr or proc.stdout).strip())
    try:
        rows = json.loads(proc.stdout or "[]")
    except ValueError as exc:
        raise QualityError("ruff produced invalid JSON: %s" % exc) from exc
    groups = {}
    for row in rows:
        _add(groups, row["filename"], row.get("code") or "syntax")
    return groups


def _mypy_findings():
    command = [sys.executable, "-m", "mypy", "--config-file", str(MYPY_CONFIG),
               "--no-error-summary", "--show-error-codes", "--no-pretty"]
    command.extend(MYPY_TARGETS)
    proc = _run(command, "mypy")
    if proc.returncode not in (0, 1):
        raise QualityError("mypy failed: %s" % (proc.stderr or proc.stdout).strip())
    groups = {}
    for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines():
        if ": error: " not in line:
            continue
        location, detail = line.split(": error: ", 1)
        path, _, _line = location.rpartition(":")
        match = re.search(r"\[([a-z0-9-]+)\]$", detail)
        _add(groups, path, match.group(1) if match else "no-code")
    if proc.returncode == 1 and not groups:
        raise QualityError("mypy failed without a parseable error: %s"
                           % (proc.stderr or proc.stdout).strip())
    return groups


def _totals(groups):
    return sum(count for by_file in groups.values() for count in by_file.values())


def _regressions(baseline, current):
    problems = []
    for path, by_code in sorted(current.items()):
        allowed = baseline.get(path, {})
        for code, count in sorted(by_code.items()):
            previous = int(allowed.get(code, 0))
            if count > previous:
                problems.append("%s %s: %d > baseline %d" % (path, code, count, previous))
    return problems


def _counts(ruff, ruff_findings, mypy_findings):
    return {
        "ruff": {"version": _tool_version([ruff, "--version"], "ruff --version"),
                 "findings": ruff_findings, "total": _totals(ruff_findings)},
        "mypy": {"version": _tool_version([sys.executable, "-m", "mypy", "--version"], "mypy --version"),
                 "findings": mypy_findings, "total": _totals(mypy_findings)},
        "scope": {"ruff_rules": "F,E9,B,I", "mypy_targets": list(MYPY_TARGETS)},
    }


def _load_baseline():
    try:
        with BASELINE_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise QualityError("cannot read %s: %s (run with --update)" % (BASELINE_PATH, exc)) from exc
    except ValueError as exc:
        raise QualityError("invalid %s: %s" % (BASELINE_PATH, exc)) from exc


def _check(ruff, ruff_findings, mypy_findings):
    baseline = _load_baseline()
    if baseline.get("schema") != 1:
        raise QualityError("unsupported baseline schema: %r" % (baseline.get("schema"),))
    current = _counts(ruff, ruff_findings, mypy_findings)
    problems = []
    for tool in ("ruff", "mypy"):
        expected = str(baseline.get(tool, {}).get("version") or "")
        actual = current[tool]["version"]
        if expected != actual:
            problems.append("%s version is %s; baseline was measured with %s"
                            % (tool, actual, expected or "unknown"))
            continue
        problems.extend("%s: %s" % (tool, item) for item in _regressions(
            baseline.get(tool, {}).get("findings") or {}, current[tool]["findings"]))
    if problems:
        print("Python quality ratchet failed: %d new or increased finding(s)." % len(problems),
              file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        print("Fix the finding, or after review run "
              "`python tools/check_python_quality.py --update`.", file=sys.stderr)
        return 1
    for tool in ("ruff", "mypy"):
        now = current[tool]["total"]
        was = int(baseline[tool].get("total") or 0)
        note = "" if now == was else " (%+d)" % (now - was)
        print("%s: %d findings%s; baseline allows %d" % (tool, now, note, was))
    return 0


def _update(ruff, ruff_findings, mypy_findings):
    current = _counts(ruff, ruff_findings, mypy_findings)
    payload = {"schema": 1}
    payload.update(current)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote %s: ruff=%d, mypy=%d" % (
        BASELINE_PATH.relative_to(ROOT), current["ruff"]["total"], current["mypy"]["total"]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true",
                        help="rewrite the baseline from the current tree after review")
    args = parser.parse_args(argv)

    ruff = shutil.which("ruff")
    if not ruff:
        print("check_python_quality: ruff is not installed (pip install -r tests/requirements-test.txt)",
              file=sys.stderr)
        return 2
    try:
        ruff_findings = _ruff_findings(ruff)
        mypy_findings = _mypy_findings()
        return (_update(ruff, ruff_findings, mypy_findings) if args.update
                else _check(ruff, ruff_findings, mypy_findings))
    except QualityError as exc:
        print("check_python_quality: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
