#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Ratchet Python lint and type checking against a measured baseline.

The tree predates a lint/type gate, so making either tool fail on all current
findings would block every change. Instead, this gate records the current
(file, rule/code) counts in `tools/python_quality_baseline.json` and fails only
when a count grows or a new file/rule pair appears. Fixing debt is always allowed;
`--update --reason ...` lowers the baseline deliberately after a reviewed change. An update
refuses any file/rule count that grew, so the flag cannot be used to launder a new finding, and
it records the reason, the measured commit and the per-tool delta in the baseline.

Ruff covers the whole Python tree with correctness rules F, E9, B and I (see
`ruff.toml`). Mypy and Pyright cover first-party production code only (`kindi`,
`tools`, `examples` and `bridge/kindi_bridge`) with third-party imports ignored
(see `mypy.ini` and `pyrightconfig.json`); the test tree is not type-checked
because its deliberately dynamic fixtures and duplicate `conftest.py` module
names are not a useful signal. Pylint is run on the same production targets for
errors and fatal messages only (`E`, `F`) at the other tools' Python 3.10 target, with
optional-import errors ignored.

Usage:
    python tools/check_python_quality.py
    python tools/check_python_quality.py --update --reason "removed the B904 sites in kindi/cache.py"
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
PYRIGHT_CONFIG = ROOT / "pyrightconfig.json"
PYRIGHT_TARGETS = MYPY_TARGETS
PYLINT_TARGETS = MYPY_TARGETS
PYLINT_PY_VERSION = "3.10"          # Ruff/mypy/Pyright target 3.10; Pylint would otherwise use the host Python
# 0.7.0's Flash-Next GPU modules (and the qx_* kernels they call) are vendored but not wired into this
# fork: no host here can run them, so their torch/CUDA type idioms stay outside the analyser scope until
# a GPU run makes them part of the served surface (AGENTS.md records the decision). Ruff and mypy do
# cover them; this tuple gates only the two tools whose findings are torch/CUDA false positives.
VENDORED_UNWIRED = ("kindi/qwen4exp_", "kindi/qx_")


def _dedupe_targets(targets: tuple[str, ...]) -> tuple[str, ...]:
    """Keep the outermost target when a path is nested inside another target.

    Mypy maps a file and its containing directory to the same module and exits 2
    with ``Duplicate module named ...`` when both are passed. The directory already
    covers the nested path, so dropping the nested entry checks the same files.
    """
    paths = [Path(target) for target in targets]
    keep = []
    for index, (target, path) in enumerate(zip(targets, paths, strict=True)):
        if path in paths[:index]:
            continue
        if any(path != other and other in path.parents for other in paths):
            continue
        keep.append(target)
    return tuple(keep)


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
    command.extend(_dedupe_targets(MYPY_TARGETS))
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


def _pyright_findings(pyright):
    proc = _run([pyright, "--project", str(PYRIGHT_CONFIG), "--outputjson"], "pyright")
    if proc.returncode not in (0, 1):
        raise QualityError("pyright failed: %s" % (proc.stderr or proc.stdout).strip())
    try:
        payload = json.loads(proc.stdout or "{}")
    except ValueError as exc:
        raise QualityError("pyright produced invalid JSON: %s" % exc) from exc
    groups = {}
    for row in payload.get("generalDiagnostics", []):
        if row.get("severity") != "error":
            continue
        _add(groups, row.get("file") or "?", row.get("rule") or "no-rule")
    if proc.returncode == 1 and not groups:
        raise QualityError("pyright failed without a parseable error: %s"
                           % (proc.stderr or proc.stdout).strip())
    return groups


def _pylint_findings(pylint):
    command = [pylint, "--errors-only", "--disable=import-error",
               "--py-version=" + PYLINT_PY_VERSION,
               "--output-format=json", "--score=n", "--reports=n"]
    command.extend(PYLINT_TARGETS)
    proc = _run(command, "pylint")
    if proc.returncode not in (0, 2):
        raise QualityError("pylint failed: %s" % (proc.stderr or proc.stdout).strip())
    try:
        rows = json.loads(proc.stdout or "[]")
    except ValueError as exc:
        raise QualityError("pylint produced invalid JSON: %s" % exc) from exc
    groups = {}
    for row in rows:
        rel = (row.get("path") or "?").replace("\\", "/")
        if any(rel.startswith(prefix) for prefix in VENDORED_UNWIRED):
            continue
        _add(groups, row.get("path") or "?", row.get("message-id") or "no-code")
    if proc.returncode == 2 and not groups:
        raise QualityError("pylint failed without a parseable error: %s"
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


def _counts(ruff, ruff_findings, mypy_findings, pyright, pyright_findings, pylint, pylint_findings):
    return {
        "ruff": {"version": _tool_version([ruff, "--version"], "ruff --version"),
                 "findings": ruff_findings, "total": _totals(ruff_findings)},
        "mypy": {"version": _tool_version([sys.executable, "-m", "mypy", "--version"], "mypy --version"),
                 "findings": mypy_findings, "total": _totals(mypy_findings)},
        "pyright": {"version": _tool_version([pyright, "--version"], "pyright --version"),
                    "findings": pyright_findings, "total": _totals(pyright_findings)},
        "pylint": {"version": _tool_version([pylint, "--version"], "pylint --version"),
                   "findings": pylint_findings, "total": _totals(pylint_findings)},
        "scope": {"ruff_rules": "F,E9,B,I",
                  "mypy_targets": list(MYPY_TARGETS),
                  "pyright_targets": list(PYRIGHT_TARGETS),
                  "pyright_mode": "basic",
                  "pyright_platform": "All",
                  "pylint_targets": list(PYLINT_TARGETS),
                  "pylint_checks": "E,F (import-error disabled)",
                  "pylint_py_version": PYLINT_PY_VERSION},
    }


def _head():
    """The measured commit, for the baseline's review block; empty when the tree is not a checkout."""
    try:
        proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def _load_baseline():
    try:
        with BASELINE_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise QualityError("cannot read %s: %s (run with --update --reason ...)"
                           % (BASELINE_PATH, exc)) from exc
    except ValueError as exc:
        raise QualityError("invalid %s: %s" % (BASELINE_PATH, exc)) from exc


def _check(ruff, ruff_findings, mypy_findings, pyright, pyright_findings, pylint, pylint_findings):
    baseline = _load_baseline()
    if baseline.get("schema") != 2:
        raise QualityError("unsupported baseline schema: %r" % (baseline.get("schema"),))
    current = _counts(ruff, ruff_findings, mypy_findings, pyright, pyright_findings,
                       pylint, pylint_findings)
    problems = []
    for tool in ("ruff", "mypy", "pyright", "pylint"):
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
              "`python tools/check_python_quality.py --update --reason \"...\"`.", file=sys.stderr)
        return 1
    for tool in ("ruff", "mypy", "pyright", "pylint"):
        now = current[tool]["total"]
        was = int(baseline[tool].get("total") or 0)
        note = "" if now == was else " (%+d)" % (now - was)
        print("%s: %d findings%s; baseline allows %d" % (tool, now, note, was))
    return 0


def _update(ruff, ruff_findings, mypy_findings, pyright, pyright_findings, pylint, pylint_findings,
            reason):
    baseline = _load_baseline()
    if baseline.get("schema") != 2:
        raise QualityError("unsupported baseline schema: %r" % (baseline.get("schema"),))
    current = _counts(ruff, ruff_findings, mypy_findings, pyright, pyright_findings,
                       pylint, pylint_findings)
    problems = []
    for tool in ("ruff", "mypy", "pyright", "pylint"):
        problems.extend("%s: %s" % (tool, item) for item in _regressions(
            baseline.get(tool, {}).get("findings") or {}, current[tool]["findings"]))
    if problems:
        print("refusing to update: %d new or increased finding(s) must be fixed first." % len(problems),
              file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        return 1
    payload: dict[str, object] = {"schema": 2}
    payload.update(current)
    payload["review"] = {
        "reason": reason.strip(),
        "head": _head(),
        "previous_head": str((baseline.get("review") or {}).get("head") or ""),
        "deltas": {tool: current[tool]["total"] - int(baseline.get(tool, {}).get("total") or 0)
                   for tool in ("ruff", "mypy", "pyright", "pylint")},
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        shown = BASELINE_PATH.relative_to(ROOT)
    except ValueError:                 # a caller may point --update at a baseline outside the tree
        shown = BASELINE_PATH
    print("wrote %s: ruff=%d, mypy=%d, pyright=%d, pylint=%d" % (
        shown, current["ruff"]["total"], current["mypy"]["total"],
        current["pyright"]["total"], current["pylint"]["total"]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--update", action="store_true",
                        help="lower the baseline after review; refuses any growth")
    parser.add_argument("--reason", default="",
                        help="why the baseline is being lowered (required with --update)")
    args = parser.parse_args(argv)
    if args.update and not args.reason.strip():
        print("check_python_quality: --update requires --reason so the change is reviewable",
              file=sys.stderr)
        return 2
    if args.reason.strip() and not args.update:
        print("check_python_quality: --reason is only valid with --update", file=sys.stderr)
        return 2

    ruff = shutil.which("ruff") or ""
    pyright = shutil.which("pyright") or ""
    pylint = shutil.which("pylint") or ""
    missing = [name for name, path in (("ruff", ruff), ("pyright", pyright),
                                       ("pylint", pylint)) if not path]
    if missing:
        print("check_python_quality: %s not installed (pip install -r tests/requirements-test.txt)"
              % ", ".join(missing), file=sys.stderr)
        return 2
    try:
        ruff_findings = _ruff_findings(ruff)
        mypy_findings = _mypy_findings()
        pyright_findings = _pyright_findings(pyright)
        pylint_findings = _pylint_findings(pylint)
        return (_update(ruff, ruff_findings, mypy_findings, pyright, pyright_findings,
                        pylint, pylint_findings, args.reason)
                if args.update else
                _check(ruff, ruff_findings, mypy_findings, pyright, pyright_findings,
                       pylint, pylint_findings))
    except QualityError as exc:
        print("check_python_quality: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
