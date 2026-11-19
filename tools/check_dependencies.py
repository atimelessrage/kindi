#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Gate dependency-audit reports against a reviewed baseline.

`pip-audit` writes JSON; this tool turns it into a set of `package advisory` pairs and
fails only on a pair the baseline has not reviewed. The CI job is report-only until the
first baseline is recorded:

    pip-audit -r requirements-runtime.txt --no-deps --format json --output audit.json
    python tools/check_dependencies.py --report audit.json --report-only

After reviewing a report, record it and make later runs gating:

    python tools/check_dependencies.py --report audit.json --update
    python tools/check_dependencies.py --report audit.json

The baseline is `docs/development/dependency-audit-baseline.json`.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINE = os.path.join(ROOT, "docs", "development", "dependency-audit-baseline.json")


class AuditError(Exception):
    pass


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise AuditError("cannot read %s: %s" % (path, exc)) from exc


def findings(report_path):
    """Return `{package advisory}` from one pip-audit JSON report."""
    data = _read_json(report_path)
    dependencies = data.get("dependencies", []) if isinstance(data, dict) else data
    if not isinstance(dependencies, list):
        raise AuditError("%s has no dependency list" % report_path)
    found = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        name = dependency.get("name") or dependency.get("package") or "?"
        for vuln in dependency.get("vulns") or []:
            advisory = vuln.get("id") or vuln.get("vulnerability_id") or "?"
            found.add("%s %s" % (name, advisory))
    return found


def load_baseline(path):
    if not path or not os.path.isfile(path):
        return set()
    data = _read_json(path)
    values = data.get("vulnerabilities", []) if isinstance(data, dict) else data
    if not isinstance(values, list):
        raise AuditError("%s has no vulnerabilities list" % path)
    return {str(value) for value in values}


def write_baseline(path, values):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"vulnerabilities": sorted(values)}, handle, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Gate dependency-audit reports against a reviewed baseline.")
    parser.add_argument("--report", action="append", default=[],
                        help="pip-audit JSON report (repeatable)")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help="reviewed baseline JSON (default: %s)" % DEFAULT_BASELINE)
    parser.add_argument("--update", action="store_true",
                        help="replace the baseline with these findings and exit 0")
    parser.add_argument("--report-only", action="store_true",
                        help="print findings but never fail on a new one")
    parser.add_argument("--allow-missing", action="store_true",
                        help="warn instead of failing when a report file is absent")
    args = parser.parse_args(argv)

    found = set()
    for report in args.report:
        if not os.path.isfile(report):
            if args.allow_missing:
                print("dependency_audit: missing report %s (skipped)" % report, file=sys.stderr)
                continue
            print("dependency_audit: missing report %s" % report, file=sys.stderr)
            return 2
        found |= findings(report)
    if not args.report:
        print("dependency_audit: no --report given", file=sys.stderr)
        return 2

    if args.update:
        write_baseline(args.baseline, found)
        print("dependency_audit: recorded %d finding(s) in %s" % (len(found), args.baseline))
        return 0

    baseline = load_baseline(args.baseline)
    new = sorted(found - baseline)
    fixed = sorted(baseline - found)
    for item in new:
        print("NEW %s" % item)
    for item in fixed:
        print("FIXED %s" % item)
    print("dependency_audit: %d finding(s), %d new, %d fixed, baseline %s"
          % (len(found), len(new), len(fixed), args.baseline))
    if args.report_only:
        return 0
    if new and not baseline:
        print("dependency_audit: no reviewed baseline; run with --update after review",
              file=sys.stderr)
        return 1
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
