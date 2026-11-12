#!/usr/bin/env python3
"""Keep .secrets.baseline in sync with the tracked files, not the whole checkout.

`detect-secrets scan --all-files` also scans untracked benchmark logs, virtualenvs
and test caches. A baseline made that way is noisy and changes with local state.
This helper always scans `git ls-files`, normalizes paths to `/` so the baseline
works on Windows and Linux, and compares only the detector results.

Examples:
  python tools/check_secret_baseline.py --check
  python tools/check_secret_baseline.py --update
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_NAME = ".secrets.baseline"


def tracked_files() -> list[str]:
    """Tracked paths, excluding the baseline itself."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True, capture_output=True,
    ).stdout
    files = [raw.decode("utf-8", "replace") for raw in out.split(b"\0") if raw]
    return [path for path in files if path.replace("\\", "/") != BASELINE_NAME]


def _normalize_results(results: dict) -> dict:
    normalized = {}
    for path, findings in results.items():
        key = path.replace("\\", "/")
        entries = []
        for item in findings:
            clean = dict(item)
            clean["filename"] = str(clean.get("filename", key)).replace("\\", "/")
            entries.append(clean)
        normalized[key] = sorted(
            entries,
            key=lambda item: (
                int(item.get("line_number", 0)),
                str(item.get("type", "")),
                str(item.get("hashed_secret", "")),
            ),
        )
    return normalized


def scan_tracked() -> dict:
    """Run detect-secrets over tracked files only."""
    files = tracked_files()
    proc = subprocess.run(
        [sys.executable, "-m", "detect_secrets", "scan", *files],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise SystemExit("detect-secrets scan failed: %s" % detail[-2000:])
    try:
        payload = json.loads(proc.stdout)
    except ValueError as exc:
        raise SystemExit("detect-secrets did not return JSON: %s" % exc)
    payload["results"] = _normalize_results(payload.get("results", {}))
    return payload


def results_differences(baseline: dict, current: dict) -> list[str]:
    """Return file-level differences without printing secret values."""
    before = baseline.get("results", {})
    after = current.get("results", {})
    problems = []
    for path in sorted(set(before) - set(after)):
        problems.append("%s: baseline-only finding(s)" % path)
    for path in sorted(set(after) - set(before)):
        problems.append("%s: new finding(s)" % path)
    for path in sorted(set(before) & set(after)):
        if before[path] != after[path]:
            problems.append("%s: finding line/hash changed" % path)
    return problems


def _load_baseline(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit("baseline not found: %s" % path)
    except (OSError, ValueError) as exc:
        raise SystemExit("cannot read baseline %s: %s" % (path, exc))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise SystemExit("baseline %s: expected a results object" % path)
    return payload


def _write_baseline(path: Path, baseline: dict, current: dict) -> None:
    baseline = dict(baseline)
    baseline["results"] = current.get("results", {})
    baseline["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    filters = baseline.setdefault("filters_used", [])
    if not any(str(item.get("path", "")).endswith("is_baseline_file") for item in filters):
        filters.append({
            "path": "detect_secrets.filters.common.is_baseline_file",
            "filename": BASELINE_NAME,
        })
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=Path, default=ROOT / BASELINE_NAME,
                    help="baseline path (default: .secrets.baseline)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="compare the baseline with the tracked tree (default)")
    mode.add_argument("--update", action="store_true",
                      help="rewrite the baseline from the current tracked tree")
    args = ap.parse_args(argv)
    baseline_path = args.baseline.expanduser().resolve()
    baseline = _load_baseline(baseline_path)
    current = scan_tracked()
    if args.update:
        _write_baseline(baseline_path, baseline, current)
        count = sum(len(findings) for findings in current["results"].values())
        print("updated %s: %d finding(s) in %d file(s)" %
              (baseline_path, count, len(current["results"])))
        return 0

    problems = results_differences(baseline, current)
    if problems:
        print("secret baseline is stale for the tracked tree:", file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        print("review the finding(s), then run: python tools/check_secret_baseline.py --update",
              file=sys.stderr)
        return 1
    count = sum(len(findings) for findings in current["results"].values())
    print("secret baseline is current: %d finding(s) in %d tracked file(s)" %
          (count, len(current["results"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
