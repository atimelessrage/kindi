#!/usr/bin/env python3
"""Run the installed 2B/4B/9B/27B packs through the tc and plain verify matrix.

Each cell is a fresh `python -m kindi --pack <pack> verify` process with a clean
KINDI_* environment, so an earlier shell setting cannot silently change the
engine. Results (status, per-prompt pass/tie/fail, throughput, exit code) go to
results.csv and results.json; the raw verify output goes to logs/.

The script exits 0 only when every selected cell passes. A 9B tc refusal is a
recorded result, not a script abort: it prints as `refused` and makes the run
exit 1, which is the correct signal that the matrix is not all-pass.

Examples:
  python tools/verify_matrix.py
  python tools/verify_matrix.py --models 2b,4b,9b,27b --modes tc \
      --pack-27b ~/.kindi/packs27/bonsai-27b
  python tools/verify_matrix.py --pack-dir /data/kindi/packs --out /tmp/verify
  python tools/verify_matrix.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("2b", "4b", "9b", "27b")
MODES = ("plain", "tc")
PACK_NAMES = {"2b": "bonsai-2b", "4b": "bonsai-4b", "9b": "bonsai-9b", "27b": "bonsai-27b"}

# Keep these explicit: the script must not inherit a KINDI_* switch from the caller.
COMMON_ENV = {
    "KINDI_DEVICE": "gpu",
    "KINDI_HEAD": "exact",
    "KINDI_KV": "q8_0",
    "KINDI_SPEC": "0",
}
MODE_ENV = {
    "plain": {
        **COMMON_ENV,
        "KINDI_TP": "0",
        "KINDI_TC": "0",
        "KINDI_FD": "0",
        "KINDI_ATT_TC": "0",
    },
    "tc": {
        **COMMON_ENV,
        "KINDI_TP": "1",
        "KINDI_TC": "1",
        "KINDI_FD": "1",
        "KINDI_ATT_TC": "1",
        "KINDI_TC_TMAX": "4",
    },
}

PROMPT_RE = re.compile(r"prompt\s+(\d+)\s+.*?\s+(PASS|TIE|FAIL)\s+(\d+)/(\d+) tokens match")
TPS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?) tokens/s over the run")
DIFF_RE = re.compile(r"first difference at token (\d+)(?: \(llama\.cpp top-2 gap there ([0-9.]+)\))?")

CSV_FIELDS = (
    "model", "mode", "status", "prompts_pass", "prompts_tie", "prompts_fail",
    "tokens_per_second", "exit_code", "duration_s", "first_diff_token",
    "first_diff_gap", "log",
)


def mode_env(mode: str, cards: int = 1) -> dict[str, str]:
    """The complete engine environment for one matrix cell."""
    if mode not in MODE_ENV:
        raise ValueError("unknown mode %r" % mode)
    env = dict(MODE_ENV[mode])
    env["KINDI_CARDS"] = str(cards)
    return env


def child_env(mode: str, cards: int = 1, base: dict[str, str] | None = None) -> dict[str, str]:
    """Copy the caller's environment except every KINDI_* setting."""
    source = os.environ if base is None else base
    env = {k: v for k, v in source.items() if not k.startswith("KINDI_")}
    env.update(mode_env(mode, cards))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_verify_output(text: str, exit_code: int) -> dict:
    """Turn the human verify log into one machine-readable result row."""
    prompts = []
    for line in text.splitlines():
        m = PROMPT_RE.search(line)
        if not m:
            continue
        prompts.append({
            "index": int(m.group(1)),
            "state": m.group(2),
            "matched": int(m.group(3)),
            "total": int(m.group(4)),
        })

    tps_matches = TPS_RE.findall(text)
    tokens_per_second = float(tps_matches[-1]) if tps_matches else None

    diffs = []
    for line in text.splitlines():
        m = DIFF_RE.search(line)
        if m:
            diffs.append({"token": int(m.group(1)), "gap": float(m.group(2)) if m.group(2) else None})

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = ""
    for line in reversed(lines):
        if ("tokens/s over the run" in line or "ALL PASS" in line
                or line.startswith("PASS:") or "FAIL" in line):
            summary = line
            break
    if not summary:
        summary = lines[-1] if lines else ""
    pass_count = sum(p["state"] == "PASS" for p in prompts)
    tie_count = sum(p["state"] == "TIE" for p in prompts)
    fail_count = sum(p["state"] == "FAIL" for p in prompts)

    if exit_code == 2 or "no llama.cpp reference" in text:
        status = "no-reference"
    elif exit_code == 124:
        status = "timeout"
    elif re.search(r"refus|supports Hadamard blocks|use the plain engine", text, re.IGNORECASE):
        status = "refused"
    elif exit_code != 0 or fail_count:
        status = "fail"
    elif prompts and pass_count + tie_count == len(prompts):
        status = "pass"
    elif "ALL PASS" in text or "PASS:" in text:
        status = "pass"
    else:
        status = "error"

    first = diffs[0] if diffs else None
    return {
        "status": status,
        "prompts": prompts,
        "prompts_pass": pass_count,
        "prompts_tie": tie_count,
        "prompts_fail": fail_count,
        "tokens_per_second": tokens_per_second,
        "first_diff_token": first["token"] if first else None,
        "first_diff_gap": first["gap"] if first else None,
        "summary": summary,
    }


def pack_for(model: str, pack_dir: Path, pack_27b: Path | None = None) -> Path:
    """Resolve the installed pack, with an optional separate 27B location."""
    if model == "27b" and pack_27b is not None:
        return pack_27b
    return pack_dir / PACK_NAMES[model]


def _display_path(path: Path) -> str:
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return path.as_posix()


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _missing_row(model: str, mode: str, pack: Path, cards: int) -> dict:
    return {
        "model": model,
        "mode": mode,
        "status": "missing",
        "prompts": [],
        "prompts_pass": 0,
        "prompts_tie": 0,
        "prompts_fail": 0,
        "tokens_per_second": None,
        "exit_code": None,
        "duration_s": 0.0,
        "first_diff_token": None,
        "first_diff_gap": None,
        "summary": "pack not found: %s" % _display_path(pack),
        "log": "",
        "config": mode_env(mode, cards),
        "command": [],
    }


def run_cell(
    model: str,
    mode: str,
    pack: Path,
    out_dir: Path,
    cards: int,
    timeout: float,
    python: str = sys.executable,
) -> dict:
    log_path = out_dir / "logs" / ("%s-%s.log" % (model, mode))
    command = [python, "-m", "kindi", "--pack", str(pack), "verify"]
    env = child_env(mode, cards)
    started = time.monotonic()
    exit_code: int | None = None
    output = ""
    try:
        proc = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
        exit_code = proc.returncode
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        output = stdout + stderr + "\nTIMEOUT after %.1f s\n" % timeout
    except OSError as exc:
        exit_code = 127
        output = "script error: %s\n" % exc
    duration = time.monotonic() - started

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", newline="\n")
    parsed = parse_verify_output(output, exit_code)
    return {
        "model": model,
        "mode": mode,
        "status": parsed["status"],
        "prompts": parsed["prompts"],
        "prompts_pass": parsed["prompts_pass"],
        "prompts_tie": parsed["prompts_tie"],
        "prompts_fail": parsed["prompts_fail"],
        "tokens_per_second": parsed["tokens_per_second"],
        "exit_code": exit_code,
        "duration_s": duration,
        "first_diff_token": parsed["first_diff_token"],
        "first_diff_gap": parsed["first_diff_gap"],
        "summary": parsed["summary"],
        "log": log_path.relative_to(out_dir).as_posix(),
        "config": mode_env(mode, cards),
        "command": [python, "-m", "kindi", "--pack", _display_path(pack), "verify"],
    }


def _write_results(out_dir: Path, rows: list[dict], args: argparse.Namespace) -> tuple[Path, Path]:
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": row["model"],
                "mode": row["mode"],
                "status": row["status"],
                "prompts_pass": row["prompts_pass"],
                "prompts_tie": row["prompts_tie"],
                "prompts_fail": row["prompts_fail"],
                "tokens_per_second": "" if row["tokens_per_second"] is None else "%.1f" % row["tokens_per_second"],
                "exit_code": "" if row["exit_code"] is None else row["exit_code"],
                "duration_s": "%.1f" % row["duration_s"],
                "first_diff_token": "" if row["first_diff_token"] is None else row["first_diff_token"],
                "first_diff_gap": "" if row["first_diff_gap"] is None else "%.4f" % row["first_diff_gap"],
                "log": row["log"],
            })
    payload = {
        "started_utc": args.started_utc,
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "packs_dir": _display_path(args.pack_dir),
        "pack_27b": None if args.pack_27b is None else _display_path(args.pack_27b),
        "cards": args.cards,
        "timeout_s": args.timeout,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return csv_path, json_path


def _print_table(rows: list[dict]) -> None:
    print("\n%-4s %-5s %-11s %4s %3s %4s %8s  %s" %
          ("model", "mode", "status", "pass", "tie", "fail", "tok/s", "log"))
    print("-" * 76)
    for row in rows:
        tps = "n/a" if row["tokens_per_second"] is None else "%.1f" % row["tokens_per_second"]
        print("%-4s %-5s %-11s %4d %3d %4d %8s  %s" %
              (row["model"], row["mode"], row["status"], row["prompts_pass"],
               row["prompts_tie"], row["prompts_fail"], tps, row["log"]))


def _split_choice(value: str, allowed: tuple[str, ...], what: str) -> list[str]:
    wanted = {part.strip().lower() for part in value.split(",") if part.strip()}
    unknown = sorted(wanted - set(allowed))
    if unknown:
        raise SystemExit("unknown %s: %s (choose from %s)" % (what, ", ".join(unknown), ", ".join(allowed)))
    if not wanted:
        raise SystemExit("no %s selected" % what)
    return [item for item in allowed if item in wanted]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack-dir", type=Path, default=Path.home() / ".kindi" / "packs",
                    help="installed pack root (default: ~/.kindi/packs)")
    ap.add_argument("--pack-27b", type=Path, default=None,
                    help="27B pack directory (default: <pack-dir>/bonsai-27b)")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated models (default: 2b,4b,9b)")
    ap.add_argument("--modes", default=",".join(MODES),
                    help="comma-separated modes (default: plain,tc)")
    ap.add_argument("--cards", type=int, default=1,
                    help="KINDI_CARDS for every cell (default: 1)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-cell timeout in seconds (default: 900)")
    ap.add_argument("--out", type=Path, default=ROOT / "bench-logs" / "verify-matrix",
                    help="output directory (default: bench-logs/verify-matrix)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and exit without running or writing")
    args = ap.parse_args(argv)

    if args.cards < 1:
        raise SystemExit("--cards must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    models = _split_choice(args.models, MODELS, "model")
    modes = _split_choice(args.modes, MODES, "mode")
    args.pack_dir = args.pack_dir.expanduser().resolve()
    if args.pack_27b is not None:
        args.pack_27b = args.pack_27b.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.dry_run:
        for model in models:
            pack = pack_for(model, args.pack_dir, args.pack_27b)
            for mode in modes:
                env_text = " ".join("%s=%s" % (k, v) for k, v in sorted(mode_env(mode, args.cards).items()))
                command = [sys.executable, "-m", "kindi", "--pack", str(pack), "verify"]
                prefix = "would run" if (pack / "kindi_pack.json").is_file() else "missing pack"
                print("%s: %s %s %s" % (prefix, model, mode, env_text))
                print("  " + shlex.join(command))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(models) * len(modes)
    index = 0
    for model in models:
        pack = pack_for(model, args.pack_dir, args.pack_27b)
        for mode in modes:
            index += 1
            if not (pack / "kindi_pack.json").is_file():
                row = _missing_row(model, mode, pack, args.cards)
                print("[%d/%d] %s %s: missing pack %s" % (index, total, model, mode, pack), flush=True)
            else:
                print("[%d/%d] %s %s: verifying..." % (index, total, model, mode), flush=True)
                row = run_cell(model, mode, pack, args.out, args.cards, args.timeout)
                print("[%d/%d] %s %s: %s" % (index, total, model, mode, row["status"]), flush=True)
            rows.append(row)

    csv_path, json_path = _write_results(args.out, rows, args)
    _print_table(rows)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("\nsummary: " + ", ".join("%s=%d" % (k, counts[k]) for k in sorted(counts)))
    print("wrote %s and %s" % (csv_path, json_path))
    return 0 if counts.get("pass", 0) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())