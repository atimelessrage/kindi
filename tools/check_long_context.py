#!/usr/bin/env python3
"""Run one `kindi bench` at a deep context and verify the requested cell exists.

The GPU CI lane needs a check that is stronger than "the process exited 0": a
pack can be clamped to a smaller context, or the deepest benchmark position can
be absent, while the command still succeeds. This tool parses the bench log and
fails unless:

* the engine reports at least the requested context;
* the deepest benchmark position below that context was measured;
* its generation throughput is greater than zero.

It writes the raw log and a small JSON result for the workflow artifact.

Example (on the self-hosted GPU runner):
  python tools/check_long_context.py --pack /data/kindi/bonsai-2b --ctx 24576
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = (100, 2000, 8000, 16000, 24000)
CONTEXT_RE = re.compile(r"with a (\d+)-token context")
READ_RE = re.compile(r"prompt reading:\s+([0-9]+(?:\.[0-9]+)?) tokens/s")
GEN_RE = re.compile(
    r"generation \((?:exact|fast) head\) at position\s+(\d+):\s+"
    r"([0-9]+(?:\.[0-9]+)?) tokens/s"
)

# Keep the benchmark path deterministic and independent of the caller's shell.
BENCH_ENV = {
    "KINDI_DEVICE": "gpu",
    "KINDI_HEAD": "exact",
    "KINDI_KV": "q8_0",
    "KINDI_SPEC": "0",
    "KINDI_TP": "0",
    "KINDI_TC": "0",
    "KINDI_FD": "0",
    "KINDI_ATT_TC": "0",
    "KINDI_CARDS": "1",
}


def required_position(ctx: int) -> int | None:
    """The deepest bench position that should fit in `ctx`."""
    positions = [position for position in POSITIONS if position + 20 < ctx]
    return max(positions) if positions else None


def parse_bench_output(text: str, ctx: int) -> dict:
    """Parse one bench log and classify whether the long-context cell is valid."""
    contexts = [int(match.group(1)) for match in CONTEXT_RE.finditer(text)]
    loaded_context = max(contexts) if contexts else None
    reads = [float(match.group(1)) for match in READ_RE.finditer(text)]
    generation = {
        int(match.group(1)): float(match.group(2))
        for match in GEN_RE.finditer(text)
    }
    required = required_position(ctx)
    required_tps = generation.get(required) if required is not None else None

    if loaded_context is None:
        status = "no-context"
    elif loaded_context < ctx:
        status = "context-clamped"
    elif required is None:
        status = "no-position"
    elif required not in generation:
        status = "missing-position"
    elif required_tps is None or required_tps <= 0:
        status = "zero-throughput"
    else:
        status = "pass"

    return {
        "status": status,
        "requested_context": ctx,
        "loaded_context": loaded_context,
        "required_position": required,
        "required_tps": required_tps,
        "read_tps": reads[-1] if reads else None,
        "generation": {str(position): tps for position, tps in sorted(generation.items())},
    }


def run_bench(pack: Path, ctx: int, timeout: float, python: str = sys.executable) -> dict:
    """Run one fresh `kindi bench` and return its parsed result."""
    command = [python, "-m", "kindi", "--pack", str(pack), "--ctx", str(ctx), "bench"]
    env = {key: value for key, value in os.environ.items() if not key.startswith("KINDI_")}
    env.update(BENCH_ENV)
    env["PYTHONUNBUFFERED"] = "1"

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

    parsed = parse_bench_output(output, ctx)
    status = parsed["status"]
    if exit_code == 124:
        status = "timeout"
    elif exit_code not in (0, None) and status == "pass":
        status = "fail"
    return {
        "pack": str(pack),
        "ctx": ctx,
        "status": status,
        "exit_code": exit_code,
        "duration_s": round(duration, 1),
        **{key: value for key, value in parsed.items() if key != "status"},
        "command": command,
        "output": output,
    }


def _write_result(out_dir: Path, row: dict, output: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bench.log"
    json_path = out_dir / "long-context.json"
    log_path.write_text(output, encoding="utf-8", newline="\n")
    payload = {
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_revision": _git_revision(),
        "pack": row["pack"],
        "ctx": row["ctx"],
        "status": row["status"],
        "loaded_context": row["loaded_context"],
        "required_position": row["required_position"],
        "required_tps": row["required_tps"],
        "read_tps": row["read_tps"],
        "generation": row["generation"],
        "duration_s": row["duration_s"],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return log_path, json_path


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _positive_int(text: str) -> int:
    """A positive integer, rejected at argparse time before the bench runs."""
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive integer (got %r)" % (text,))
    return value


def _positive_float(text: str) -> float:
    """A finite positive timeout, rejected at argparse time."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number (got %r)" % (text,))
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", type=Path, required=True,
                    help="converted pack directory on the GPU runner")
    ap.add_argument("--ctx", type=_positive_int, default=24576,
                    help="context requested from the engine, a positive integer (default: 24576)")
    ap.add_argument("--timeout", type=_positive_float, default=1800.0,
                    help="finite positive bench timeout in seconds (default: 1800)")
    ap.add_argument("--out", type=Path, default=ROOT / "bench-logs" / "ci-long-context",
                    help="output directory (default: bench-logs/ci-long-context)")
    args = ap.parse_args(argv)
    if required_position(args.ctx) is None:
        raise SystemExit("--ctx %d has no benchmark position" % args.ctx)

    row = run_bench(args.pack.expanduser().resolve(), args.ctx, args.timeout)
    log_path, json_path = _write_result(args.out, row, row.pop("output"))
    loaded = row["loaded_context"] if row["loaded_context"] is not None else "not reported"
    tps = "n/a" if row["required_tps"] is None else "%.1f" % row["required_tps"]
    read = "n/a" if row["read_tps"] is None else "%.1f" % row["read_tps"]
    print("long context %s: %s" % (args.pack, row["status"]))
    print("  requested %d tokens, engine reports %s; position %s: %s tok/s; read: %s tok/s" %
          (args.ctx, loaded, row["required_position"], tps, read))
    print("  wrote %s and %s" % (log_path, json_path))
    return 0 if row["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
