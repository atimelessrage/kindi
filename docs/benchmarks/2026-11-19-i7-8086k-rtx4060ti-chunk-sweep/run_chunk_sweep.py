#!/usr/bin/env python3
"""KINDI_CHUNK prefill sweep: 2B, 4K/8K prompts, several chunk sizes, three passes each.

Each pass is a fresh `python -m kindi bench --split-at 4096,8192 --split-generated 128
--sample-vram` process with `KINDI_CHUNK` set and `KINDI_CTX` sized so the 8K request fits.
The bench JSON already carries one timed request per prompt length: a real `feed_batch`
prefill followed by 128 greedy steps, both synced. This script only aggregates the passes:
p50/p95 are linear-interpolation percentiles over them (numpy's default definition), and
the peak is the largest sampled device-wide `memory.used` of the passes.

Usage:
  python run_chunk_sweep.py --pack ~/.kindi/packs/bonsai-2b --out .
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

GENERATED = 128
ROOT = Path(__file__).resolve().parents[3]


def percentile(values, q):
    """Linear-interpolation percentile (the numpy default), with no numpy import."""
    xs = sorted(values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def run_pass(pack, chunk, pass_no, prompts, out_dir, log_dir):
    ctx = max(prompts) + GENERATED
    log_path = log_dir / ("chunk-%d-r%d.log" % (chunk, pass_no))
    json_path = log_dir / ("chunk-%d-r%d.json" % (chunk, pass_no))
    env = dict(os.environ)
    env.update({"KINDI_PACK": str(pack), "KINDI_CTX": str(ctx), "KINDI_CHUNK": str(chunk)})
    cmd = [sys.executable, "-m", "kindi", "bench",
           "--split-at", ",".join(str(p) for p in prompts),
           "--split-generated", str(GENERATED),
           "--sample-vram", "--json", str(json_path)]
    print("chunk %4d pass %d: ctx %d -> %s" % (chunk, pass_no, ctx, log_path.name), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise SystemExit("chunk %d pass %d exited %d; see %s"
                         % (chunk, pass_no, proc.returncode, log_path))
    # tracked copies carry no machine path: the pack directory and the local home become
    # placeholders, and the bench JSON's `pack` field keeps only the pack's name
    home = str(Path.home())
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for raw in (str(pack), home.replace("\\", "/"), home):
        text = text.replace(raw, "<pack>" if raw == str(pack) else "<local-home>")
    log_path.write_text(text, encoding="utf-8")
    result = json.loads(json_path.read_text(encoding="utf-8"))
    result["pack"] = pack.name
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--chunks", default="128,256,512,1024")
    ap.add_argument("--prompts", default="4096,8192")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    pack = Path(os.path.expanduser(a.pack))
    chunks = [int(x) for x in a.chunks.split(",") if x.strip()]
    prompts = [int(x) for x in a.prompts.split(",") if x.strip()]
    out_dir = Path(a.out).resolve()
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    raw = {}
    for chunk in chunks:
        passes = []
        for pass_no in range(1, a.passes + 1):
            result = run_pass(pack, chunk, pass_no, prompts, out_dir, log_dir)
            split = result.get("split", {})
            missing = [p for p in prompts if str(p) not in split]
            if missing:
                raise SystemExit("chunk %d pass %d did not measure prompt(s) %s"
                                 % (chunk, pass_no, missing))
            vram = result.get("vram_peak") or {}
            passes.append({
                "pass": pass_no,
                "split": {str(p): split[str(p)] for p in prompts},
                "vram_peak": vram if vram.get("available") else {"available": False},
            })
        raw[str(chunk)] = passes

    summary = {"generated_tokens": GENERATED, "chunks": {}, "passes": a.passes,
               "prompts": prompts,
               "pack": pack.name, "host_pack_note": "pack path reduced to its name"}
    for chunk in chunks:
        passes = raw[str(chunk)]
        rows = {}
        for prompt in prompts:
            key = str(prompt)
            prefill = [p["split"][key]["prefill_ms"] for p in passes]
            total = [p["split"][key]["prefill_ms"] + p["split"][key]["decode_ms"] for p in passes]
            peaks = [p["vram_peak"].get("peak_used_mib") for p in passes
                     if p["vram_peak"].get("peak_used_mib") is not None]
            deltas = [p["vram_peak"].get("peak_delta_mib") for p in passes
                      if p["vram_peak"].get("peak_delta_mib") is not None]
            rows[key] = {
                "prefill_ms_p50": round(percentile(prefill, 50), 2),
                "prefill_ms_p95": round(percentile(prefill, 95), 2),
                "total_ms_p50": round(percentile(total, 50), 2),
                "total_ms_p95": round(percentile(total, 95), 2),
                "prefill_tok_s_p50": round(prompt / (percentile(prefill, 50) / 1000.0), 1),
                "prefill_tok_s_p95": round(prompt / (percentile(prefill, 95) / 1000.0), 1),
                "vram_peak_used_mib": max(peaks) if peaks else None,
                "vram_peak_delta_mib": max(deltas) if deltas else None,
                "passes": [{"pass": p["pass"],
                            "prefill_ms": p["split"][key]["prefill_ms"],
                            "decode_ms": p["split"][key]["decode_ms"],
                            "prefill_share": p["split"][key]["prefill_share"],
                            "peak_used_mib": p["vram_peak"].get("peak_used_mib")}
                           for p in passes],
            }
        summary["chunks"][str(chunk)] = rows

    payload = {"command": "python run_chunk_sweep.py --pack <pack> --out .",
               "note": ("p50/p95 over %d passes of one timed request per prompt (prefill + %d "
                        "greedy steps); peak is the largest sampled device-wide nvidia-smi "
                        "memory.used in the passes" % (a.passes, GENERATED)),
               "raw": raw, "summary": summary}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
