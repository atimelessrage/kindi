"""How long a Bonsai pack takes to load, and how much of that is the tri-code decoder.

A pack stores every ternary matrix as base-3 codes, four trits per byte; `kindi.cpu.paircodes` turns
them into the uint32 words the CPU engine reads. It was 96% of a pack's whole load (measured with
cProfile on the 2B: 20.3 s of 21.2 s), which is why the 2.5 27B's documented CPU load is 339.9 s.

The decoder now maps each byte through one 256-entry table instead of deriving four trits and
gathering a uint32 code per column pair. This tool measures that on the packs available here: it
loads each pack once with the frozen reference decoder from `tests/test_cpu_paircodes.py` and once
with the shipping one, in the same process, and reports best-of-N seconds and the ratio.

    python tools/cpu_pack_load_check.py                      # every pack in ~/.kindi/packs
    python tools/cpu_pack_load_check.py --pack ~/.kindi/packs/bonsai-9b --runs 3 --out loads.json

The 27B pack is not on this host, so its load is projection only and is not measured here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kindi import cpu  # noqa: E402  (after the path fix so `python tools/...` works)


def reference_paircodes(codes, R, C, chunk=4096):
    """The decoder `cpu.paircodes` replaced: per-trit `% 3` / `// 3`, one uint32 gather per pair."""
    G = C // 128
    out = np.empty((R // 8, G, 64), dtype=np.uint32)
    codes = codes.reshape(C // 4, R)
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        b = np.asarray(codes[:, r0:r1]).T.astype(np.uint8)
        t = np.stack([b % 3, b // 3 % 3, b // 9 % 3, b // 27 % 3], axis=2).reshape(r1 - r0, G, 64, 2)
        pc = cpu._CODE[t[..., 0].astype(np.uint32) + 3 * t[..., 1].astype(np.uint32)]
        pc = pc.reshape((r1 - r0) // 8, 8, G, 64)
        word = np.zeros(((r1 - r0) // 8, G, 64), dtype=np.uint32)
        for lane in range(8):
            word |= pc[:, lane] << np.uint32(4 * lane)
        out[r0 // 8:r1 // 8] = word
    return out


def _pack_mb(path):
    total = 0
    for name in os.listdir(path):
        f = os.path.join(path, name)
        if os.path.isfile(f):
            total += os.path.getsize(f)
    return total / 1e6


def timed_load(pack, ctx, threads, runs, decoder=None):
    """Best-of-`runs` load seconds, optionally with `cpu.paircodes` swapped for `decoder`."""
    original = cpu.paircodes
    if decoder is not None:
        cpu.paircodes = decoder
    try:
        best = float("inf")
        for _ in range(runs):
            t0 = time.time()
            eng = cpu.Engine(pack, ctx=ctx, threads=threads, log=None)
            dt = time.time() - t0
            eng.close()
            best = min(best, dt)
        return round(best, 2)
    finally:
        cpu.paircodes = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="append", default=None, help="pack directory; repeatable")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default=None, help="write the run's numbers as JSON")
    a = ap.parse_args()

    root = os.path.expanduser("~/.kindi/packs")
    packs = a.pack or [os.path.join(root, n) for n in sorted(os.listdir(root))
                       if os.path.isfile(os.path.join(root, n, "kindi_pack.json"))]
    packs = [p for p in packs if os.path.isfile(os.path.join(p, "kindi_pack.json"))]
    if not packs:
        raise SystemExit("no packs with kindi_pack.json found")

    timed_load(packs[0], a.ctx, a.threads, 1)              # warm the compiled library and the page cache
    rows, isa = [], None
    for pack in packs:
        before = timed_load(pack, a.ctx, a.threads, a.runs, reference_paircodes)
        after = timed_load(pack, a.ctx, a.threads, a.runs)
        eng = cpu.Engine(pack, ctx=a.ctx, threads=a.threads, log=None)
        isa = getattr(eng.lib, "kindi_isa", None) or isa
        eng.close()
        mb = _pack_mb(pack)
        row = {"pack": os.path.basename(pack), "pack_mb": round(mb, 1), "before_s": before,
               "after_s": after, "speedup": round(before / after, 2) if after else None,
               "mb_per_s_after": round(mb / after, 0) if after else None}
        rows.append(row)
        print("%-16s %7.1f MB  %7.2f s -> %6.2f s  %4.2fx  (%.0f MB/s)" % (
            row["pack"], mb, before, after, row["speedup"] or 0, row["mb_per_s_after"] or 0))

    out = {"measured": time.strftime("%Y-%m-%d"), "isa": isa, "ctx": a.ctx, "threads": a.threads,
           "runs": a.runs, "packs": rows,
           "note": "before = the frozen per-trit decoder in tests/test_cpu_paircodes.py, after = kindi.cpu.paircodes"}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
