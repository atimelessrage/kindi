"""Does the CPU engine's batched T-row pass reproduce plain decode - tokens, logits, and after a rollback?

Roadmap CP1 in docs/ROADMAP.md.  The pass itself is kindi/kcpu.c's kc_step_batch (one weight stream
for T rows); the evidence page is docs/benchmarks/2026-11-26-i7-8086k-27b-cpu-batch-verify/.

One engine, three runs from the same prefix:
  1. plain decode, one row at a time: the reference tokens and logits;
  2. the same tokens in batches of T: every row's argmax and full logit vector compared to (1);
  3. a batch of T rows, then kc_batch_rollback(keep), then plain rows: the state must land exactly
     where plain decode would have it, which is what a verifier needs after rejecting drafts.
Also times (1) and (2) so the rows/s cost of the T-row pass is visible.
"""
import argparse
import json
import os
import time

import numpy as np

from kindi import cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ids", default="9707,11,1879,3251,614,13,263,1758,570,220,17,79,1069,13")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=None, help="write the run's numbers as JSON")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    eng = cpu.Engine(a.pack, ctx=a.ctx, threads=a.threads, log=lambda *m: print(*m))
    start = len(ids) - 1                      # the position of the first row below

    def prefix():
        eng.reset()
        for i, t in enumerate(ids[:start]):
            eng.feed(t, i)

    # 1. reference
    prefix()
    seq, ref_tok, ref_lg = list(ids), [], []
    t0 = time.time()
    tok, pos = ids[-1], start
    for _ in range(a.rows + 1):
        tok = eng.step(tok, pos)
        ref_tok.append(tok)
        ref_lg.append(eng.logits.copy())
        seq.append(tok)
        pos += 1
    plain_s = time.time() - t0
    print("plain: %d rows in %.2fs (%.2f rows/s)" % (len(ref_tok), plain_s, len(ref_tok) / plain_s))

    # 2. batched rows
    prefix()
    got, dmax, p, done, runs = [], 0.0, start, 0, []
    t0 = time.time()
    while done < a.rows:
        n = min(a.batch, a.rows - done)
        runs.append(n)
        out = eng.step_batch(seq[p:p + n], p)
        for r in range(n if n > 1 else 0):     # a 1-row trailing pass runs the plain path
            dmax = max(dmax, float(np.max(np.abs(eng.batch_logits(r) - ref_lg[done + r]))))
        got += out
        p += n
        done += n
    batch_s = time.time() - t0
    same = got == ref_tok[:a.rows]
    print("batch: %d rows in %.2fs (%.2f rows/s), batches %s" % (len(got), batch_s, len(got) / batch_s, runs))
    print("tokens equal:", same, "| max |logit delta|: %.3e" % dmax)
    if not same:
        for i, (g, r) in enumerate(zip(got, ref_tok, strict=False)):
            if g != r:
                print("  row %d: batch %d, plain %d" % (i, g, r))
    gain = (len(got) / batch_s) / (len(ref_tok) / plain_s)
    print("rows/s: %.2fx plain" % gain)

    # 3. batch, reject the rows past `keep`, then plain rows must equal the reference
    prefix()
    keep = max(1, min(a.keep, a.batch - 1))
    eng.step_batch(seq[start:start + a.batch], start)
    eng.batch_rollback(keep)
    tail = []
    t0 = time.time()
    for i in range(keep, a.rows):
        pos = start + i
        tail.append(eng.step(seq[pos], pos))
    after_s = time.time() - t0
    want = ref_tok[keep:a.rows]
    print("rollback to %d rows, then %d plain rows: %s (%.2fs)"
          % (keep, len(tail), "equal" if tail == want else "DIFFERENT", after_s))
    if tail != want:
        for i, (g, r) in enumerate(zip(tail, want, strict=False)):
            if g != r:
                print("  row %d: %d, plain %d" % (keep + i, g, r))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"pack": os.path.basename(a.pack), "rows": a.rows, "batch_size": a.batch, "keep": keep,
                       "ctx": a.ctx, "threads": a.threads,
                       "plain": {"rows": len(ref_tok), "s": plain_s, "rows_s": len(ref_tok) / plain_s},
                       "batch": {"rows": len(got), "s": batch_s, "rows_s": len(got) / batch_s,
                                 "sizes": runs, "tokens_equal": same, "max_logit_delta": dmax},
                       "rows_s_gain": gain,
                       "batch_rows_cost_in_plain_rows": (1.0 / gain if gain else None),
                       "logit_rows_compared": sum(n for n in runs if n > 1),
                       "rollback": {"keep": keep, "plain_rows": len(tail), "s": after_s,
                                    "equal": tail == want}}, f, indent=1)


if __name__ == "__main__":
    main()
