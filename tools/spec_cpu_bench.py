"""Plain CPU decode against k-draft batched verification, on the same prompts.

Roadmap CP1: draft with a smaller pack, confirm with one T-row pass (kindi/mtp_cpu.py:
speculative_decode, kindi/kcpu.c: kc_step_batch).  The evidence page is
docs/benchmarks/2026-11-26-i7-8086k-27b-cpu-batch-verify/; a run writes its own JSON with --out.

For every prompt: plain greedy decode (one weight pass per token) and speculative decode with
k drafts confirmed by one T-row pass, both timed on the same engine, with the two token lists
compared.  The acceptance rate, the batch rows and the wall time inside the batch and the rollback
are recorded, because the speedup is a function of all of them.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kindi import cpu, mtp_cpu, tokenizer  # noqa: E402

PROMPTS = ("In one sentence: what is a ternary weight?",
           "2 + 2 = ? Answer with the number only.",
           "Name three colours and stop.",
           "Write a four-word sentence about the sea.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--drafter-pack", required=True)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--replay-drafter", action="store_true", help="replay the prefix every round (the old drafter)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(os.path.join(a.pack, "vocab.json"), encoding="utf-8") as f:
        tok = tokenizer.Tok(json.load(f), os.path.join(a.pack, "merges.txt"))
    eng = cpu.Engine(a.pack, ctx=a.ctx, threads=a.threads, log=lambda *m: print(*m))
    drafter = mtp_cpu.PackDrafter(a.drafter_pack, ctx=a.ctx, threads=a.threads, k=a.k,
                                  log=lambda *m: print(*m), incremental=not a.replay_drafter)
    runs = []
    for text in PROMPTS[:a.prompts]:
        ids = tok.encode(text)
        t0 = time.time()
        plain = eng.generate(ids, a.tokens)
        plain_s = time.time() - t0
        t0 = time.time()
        spec, stats = mtp_cpu.speculative_decode(eng, ids, a.tokens, drafter, k=a.k)
        spec_s = time.time() - t0
        row = {"prompt": text, "ids": ids, "plain_ids": plain, "spec_ids": spec,
               "exact": plain == spec, "plain_s": plain_s, "spec_s": spec_s,
               "plain_tok_s": len(plain) / plain_s, "spec_tok_s": len(spec) / spec_s,
               "generated": {"plain": len(plain), "spec": len(spec)},
               "speedup": plain_s / spec_s, "k": a.k, "stats": stats}
        runs.append(row)
        print("%-44s exact=%s plain %.2f tok/s, spec %.2f tok/s (%.2fx), accept %d/%d, rows %d, batch %.0f ms"
              % (text[:43], row["exact"], row["plain_tok_s"], row["spec_tok_s"], row["speedup"],
                 stats["matched"], stats["offered"], stats["rows"], stats["batch_ms"]))
    ok = [r for r in runs if r["exact"]]
    print("%d/%d prompts token-exact; mean speedup %.2fx; acceptance %d/%d (%.1f%%)"
          % (len(ok), len(runs), sum(r["speedup"] for r in runs) / len(runs),
             sum(r["stats"]["matched"] for r in runs), sum(r["stats"]["offered"] for r in runs),
             100.0 * sum(r["stats"]["matched"] for r in runs) / max(1, sum(r["stats"]["offered"] for r in runs))))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"pack": os.path.basename(a.pack), "drafter": os.path.basename(a.drafter_pack),
                       "tokens": a.tokens, "k": a.k, "threads": a.threads, "ctx": a.ctx,
                       "incremental_drafter": not a.replay_drafter, "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
