# SPDX-License-Identifier: AGPL-3.0-only
"""CPU drafting for Bonsai 2.5: the ``-mtp`` head or an external pack, and how often each is right.

Two drafters can be measured against the same greedy run of the 27B on the CPU engine:

  - ``--gguf`` the ``-mtp`` GGUF's 65th block, the integrated head (``kindi.mtp_cpu.Head``),
  - ``--drafter-pack`` a second, smaller converted pack already on this host (the 2B), whose own
    greedy decode supplies the drafts (``kindi.mtp_cpu.PackDrafter``).

Either way the drafts are *scored* against the model's own tokens, never substituted:
``kindi/cpu.py`` verifies one token per forward pass, so a draft cannot shorten a run yet.  This
tool measures what is measurable today: whether the drafts are right, how acceptance falls with
draft depth, and what one drafted token costs on this host.

    python tools/mtp_cpu_bench.py --pack PACK --gguf ../Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf \\
        --out run.json --tokens 24 --k 4
    python tools/mtp_cpu_bench.py --pack PACK --drafter-pack ~/.kindi/packs/bonsai-2b --tokens 24

The decode rates are facts of one run on one host, not a tuning comparison; take them on an idle
machine.  ``--k`` drafts per step; the head's window is ``--window`` positions (default 64).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kindi import cpu, mtp_cpu, tokenizer  # noqa: E402

PROMPTS = ("In one sentence: what is a ternary weight?",
           "2 + 2 = ? Answer with the number only.")


def _load_tok(pack):
    with open(os.path.join(pack, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    return tokenizer.Tok(vocab, os.path.join(pack, "merges.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gguf", default=None, help="the -mtp GGUF the integrated head is read from")
    ap.add_argument("--drafter-pack", default=None,
                    help="an external drafter: a smaller converted pack (2B/4B) on the same CPU engine")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if bool(a.gguf) == bool(a.drafter_pack):
        ap.error("give exactly one drafter: --gguf (the integrated head) or --drafter-pack (an external pack)")

    def log(*m):
        print(*m, flush=True)

    t0 = time.time()
    eng = cpu.Engine(a.pack, ctx=a.ctx, threads=a.threads, log=log)
    load_s = time.time() - t0
    if a.drafter_pack:
        drafter = mtp_cpu.PackDrafter(a.drafter_pack, ctx=a.ctx, threads=a.threads, k=a.k, log=log)
        desc = {"kind": "pack", "path": os.path.abspath(a.drafter_pack),
                "model": os.path.basename(os.path.normpath(a.drafter_pack))}
    else:
        drafter = mtp_cpu.Head(a.gguf, eng, k=a.k, window=a.window, log=log)
        desc = {"kind": "mtp-head", "path": os.path.abspath(a.gguf), "block": drafter.dims["block"],
                "bytes": drafter.bytes, "dims": drafter.dims}
    tk = _load_tok(a.pack)
    runs = []
    for text in (a.prompt or PROMPTS):
        ids = tk.encode(text)
        t = time.time()
        plain = eng.generate(ids, a.tokens)
        plain_s = time.time() - t
        t = time.time()
        scored, stats = mtp_cpu.score_drafts(eng, ids, a.tokens, drafter, k=a.k)
        scored_s = time.time() - t
        rows = int(drafter.stats.get("rows", 0))
        runs.append({"prompt": text, "ids": ids, "plain_ids": plain, "scored_ids": scored,
                     "plain_text": tk.decode(plain), "scored_text": tk.decode(scored),
                     "exact": scored == plain, "plain_s": plain_s, "scored_s": scored_s,
                     "plain_tok_s": len(plain) / plain_s, "scored_tok_s": len(scored) / scored_s,
                     "drafts": stats, "drafter_rows": rows,
                     "drafter_ms_per_row": (1000.0 * scored_s / rows) if rows else None})
        log("%-46s exact=%s  %.2f -> %.2f tok/s  accepted %d/%d (%.0f%%)" % (
            text[:46], runs[-1]["exact"], runs[-1]["plain_tok_s"], runs[-1]["scored_tok_s"],
            stats["matched"], stats["offered"], 100.0 * stats["rate"]))
        log("  reply: %r" % (runs[-1]["plain_text"][:120],))

    offered = sum(r["drafts"]["offered"] for r in runs)
    matched = sum(r["drafts"]["matched"] for r in runs)
    depth = [sum(r["drafts"]["depth"][i] for r in runs if i < len(r["drafts"]["depth"]))
             for i in range(max([len(r["drafts"]["depth"]) for r in runs] or [0]))]
    desc["rows"] = int(drafter.stats.get("rows", 0))
    doc = {"pack": os.path.abspath(a.pack), "drafter": desc, "tokens": a.tokens, "k": a.k,
           "window": a.window if desc["kind"] == "mtp-head" else None, "ctx": a.ctx, "threads": eng.threads,
           "load_s": load_s, "accepted": matched, "offered": offered,
           "rate": (matched / float(offered)) if offered else 0.0, "depth": depth, "runs": runs}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
        log("wrote %s" % a.out)
    log("drafter %s %s, accepted %d/%d (%.0f%%), depth %s" % (
        desc["kind"], desc.get("model") or ("blk.%s" % desc.get("block")), matched, offered,
        100.0 * doc["rate"], depth))
    if not all(r["exact"] for r in runs):
        log("FAIL: the scored run's tokens differ from generate()")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
