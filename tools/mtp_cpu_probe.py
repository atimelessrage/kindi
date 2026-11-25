# SPDX-License-Identifier: AGPL-3.0-only
"""Try several MTP-head variants against one loaded 27B: the fast loop for the head's acceptance.

Loading the 27B pack costs about six minutes and the decode is one row per forward pass, so the
expensive part is the model, not the attempt. This tool loads the pack and the head once, then runs
every command in ``--commands`` through ``kindi.mtp_cpu.score_drafts`` and appends one JSON line per
prompt per attempt. Changing a head-side choice (today: the ``eh_proj`` input order, ``eh`` =
[embedding, hidden] like the current code, ``he`` = the reverse) costs an attempt, not a reload.

    python tools/mtp_cpu_probe.py --pack PACK --gguf MMTP.gguf --commands cmds.jsonl --out out.jsonl

A command is one JSON object per line::

    {"variant": "eh", "k": 4, "tokens": 16, "prompts": ["..."]}

Nothing here is a support or speed claim: the scored tokens are asserted equal to plain greedy decode
(the head is scored, never substituted), and the acceptance rate is what the attempt measures.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kindi import cpu, mtp_cpu, tokenizer  # noqa: E402

VARIANTS = ("eh", "he", "eh-normed", "he-normed", "eh-folded", "eh-normed-folded", "eh-normed-rope1",
            "eh-normed-seeded", "eh-seeded")


def _tok(pack):
    with open(os.path.join(pack, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    return tokenizer.Tok(vocab, os.path.join(pack, "merges.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gguf", required=True, help="the -mtp GGUF the head is read from, once")
    ap.add_argument("--commands", required=True, help="JSONL, one attempt per line")
    ap.add_argument("--out", default=None, help="JSONL results (also printed)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--window", type=int, default=64)
    a = ap.parse_args()

    def log(*m):
        print(*m, flush=True)

    cmds = []
    with open(a.commands, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cmds.append(json.loads(line))
    for c in cmds:
        if c.get("variant", "eh") not in VARIANTS:
            ap.error("variant must be one of %s" % ", ".join(VARIANTS))

    t0 = time.time()
    eng = cpu.Engine(a.pack, ctx=a.ctx, threads=a.threads, log=log)
    log("engine loaded in %.0fs; %d attempt(s) queued" % (time.time() - t0, len(cmds)))
    head = mtp_cpu.Head(a.gguf, eng, k=cmds[0].get("k", 4), window=a.window, log=log)
    tk = _tok(a.pack)
    plain = {}                                    # prompt -> (ids, tokens, seconds): variant-free

    out = open(a.out, "w", encoding="utf-8") if a.out else None
    seen_drafts = []
    _orig_draft = head.draft

    def _recording_draft(prefix, hin, k=None):
        d = _orig_draft(prefix, hin, k)
        seen_drafts.extend(int(t) for t in d)
        return d

    head.draft = _recording_draft
    try:
        for attempt in cmds:
            variant = attempt.get("variant", "eh")
            k = int(attempt.get("k", 4))
            tokens = int(attempt.get("tokens", 16))
            head.order = variant.split("-")[0]
            head.hidden = "normed" if "normed" in variant else "raw"
            head.emb_mode = "folded" if "folded" in variant else "raw"
            head.rope_off = 1 if "rope1" in variant else 0
            head.seed_enabled = "seeded" in variant
            head.k = k
            rows = []
            for text in attempt.get("prompts") or []:
                ids = tk.encode(text)
                if text not in plain:
                    t = time.time()
                    got = eng.generate(ids, tokens)
                    plain[text] = (ids, got, time.time() - t)
                ids, want, plain_s = plain[text]
                t = time.time()
                seen_drafts.clear()
                scored, stats = mtp_cpu.score_drafts(eng, ids, tokens, head, k=k)
                scored_s = time.time() - t
                row = {"variant": variant, "k": k, "tokens": tokens, "prompt": text,
                       "exact": list(scored) == list(want), "plain_s": round(plain_s, 3),
                       "scored_s": round(scored_s, 3),
                       "plain_tok_s": round(len(want) / plain_s, 3),
                       "scored_tok_s": round(len(scored) / scored_s, 3),
                       "accepted": stats["matched"], "offered": stats["offered"],
                       "rate": round(stats["rate"], 4), "depth": stats["depth"],
                       "draft_tokens": list(seen_drafts[:40]),
                       "model_tokens": [int(t) for t in want]}
                rows.append(row)
                log(json.dumps(row, ensure_ascii=False))
                if out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
            if rows:
                acc = sum(r["accepted"] for r in rows)
                off = sum(r["offered"] for r in rows)
                log("ATTEMPT variant=%s k=%d: %d/%d (%.1f%%), exact=%s"
                    % (variant, k, acc, off, 100.0 * acc / off if off else 0.0,
                       all(r["exact"] for r in rows)))
    finally:
        if out:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
