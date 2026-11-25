# SPDX-License-Identifier: AGPL-3.0-only
"""Repeatable benchmark of the Bonsai 2.5 27B publisher sampling settings, on one pack.

Prism ML's card for Ternary Bonsai 2.5 27B gives one sampling set per mode:

    thinking: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence 0.0, repetition 1.0
    instruct: temperature 0.7, top_p 0.80, top_k 20, min_p 0.0, presence 1.5, repetition 1.0

and `xhigh` as the default reasoning effort, `medium` for shorter answers. `low` is not supported:
the card says the model then behaves close to `xhigh`, so Kindi renders the `xhigh` instruction for
it too (kindi/chat.py).

This tool requests exactly those values through the same code the server uses
(``kindi.profiles.sampling_defaults`` with ``sampling_profile = publisher``), renders the 27B chat
template, and samples with ``kindi.sampling.Sampler`` on the engine that is loaded. It works on the
CPU engine (no CUDA needed) and on a card; it records what each run used, so the JSON is the
evidence and no number has to be retyped.

The decode rate it prints is a fact of that run, not a performance measurement: a busy host makes it
unfit for comparison with another run, mode, pack or machine. Use the ids, the settings and the text
as the check that the profile applies and the engine answers, and take any rate on an idle host.

    KINDI_DEVICE=cpu python tools/bonsai_sampling_bench.py --pack PACK --out run.json --tokens 32

The pack may be the plain `-mtp`-less Bonsai 2.5 27B or the `-mtp` file's pack; the tool reads the
model id from the manifest and refuses a pack that is not Bonsai 2.5.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

PROMPTS = (
    "In one sentence: what is a ternary weight?",
    "2 + 2 = ? Answer with the number only.",
)
EFFORTS = ("xhigh", "medium", "low")


def fingerprint(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()[:16]


def combos(efforts=EFFORTS):
    """(mode, reasoning effort or None) for the publisher table: instruct once, thinking per effort."""
    out = [("instruct", None)]
    out += [("thinking", e) for e in efforts]
    return out


def settings(profile_id, thinking):
    """The profile's sampling values, through the server's own resolver (sampling_profile=publisher)."""
    from kindi import profiles

    cfg = {"model_name": profile_id, "sampling_profile": "publisher", "explicit": frozenset()}
    for k, v in (("max_tokens", 8192), ("temperature", 0.7), ("top_p", 1.0), ("top_k", 0),
                 ("min_p", 0.0), ("repeat_penalty", 1.0), ("repeat_last_n", 64), ("presence_penalty", 0.0)):
        cfg[k] = v
    sd = profiles.sampling_defaults(cfg, thinking)
    return {k: sd[k] for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                               "repeat_last_n", "presence_penalty")}


def _sample(eng, prompt_ids, n, p, seed):
    """Greedy-shaped loop with kindi.sampling.Sampler: replay the prompt, then sample n tokens."""
    import numpy as np

    from kindi.sampling import Sampler

    if not prompt_ids or len(prompt_ids) + n + 1 >= eng.ctx:
        raise SystemExit("kindi bench: prompt %d + %d tokens does not fit ctx %d"
                         % (len(prompt_ids), n + 1, eng.ctx))
    sampler = Sampler(eng)
    rng = getattr(np.random, "RandomState")(seed)   # noqa: B009 - pylint's numpy stubs miss RandomState
    eng.reset()
    for i, t in enumerate(prompt_ids[:-1]):
        eng.feed(t, i)
    pos = len(prompt_ids)
    greedy = eng.step(prompt_ids[-1], pos - 1)
    vocab = int(getattr(eng, "ylog_n", 0))
    rlast = int(p["repeat_last_n"] or 0)
    rp = float(p["repeat_penalty"] or 1.0)
    pp = float(p["presence_penalty"] or 0.0)
    hist = list(prompt_ids)
    out = []
    presence_mask = None
    t0 = time.perf_counter()
    for _ in range(n):
        penalty = None
        if rp != 1.0 and rlast:
            penalty = (rp, hist[-rlast:])
        presence = None
        if pp:
            if presence_mask is None:
                presence_mask = np.zeros(vocab, dtype=bool)
            for t in hist:
                if 0 <= t < vocab:
                    presence_mask[t] = True
            presence = (pp, presence_mask)
        tok = int(sampler(greedy, float(p["temperature"]), float(p["top_p"]), int(p["top_k"]), rng,
                          float(p["min_p"]), penalty, presence))
        out.append(tok)
        hist.append(tok)
        greedy = eng.step(tok, pos)
        pos += 1
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description="benchmark the Bonsai 2.5 27B publisher sampling settings")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--efforts", default=",".join(EFFORTS))
    args = ap.parse_args()

    from kindi import chat, tokenizer
    from kindi.backends.bonsai import model_id_for_manifest
    from kindi.engine import make_engine
    from kindi.mtp_heads import report as spec_report

    man = json.load(open(os.path.join(args.pack, "kindi_pack.json"), encoding="utf-8"))
    profile_id = model_id_for_manifest(man)
    if profile_id != "bonsai2.5-27b":
        raise SystemExit("kindi bench: this tool benchmarks the Bonsai 2.5 27B profile, not %r (%s)"
                         % (profile_id, man.get("source_file")))
    said = []
    t0 = time.perf_counter()
    eng = make_engine(args.pack, ctx=args.ctx, log=said.append)
    load_s = time.perf_counter() - t0
    tok = tokenizer.Tok(eng.vocab, os.path.join(args.pack, "merges.txt"))
    variant = chat.template_variant(man)

    efforts = tuple(e for e in args.efforts.split(",") if e)
    rows = []
    for mode, effort in combos(efforts):
        thinking = mode == "thinking"
        p = settings(profile_id, thinking)
        for pi, prompt in enumerate(PROMPTS):
            body, gp = chat.render([{"role": "user", "content": prompt}], variant=variant,
                                   enable_thinking=thinking, reasoning_effort=effort)
            ids = tok.encode(body) + tok.encode(gp)
            row_seed = args.seed + pi          # one seed per prompt: efforts differ only by instruction
            out, seconds = _sample(eng, ids, args.tokens, p, row_seed)
            rows.append({
                "mode": mode,
                "reasoning_effort": effort if thinking else None,
                "prompt": prompt,
                "prompt_tokens": len(ids),
                "sampling": p,
                "reply_ids": out,
                "reply_text": tok.decode(out),
                "decode_seconds": round(seconds, 3),
                "decode_tokens_per_second": round(len(out) / seconds, 2) if seconds else 0.0,
                "fingerprint": fingerprint(out),
                "seed": row_seed,
            })
            print("%-8s %-6s %2d tok  %5.2f tok/s  %s" % (
                mode, effort or "-", len(out), rows[-1]["decode_tokens_per_second"], prompt[:40]))

    out = {
        "model": man.get("model"),
        "profile_id": profile_id,
        "source_file": man.get("source_file"),
        "source_sha256": man.get("source_sha256"),
        "pack": "<pack>",
        "engine": type(eng).__module__ + "." + type(eng).__name__,
        "device": os.environ.get("KINDI_DEVICE"),
        "ctx_requested": args.ctx,
        "tokens_requested": args.tokens,
        "seed": args.seed,
        "load_seconds": round(load_s, 1),
        "engine_log": said,
        "speculation": spec_report(),
        "pack_facts": {"tensors": man.get("tensors"), "dimensions": man.get("dimensions"),
                       "vocab": man.get("vocab"), "mtp_head": man.get("mtp_head")},
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "cpu_count": os.cpu_count(), "python": sys.version.split()[0]},
        "requests": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
