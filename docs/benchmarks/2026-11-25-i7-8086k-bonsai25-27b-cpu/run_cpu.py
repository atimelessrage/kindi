"""Greedy CPU runs of a Bonsai 2.5 27B pack: load, two prompts, determinism, decode rate.

    KINDI_DEVICE=cpu python docs/benchmarks/2026-11-25-i7-8086k-bonsai25-27b-cpu/run_cpu.py \
        --pack <pack> --out run1.json --tokens 16

The same command twice writes two files whose `fingerprint` fields must match. Nothing here compares
against llama.cpp: there is no reference file for this release, which is what `kindi verify` reports.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

os.environ.setdefault("KINDI_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

PROMPTS = (
    "In one sentence: what is a ternary weight?",
    "2 + 2 = ? Answer with the number only.",
)


def fingerprint(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=2048)
    args = ap.parse_args()

    import numpy as np

    from kindi import tokenizer
    from kindi.engine import make_engine

    man = json.load(open(os.path.join(args.pack, "kindi_pack.json"), encoding="utf-8"))
    said = []
    t0 = time.perf_counter()
    eng = make_engine(args.pack, ctx=args.ctx, log=said.append)
    load_s = time.perf_counter() - t0
    tok = tokenizer.Tok(eng.vocab, os.path.join(args.pack, "merges.txt"))

    body = "<|im_start|>user\n{0}<|im_end|>\n".format
    gen = "<|im_start|>assistant\n thinking\n\n</think>\n\n"
    rows = []
    for text in PROMPTS:
        ids = tok.encode(body(text) ) + tok.encode(gen)
        t1 = time.perf_counter()
        warm = eng.generate(ids, 1)                      # one token: the prefill is now cached work
        t_one = time.perf_counter() - t1
        t2 = time.perf_counter()
        got = eng.generate(ids, args.tokens)
        t_many = time.perf_counter() - t2
        again = eng.generate(ids, args.tokens)
        decode_tps = (args.tokens - 1) / (t_many - t_one) if t_many > t_one else 0.0
        rows.append({
            "prompt": text,
            "prompt_tokens": len(ids),
            "reply_ids": got,
            "reply_text": tok.decode(got),
            "first_token_seconds": round(t_one, 3),
            "reply_seconds": round(t_many, 3),
            "decode_tokens_per_second": round(decode_tps, 2),
            "deterministic": got == again,
            "fingerprint": fingerprint(got),
            "warm_fingerprint": fingerprint(warm),
        })
        print("%-45s %2d tokens  %.2f tok/s  %.2fs  %s" % (
            text[:45], args.tokens, decode_tps, t_many, "same" if got == again else "DIFFERS"))

    sizes = {name: os.path.getsize(os.path.join(args.pack, name))
             for name in ("bcodes.bin", "dpm.bin", "f32.bin")}
    out = {
        "pack": "<pack>",                     # never write a machine path into an evidence file
        "model": man.get("model"),
        "backend": man.get("backend"),
        "source_file": man.get("source_file"),
        "pack_facts": {
            "tensors": man.get("tensors"),
            "dimensions": man.get("dimensions"),
            "vocab": man.get("vocab"),
            "context_length": man.get("context_length"),
            "hadamard_block": man.get("hadamard_block"),
            "chat_template_chars": len(man.get("chat_template") or ""),
            "source_sha256": man.get("source_sha256"),
            "file_sizes": sizes,
            "declared": {"bcodes": man.get("bcodes_bytes"), "dpm": man.get("dpm_bytes"),
                         "f32": man.get("f32_bytes")},
            "totals_match": (sizes["bcodes.bin"] == man.get("bcodes_bytes"),
                             sizes["dpm.bin"] == man.get("dpm_bytes"),
                             sizes["f32.bin"] == man.get("f32_bytes")),
        },
        "ctx_requested": args.ctx,
        "engine": type(eng).__module__ + "." + type(eng).__name__,
        "device": os.environ.get("KINDI_DEVICE"),
        "load_seconds": round(load_s, 1),
        "engine_log": said,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "requests": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
