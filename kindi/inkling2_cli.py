"""Local text inference for an Inkling 2 Mini GGUF, and kindi convert for it (backend "kindi_inkling2": the pack
holds only kindi_pack.json; the engine reads the weights from the GGUF file itself).

`kindi verify` and `kindi bench` dispatch here for a kindi_inkling2 pack (verify.py / bench.py), the same way
Nanbeige and Lightning dispatch to their own *_cli modules; the logic is the one in tools/verify_inkling2.py,
which stays as the standalone script.
"""
import json
import os
import sys
import time

from .inkling2 import Engine
from .inkling2_tokenizer import Inkling2Tokenizer

TIE_GAP = 0.08  # logprob gap under which a different-but-close token counts as a tie, not a failure


def run(args):
    def log(message):
        print(message, file=sys.stderr)

    if not os.path.isfile(args.gguf):
        log("inkling: no such GGUF: %s" % args.gguf)
        return 2
    eng = Engine(args.gguf, ctx=args.ctx or 2048, log=log)
    try:
        tok = Inkling2Tokenizer(eng.gguf.kv)
        prompt = args.prompt
        if not args.raw:
            from .inkling2_chat import render
            prompt = render(eng.model_path, [{"role": "user", "content": prompt}], enable_thinking=False)
        ids = tok.encode(prompt, special=not args.raw)
        if not ids or len(ids) > eng.ctx:
            log("inkling: prompt must encode to between 1 and %d tokens" % eng.ctx)
            return 2
        stop = {tok.eos, tok.eot}
        out = eng.generate(ids, args.max_tokens, stop=stop)
        if out and out[-1] in stop:
            out.pop()
        print(tok.decode(out))
        return 0
    finally:
        eng.close()


def is_inkling(g):
    return g.kv.get("general.architecture") == "inkling2_loop"


def convert(g, gguf_path, out_dir, version, pack_format, sha256, log=print):
    from collections import Counter
    os.makedirs(out_dir, exist_ok=True)
    kv = g.kv
    man = {
        "kindi_version": version, "pack_format": pack_format, "backend": "kindi_inkling2",
        "model": kv.get("general.name"), "model_id": "inkling-2-mini",
        "source_file": os.path.basename(gguf_path), "model_path": os.path.abspath(gguf_path),
        "source_sha256": sha256(gguf_path), "context_length": kv.get("inkling2_loop.context_length"),
        "tensor_types": dict(Counter(str(t) for _, _, t, _ in g.tensors)),
    }
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    log("recognized %s (Inkling 2, %d tensors): Kindi Inkling engine; the weights stay in the GGUF file"
        % (man["model"], len(g.tensors)))
    return man


def _open_engine(man, ctx, log):
    path = man.get("model_path")
    if not path or not os.path.isfile(path):
        log("kindi: this Inkling pack records no usable model_path (%r); re-run `kindi convert` for the GGUF" % path)
        return None
    return Engine(path, ctx=max(512, int(ctx or 2048)), log=log)


def verify(cfg, man, log=print):
    """Greedy output against the bundled llama.cpp reference (reference-inkling2-mini.json, UD-Q4_K_M).

    Teacher-forced: each step compares the engine's greedy token with the reference's, then feeds the reference
    token, so one divergence cannot cascade. A mismatch whose reference top-2 gap is under TIE_GAP (or whose
    recorded alt token matches) is a tie, not a failure - the same rule tools/verify_inkling2.py uses.
    Returns 0 when every step passes or ties, 1 on a real mismatch, 2 when the pack cannot be opened.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(here, "reference-inkling2-mini.json")
    if not os.path.isfile(ref_path):
        log("no llama.cpp reference for Inkling 2 Mini in this version of Kindi")
        return 2
    ctx = int((cfg or {}).get("ctx") or 2048)
    eng = _open_engine(man, max(ctx, 1600), log)  # long_swa_1060 needs 1456 + 48 of context
    if eng is None:
        return 2
    try:
        tok = Inkling2Tokenizer(eng.gguf.kv)
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        log("reference %s: %s" % (os.path.basename(ref_path), ref.get("model", "")))
        n_pass = n_tie = n_fail = total = 0
        for case in ref["cases"]:
            eng.reset()
            pred, pgap = eng.prefill(case["ids"], pos=0, return_gap=True)
            c_pass = c_tie = c_fail = 0
            first_fail = None
            for step, ref_tok in enumerate(case["expected"]):
                ref_gap, ref_alt = case["gap"][step], case["alt"][step]
                if pred == ref_tok:
                    c_pass += 1
                elif (pred == ref_alt and ref_gap < TIE_GAP) or (ref_gap < 0.02 and pgap < 0.05):
                    c_tie += 1
                else:
                    c_fail += 1
                    if first_fail is None:
                        first_fail = (step, pred, ref_tok, ref_alt, ref_gap, pgap)
                if step + 1 < len(case["expected"]):
                    pred, pgap = eng.step(ref_tok, return_gap=True)
            total += len(case["expected"])
            n_pass += c_pass
            n_tie += c_tie
            n_fail += c_fail
            msg = "CASE %-14s steps=%3d PASS=%3d TIE=%2d FAIL=%2d" % (
                case["name"], len(case["expected"]), c_pass, c_tie, c_fail)
            if first_fail:
                s, pr, rt, ra, rg, pg = first_fail
                msg += " FIRST_FAIL@%d: got %d(%r) vs ref %d(%r) alt=%d ref_gap=%.4f our_gap=%.4f" % (
                    s, pr, tok.decode([pr]), rt, tok.decode([rt]), ra, rg, pg)
            log(msg)
        log("SUMMARY: total=%d PASS=%d TIE=%d FAIL=%d" % (total, n_pass, n_tie, n_fail))
        return 0 if n_fail == 0 else 1
    finally:
        eng.close()


def bench(cfg, man, log=print):
    """Reading (prefill) and writing (decode) rates on this host, pack and context; no claim is adopted from it."""
    import numpy as np
    ctx = int((cfg or {}).get("ctx") or 2048)
    eng = _open_engine(man, ctx, log)
    if eng is None:
        return 2
    try:
        prompt = [100 + (i * 37) % 5000 for i in range(512)]
        pp, tg = [], []
        for _ in range(3):
            eng.reset()
            t0 = time.time()
            cur, _gap = eng.prefill(prompt, pos=0, return_gap=True)
            pp.append(512 / max(0.001, time.time() - t0))
            t0 = time.time()
            for _i in range(127):
                cur = eng.step(cur)
            tg.append(127 / max(0.001, time.time() - t0))
        log("reading a 512-token prompt: %6.1f tok/s" % float(np.mean(pp)))
        log("writing at 128 tokens: %6.1f tok/s" % float(np.mean(tg)))
        log("engine mode: %s, %d CPU threads, ctx %d" % (eng.mode, eng.n_threads, eng.ctx))
        return 0
    finally:
        eng.close()
