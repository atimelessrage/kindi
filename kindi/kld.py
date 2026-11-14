"""kindi kld: a quality gate for lossy K/V cache formats (roadmap gate 3).

Teacher-forced KL divergence (nats) + same-top-token share + PPL ratio of a lossy cache
against the fp16 whole-model reference (the plain engine, kv="f16"): both engines are fed the
same token ids (teacher forcing) and their next-token distributions are compared at every
position after a warmup. Three numbers per lossy format, at 2K and 8K on the 2B:

  mean KLD        KL(ref || variant), reference first
  same-top share  fraction of positions where the argmax token agrees
  PPL ratio       exp(mean NLL_variant - mean NLL_ref) over the same positions

Variants measured in one run (all against the same reference pass):
  plain-q8   the plain engine with kv="q8_0"  (cache drift, same engine)
  tp-q8      the plan engine with the q8_0 cache  (engine + cache, the control)
  tp-k8v4    the plan engine with the experimental 4-bit V cache

`tp-k8v4`'s quarantine is then re-derived from the numbers: its excess over the plan engine's
own q8_0 row is the loss the 4-bit V itself adds. Like `kindi bench`, this times the exact head;
an engine with a reordered (fast) head is refused - its logits are not a full vocabulary.
"""
import gc
import json
import math
import os
import time

import numpy as np

WARMUP = 64                                   # positions are scored only after this many tokens
CONTEXTS = (2048, 8192)
PROMPT = "The quick brown fox jumps over the lazy dog. " * 400   # ~4K tokens at 2B tokenization
VARIANTS = ("plain-q8", "tp-q8", "tp-k8v4")


def softmax(logits):
    """Full-vocabulary softmax in float64 (logits: 1-D, one score per token id)."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_nats(p, q):
    """KL(p || q) in nats, both full-vocabulary probability vectors over the same token ids."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    tiny = 1e-300
    return float(np.sum(p * (np.log(np.maximum(p, tiny)) - np.log(np.maximum(q, tiny)))))


def summarize(steps):
    """steps: iterable of (kld, same_top, nll_ref, nll_variant) -> the three gate numbers."""
    steps = list(steps)
    n = len(steps)
    if not n:
        raise ValueError("no measured steps (context too short for the warmup?)")
    kld = sum(s[0] for s in steps) / n
    share = sum(s[1] for s in steps) / n
    nll_r = sum(s[2] for s in steps) / n
    nll_v = sum(s[3] for s in steps) / n
    return {"steps": n, "kld_nats": kld, "same_top_share": share,
            "ppl_ref": math.exp(nll_r), "ppl_ratio": math.exp(nll_v - nll_r)}


def _tokens(tok, n):
    """A fixed token-id sequence of length n (teacher forcing needs the same ids for every engine)."""
    ids = tok.encode(PROMPT)
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def _logits(eng, cp):
    """The full-vocabulary logits left by the last step/head of an exact-head engine."""
    if getattr(eng, "ylog_map", None) is not None:
        raise SystemExit("kindi kld: this engine has a reordered (fast) head; only the exact head "
                         "produces a full vocabulary to compare")
    return cp.asnumpy(eng.ylog[:eng.ylog_n]).copy()


def _engine(pack, ctx, kind, log, cp):
    """One engine of the given kind: the fp16 reference is always the plain engine (kv="f16")."""
    if kind == "ref":
        from .engine import Engine
        return Engine(pack, ctx=ctx, log=log, kv="f16")
    if kind == "plain-q8":
        from .engine import Engine
        return Engine(pack, ctx=ctx, log=log, kv="q8_0")
    from .tp import TPEngine
    old_kv = os.environ.get("KINDI_KV")
    old_exp = os.environ.get("KINDI_KV_EXPERIMENTAL")
    try:
        os.environ["KINDI_KV"] = "k8v4" if kind == "tp-k8v4" else "q8_0"
        if kind == "tp-k8v4":
            os.environ["KINDI_KV_EXPERIMENTAL"] = "1"
        return TPEngine(pack, ctx=ctx, log=log)
    finally:
        if old_kv is None:
            os.environ.pop("KINDI_KV", None)
        else:
            os.environ["KINDI_KV"] = old_kv
        if old_exp is None:
            os.environ.pop("KINDI_KV_EXPERIMENTAL", None)
        else:
            os.environ["KINDI_KV_EXPERIMENTAL"] = old_exp


def _free(cp):
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def compare(pack, ctx, variant, log=print):
    """One teacher-forced pass of the reference and one variant in lockstep at context `ctx`."""
    import cupy as cp
    from .tokenizer import Tok
    ref = _engine(pack, ctx, "ref", log, cp)
    var = _engine(pack, ctx, variant, log, cp)
    steps = []
    try:
        ids = _tokens(Tok(ref.vocab, os.path.join(pack, "merges.txt")), ctx)
        ref.reset()
        var.reset()
        t0 = time.time()
        for pos, t in enumerate(ids):
            ref.step(t, pos)
            var.step(t, pos)
            if pos < WARMUP or pos + 1 >= len(ids):
                continue
            lr = _logits(ref, cp)
            lv = _logits(var, cp)
            p = softmax(lr)
            q = softmax(lv)
            steps.append((kl_nats(p, q), int(np.argmax(lr)) == int(np.argmax(lv)),
                          -float(np.log(p[ids[pos + 1]])), -float(np.log(q[ids[pos + 1]]))))
        out = summarize(steps)
        out.update(ctx=ctx, variant=variant, seconds=round(time.time() - t0, 1),
                   warmup=WARMUP)
        return out
    finally:
        del ref, var
        _free(cp)


def run(cfg, log=print, contexts=CONTEXTS, variants=VARIANTS):
    """The gate: for each context, the three numbers for every lossy variant against fp16."""
    pack = cfg["pack"]
    results = []
    for ctx in contexts:
        log("kindi kld: context %d (teacher-forced, warmup %d, exact head)" % (ctx, WARMUP))
        for v in variants:
            r = compare(pack, ctx, v, log)
            r["pack"] = pack
            results.append(r)
            log("  %-8s KLD %.4f nats   same-top %.1f%%   PPL ratio %.4f   (%d steps, %.0f s)"
                % (r["variant"], r["kld_nats"], 100 * r["same_top_share"], r["ppl_ratio"],
                   r["steps"], r["seconds"]))
    return results


def write_json(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
