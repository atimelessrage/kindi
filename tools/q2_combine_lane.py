"""Gate the mixed card/CPU combine contract against the real ``lt_mv_q2_down``.

    python tools/q2_combine_lane.py --json docs/benchmarks/<run>/results.json

A layer that puts some routed experts on the host cannot call ``lt_mv_q2_down``
for the whole slot set: the host produces the CPU experts' per-lane pre-tree
partials, while the card produces the card experts' and the shared expert's.
The contract is:

* host: ``kindi.moe_expert_lane.Q2Matrix.partials`` returns float32
  ``[rows, 32]`` per expert, the same accumulators ``lt_mv_q2_down`` would hold;
* card experts: ``lt_mv_q2_down_parts`` writes float32 ``[nslot, rows, 32]``;
* shared expert: ``lt_q2_shared_parts`` writes float32 ``[rows, 32]``;
* combine: ``lt_q2_combine(parts, nslot, gw, shparts, rows, x)`` adds
  ``gw[k] * parts[k]`` per lane to the shared partial, runs the same five-step
  XOR tree and does ``x[r] += tot``.

This tool builds synthetic Q2_0 weights at the real down geometry (rows 2688,
expert width 1856, shared width 3712), puts experts 0 and 2 on the host and 1
and 3 on the card, and compares the mixed result bit for bit with one
``lt_mv_q2_down`` launch over all four slots. It also compares the host
partials with ``lt_mv_q2_down_parts`` for the same experts. No model result is
implied: there is no Lightning pack here, so no verify/tok/s is claimed.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kindi import moe_expert_lane as lane  # noqa: E402

HIDDEN = 2688                                               # Spec.n_embd of the 30B-A3B
FF = 1856                                                   # Spec.n_ff
EXPERTS = 4
CPU_SLOTS = (0, 2)
CARD_SLOTS = (1, 3)


def _random_q2(rng, experts, rows, cols):
    runs, nb = cols // 32, cols // 64
    codes = rng.integers(0, 3, size=(experts, rows, runs * 2), dtype=np.uint32)
    scales = (rng.standard_normal((experts, rows, nb)) * 0.01).astype(np.float16)
    return codes, scales


def _quantize(cp, quant, x, relu2=1):
    """Run the engine's own ``lt_quant`` on one vector, returning device and host runs."""
    x_dev = cp.asarray(x.reshape(1, -1))
    runs = x.size // 32
    q = cp.zeros((runs, 8), np.int32)
    d = cp.zeros(runs, np.float32)
    s = cp.zeros(runs, np.int32)
    quant(((runs + 7) // 8,), (256,), (x_dev, np.int32(1), np.int32(x.size), np.int32(x.size),
                                       np.int32(relu2), q, d, s))
    cp.cuda.Stream.null.synchronize()
    return q, d, s, cp.asnumpy(q).reshape(-1), cp.asnumpy(d), cp.asnumpy(s)


def _ptrs(cp, arrays):
    return cp.asarray(np.array([a.data.ptr for a in arrays], dtype=np.uint64))


def _bits(a):
    return np.asarray(a, np.float32).view(np.int32)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python tools/q2_combine_lane.py")
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--ff", type=int, default=FF)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--json", default=None, help="write the result object here")
    a = ap.parse_args(argv)
    if a.hidden % 32 or a.ff % 64:
        ap.error("hidden must be a multiple of 32 and ff a multiple of 64")
    if len(set(CPU_SLOTS) | set(CARD_SLOTS)) != EXPERTS:
        ap.error("the four slots must partition 0..3")
    if not lane.available():
        print("q2 combine: unavailable - numba with its fmaf intrinsic is required")
        return 2
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:                                # pragma: no cover - host without CUDA
        print("q2 combine: unavailable - no CUDA device (%s)" % exc)
        return 2

    rng = np.random.default_rng(a.seed)
    down_codes, down_scales = _random_q2(rng, EXPERTS, a.hidden, a.ff)
    sh_codes, sh_scales = _random_q2(rng, 1, a.hidden, 2 * a.ff)
    sh_codes, sh_scales = sh_codes[0], sh_scales[0]

    src = open(os.path.join(ROOT, "kindi", "lightning.cu"), encoding="utf-8").read()
    mod = cp.RawModule(code=src, options=("--use_fast_math", "-std=c++14"))
    quant = mod.get_function("lt_quant")
    mv_down = mod.get_function("lt_mv_q2_down")
    mv_parts = mod.get_function("lt_mv_q2_down_parts")
    mv_shared = mod.get_function("lt_q2_shared_parts")
    combine = mod.get_function("lt_q2_combine")

    # Each routed expert has its own up-projection output, so lt_mv_q2_down's stacked input
    # is nslot blocks of runs*8 words. Quantizing one vector per expert is what the engine
    # does; passing one block for every slot would read past it and test the wrong thing.
    dev_hq, dev_hd, dev_hs = [], [], []
    host_hq, host_hd, host_hs = [], [], []
    for _ in range(EXPERTS):
        q, d, s, qh, dh, sh = _quantize(cp, quant, rng.standard_normal(a.ff).astype(np.float32) * 2)
        dev_hq.append(q.reshape(-1)); dev_hd.append(d); dev_hs.append(s)
        host_hq.append(qh); host_hd.append(dh); host_hs.append(sh)
    hq = cp.concatenate(dev_hq)
    hd = cp.concatenate(dev_hd)
    hs = cp.concatenate(dev_hs)
    gq, gd, gs, *_ = _quantize(cp, quant, rng.standard_normal(2 * a.ff).astype(np.float32) * 2)

    down = lane.Q2Matrix(down_codes, down_scales, a.hidden, a.ff, experts=EXPERTS)
    dev_dn = [cp.asarray(down_codes[e].reshape(-1)) for e in range(EXPERTS)]
    dev_ds = [cp.asarray(down_scales[e].reshape(-1)) for e in range(EXPERTS)]
    ptr_dn, ptr_ds = _ptrs(cp, dev_dn), _ptrs(cp, dev_ds)
    dev_sh = cp.asarray(sh_codes.reshape(-1))
    dev_ssh = cp.asarray(sh_scales.reshape(-1))
    ids_all = cp.asarray(np.arange(EXPERTS, dtype=np.int32))
    gw = np.array([0.13, 0.27, 0.41, 0.19], np.float32)
    gw_dev = cp.asarray(gw)

    # Reference: one lt_mv_q2_down launch with all four routed slots plus the shared expert.
    # Two inputs: x = 0 checks the combined sum, x = residual checks the x[r] += tot path.
    residual = cp.asarray(rng.standard_normal(a.hidden).astype(np.float32))
    ref_zero = cp.zeros(a.hidden, np.float32)
    ref_res = residual.copy()
    for x in (ref_zero, ref_res):
        mv_down(((a.hidden + 3) // 4,), (128,),
                (ptr_dn, ptr_ds, ids_all, gw_dev, np.int32(EXPERTS), np.int32(a.hidden),
                 np.int32(a.ff), hq, hd, hs, dev_sh, dev_ssh, np.int32(2 * a.ff),
                 gq, gd, gs, x))
    cp.cuda.Stream.null.synchronize()

    # Mixed path: host partials for CPU slots, the parts kernel for card slots, shared parts
    # on the card, then one combine. parts is [slot][row][lane], exactly what combine reads.
    parts = cp.zeros((EXPERTS, a.hidden, 32), np.float32)
    host_parts = {}
    host_diff = {}
    for slot in CPU_SLOTS:
        host_parts[slot] = down.partials(slot, host_hq[slot], host_hd[slot], host_hs[slot])
        parts[slot] = cp.asarray(host_parts[slot])
        card_copy = cp.empty((a.hidden, 32), np.float32)
        mv_parts(((a.hidden + 3) // 4,), (128,),
                 (ptr_dn, ptr_ds, cp.asarray(np.array([slot], np.int32)), np.int32(1),
                  np.int32(a.hidden), np.int32(a.ff), dev_hq[slot], dev_hd[slot], dev_hs[slot],
                  card_copy))
        cp.cuda.Stream.null.synchronize()
        host_diff[slot] = int(np.sum(_bits(cp.asnumpy(card_copy)) != _bits(host_parts[slot])))
    for slot in CARD_SLOTS:
        mv_parts(((a.hidden + 3) // 4,), (128,),
                 (ptr_dn, ptr_ds, cp.asarray(np.array([slot], np.int32)), np.int32(1),
                  np.int32(a.hidden), np.int32(a.ff), dev_hq[slot], dev_hd[slot], dev_hs[slot],
                  parts[slot]))
    shparts = cp.empty((a.hidden, 32), np.float32)
    mv_shared(((a.hidden + 3) // 4,), (128,),
              (dev_sh, dev_ssh, np.int32(a.hidden), np.int32(2 * a.ff), gq, gd, gs, shparts))

    mixed_zero = cp.zeros(a.hidden, np.float32)
    mixed_res = residual.copy()
    for x in (mixed_zero, mixed_res):
        combine(((a.hidden + 3) // 4,), (128,),
                (parts, np.int32(EXPERTS), gw_dev, shparts, np.int32(a.hidden), x))
    cp.cuda.Stream.null.synchronize()

    zero_diff = int(np.sum(_bits(cp.asnumpy(mixed_zero)) != _bits(cp.asnumpy(ref_zero))))
    res_diff = int(np.sum(_bits(cp.asnumpy(mixed_res)) != _bits(cp.asnumpy(ref_res))))

    # Isolate the shared branch: no routed slots on either side.
    ref_shared = cp.zeros(a.hidden, np.float32)
    mix_shared = cp.zeros(a.hidden, np.float32)
    mv_down(((a.hidden + 3) // 4,), (128,),
            (ptr_dn, ptr_ds, ids_all, gw_dev, np.int32(0), np.int32(a.hidden),
             np.int32(a.ff), hq, hd, hs, dev_sh, dev_ssh, np.int32(2 * a.ff),
             gq, gd, gs, ref_shared))
    combine(((a.hidden + 3) // 4,), (128,),
            (parts, np.int32(0), gw_dev, shparts, np.int32(a.hidden), mix_shared))
    cp.cuda.Stream.null.synchronize()
    shared_diff = int(np.sum(_bits(cp.asnumpy(mix_shared)) != _bits(cp.asnumpy(ref_shared))))

    result = {
        "measured": time.strftime("%Y-%m-%d"),
        "host": {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                 if isinstance(cp.cuda.runtime.getDeviceProperties(0)["name"], bytes)
                 else str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
                 "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                 "python": platform.python_version(), "cupy": cp.__version__,
                 "numba": __import__("numba").__version__},
        "geometry": {"hidden": a.hidden, "ff": a.ff, "experts": EXPERTS,
                     "shared_cols": 2 * a.ff},
        "slots": {"cpu": list(CPU_SLOTS), "card": list(CARD_SLOTS), "shared": "card"},
        "contract": {
            "host_partials": "Q2Matrix.partials -> float32 [rows, 32]",
            "card_partials": "lt_mv_q2_down_parts -> float32 [nslot, rows, 32]",
            "shared_partials": "lt_q2_shared_parts -> float32 [rows, 32]",
            "combine": "lt_q2_combine(parts, nslot, gw, shparts, rows, x): x[r] += tree("
                       "shparts[r] + sum_k gw[k] * parts[k][r])",
            "reference": "lt_mv_q2_down with all four slots and the shared expert",
        },
        "parity": {
            "combine_zero_differing": zero_diff,
            "combine_residual_differing": res_diff,
            "combine_zero": zero_diff == 0,
            "combine_residual": res_diff == 0,
            "shared_only_differing": shared_diff,
            "shared_only": shared_diff == 0,
            "host_partials_vs_parts": all(v == 0 for v in host_diff.values()),
            "host_partials_differing": {str(k): v for k, v in host_diff.items()},
        },
        "pack": None,
        "blocker": ("no kindi_lightning pack on this host: this gate runs on synthetic Q2_0 weights; "
                    "the capacity-bound model row (verify 4/4 and tok/s) needs the Lightning "
                    "GGUF/pack, which is absent"),
    }
    packs = os.path.join(os.path.expanduser("~"), ".kindi", "packs")
    for name in sorted(os.listdir(packs)) if os.path.isdir(packs) else []:
        p = os.path.join(packs, name, "kindi_pack.json")
        try:
            manifest = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("backend") == "kindi_lightning":
            result["pack"] = p
            result["blocker"] = None
            break

    print("combine vs lt_mv_q2_down: zero %s (%d differing), residual %s (%d differing), "
          "shared-only %s (%d differing)"
          % ("IDENTICAL" if zero_diff == 0 else "DIFFERS", zero_diff,
             "IDENTICAL" if res_diff == 0 else "DIFFERS", res_diff,
             "IDENTICAL" if shared_diff == 0 else "DIFFERS", shared_diff))
    print("host partials vs lt_mv_q2_down_parts: %s (%s)"
          % ("IDENTICAL" if result["parity"]["host_partials_vs_parts"] else "DIFFERS",
             ", ".join("slot %s: %d" % (k, v) for k, v in host_diff.items())))
    print("slots: CPU %s, card %s, shared card" % (list(CPU_SLOTS), list(CARD_SLOTS)))
    if result["blocker"]:
        print("model row: BLOCKED - " + result["blocker"])
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    return 0 if (zero_diff == 0 and res_diff == 0 and shared_diff == 0
                 and result["parity"]["host_partials_vs_parts"]) else 1


if __name__ == "__main__":
    sys.exit(main())
