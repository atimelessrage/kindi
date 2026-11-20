"""kindi verify: greedy output vs llama.cpp's reference for the same model file (4 prompts, the first 40 tokens
llama.cpp generated from them at temperature 0). Every token must match. reference.json is the 2B; reference-4b.json,
-9b.json and -27b.json the others (the 27B with llama.cpp's q8_0 K/V cache, as Kindi's 27B engine)."""
import json
import os
import re
import time


def reference_for(pack):
    """the reference file of the pack's model (by the size in its source file name), or None"""
    here = os.path.dirname(os.path.abspath(__file__))
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    m = re.search(r"-(\d+)B[-_.]", man.get("source_file", "") + ".") or re.search(r"(\d+)B", man.get("model", ""))
    size = (m.group(1) + "b") if m else "2b"
    p = os.path.join(here, "reference.json" if size == "2b" else "reference-%s.json" % size)
    return p if os.path.exists(p) else None


GEMMA4_MIN_PREFIX = 12      # Gemma 4: every prompt must agree with llama.cpp on at least this many first tokens


def run_gemma4(cfg, man, log=print):
    """Nemotron / Gemma 4: Kindi quantizes activations to int8 and sums in its own order, so it is not token-exact
    with llama.cpp. Each prompt must agree with llama.cpp's greedy output on its first GEMMA4_MIN_PREFIX tokens."""
    import cupy as cp
    here = os.path.dirname(os.path.abspath(__file__))
    rp = os.path.join(here, "reference-%s.json" % man.get("model_id", ""))
    if not os.path.exists(rp):
        log("no llama.cpp reference for this model in this version of Kindi")
        return 2
    ref = json.load(open(rp))
    man = dict(man, pack_path=cfg["pack"])
    from .server import _two_cards
    if _two_cards():
        from .gemma4_tp import TPEngine as E
    else:
        from .gemma4 import Engine as E
    eng = E(man, ctx=2048, log=log)
    ok_all, tok, dt = True, 0, 0.0
    for i, r in enumerate(ref["prompts"]):
        ids, want = r["ids"], r["expected"]
        cp.cuda.Device().synchronize(); t0 = time.time()
        eng.forward_batch(ids[:-1], 0); t = int(cp.argmax(eng.forward(ids[-1], len(ids) - 1)).item()); got = [t]
        for pos in range(len(ids), len(ids) + len(want) - 1):
            t = int(cp.argmax(eng.forward(t, pos)).item()); got.append(t)
        cp.cuda.Device().synchronize(); dt += time.time() - t0; tok += len(ids) + len(got)
        pre = next((k for k in range(len(want)) if got[k] != want[k]), len(want))
        ok = pre >= GEMMA4_MIN_PREFIX
        ok_all &= ok
        log("prompt %d %-60s %s  the first %d of %d tokens agree" % (i + 1, repr(r["prompt"])[:60], "PASS" if ok else "FAIL",
                                                                     pre, len(want)))
    log("%s - %.1f tokens/s over the run" % (("ALL PASS: each prompt agrees with llama.cpp on its first %d tokens or more"
                                              % GEMMA4_MIN_PREFIX) if ok_all else "FAIL", tok / dt))
    return 0 if ok_all else 1


# a first difference where llama.cpp's own top-2 log-probability gap (reference "gap", tools/bonsai_gaps.py) is under
# TIE_GAP nats, or where Kindi gives the token of another llama.cpp run ("alt"), is a tie, as in nanbeige_cli
TIE_GAP = 0.01

OFF = ("0", "", "off")


def _engine_path(eng):
    """The engine path a verify line names: the plan engine is not the plain engine."""
    cls = type(eng)
    module, name = cls.__module__, cls.__name__
    if module == "kindi.tp" and name == "TPEngine":
        return ("the tensor-core plan engine" if getattr(eng, "tc", False)
                else "the plan engine's 5-trit lane")
    if module == "kindi.engine" and name == "Engine":
        return "the plain engine"
    if module == "kindi.cpu" and name == "Engine":
        return "the CPU engine"
    if module == "kindi.tandem" and name == "TandemEngine":
        return "the tandem engine"
    if name == "Engine":
        return "the %s engine" % module.rsplit(".", 1)[-1]
    return "the %s engine" % name


def spec_state(eng, requested=None):
    """The speculation `kindi verify` will run, the line that says so, and a refusal (REQ-0025 A1/A4).

    Returns (mode, line, refusal). A model-drafter mode whose drafter did not attach is a refusal,
    not a quiet fallback to the plain engine: `draft2b` with no pack resolves to `ngram`, so the
    requested mode would otherwise never be tested. `auto` is the one mode allowed to resolve to a
    different name, since choosing is what it is for.
    """
    requested = (os.environ.get("KINDI_SPEC", "0") or "0") if requested is None else requested
    off = requested in OFF
    if not hasattr(eng, "spec_begin"):
        line = ("speculation: unavailable (this engine has no spec_begin); verifying %s"
                % _engine_path(eng))
        return None, line, None if off else "KINDI_SPEC=%s was requested but this engine cannot speculate" % requested
    from . import spec as _sp
    st = _sp.effective_state(eng, requested)
    mode, head = st["mode"], st["head"] or {}
    if off:
        return None, "speculation: off (KINDI_SPEC=%s)" % requested, None
    parts = ["requested %s, effective %s" % (requested, mode or "0")]
    if head.get("state") == "loaded":
        parts.append("head %s sha256 %s" % (head["file"], (head.get("sha256") or "")[:12]))
    if st["reason"]:
        parts.append(st["reason"])
    refusal = None
    if mode is None or (requested != "auto" and mode != requested):
        refusal = "%s was requested but the effective mode is %s%s" % (
            requested, mode or "0", (": " + st["reason"]) if st["reason"] else "")
    return mode, "speculation: " + ", ".join(parts), refusal


def run(cfg, log=print):
    try:
        import cupy as cp
    except ImportError:
        cp = None
    from .engine import make_engine
    import numpy as np
    pack = cfg["pack"]
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    if man.get("backend") == "kindi_gemma4":
        return run_gemma4(cfg, man, log=log)
    if man.get("backend") == "kindi_nanbeige":
        from .nanbeige_cli import verify
        return verify(cfg, man, log=log)
    if man.get("backend") == "kindi_lightning":
        from .lightning_cli import verify
        return verify(cfg, man, log=log)
    if man.get("backend") == "kindi_cascade":
        from .cascade import verify
        return verify(cfg, man, log=log)
    if man.get("backend") == "kindi_dsv41":
        log("kindi verify: DeepSeek V4.1's release gate is `dsv41.verify` (teacher-forced against the "
            "release gate data), not this llama.cpp reference matrix; run `python -m kindi.dsv41_server` "
            "with the gate directory to use it")
        return 2
    rp = reference_for(pack)
    if rp is None:
        log("no llama.cpp reference for this model in this version of Kindi")
        return 2
    ref = json.load(open(rp))
    cl = os.path.join(pack, "head_clusters.npz")
    eng = make_engine(pack, ctx=1024, clusters=dict(np.load(cl)) if os.path.exists(cl) else None, log=log,
                       tandem=cfg.get("tandem", "off"), cpu_layers=cfg.get("tandem_cpu_layers"),
                       gpu=cfg.get("tandem_gpu"), kv_cache=cfg.get("kv_cache"))
    # the first line names the mode that will generate, so a PASS is never read as the plain engine's
    # when speculation was on - or the reverse (REQ-0025 A1); a drafter that did not attach stops here (A4)
    spec_mode, spec_line, refusal = spec_state(eng)
    log(spec_line)
    if refusal:
        log("FAIL: %s" % refusal)
        return 2
    ok_all, tok, dt, ties = True, 0, 0.0, 0
    for i, r in enumerate(ref["prompts"]):
        sync = (lambda: cp.cuda.Device().synchronize()) if cp is not None and hasattr(eng, "k") else (lambda: None)
        sync()
        t0 = time.time()
        if spec_mode is not None:
            eng.spec_begin(True)                     # greedy, exactly the server's request path
            try:
                got = eng.generate(r["ids"], len(r["expected"]))
            finally:
                sp = eng.spec_end()
        else:
            got, sp = eng.generate(r["ids"], len(r["expected"])), None
        sync()
        t = time.time() - t0
        n = sum(1 for a, b in zip(got, r["expected"]) if a == b)
        ok = got == r["expected"]
        state, j, gap = "PASS" if ok else "FAIL", None, None
        if not ok:
            j = next(k for k, (a, b) in enumerate(zip(got, r["expected"])) if a != b)
            gap = r["gap"][j] if r.get("gap") and j < len(r["gap"]) else None
            alt = r.get("alt") or []
            if (gap is not None and gap < TIE_GAP) or (j < len(alt) and alt[:j + 1] == got[:j + 1]):
                # a dead heat in llama.cpp's own top two, or the token another llama.cpp run of the same file gives
                # there (tools/bonsai_gaps.py "alt"): the Nanbeige backend's rule
                state = "TIE"
                ties += 1
        ok_all &= state != "FAIL"
        tok += len(r["ids"]) + len(got)
        dt += t
        spec_note = ""
        if sp and sp.get("drafted"):
            spec_note = ", %d/%d drafts accepted" % (sp.get("accepted", 0), sp.get("drafted", 0))
        log("prompt %d %-60s %s  %d/%d tokens match  (%.1fs%s)" % (
            i + 1, repr(r["prompt"])[:60], state, n, len(r["expected"]), t, spec_note))
        if not ok:
            log("   first difference at token %d%s" % (j, " (llama.cpp top-2 gap there %.4f)" % gap if gap is not None else ""))
    where = " with KINDI_SPEC=%s" % spec_mode if spec_mode is not None else " (speculation off)"
    if ok_all and ties:
        log("PASS%s: %d of %d prompts token-exact, %d end at a llama.cpp tie, 0 fail - %.1f tokens/s over the run" % (
            where, len(ref["prompts"]) - ties, len(ref["prompts"]), ties, tok / dt))
    else:
        log("%s%s - %.1f tokens/s over the run" % ("ALL PASS: token-exact with llama.cpp" if ok_all else "FAIL",
                                                    where, tok / dt))
    return 0 if ok_all else 1
