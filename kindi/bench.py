"""kindi bench: generation speed at several context positions, prompt-reading speed, the prefill/
decode share of one whole request, and GPU memory.
With KINDI_DEVICE=cpu (or without CuPy) it times the CPU engine with the host clock.

`--json FILE` additionally records the numbers in a machine-readable JSON file, so a recorded
run does not have to be scraped back out of the human log.
"""
import json
import os
import time

POSITIONS = (100, 2000, 8000, 16000, 24000)
# A request is a prompt plus a reply, so the two rates decide different things: a 4K prompt with a
# 128-token reply spends most of its wall clock in prefill, and tuning decode would not move it.
# `--split-at` times that whole request and reports each phase's share of the clock.
SPLIT_PROMPTS = (4096, 8192)
SPLIT_GENERATED = 128


def _cupy():
    if os.environ.get("KINDI_DEVICE", "gpu") == "cpu":
        return None
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None


def split_prompts(cfg):
    """The prompt lengths `--split-at` asked for ("none"/empty turns the report off).

    Every other token must be a positive integer.  A token used to be dropped when it was
    zero or negative, and text raised a bare ``ValueError``; both now name the valid choices
    before a model is loaded.
    """
    raw = cfg.get("split_at", SPLIT_PROMPTS)
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        text = str(raw).strip().lower()
        if text in ("", "none", "off"):
            return []
        values = [part.strip() for part in text.split(",") if part.strip()]
    out = []
    for value in values:
        try:
            n = int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError("prompt lengths must be comma-separated positive integers, or "
                             "none/off (got %r)" % (value,))
        if n <= 0:
            raise ValueError("prompt lengths must be comma-separated positive integers, or "
                             "none/off (got %r)" % (value,))
        out.append(n)
    return out


def request_split(eng, sync, prompt, generated, warmup=True):
    """Time one whole request: a real `prompt`-token prefill, then `generated` greedy steps.

    The prefill goes through `feed_batch` when the engine has one - the chunked path the server
    uses - so this is the request a user waits for, not two per-token rates multiplied together.
    Returns (prefill_ms, decode_ms); both blocks are synced before the clock is read.
    """
    ids = [1 + (i % 977) for i in range(prompt)]

    def feed_all(tokens, start=0):
        if hasattr(eng, "feed_batch"):
            eng.feed_batch(tokens, start)
        else:
            for i, tok in enumerate(tokens):
                eng.feed(tok, start + i)

    if warmup:
        eng.reset()
        feed_all(ids[:8])
        eng.step(ids[0], 8)
        sync()
    eng.reset()
    sync()
    t0 = time.perf_counter()
    feed_all(ids)
    sync()
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    sync()
    t0 = time.perf_counter()
    for i in range(generated):
        eng.step(ids[-1], prompt + i)
    sync()
    decode_ms = (time.perf_counter() - t0) * 1000.0
    return prefill_ms, decode_ms


def bench_json(result, path):
    """Write one benchmark result as JSON (per-run row)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)      # a new directory is cheaper than losing a 27B run
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def run(cfg, log=print):
    """Run one benchmark; an explicit `--dcache` is the engine's slot budget for this run only."""
    dcache = cfg.get("dcache")
    previous = os.environ.get("KINDI_LT_DCACHE")
    if dcache is not None:
        os.environ["KINDI_LT_DCACHE"] = str(int(dcache))
    try:
        return _run(cfg, log=log)
    finally:
        if dcache is not None:
            if previous is None:
                os.environ.pop("KINDI_LT_DCACHE", None)
            else:
                os.environ["KINDI_LT_DCACHE"] = previous


def _run(cfg, log=print):
    try:
        requested_splits = split_prompts(cfg)
    except ValueError as exc:
        log("kindi bench: %s" % exc)
        return 2
    import numpy as np
    cp = _cupy()
    from .engine import make_engine
    pack = cfg["pack"]
    mp = os.path.join(pack, "kindi_pack.json")
    man = __import__("json").load(open(mp)) if os.path.exists(mp) else {}
    if man.get("backend") == "kindi_nanbeige":
        if cp is None:
            log("kindi bench: the Nanbeige backend runs only on an NVIDIA GPU (KINDI_DEVICE=cpu is set or CuPy is missing)")
            return 2
        from .nanbeige_cli import bench
        return bench(cfg, man, log=log)
    if man.get("backend") == "kindi_lightning":
        if cp is None:
            log("kindi bench: the Lightning backend runs only on an NVIDIA GPU (KINDI_DEVICE=cpu is set or CuPy is missing)")
            return 2
        from .lightning_cli import bench
        return bench(cfg, man, log=log)
    if man.get("backend") == "kindi_cascade":
        log("kindi bench: the Cascade backend is converter/validator only; GPU K-quant Nemotron-H kernels are pending")
        return 2
    if man.get("backend") == "kindi_gemma4":
        log("kindi bench: the Gemma 4 engine is not wired into this benchmark (the dense loop below drives "
            "prefill/step on Bonsai engines); its gate is `kindi verify`, which has a Gemma 4 reference")
        return 2
    if man.get("backend") == "kindi_dsv41":
        log("kindi bench: DeepSeek V4.1 runs under `python -m kindi.dsv41_server` and its release gate is "
            "dsv41.verify; this benchmark does not drive that engine")
        return 2
    from .vram import VramMonitor
    # sample the card the engine will run on (the cupy device ordinal), not unconditionally GPU 0
    dev = 0
    if cp is not None:
        try:
            dev = int(cp.cuda.Device().id)
        except Exception:                                              # noqa: BLE001 - no usable device: GPU 0
            dev = 0
    vram = VramMonitor(cfg.get("vram_sample_s", 0.5), device=dev) if cfg.get("sample_vram") else None
    if vram is not None:
        vram.start()
    cl = os.path.join(pack, "head_clusters.npz")
    eng = make_engine(pack, ctx=cfg["ctx"], clusters=dict(np.load(cl)) if os.path.exists(cl) else None, log=log,
                       tandem=cfg.get("tandem", "off"), cpu_layers=cfg.get("tandem_cpu_layers"),
                       gpu=cfg.get("tandem_gpu"), kv_cache=cfg.get("kv_cache"),
                       mtp_head=cfg.get("mtp_head"))
    if vram is not None:
        vram.mark("loaded")
    # eng.ctx is the *actual* context (the loader reduces it to fit); the requested one is
    # recorded separately so a reduced run is never mistaken for the requested context.
    result = {"pack": pack, "model": man.get("model"), "ctx": eng.ctx,
              # exact: eng.ctx is the requested context unless the loader reduced it to fit (the
              # 256-token rounding is inside the fit, so any shortfall is a real reduction)
              "ctx_reduced": cfg["ctx"] > eng.ctx, "ctx_requested": cfg["ctx"],
              "tandem": cfg.get("tandem", "off"), "cpu_layers": cfg.get("tandem_cpu_layers"),
              "device": "gpu" if cp is not None else "cpu", "read_tok_s": None, "gen_tok_s": {}}
    if hasattr(eng, "cpu_range"):
        result["cpu_range"] = list(eng.cpu_range)
        result["gpu_range"] = list(eng.gpu_range)
    if cp is not None:
        free, total = cp.cuda.runtime.memGetInfo()
        log("GPU memory: %.0f MiB free of %.0f with a %d-token context" % (free / 2 ** 20, total / 2 ** 20, eng.ctx))
        result["free_mib_after_load"] = round(free / 2 ** 20)
        result["total_mib"] = round(total / 2 ** 20)
        sync = cp.cuda.Device().synchronize
    else:
        log("CPU engine: %s threads, %d-token context" % (os.environ.get("OMP_NUM_THREADS", os.cpu_count()), eng.ctx))
        sync = lambda: None                    # noqa: E731 - the CPU engine returns when the work is done
    eng.reset()

    def timed(fn, n):
        fn(0)                                  # warm-up (first launches compile/load kernels)
        sync()
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        sync()
        return (time.perf_counter() - t0) / n * 1000

    ms = timed(lambda i: eng.feed(1000 + i, 10 + i), 20)
    result["read_tok_s"] = round(1000 / ms, 1)
    log("prompt reading:       %5.1f tokens/s" % (1000 / ms))
    modes = [("exact", 0)] + ([("fast", cfg["fast_clusters"])] if eng.hsl is not None else [])
    for name, M in modes:
        eng.shortlist = M
        for p in [x for x in POSITIONS if x + 20 < eng.ctx]:
            ms = timed(lambda i: eng.step(2000 + i, p + i), 10)
            result["gen_tok_s"]["%s:%d" % (name, p)] = round(1000 / ms, 1)
            log("generation (%s head) at position %5d: %5.1f tokens/s" % (name, p, 1000 / ms))
    eng.shortlist = 0
    generated = int(cfg.get("split_generated", SPLIT_GENERATED) or 0)
    splits = {}
    for prompt in requested_splits:
        if generated and prompt + generated > eng.ctx:
            log("request split at %d prompt tokens skipped: %d + %d does not fit the %d-token context"
                % (prompt, prompt, generated, eng.ctx))
            continue
        prefill_ms, decode_ms = request_split(eng, sync, prompt, generated)
        # every reported figure is derived from the rounded milliseconds, so the file is consistent
        # with itself: a rate recomputed from the printed ms must be the rate printed next to it
        prefill_ms, decode_ms = round(prefill_ms, 2), round(decode_ms, 2)
        total = prefill_ms + decode_ms
        row = {"prompt_tokens": prompt, "generated_tokens": generated,
               "prefill_ms": prefill_ms, "decode_ms": decode_ms,
               "prefill_share": round(prefill_ms / total * 100.0, 1),
               "decode_share": round(decode_ms / total * 100.0, 1),
               "prefill_tok_s": round(prompt / (prefill_ms / 1000.0), 1) if prefill_ms > 0 else None,
               "decode_tok_s": round(generated / (decode_ms / 1000.0), 1) if decode_ms > 0 else None}
        splits[str(prompt)] = row
        log("request split: %d prompt + %d generated = %.2f s; prefill %.2f s (%4.1f%%) at %.1f "
            "tokens/s, decode %.2f s (%4.1f%%) at %.1f tokens/s"
            % (prompt, generated, total / 1000.0, prefill_ms / 1000.0, row["prefill_share"],
               row["prefill_tok_s"] or 0.0, decode_ms / 1000.0, row["decode_share"],
               row["decode_tok_s"] or 0.0))
    if splits:
        result["split"] = splits
    timing = eng.timing_summary() if hasattr(eng, "timing_summary") else None
    if timing is not None:
        result["tandem_timing"] = timing
        log("tandem timing/token: cpu %.1f ms (%d layers) + gpu %.1f ms (%d) + transfer %.2f ms "
            "(%0.1f/token) + head %.1f ms = %.1f ms over %d tokens"
            % (timing["cpu_ms_per_token"], timing["cpu_layers"], timing["gpu_ms_per_token"],
               timing["gpu_layers"], timing["transfer_ms_per_token"], timing["transfers_per_token"],
               timing["head_ms_per_token"], timing["total_ms_per_token"], timing["tokens"]))
    if vram is not None:
        vram.stop()
        s = vram.summary()
        if s.get("available"):
            result["vram_peak"] = s
            log("VRAM peak (sampled, device-wide): %d MiB used (%d MiB over idle), %d MiB free, peak in %s, %d samples"
                % (s["peak_used_mib"], s["peak_delta_mib"], s["peak_free_mib"], s["peak_phase"], s["sample_count"]))
        else:
            result["vram_peak"] = {"available": False, "interval_s": vram.interval}
            log("VRAM peak (sampled): nvidia-smi unavailable, no samples taken")
    if cfg.get("json"):
        try:
            bench_json(result, cfg["json"])
            log("wrote %s" % cfg["json"])
        except OSError as e:                     # the numbers above are still the result
            log("could not write %s (%s); the numbers above are the run" % (cfg["json"], e))
    return 0
