"""kindi bench: generation speed at several context positions, prompt-reading speed, and GPU memory.
With KINDI_DEVICE=cpu (or without CuPy) it times the CPU engine with the host clock.

`--json FILE` additionally records the numbers in a machine-readable JSON file, so a recorded
run does not have to be scraped back out of the human log.
"""
import json
import os
import time

POSITIONS = (100, 2000, 8000, 16000, 24000)


def _cupy():
    if os.environ.get("KINDI_DEVICE", "gpu") == "cpu":
        return None
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None


def bench_json(result, path):
    """Write one benchmark result as JSON (per-run row)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def run(cfg, log=print):
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
    from .vram import VramMonitor
    vram = VramMonitor(cfg.get("vram_sample_s", 0.5)) if cfg.get("sample_vram") else None
    if vram is not None:
        vram.start()
    cl = os.path.join(pack, "head_clusters.npz")
    eng = make_engine(pack, ctx=cfg["ctx"], clusters=dict(np.load(cl)) if os.path.exists(cl) else None, log=log,
                       tandem=cfg.get("tandem", "off"), cpu_layers=cfg.get("tandem_cpu_layers"))
    if vram is not None:
        vram.mark("loaded")
    result = {"pack": pack, "model": man.get("model"), "ctx": eng.ctx,
              "tandem": cfg.get("tandem", "off"), "cpu_layers": cfg.get("tandem_cpu_layers"),
              "device": "gpu" if cp is not None else "cpu", "read_tok_s": None, "gen_tok_s": {}}
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
        bench_json(result, cfg["json"])
        log("wrote %s" % cfg["json"])
    return 0
