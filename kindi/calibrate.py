"""kindi calibrate: measure the per-machine tuning once and persist it (`kindi/calib.py`, roadmap gate 2).

On the paths that time them (non-TC plans, HIP), building the engine times the 5-trit kernel
shapes (`tp.g5_tuned`), and those configs are stored in the calibration file, so later starts skip
the timing; on the NVIDIA tensor-core path there is no g5 timing to persist. This command additionally records the
other measured answers as data - the split-ratio snapshot, the planner's placement picks for the
configured context, and the effective draft length - so they are stored in one keyed place instead
of living in chat logs. KINDI_SPEC_K and an explicit `--tandem-cpu-layers` still win at runtime;
the file records what was measured, it does not override the user.
"""
import os

from .jsonio import load_json


def run(cfg, log=print):
    import cupy as cp

    from . import calib
    from .engine import make_engine

    pack = cfg["pack"]
    man_path = os.path.join(pack, "kindi_pack.json")
    man = load_json(man_path) if os.path.exists(man_path) else {}
    pack_sha = man.get("source_sha256")
    gpu = calib.gpu_identity()
    if not gpu:
        log("kindi calibrate: no GPU device found; calibration needs a card")
        return 2
    if not pack_sha:
        log("kindi calibrate: the pack manifest names no source_sha256; calibration keys on the pack hash")
        return 2

    make_engine(pack, ctx=cfg.get("ctx", 24576), log=log,
                      kv_cache=cfg.get("kv_cache"))   # times + stores the g5 configs
    saved = calib.read(gpu, pack_sha) or {}
    shapes = len((saved.get("g5") or {}).get("lane") or {}) + len((saved.get("g5") or {}).get("row") or {})
    log("kindi calibrate: 5-trit configs for %d shapes stored" % shapes)

    try:
        from . import tandem
        ratio = tandem.load_ratio(pack_sha=pack_sha)
    except Exception:
        ratio = None
    if ratio is not None:
        calib.record(gpu, pack_sha, "ratio", {
            "measured_utc": ratio.get("measured_utc"),
            "predicted_tok_s": ratio.get("predicted_tok_s"),
        })
        log("kindi calibrate: split-ratio snapshot recorded (%s)" % ratio.get("measured_utc", "?"))
    else:
        log("kindi calibrate: no split-ratio calibration for this pack (tools/measure_split_ratio.py)")

    try:
        from . import tandem
        dims = dict(man.get("dimensions", {}))
        sizes = tandem.pack_bytes(pack, man)
        ctx = cfg.get("ctx", 24576)
        free, total = cp.cuda.runtime.memGetInfo()
        # As if the card were empty, but with the user's ceiling and display reserve: calibration
        # must recommend the placement the runtime would actually build.
        room = tandem.vram_budget_bytes(total, cfg.get("tandem_reserve_mb", 512),
                                        cfg.get("tandem_vram_mb"))
        auto = tandem.plan_cpu_layers(dims, sizes, ctx, room, reserve_mib=0, owner="plan")
        picks = {"ctx": ctx, "budget_mib": room // 2 ** 20, "auto": [int(x) for x in auto],
                 "owner": "plan", "vram_budget_mib": cfg.get("tandem_vram_mb") or None,
                 "reserve_mib": cfg.get("tandem_reserve_mb", 512)}
        rate = tandem.plan_cpu_layers_rate(dims, sizes, ctx, room, reserve_mib=0, owner="plan",
                                           pack_sha=pack_sha)
        picks["rate"] = [int(x) for x in rate]
        calib.record(gpu, pack_sha, "placement", picks)
        log("kindi calibrate: placement picks at ctx %d: auto %d layers, rate %d layers"
            % (ctx, len(picks["auto"]), len(picks["rate"])))
    except Exception as e:                      # recorded as data: a planner error must not fail calibration
        log("kindi calibrate: placement picks not recorded (%s)" % e)

    calib.record(gpu, pack_sha, "draft_k", {
        "env": os.environ.get("KINDI_SPEC_K"),
        "default": 4, "auto_draft2b": 3,
    })
    path = calib.path_for(gpu, pack_sha)
    log("kindi calibrate: wrote %s" % path)
    return 0
