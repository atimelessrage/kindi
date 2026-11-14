# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi plan`: the placement dry-run, without loading a single weight tensor.

`--tandem-cpu-layers auto` decides silently, and the only way to check its pick was to load a 27B.
This command prints the model, the hardware (live or `--assume-*`), the placement, one reason per
choice, and the exact command that follows it — from the manifest, `index.json` /
`bcodes_index.json` and the same planners (`plan_cpu_layers` / `plan_cpu_layers_rate` /
`gpu_estimate`) the runtime uses, so the reported pick is the pick a real run makes on the same
numbers. The `--assume-*` flags let the P106/M4000 roadmaps' placements be tested before the
hardware exists. Always exits 0; a placement that fits nothing is reported, not raised.
"""
import json
import os
import sys

from . import tandem as _t


def _manifest(pack):
    path = os.path.join(pack, "kindi_pack.json")
    if not os.path.isfile(path):
        raise SystemExit("kindi: %s does not hold a kindi pack (no kindi_pack.json)" % pack)
    return json.load(open(path))


def _free_vram_gib():
    """(free, total) GiB from the engine's own CUDA handle; None when no CUDA device is usable."""
    try:
        from .engine import cp
    except Exception:
        return None
    if cp is None:
        return None
    try:
        free, total = cp.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return free / 2 ** 30, total / 2 ** 30


def _physical_ram_gib():
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MS()
            st.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 2 ** 30, st.ullAvailPhys / 2 ** 30
        pages = os.sysconf("SC_PAGE_SIZE")
        count = os.sysconf("SC_PHYS_PAGES")
        return pages * count / 2 ** 30, None
    except (AttributeError, OSError, ValueError):
        return None, None


def _cores():
    try:
        from .cpu import feature_report
        rep = feature_report()
        return rep.get("physical"), rep.get("logical")
    except Exception:
        return None, os.cpu_count()


def _owner(kind, assume_vram):
    kind = str(kind or "auto").strip().lower()
    if kind not in _t.GPU_OWNERS:
        raise SystemExit("kindi: the tandem GPU owner must be auto, plan or plain (got %r)" % kind)
    if kind != "auto":
        return kind
    if assume_vram:
        return "plain" if os.environ.get("KINDI_TP", "") == "0" else "plan"
    return _t._pick_owner("auto")


def describe(pack, ctx, spec, mode="cpu-first", gpu_kind=None,
             assume_vram_gib=None, assume_ram_gib=None, assume_cores=None, reserve_mib=512):
    """The placement report as a dict; no weight tensor is read, only indexes and the manifest."""
    man = _manifest(pack)
    dims = dict(man.get("dimensions") or {})
    lines = []          # (choice, reason) pairs, printed in order
    out = {"pack": pack, "model": man.get("model"), "backend": man.get("backend"),
           "ctx": int(ctx), "mode": mode, "lines": lines}

    lines.append(("model", "%s (%s, %d layers)" % (man.get("model"), man.get("backend"),
                                                   int(dims.get("layers", 0)))))
    lines.append(("context", "%d tokens requested; the loader clamps to device memory"
                  % int(ctx)))

    sizes = _t.pack_bytes(pack, man)
    pack_gib = sum(sizes["layers"].values()) / 2 ** 30 + sizes["f32"] / 2 ** 30

    vram = _free_vram_gib()
    if assume_vram_gib is not None:
        free_gib = float(assume_vram_gib)
        lines.append(("vram", "assumed %.1f GiB free; the planner runs on this number, "
                      "not the local card's" % free_gib))
    elif vram is None:
        # No usable CUDA here: the split planner needs a free-VRAM number, so report the model
        # byte accounting only and name the live command that needs the hardware.
        free_gib = None
        lines.append(("vram", "no CUDA device visible here; re-run `kindi plan` on the rig "
                      "or pass --assume-vram-gib"))
    else:
        free_gib = vram[0]
        lines.append(("vram", "%.2f of %.2f GiB free now" % vram))

    ram_total, _ = _physical_ram_gib()
    ram = float(assume_ram_gib) if assume_ram_gib is not None else ram_total
    if ram is not None:
        lines.append(("host ram", "~%.1f GiB of weights must fit host RAM; %.1f GiB %s"
                      % (pack_gib, ram, "assumed" if assume_ram_gib is not None else "probed")))
    else:
        lines.append(("host ram", "~%.1f GiB of weights; host RAM not probed on this platform"
                      % pack_gib))

    phys, logical = _cores()
    cores = int(assume_cores) if assume_cores is not None else phys
    if cores:
        lines.append(("cpu threads", "the CPU engine takes %d thread%s (%s physical%s)"
                      % (cores, "" if cores == 1 else "s",
                         "assumed" if assume_cores is not None else str(phys),
                         "" if logical == phys else " of %d logical" % logical)))
    else:
        lines.append(("cpu threads", "thread count not probed on this platform"))

    owner = _owner(gpu_kind or os.environ.get("KINDI_TANDEM_GPU", "auto"),
                   assume_vram_gib is not None)
    lines.append(("gpu owner", "%s (the plan engine reads per tensor, so offloaded layers free "
                  "their weights and K/V)" % owner if owner == "plan"
                  else "%s (whole f32 file uploaded; offloaded layers free weights and K/V only)"
                  % owner))
    out["owner"] = owner

    if free_gib is None:
        out["cpu_layers"] = None
        out["fit"] = None
        lines.append(("placement", "not decided without a free-VRAM number; the full model needs "
                      "~%.1f GiB of weights + K/V at %d tokens"
                      % (pack_gib, int(ctx))))
        out["command"] = _command(pack, mode, spec)
        return out

    free_bytes = int(free_gib * 2 ** 30)
    full = _t.gpu_estimate(dims, sizes, int(ctx), (), owner=owner)
    budget = free_bytes - reserve_mib * 2 ** 20
    out["estimate_full_mib"] = full // 2 ** 20

    parsed = spec if not isinstance(spec, str) else _t._parse_layers(spec)
    chosen, reason = [], ""
    try:
        if parsed in ("auto", "rate"):
            if parsed == "auto":
                chosen = _t.plan_cpu_layers(dims, sizes, int(ctx), free_bytes,
                                            reserve_mib=reserve_mib, owner=owner)
                reason = ("smallest GDN-first set that makes %d tokens fit in %.2f GiB free "
                          "(%d MiB reserve)" % (int(ctx), free_gib, reserve_mib))
            else:
                chosen = _t.plan_cpu_layers_rate(dims, sizes, int(ctx), free_bytes,
                                                 reserve_mib=reserve_mib, owner=owner,
                                                 pack_sha=man.get("source_sha256"))
                reason = ("rate rule: the fastest fitting GDN set (CPU/PCIe ratio "
                          "calibration %s)" % ("found" if _t.load_ratio(man.get("source_sha256"))
                                               else "not found for this pack - fit-only pick"))
        elif isinstance(parsed, str) and parsed.startswith("gdn:"):
            chosen = _t.gdn_first(dims, int(parsed.partition(":")[2]))
            reason = "explicit gdn set: the first %d gated-delta layers, attention stays on the GPU" % len(chosen)
        elif parsed is None:
            chosen = ()
            reason = "no split requested: the full model at %d tokens needs ~%d MiB against a %d MiB budget" % (
                int(ctx), full // 2 ** 20, budget // 2 ** 20)
        else:
            chosen = tuple(parsed)
            reason = "explicit layer set: %d layer%s on the CPU" % (
                len(chosen), "" if len(chosen) <= 1 else "s")
    except SystemExit as e:                     # the planner's refusal is the answer, not a crash
        lines.append(("placement", str(e)[len("kindi: "):]))
        out["cpu_layers"], out["fit"], out["estimate_mib"] = None, False, None
        out["command"] = _command(pack, mode, spec)
        return out

    est = _t.gpu_estimate(dims, sizes, int(ctx), chosen, owner=owner)
    out["estimate_mib"] = est // 2 ** 20
    out["cpu_layers"] = list(chosen)
    fits = est <= budget
    out["fit"] = fits
    if not chosen:
        lines.append(("placement", "no split — the full model fits (%d MiB with the %s owner against "
                      "a %d MiB budget): %s" % (est // 2 ** 20, owner, budget // 2 ** 20, reason)))
    elif fits:
        lines.append(("placement", "%d CPU layer%s (%s) takes the %s GPU side to ~%d MiB "
                      "(%d freed; %d MiB budget): %s"
                      % (len(chosen), "" if len(chosen) <= 1 else "s",
                         ",".join(str(x) for x in chosen), owner, est // 2 ** 20,
                         (full - est) // 2 ** 20, budget // 2 ** 20, reason)))
    else:
        lines.append(("placement", "%d CPU layers still do not fit (~%d MiB against a %d MiB "
                      "budget): the pick is insufficient; reduce --ctx or add free VRAM"
                      % (len(chosen), est // 2 ** 20, budget // 2 ** 20)))
    out["command"] = _command(pack, mode, spec)
    return out


def _command(pack, mode, spec):
    tail = " bench --tandem %s --tandem-cpu-layers %s" % (mode, spec or "auto")
    return "%s -m kindi --pack %s%s" % (sys.executable, pack, tail)


def run(cfg, a, log=print):
    """CLI entry point: print the report (and optionally its JSON), always 0."""
    pack = cfg.get("pack")
    if not pack:
        raise SystemExit("kindi: no pack - pass --pack DIR or set `pack` in kindi.conf")
    spec = getattr(a, "tandem_cpu_layers", None) or cfg.get("tandem_cpu_layers") or "auto"
    med = getattr(a, "tandem", None) or cfg.get("tandem") or "cpu-first"
    gpu_kind = getattr(a, "tandem_gpu", None) or cfg.get("tandem_gpu") or os.environ.get("KINDI_TANDEM_GPU")
    rep = describe(pack, cfg.get("ctx", 24576), spec, mode=med, gpu_kind=gpu_kind,
                   assume_vram_gib=getattr(a, "assume_vram_gib", None),
                   assume_ram_gib=getattr(a, "assume_ram_gib", None),
                   assume_cores=getattr(a, "assume_cores", None),
                   reserve_mib=int(os.environ.get("KINDI_TANDEM_RESERVE_MB", "512")))
    for choice, reason in rep["lines"]:
        log("%-11s %s" % (choice + ":", reason))
    log("follow up: %s" % rep["command"])
    log("verify:    %s -m kindi --pack %s verify --tandem %s --tandem-cpu-layers %s"
        % (sys.executable, pack, med, spec or "auto"))
    if getattr(a, "json", None):
        from .bench import bench_json
        bench_json(rep, a.json)
        log("wrote %s" % a.json)
    return 0
