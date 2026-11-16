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

import numpy as np

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
    """The RAM budget Kindi should plan against, in GiB.

    Uses `hostram.total()`, which caps the host's physical RAM at the process's cgroup limit so a
    container's placement follows its actual budget rather than the host's. Returns (total, free);
    free may be None."""
    try:
        from . import hostram

        total = hostram.total()
        if total:
            return total / 2 ** 30, None
    except Exception:
        pass
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


def _moe_free_bytes(assume_vram_gib):
    if assume_vram_gib is not None:
        return int(float(assume_vram_gib) * 2 ** 30)
    vram = _free_vram_gib()
    if vram is None:
        raise SystemExit("kindi: no CUDA device visible here; plan an MoE placement with "
                         "--assume-vram-gib (or run this on the rig)")
    return int(vram[0] * 2 ** 30)


def describe(pack, ctx, spec, mode="cpu-first", gpu_kind=None,
             assume_vram_gib=None, assume_ram_gib=None, assume_cores=None, reserve_mib=512,
             usage_path=None, moe_cache_mib=None, dcache=None,
             chunk_tokens=None, chunk_lane="copy"):
    """The placement report as a dict; no weight tensor is read, only indexes and the manifest."""
    man = _manifest(pack)
    if man.get("backend") == "kindi_lightning":
        return describe_moe(pack, ctx, _moe_free_bytes(assume_vram_gib),
                            usage_path=usage_path, moe_cache_mib=moe_cache_mib,
                            dcache=dcache, reserve_mb=reserve_mib,
                            chunk_tokens=chunk_tokens, chunk_lane=chunk_lane)
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


_Q2, _Q8, _F32 = 42, 8, 0                     # lightning.py's tensor-type ids


def _lt_dense_bytes(tinfo):
    """Lightning's `_plan` dense loop (non-expert tensors; F32/Q8/Q2 byte rules)."""
    total = 0
    for name, (ne, ty, _off) in tinfo.items():
        if "_exps" in name or name == "token_embd.weight":
            continue
        n = int(np.prod(ne)) if ne else 0
        total += n // 64 * 18 if ty == _Q2 else n // 32 * 34 if ty == _Q8 else n * 4
    return total


def _lt_buf_bytes(s, chunk):
    f = chunk * (2 * s.n_embd + s.d_in + s.d_xbc + 2 * s.d_inner + s.n_exp + s.n_ffsh + s.n_embd) * 4
    f += chunk * s.n_used * (s.n_ff + s.n_embd) * 4 + chunk * s.n_used * s.n_ff * 1.5
    return int(f + chunk * max(s.d_in, s.n_ffsh) * 1.5 + s.vocab * 4 + (64 << 20))


def _lt_fit(s, free_bytes, dense, kv, state, buf_bytes, reserve, eb, moe_layers):
    """Lightning's placement math (`_plan` plus the min-card rule), free of the engine's GPU work.

    Returns (n_card, room) or None when even n_used experts per layer do not fit."""
    n_moe = len(moe_layers)
    room = free_bytes - dense - kv - state - buf_bytes - reserve
    fit = max(0, room // (n_moe * eb)) if n_moe else 0
    if fit < s.n_exp and n_moe > 2:
        dcache = max(s.n_used, int(os.environ.get("KINDI_LT_DCACHE", "10")))
        fit = max(0, int((room - 2 * s.n_exp * eb - n_moe * dcache * eb) // ((n_moe - 2) * eb)))
    if fit < s.n_used:
        return None, room
    return min(s.n_exp, fit), room


def describe_moe(pack, ctx, free_bytes, usage_path=None, moe_cache_mib=None,
                 dcache=None, reserve_mb=None, chunk_tokens=None, chunk_lane="copy"):
    """The MoE strategy for a `kindi_lightning` pack: expert tiers (+ decode slots), not a layer split.

    Same arithmetic as `lightning._plan`; the weights are never read - the GGUF header, `Spec` and
    `tinfo` carry everything the fit needs. `--moe-cache-mib` caps the experts' card budget; the
    weighted placement and measured-use share come from `kindi/moe.py` and the usage file."""
    from .gguf import GGUF
    from .lightning import Spec
    from .moe import MoESpec, Usage, plan as moe_plan

    man = _manifest(pack)
    gguf_path = man.get("model_path")
    if not gguf_path or not os.path.isfile(gguf_path):
        raise SystemExit("kindi: the Lightning pack manifest names no readable GGUF (model_path: %r)"
                         % gguf_path)
    g = GGUF(gguf_path)
    s = Spec(g)
    tinfo = {n: (ne, ty, off) for n, ne, ty, off in g.tensors}
    moe_layers = [il for il, k in enumerate(s.kind) if k == "moe"]
    eb = 2 * s.n_ff * s.n_embd // 64 * 18                     # one expert: up + down
    n_attn, n_moe = s.kind.count("attn"), s.kind.count("moe")
    dense = _lt_dense_bytes(tinfo)
    kv = n_attn * 2 * s.n_kv * s.hd * 2 * int(ctx)
    state = s.kind.count("ssm") * (s.n_sh * 64 * s.d_state + s.d_xbc * 3) * 4
    reserve = (int(os.environ.get("KINDI_LT_RESERVE_MB", "350")) if reserve_mb is None
               else int(reserve_mb)) * 2 ** 20
    if dcache is None:
        dcache = max(s.n_used, int(os.environ.get("KINDI_LT_DCACHE", "10")))
    else:
        dcache = int(dcache)
    room = free_bytes - dense - kv - state - reserve
    lines = [("model", "%s (%s, %d layers, %d MoE layers, %d experts, %d per token)"
              % (man.get("model"), man.get("backend"), s.n_layer, n_moe, s.n_exp, s.n_used)),
             ("strategy", "moe-expert-tiers: the dense core and shared expert live on the card; "
              "routed experts are tiered card / pinned host / mmap, and the last-used dcache="
              "%d per layer stay in device slots (the engine's LRU)" % dcache),
             ("host ram", "the whole expert set + dense core (~%.1f GiB) stays readable from host RAM"
              % ((n_moe * s.n_exp * eb + dense) / 2 ** 30)),
             ("reserve", "%d MiB kept free (the CUDA context, kernels and pool rounding; the engine "
              "adds more after a failed load)" % (reserve // 2 ** 20))]
    n_card = None
    cap = int(moe_cache_mib) * 2 ** 20 if moe_cache_mib is not None else None
    if cap is not None:
        n_card = max(0, cap // (n_moe * eb))
        lines.append(("cache cap", "--moe-cache-mib caps the experts' card budget at %d MiB: %d of %d "
                      "experts per layer on the card" % (int(moe_cache_mib), n_card, s.n_exp)))
    else:
        n_card, room = _lt_fit(s, free_bytes, dense, kv, state, _lt_buf_bytes(s, 512),
                               reserve, eb, moe_layers)
    if n_card is None or n_card < s.n_used:
        lines.append(("experts", "nothing fits (%d experts per MoE layer cannot be resident at "
                      "this context and free VRAM; the engine needs at least %d per token): "
                      "reduce --ctx or add free VRAM" % (n_card or 0, s.n_used)))
        return {"pack": pack, "model": man.get("model"), "strategy": "moe-expert-tiers",
                "lines": lines, "n_card": n_card if n_card is not None else 0,
                "n_exp": s.n_exp, "fit": False,
                "command": "%s -m kindi --pack %s bench" % (sys.executable, pack)}
    # weighted placement, mirroring lightning._plan_weighted's candidate scan
    usage_path = usage_path or os.environ.get("KINDI_LT_USAGE") or os.path.join(
        os.path.dirname(__file__), "lightning-experts.json")
    spec = MoESpec(moe_layers, s.n_exp, s.n_used, eb, shared=True)
    usage = Usage.load(spec, usage_path)
    placement = None
    if usage.has_counts() and os.environ.get("KINDI_MOE_WEIGHTED", "1") != "0":
        room2 = free_bytes - dense - kv - state - _lt_buf_bytes(s, 512) - reserve
        best = None
        for m in sorted(set(list(range(max(s.n_used, n_card - 64), n_card + 1, 4)) + [n_card])):
            b = room2 - 2 * (s.n_exp - m) * eb - n_moe * dcache * eb
            if cap is not None:
                # --moe-cache-mib is a cap on the experts' bytes: the weighted scan must not spend
                # past it, or the report names more card experts than the flag allows
                b = min(b, cap)
            if b < m * n_moe * eb:
                continue
            try:
                p = moe_plan(spec, usage, b, min_card=m, weighted=True)
            except MemoryError:
                continue
            f = p.served("card")
            if best is None or (f is not None and f > best[0] + 1e-9):
                best = (f, p, m)
        if best is not None:
            placement = best[1]
            n_card = min(placement.counts("card").values())
    if placement is not None:
        head = placement.describe()
        lines.append(("experts", "%s (usage-weighted placement; the most-used experts stay on the card)"
                      % head))
    else:
        suffix = "" if n_card >= s.n_exp else ", the rest in pinned host memory"
        lines.append(("experts", "%d of %d experts per layer on the card%s"
                      % (n_card, s.n_exp, suffix)))
    slot_gib = n_moe * dcache * eb / 2 ** 30
    if n_card < s.n_exp:
        why = ("%d per MoE layer (%.2f GiB): the last-used host experts kept in device slots so the next "
               "use is not a PCIe read. Each slot displaces one expert per layer" % (dcache, slot_gib))
        if placement is not None:
            # The weighted plan already spent these bytes before it chose m, so the honest number
            # is the share that placement serves, not the top-n share a uniform split would get.
            share = placement.served("card")
            if share is not None:
                why += ("; the usage-weighted placement already leaves room for them and serves "
                        "%.1f%% of the measured picks from the card" % (100 * share))
            else:
                why += "; the usage-weighted placement already leaves room for them"
        else:
            if moe_cache_mib is None:
                why += ("; the fit already leaves room for them, and as residency the same bytes would "
                        "hold %d experts per layer" % (n_card + dcache))
            cov_slots = usage.coverage(n_card)
            cov_resident = usage.coverage(n_card + dcache) if moe_cache_mib is None else None
            if cov_slots is not None and cov_resident is not None:
                why += (" - %.1f%% of the measured picks against %.1f%%"
                        % (100 * cov_slots, 100 * cov_resident))
            else:
                why += " (no usage counts: run bench to measure use)"
        why += "; --dcache or KINDI_LT_DCACHE moves the split"
        lines.append(("slots", why))
    else:
        lines.append(("slots", "%d per MoE layer (%.2f GiB) are reserved but unused: all %d experts per "
                      "layer fit on the card, so no host expert needs a slot (--dcache still sizes them)"
                      % (dcache, slot_gib, s.n_exp)))
    est = dense + kv + state + reserve + n_moe * n_card * eb
    lines.append(("placement", "~%.2f GiB on the card at ctx %d (%d MiB of %d MiB free); host experts are "
                  "read over PCIe through per-expert pointer tables" % (
                      est / 2 ** 30, int(ctx), est // 2 ** 20, int(free_bytes) // 2 ** 20)))
    out = {"pack": pack, "model": man.get("model"), "strategy": "moe-expert-tiers",
           "lines": lines, "n_card": n_card, "n_exp": s.n_exp, "dcache": dcache, "fit": True,
           "estimate_mib": est // 2 ** 20,
           "command": "%s -m kindi --pack %s bench" % (sys.executable, pack)}
    if chunk_tokens:
        chunk = _sample_tandem(spec, usage, placement, moe_layers, n_card, eb,
                               int(chunk_tokens), chunk_lane)
        if isinstance(chunk, dict) and "lines" in chunk:
            lines.extend(chunk.pop("lines"))
        out["chunk"] = chunk
    return out


def _sample_tandem(spec, usage, placement, moe_layers, n_card, eb, tokens, lane):
    """What `MoeTandem` would do for a sample prefill chunk, from the placement and the usage file.

    There is no router here: the sample spreads over each layer's measured order, token t starting
    at rank (3t + 7il) mod n_exp and taking the next n_used ranks, so adjacent tokens share
    experts (the dedupe a real chunk gets) and cold experts get picked too (the copy lane, which a
    hottest-experts-only sample would never show). Only the scheduler's bookkeeping runs.
    """
    from .expert_cache import ExpertCache
    from .moe import Placement
    from .moe_tandem import MoeTandem

    tiers = {}
    for il in moe_layers:
        tiers[il] = list(placement.tier[il]) if placement is not None else ["host"] * spec.n_exp
        if placement is None:
            for e in usage.order(il)[:n_card]:
                tiers[il][e] = "card"
    sample = Placement(spec, usage, tiers)
    picks = []
    for t in range(tokens):
        keys = []
        for li, il in enumerate(moe_layers):
            order = usage.order(il)
            start = (3 * t + 7 * li) % spec.n_exp
            for k in range(spec.n_used):
                keys.append((il, order[(start + k) % spec.n_exp]))
        picks.append(keys)

    host_keys = set(key for keys in picks for key in keys if tiers[key[0]][key[1]] != "card")
    cache = ExpertCache(max(1, len(host_keys)), slot_bytes=eb)
    try:
        plan = MoeTandem(sample, cache, lane=lane).forward(list(enumerate(picks)))
    except (ValueError, TypeError) as e:
        return {"error": str(e)}
    m = plan.metrics()
    plan.release()
    split = ("%d-token sample chunk: %d selection(s) over %d MoE layer(s) x %d routed expert(s); "
             "lanes card %d, copy %d, cpu %d (spread starts of the measured order: rank 3t+7l per "
             "token and layer, n_used consecutive ranks each)"
             % (tokens, m["pairs"], len(moe_layers), spec.n_used, m["card_pairs"],
                m["copy_pairs"], m["cpu_pairs"]))
    copies = ("%d distinct expert(s) copied = %.2f MiB in the sample; %d selection(s) shared a copy "
              "(%.1f%% of the copy lane). Planner output only: no bytes are moved."
              % (m["distinct_copies"], m["copy_bytes"] / 2 ** 20, m["saved_copies"],
                 100.0 * m["saved_copies"] / m["copy_pairs"] if m["copy_pairs"] else 0.0))
    return {"tokens": tokens, "lane": lane, "layers": len(moe_layers), "n_used": spec.n_used,
            "selections": m["pairs"],
            "card_pairs": m["card_pairs"], "copy_pairs": m["copy_pairs"],
            "cpu_pairs": m["cpu_pairs"], "copies": m["distinct_copies"],
            "saved_copies": m["saved_copies"], "copy_bytes": m["copy_bytes"],
            "hits": m["hits"], "lines": [("chunk", split), ("copies", copies)]}


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
                   usage_path=getattr(a, "usage", None),
                   moe_cache_mib=getattr(a, "moe_cache_mib", None),
                   dcache=getattr(a, "dcache", None),
                   chunk_tokens=getattr(a, "moe_chunk", None),
                   chunk_lane=getattr(a, "moe_lane", None) or "copy",
                   reserve_mib=int(os.environ.get("KINDI_TANDEM_RESERVE_MB", "512")))
    for choice, reason in rep["lines"]:
        log("%-11s %s" % (choice + ":", reason))
    log("follow up: %s" % rep["command"])
    if rep.get("strategy") == "moe-expert-tiers":
        log("verify:    %s -m kindi --pack %s verify" % (sys.executable, pack))
    else:
        log("verify:    %s -m kindi --pack %s verify --tandem %s --tandem-cpu-layers %s"
            % (sys.executable, pack, med, spec or "auto"))
    if getattr(a, "json", None):
        from .bench import bench_json
        bench_json(rep, a.json)
        log("wrote %s" % a.json)
    return 0
