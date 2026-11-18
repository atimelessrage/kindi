"""`kindi autotune`: measure the placements a chosen model will be served with, and rank them.

Every candidate is a real `kindi bench` run in a child process, so the numbers come from the same
code path `serve` uses and nothing here estimates a throughput. The candidate list comes from
`kindi/caps.py` (how many cards this host has, what the CPU can do) and the pack manifest (dense
Bonsai vs a MoE backend), which is why the tuner only proposes what this machine could actually run:
a two-card split is not offered on a one-card host, and the Lightning decode-slot knob is not
offered for a pack that has no expert cache.

Ranking is by measured decode tokens/s at the requested context. A run whose requested context did
not fit is flagged and ranked below every run that got it, because a faster number at a smaller
context is a different question. Exactness is not re-measured here: `kindi verify` owns that, and a
placement-changing winner (tandem, two-card split) has to pass it before it is served.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

MOE_BACKENDS = ("kindi_lightning",)
NON_DENSE_BACKENDS = ("kindi_lightning", "kindi_nanbeige", "kindi_gemma4", "kindi_dsv41",
                      "kindi_cascade")


def _manifest(pack):
    path = os.path.join(pack or "", "kindi_pack.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def io_rates(a, mib=64, log=print):
    """Saved `kindi io` rates when available, else one probe with a visible fallback."""
    from . import io_probe
    path = getattr(a, "io_report", None) or os.environ.get("KINDI_IO_REPORT")
    return io_probe.load_or_probe(path, mib=mib, log=log, label="kindi autotune")


def _expert_cache_mib(moe, rates):
    """The pinned-host cache size for the cold tier, from the rate-sized slot working set.

    The host cache is the RAM side of the same prediction as the decode slots: keep the
    rate-sized working set of each MoE layer in host memory so a slot refill is a RAM copy
    instead of a cold file read. The size is capped by the experts that actually live on the
    host tier (`n_card` is the card-resident count), so a pack that fits on the card gets no
    cache candidate. Returns None without a usable rate report or a host tier smaller than one
    token's selection.
    """
    m = moe or {}
    if not rates or m.get("fit") is False:
        return None
    try:
        n_used = int(m.get("n_used") or 0)
        n_exp = int(m.get("n_exp") or 0)
        n_moe = int(m.get("n_moe") or 0)
        eb = int(m.get("expert_bytes") or 0)
        n_card = max(0, int(m.get("n_card") or 0))
    except (TypeError, ValueError):
        return None
    host_per_layer = max(0, n_exp - min(n_exp, n_card))
    if n_used <= 0 or n_moe <= 0 or eb <= 0 or host_per_layer < n_used:
        return None
    from . import ltfit
    # Size against the full expert count and the plan's room, so `rate_slots` charges the same
    # staging buffers and decode slots as `fit_experts`. Without the room it could name a slot
    # count the engine's own planner would refuse. The cache itself covers only the non-card
    # experts, so its size is capped by the host tier after that fit.
    slots = ltfit.rate_slots(n_used, n_exp, eb, rates, room=m.get("room"), n_moe=n_moe)
    if not slots:
        return None
    host_slots = min(host_per_layer, slots)
    mib = max(1, int(math.ceil(host_slots * n_moe * eb / 2 ** 20)))
    # Never propose more pinned memory than this host can spare now. Use the engine's automatic
    # budget (available RAM less max(1 GiB, 10%)); passing an empty environment ignores an explicit
    # KINDI_LT_HOST_MB, which is the setting this candidate itself proposes.
    from .cold_experts import host_budget_bytes
    host_mib = host_budget_bytes(environ={}) >> 20
    if host_mib < 1:
        return None
    return min(mib, host_mib)


def candidates(report, manifest, ctx, max_runs=6, include_cpu=False, rates=None, moe=None):
    """The configurations worth measuring on this host, most likely useful first.

    Every entry is a dict: `name`, `why`, `args` (bench flags) and `env` (process environment).
    `moe` is the Lightning sizing report (`dcache`, `n_used`, `n_exp`); with `rates` it lets the
    slot candidates follow the measured I/O model instead of the fixed 16/6 pair.
    """
    cards = (report or {}).get("gpus") or []
    cpu = (report or {}).get("cpu") or {}
    backend = (manifest or {}).get("backend") or ""
    dense = backend not in NON_DENSE_BACKENDS
    physical = cpu.get("physical") or cpu.get("logical") or 0

    out = []
    if cards:
        out.append({"name": "gpu0", "why": "one card holds the whole model",
                    "args": [], "env": {}})
    if len(cards) >= 2:
        for split in ("auto", "equal"):
            out.append({"name": "tp-2-%s" % split, "why": "tensor-parallel row split over two cards",
                        "args": [], "env": {"KINDI_TP": "1", "KINDI_TP_SPLIT": split}})
    if cards and dense and physical >= 4:
        out.append({"name": "tandem-auto", "args": ["--tandem", "cpu-first",
                                                    "--tandem-cpu-layers", "auto"],
                    "why": "the smallest CPU slice that makes the context fit", "env": {}})
        for n in (8, 16):
            out.append({"name": "tandem-gdn:%d" % n,
                        "args": ["--tandem", "cpu-first", "--tandem-cpu-layers", "gdn:%d" % n],
                        "why": "fixed CPU slice: the first %d non-attention layers" % n, "env": {}})
    if backend in MOE_BACKENDS:
        sized = int((moe or {}).get("dcache") or 0)
        n_used = int((moe or {}).get("n_used") or 4)
        n_exp = int((moe or {}).get("n_exp") or 32)
        values = ([sized, max(n_used, sized // 2), min(n_exp, sized * 2)]
                  if sized else [16, 6])
        values = list(dict.fromkeys(max(1, int(v)) for v in values))
        from . import io_probe
        rate_note = io_probe.rate_line(rates) if rates else ""
        cache_mib = _expert_cache_mib(moe, rates)
        cache_why = ("pinned host expert cache for the cold tier: the rate-sized slot working "
                     "set kept in RAM; KINDI_LT_COLD=1 and KINDI_LT_HOST_MB")
        if rate_note:
            cache_why += " (%s)" % rate_note
        for i, slots in enumerate(values):
            why = "decode LRU slots per MoE layer against VRAM"
            if sized:
                why += "; sized from the measured I/O rates"
            elif rate_note:
                why += "; default pair (rates were not usable for sizing)"
            if rate_note:
                why += " (%s)" % rate_note
            out.append({"name": "dcache-%d" % slots,
                        "args": [], "env": {"KINDI_LT_DCACHE": str(slots)},
                        "why": why})
            # Keep the explicit cache candidate visible under the default --max-runs 6:
            # two cards already spend three slots on gpu0/tp-2-auto/tp-2-equal.
            if i == 0 and cache_mib:
                out.append({"name": "expert-cache-%d" % cache_mib,
                            "args": [],
                            "env": {"KINDI_LT_COLD": "1",
                                    "KINDI_LT_HOST_MB": str(cache_mib)},
                            "why": cache_why})
    if include_cpu or not cards:
        out.append({"name": "cpu", "why": "the CPU engine (KINDI_DEVICE=cpu)",
                    "args": [], "env": {"KINDI_DEVICE": "cpu"}})
    return out[:max(1, int(max_runs))]


def bench_argv(ctx, json_path, candidate, python=None):
    """The child command: `kindi --ctx N bench --json FILE` plus the candidate's own flags."""
    argv = [python or sys.executable, "-m", "kindi", "--ctx", str(int(ctx)), "bench",
            "--json", json_path, "--sample-vram"]
    return argv + list((candidate or {}).get("args") or [])


def _best_gen(gen):
    """The fastest decode rate in a bench result's {label: tok_s} mapping."""
    values = []
    for value in (gen or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def run_candidate(candidate, ctx, timeout=600.0, python=None, workdir=None):
    """One bench run. Returns a result dict; a failed run is recorded, never scored."""
    fd, json_path = tempfile.mkstemp(prefix="kindi-autotune-", suffix=".json", dir=workdir)
    os.close(fd)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in ((candidate or {}).get("env") or {}).items()})
    argv = bench_argv(ctx, json_path, candidate, python=python)
    result = {"name": (candidate or {}).get("name"), "ok": False, "gen_tok_s": None,
              "read_tok_s": None, "ctx_reduced": None, "vram_peak": None, "error": None,
              "argv": argv, "args": list((candidate or {}).get("args") or []),
              "env": dict((candidate or {}).get("env") or {})}
    try:
        done = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout)
        result["returncode"] = done.returncode
        text = done.stdout.decode("utf-8", "replace") if done.stdout else ""
        if done.returncode != 0:
            tail = [ln for ln in text.strip().splitlines() if ln.strip()]
            result["error"] = tail[-1] if tail else "exit %d" % done.returncode
            return result
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except subprocess.TimeoutExpired:
        result["error"] = "timed out after %.0f s" % timeout
        return result
    except (OSError, ValueError) as e:
        result["error"] = str(e)
        return result
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass
    result["ok"] = True
    result["gen_tok_s"] = _best_gen(payload.get("gen_tok_s"))
    result["read_tok_s"] = payload.get("read_tok_s")
    result["ctx_reduced"] = payload.get("ctx_reduced")
    result["ctx_requested"] = payload.get("ctx_requested")
    result["vram_peak"] = payload.get("vram_peak")
    return result


def rank(results):
    """Fastest decode first; a reduced context or a failed run never outranks a full one."""
    def key(item):
        reduced = 1 if item.get("ctx_reduced") else 0
        gen = item.get("gen_tok_s")
        return (0 if item.get("ok") else 1, reduced, -(gen if gen is not None else -1.0),
                item.get("name") or "")
    return sorted(results, key=key)


def _peak_mib(vram_peak):
    """Device-wide peak from `kindi/vram.py`'s summary (`peak_used_mib`), or None."""
    if isinstance(vram_peak, dict):
        for key in ("peak_used_mib", "peak_delta_mib", "peak_mib", "peak"):
            try:
                return int(vram_peak[key])
            except (KeyError, TypeError, ValueError):
                continue
    return None


def render(results, ctx, cards, cpu):
    """The ranked table plus the exact command that reproduces the winner."""
    lines = []
    lines.append("kindi autotune: ctx %d, %d card%s, %s (%s physical cores), %d run%s" %
                 (ctx, len(cards), "" if len(cards) == 1 else "s",
                  cpu.get("model") or "unknown CPU", cpu.get("physical"),
                  len(results), "" if len(results) == 1 else "s"))
    lines.append("  %-14s %10s %10s %10s  %s" % ("config", "decode", "read", "peak", "note"))
    ordered = rank(results)
    for item in ordered:
        if not item.get("ok"):
            lines.append("  %-14s %10s %10s %10s  failed: %s" %
                         (item.get("name"), "-", "-", "-", item.get("error") or "?"))
            continue
        gen = item.get("gen_tok_s")
        peak = _peak_mib(item.get("vram_peak"))
        note = ""
        if item.get("ctx_reduced"):
            note = "context reduced to fit"
        lines.append("  %-14s %10s %10s %10s  %s" %
                     (item.get("name"),
                      "%.1f" % gen if gen is not None else "-",
                      "%.1f" % item["read_tok_s"] if item.get("read_tok_s") is not None else "-",
                      "%d MiB" % peak if peak is not None else "-", note))
    best = next((item for item in ordered if item.get("ok") and item.get("gen_tok_s") is not None), None)
    if best is None:
        lines.append("no candidate produced a decode number; nothing is recommended")
        return lines
    lines.append("best: %s - %.1f tok/s decode at ctx %d" %
                 (best.get("name"), best["gen_tok_s"], ctx))
    env = best.get("env") or {}
    prefix = " ".join("%s=%s" % (k, v) for k, v in sorted(env.items()))
    argv = ["kindi", "--ctx", str(ctx), "serve"] + list(best.get("args") or [])
    command = ("%s %s" % (prefix, " ".join(argv))).strip()
    lines.append("serve it: %s" % command)
    if env:
        lines.append("(set those variables in the server's environment; the CLI has no flag for them)")
    if best.get("name") != "gpu0":
        lines.append("a different placement than one card: run `kindi verify` on it before serving")
    return lines


def run(cfg, a, log=print):
    from . import caps

    pack = cfg.get("pack")
    manifest = _manifest(pack)
    if not manifest:
        log("kindi autotune: no pack at %s (kindi convert MODEL.gguf, or --pack DIR)" % pack)
        return 2
    ctx = int(getattr(a, "ctx", None) or cfg.get("ctx") or 24576)
    report = caps.probe()
    rates, moe = None, None
    if manifest.get("backend") in MOE_BACKENDS:
        from . import plan as _plan
        rates = io_rates(a, log=log)
        gpu = (report.get("gpus") or [{}])[0]
        free = gpu.get("memory_free_mib")
        try:
            moe = _plan.describe(pack, ctx, None,
                                 assume_vram_gib=(free / 1024.0 if free else None),
                                 reserve_mib=int(os.environ.get("KINDI_LT_RESERVE_MB", "350")),
                                 rates=rates)
        except (Exception, SystemExit):
            moe = None
    options = candidates(report, manifest, ctx, max_runs=getattr(a, "max_runs", 6),
                         include_cpu=getattr(a, "include_cpu", False),
                         rates=rates, moe=moe)
    if getattr(a, "dry_run", False):
        log("kindi autotune: %d candidate%s at ctx %d (dry run: nothing is loaded)" %
            (len(options), "" if len(options) == 1 else "s", ctx))
        for item in options:
            env = " ".join("%s=%s" % (k, v) for k, v in sorted(item["env"].items()))
            log("  %-14s %s%s" % (item["name"], item["why"], (" [" + env + "]") if env else ""))
        return 0
    results = []
    timeout = float(getattr(a, "timeout_s", None) or 600.0)
    for item in options:
        log("kindi autotune: measuring %s (%s) ..." % (item["name"], item["why"]))
        result = run_candidate(item, ctx, timeout=timeout)
        if result.get("ok"):
            gen = result.get("gen_tok_s")
            log("kindi autotune: %s -> %s tok/s decode%s" %
                (item["name"], "%.1f" % gen if gen is not None else "no number",
                 " (context reduced)" if result.get("ctx_reduced") else ""))
        else:
            log("kindi autotune: %s failed (%s)" % (item["name"], result.get("error")))
        results.append(result)
    for line in render(results, ctx, report.get("gpus") or [], report.get("cpu") or {}):
        log(line)
    if getattr(a, "json", None):
        payload = {"pack": pack, "model": manifest.get("model"), "ctx": ctx,
                   "cards": report.get("gpus") or [], "cpu": report.get("cpu") or {},
                   "rates": rates, "moe": ({k: moe.get(k) for k in
                                           ("dcache", "n_used", "n_exp", "n_moe", "n_card",
                                            "expert_bytes", "room", "fit")}
                                          if moe else None),
                   "candidates": options, "results": results,
                   "best": (next((r for r in rank(results) if r.get("ok")
                                  and r.get("gen_tok_s") is not None), None) or {}).get("name")}
        try:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            log("kindi autotune: wrote %s" % a.json)
        except OSError as e:
            log("kindi autotune: could not write %s (%s)" % (a.json, e))
    return 0
