"""kindi - command line.

  kindi convert MODEL.gguf      build the weight pack from a PTQ1_0/PQ2_0/Q2_0 GGUF (once; ~1 minute)
  kindi fast-head               build the cluster file for the fast head (optional; ~2 minutes)
  kindi verify                  check greedy output against llama.cpp's reference (token for token)
  kindi bench                   measure speed and GPU memory
  kindi kld                     teacher-forced KLD quality gate for lossy K/V caches (vs fp16)
  kindi calibrate               measure the per-machine tuning once (5-trit configs, placement, draft length)
  kindi plan                    dry-run placement report (no weights loaded; --assume-vram-gib tests a card's split)
  kindi serve                   run the OpenAI-compatible server
  kindi chat                    chat in the terminal (no server needed; saves the session unless --no-record;
                                --resume SESSION_ID continues a saved session in place)
  kindi sessions                list, triage, tag and export chat sessions as raw markdown
  kindi head exact|fast         choose the output head of a running server (next request, no restart)
  kindi info                    show the configuration, the pack and the GPU
  kindi caps                    what each card and this CPU can do (and which Kindi paths that enables)
  kindi autotune                measure the placements for this model here and rank them (one bench each)
  kindi backends                list backend plugins registered in the plugin registry
Global options: --config FILE, --pack DIR, --state-dir DIR, --ctx N.
Tandem: verify/bench/serve/chat accept --tandem cpu-first|gpu-first --tandem-cpu-layers N|LIST|gdn:N
        and --tandem-gpu auto|plan|plain (the plan engine owner frees the offloaded layers' VRAM),
where LIST is an explicit layer set (0,1,2,4 or 0-2,4-6) and gdn:N moves the first N gated-delta
(non-attention) layers to the CPU, which keeps decode flat with context."""
import argparse
import json
import os
import re
import sys
import time

from . import __version__
from . import config as C


def log(msg):
    sys.stdout.write("[kindi %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def cmd_sessions(cfg, a):
    """kindi sessions list|show|export|triage|tag|status|import (host-only; reads state_dir/sessions)."""
    from .sessions import SessionStore, STATUSES
    st = SessionStore(os.path.join(cfg["state_dir"], "sessions"))

    def _flags(r):
        out = ""
        if r.get("unread"):
            out += "  [unread]"
        if r.get("protected"):
            out += "  [protected]"
        return out
    if a.action == "list":
        rows = st.list(status=a.status, tag=a.tag)
        if not rows:
            log("no sessions")
            return 0
        for r in rows:
            log("%s  %-24s %-13s %d msgs  %s%s" % (r["id"], r["title"][:24], r["status"],
                                               len(r.get("messages") or []),
                                               ",".join(r.get("tags") or []), _flags(r)))
        return 0
    if a.action == "show":
        if not a.id:
            raise SystemExit("kindi sessions show: give a session id")
        rec = st.get(a.id[0])
        st.mark_read(a.id[0])
        sys.stdout.write(st.export_markdown(rec))
        return 0
    if a.action == "triage":
        rows = st.triage()
        if not rows:
            log("triage queue is empty")
            return 0
        for r in rows:
            log("%s  %-24s %-13s %d msgs  %s%s" % (r["id"], r["title"][:24], r["status"],
                                               len(r.get("messages") or []),
                                               ",".join(r.get("tags") or []), _flags(r)))
        return 0
    if a.action == "protect":
        if not a.id:
            raise SystemExit("kindi sessions protect: give a session id")
        rec = st.set_protected(a.id[0], on=not a.off)
        log("session %s %sprotected" % (a.id[0], "un" if a.off else ""))
        return 0
    if a.action == "export":
        if not a.id:
            raise SystemExit("kindi sessions export: give at least one session id")
        for sid in a.id:
            path = st.export_file(st.get(sid), a.out or os.getcwd())
            log("exported %s -> %s" % (sid, path))
        return 0
    if a.action == "tag":
        if not a.id or not a.tag:
            raise SystemExit("kindi sessions tag: give a session id and --tag")
        st.add_tag(a.id[0], a.tag)
        log("tagged %s with %s" % (a.id[0], a.tag))
        return 0
    if a.action == "status":
        if not a.id or not a.status:
            raise SystemExit("kindi sessions status: give a session id and --status")
        if a.status not in STATUSES:
            raise SystemExit("kindi sessions: status must be one of %s (got %r)" % (STATUSES, a.status))
        st.set_status(a.id[0], a.status)
        log("session %s -> %s" % (a.id[0], a.status))
        return 0
    if a.action == "import":
        if not a.id:
            raise SystemExit("kindi sessions import: give a markdown file (or - for stdin)")
        text = sys.stdin.read() if a.id[0] == "-" else open(a.id[0], encoding="utf-8").read()
        rec = st.import_markdown(text)
        log("imported %s (%d msgs)" % (rec["id"], len(rec["messages"])))
        return 0
    if a.action == "prune":
        rows = st.prune(older_than_days=a.days, statuses=None if a.all else ("resolved",),
                        dry_run=bool(a.dry_run))
        if not rows:
            log("no sessions older than %d days" % a.days)
            return 0
        kept = sum(1 for r in rows if r.get("protected"))
        for r in rows:
            log("%s  %-24s %-13s updated %s%s" % (r["id"], r["title"][:24], r["status"], r["updated"],
                                                  "  (protected: kept)" if r.get("protected") else ""))
        log("%s %d session(s)" % ("would prune" if a.dry_run else "pruned", len(rows) - kept))
        if kept:
            log("%d protected session(s) kept" % kept)
        return 0
    if a.action == "stats":
        stats = st.stats()
        if a.json:
            import json as _json
            sys.stdout.write(_json.dumps(stats, indent=2) + "\n")
            return 0
        log("sessions: %d (open %d, needs-review %d, resolved %d, unread %d, protected %d)" % (
            stats["sessions"], stats["by_status"]["open"],
            stats["by_status"]["needs-review"], stats["by_status"]["resolved"],
            stats["unread"], stats["protected"]))
        log("messages: %d, stored bytes: %d" % (stats["messages"], stats["bytes"]))
        if stats["sessions"]:
            log("oldest updated %s, newest updated %s" % (stats["oldest_updated"], stats["newest_updated"]))
        return 0
    raise SystemExit("kindi sessions: unknown action")


def _record_chat(cfg, msgs):
    """Save a chat transcript as a session (host-only; no GPU touched)."""
    from .sessions import SessionStore
    convo = [m for m in msgs if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()]
    if not convo:
        return None
    title = next((str(m["content"]).strip().splitlines()[0][:60] for m in convo if m["role"] == "user"), "chat")
    st = SessionStore(os.path.join(cfg["state_dir"], "sessions"))
    rec = st.create(title=title, messages=[(m["role"], m["content"]) for m in convo])
    return rec["id"]


def _resume_chat(cfg, sid):
    """Load a stored session's turns as the chat seed; unknown or malformed ids refuse loudly."""
    from .sessions import SessionStore
    if not sid or not re.fullmatch(r"[0-9a-f]{12}", str(sid)):
        raise SystemExit("kindi chat --resume: the session id must be 12 hex characters (got %r)" % sid)
    st = SessionStore(os.path.join(cfg["state_dir"], "sessions"))
    try:
        rec = st.get(sid)
    except KeyError:
        raise SystemExit("kindi chat --resume: no session %s (kindi sessions list)" % sid)
    return rec, [{"role": m["role"], "content": m["content"]}
                  for m in rec.get("messages", [])
                  if m.get("role") in ("system", "user", "assistant")]


def _append_chat(cfg, sid, msgs):
    """Append the new turns of a resumed chat back to its stored session."""
    from .sessions import SessionStore
    convo = [m for m in msgs if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()]
    if not convo:
        return None
    st = SessionStore(os.path.join(cfg["state_dir"], "sessions"))
    rec = st.append_messages(sid, [(m["role"], m["content"]) for m in convo])
    return rec["id"]


def _setup_gpu_env(cfg):
    """Kernel caches under state_dir, set before CUDA starts: CuPy's (CUDA C -> PTX) and the driver's JIT cache
    (PTX -> sm_30 code, ~1 minute on a K420 - without a writable cache the driver redoes it on every start)."""
    kdir = os.path.join(cfg["state_dir"], "kernels")
    jdir = os.path.join(cfg["state_dir"], "kernels", "driver")
    os.makedirs(jdir, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", kdir)
    os.environ.setdefault("CUDA_CACHE_PATH", jdir)
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(512 * 1024 * 1024))
    # The tandem engine reads these before it builds the GPU owner. The config loader has already
    # applied file < environment < CLI precedence; writing them here makes that one decision the
    # value every backend sees, including a direct TandemEngine construction.
    os.environ["KINDI_TANDEM_VRAM_MB"] = str(int(cfg.get("tandem_vram_mb", 0) or 0))
    os.environ["KINDI_TANDEM_RESERVE_MB"] = str(int(cfg.get("tandem_reserve_mb", 512) or 0))


def cmd_convert(cfg, a):
    if not a.gguf:
        raise SystemExit("kindi convert: give the model GGUF")
    from .convert import convert
    out = a.out or cfg["pack"]
    if os.path.exists(os.path.join(out, "kindi_pack.json")) and not a.force:
        raise SystemExit("kindi: %s already holds a pack (use --force to rebuild)" % out)
    manifest = convert(a.gguf, out, log=log)
    if manifest.get("backend") == "kindi_gemma4":
        log("Gemma 4 model ready for Kindi native CUDA inference")
    else:
        log("next: kindi verify")
    return 0


def cmd_fast_head(cfg, a):
    _setup_gpu_env(cfg)
    manifest_path=os.path.join(cfg["pack"],"kindi_pack.json")
    manifest=json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}
    if manifest.get("backend")=="kindi_gemma4":
        from .gemma4 import build_fast_head
        build_fast_head(manifest["model_path"],cfg["pack"],K=a.clusters,iters=a.iters,log=log)
    else:
        from .cluster import build
        build(cfg["pack"], K=a.clusters, iters=a.iters, log=log)
    log("restart `kindi serve`, then `kindi head fast` to use it")
    return 0


def cmd_verify(cfg, a):
    spec = getattr(a, "spec", None)
    if spec:
        os.environ["KINDI_SPEC"] = spec          # the mode's own variables (KINDI_MTP_HEAD, KINDI_SPEC_PACK) stay as set
    _setup_gpu_env(cfg)
    from .verify import run
    return run(cfg, log=log)


def cmd_bench(cfg, a):
    _setup_gpu_env(cfg)
    from .bench import run
    return run(cfg, log=log)


def cmd_plan(cfg, a):
    from . import plan as _plan
    return _plan.run(cfg, a)


def cmd_kld(cfg, a):
    _setup_gpu_env(cfg)
    from .kld import run, write_json
    results = run(cfg, log=log)
    if getattr(a, "json", None):
        write_json(results, a.json)
        log("wrote %s" % a.json)
    return 0


def cmd_calibrate(cfg, a):
    _setup_gpu_env(cfg)
    from .calibrate import run
    return run(cfg, log=log)


def _require_pack(cfg):
    """Fail before CUDA starts when the configured pack directory or manifest is missing."""
    pack = cfg["pack"]
    manifest = os.path.join(pack, "kindi_pack.json")
    source = cfg.get("source") or "the config file"
    if source.startswith("("):
        source = "the config file"
    if not os.path.isdir(pack):
        raise SystemExit(
            "kindi: the configured pack directory does not exist:\n"
            "  %s\n"
            "Set 'pack' in %s or KINDI_PACK, or run `kindi use <model>`." % (pack, source))
    if not os.path.isfile(manifest):
        raise SystemExit(
            "kindi: the configured pack is incomplete:\n"
            "  %s\n"
            "kindi_pack.json is missing. Run `kindi use <model>` or "
            "`kindi convert MODEL.gguf --out %s`." % (pack, pack))
    return pack


def cmd_serve(cfg, a):
    _require_pack(cfg)
    _setup_gpu_env(cfg)
    from .server import serve
    serve(cfg, log=log)
    return 0


def cmd_chat(cfg, a):
    _require_pack(cfg)
    _setup_gpu_env(cfg)
    from .server import Kindi
    K = Kindi(cfg, log=log)
    msgs, seed_len = [], 0
    if a.resume:
        rec, seeded = _resume_chat(cfg, a.resume)
        msgs = [{"role": "system", "content": a.system}] if a.system else []
        msgs += seeded
        seed_len = len(msgs)
        print("resumed session %s (%d prior turns)" % (rec["id"], len(seeded)))
    elif a.system:
        msgs.append({"role": "system", "content": a.system})
    print("Kindi %s - type a message, /reset to start over, /exit to quit." % __version__)
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() in ("/exit", "/quit"):
            break
        if line.strip() == "/reset":
            msgs = msgs[:1] if a.system else []
            continue
        if not line.strip():
            continue
        msgs.append({"role": "user", "content": line})
        sys.stdout.write("kindi> ")
        reply = ""
        for kind, val in K.generate({"messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens}):
            if kind == "text":
                reply += val
                sys.stdout.write(val)
                sys.stdout.flush()
            else:
                sys.stdout.write("\n  (%.1f tokens/s)" % val["timing"]["gen_tok_s"])
        msgs.append({"role": "assistant", "content": reply})
    if not a.no_record:
        if a.resume:
            sid = _append_chat(cfg, a.resume, msgs[seed_len:])
            if sid:
                print("\nappended to session %s (kindi sessions list)" % sid)
        else:
            sid = _record_chat(cfg, msgs)
            if sid:
                print("\nsaved session %s (kindi sessions list)" % sid)


def cmd_head(cfg, a):
    os.makedirs(cfg["state_dir"], exist_ok=True)
    if a.mode == "fast" and not os.path.exists(os.path.join(cfg["pack"], "head_clusters.npz")):
        raise SystemExit("kindi: the fast head needs the cluster file - run: kindi fast-head")
    with open(os.path.join(cfg["state_dir"], "head_mode"), "w") as f:
        f.write(a.mode + "\n")
    log("head mode: %s (the next request uses it)" % a.mode)
    return 0


def cmd_models(cfg, a):
    from .catalog import list_models
    list_models(cfg, log=log)
    return 0


def cmd_use(cfg, a):
    from .catalog import use
    return use(cfg, a.model, gguf=a.gguf, log=log)


def cmd_backends(cfg, a):
    """List the backend plugin registry (host-only; no pack or GPU touched)."""
    from .backends import list_backends, load_configured_backends
    load_configured_backends()
    rows = list_backends()
    if getattr(a, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        log("no backend plugins registered")
        return 0
    for row in rows:
        log("%-16s devices=%-10s architectures=%-20s model_ids=%s" % (
            row.get("name", ""),
            ",".join(row.get("devices") or []) or "-",
            ",".join(row.get("architectures") or []) or "-",
            ",".join(row.get("model_ids") or []) or "-"))
    return 0


def cmd_caps(cfg, a):
    from . import caps as _caps
    report = _caps.probe()
    if getattr(a, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    for line in _caps.render(report):
        print(line)
    return 0


def cmd_intel_info(cfg, a):
    """Read-only Intel M1 probe; it never loads a backend or enters an engine path."""
    from . import intel
    report = intel.probe()
    if getattr(a, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(intel.render(report))
    return 0


def cmd_intel_smoke(cfg, a):
    """Level Zero device-memory round trip on an Intel GPU; no kernel or model."""
    from . import intel_l0

    report = intel_l0.run_smoke(a.mib, loader=a.loader, device=a.device)
    if a.json:
        print(intel_l0.dumps(report))
    else:
        for line in intel_l0.render(report):
            print(line)
    return 0 if report["result"] == "pass" else 1


def cmd_intel_cl(cfg, a):
    """Runtime-compiled OpenCL vector-add and GEMM on an Intel GPU; no model."""
    from . import intel_cl

    report = intel_cl.run_smoke(a.n, dim=a.dim, repeats=a.repeats, loader=a.loader,
                                device=a.device)
    if a.json:
        print(intel_cl.dumps(report))
    else:
        for line in intel_cl.render(report):
            print(line)
    return 0 if report["result"] == "pass" else 1


def cmd_loop_info(cfg, a):
    """Inspect a looped-transformer GGUF map without loading any tensor data (REQ-0031)."""
    from . import looped
    from .gguf import GGUF

    try:
        g = GGUF(a.gguf)
    except Exception as e:  # noqa: BLE001 - a malformed GGUF is a CLI error, not a traceback
        raise SystemExit("kindi loop-info: %s" % e) from None
    changed = looped.mismatch(g)
    if changed:
        raise SystemExit("kindi loop-info: " + changed)
    report = looped.layout(g)
    if report is None:
        raise SystemExit("kindi loop-info: %s is not a recognized looped GGUF (REQ-0031)" % a.gguf)
    if a.json:
        print(json.dumps(looped.report_dict(report), indent=2, sort_keys=True))
    else:
        for line in looped.render(report):
            print(line)
    return 1 if (report.missing or report.unexpected) else 0


def cmd_io(cfg, a):
    from . import io_probe
    return io_probe.run(a.mib, a.disk_dir, a.skip, a.json)

def cmd_autotune(cfg, a):
    from . import autotune
    return autotune.run(cfg, a, log=log)


def cmd_info(cfg, a):
    print("Kindi %s" % __version__)
    from . import cpu as cpu_backend
    report = cpu_backend.feature_report()
    print("CPU: %s, %s physical / %s logical cores" %
          (report["model"], report["physical"], report["logical"] if report["logical"] is not None else "?"))
    print("CPU engine: %s" % ", ".join(
        "%s=%s" % (label, "unknown" if report["features"].get(name) is None else
                   ("yes" if report["features"][name] else "no"))
        for name, label in cpu_backend.CPU_FEATURES))
    tier = cpu_backend.cpu_tier(report)
    print("CPU engine path: %s" % cpu_backend.CPU_TIER_LABELS.get(tier, tier))
    forced = os.environ.get("KINDI_CPU_ISA", "")
    build = cpu_backend.select_isa(report)
    print("CPU engine build: %s%s" % (build or "native (-march=native)",
                                      " (KINDI_CPU_ISA)" if forced and forced != "auto" else ""))
    from . import mtp_heads
    # the requested mode and the head's registry verdict, which is what can be known before an engine
    # loads; the effective mode is the serve log's "speculation: requested ... effective ..." (REQ-0024)
    print("speculation: %s" % mtp_heads.report())
    from . import runtime as runtime_view
    print(runtime_view.render(runtime_view.describe(cfg)))
    print("config: %s" % cfg["source"])
    for k in sorted(C.DEFAULTS):
        v = cfg[k]
        print("  %-17s %s" % (k, "(set)" if k == "api_key" and v else v))
    man = os.path.join(cfg["pack"], "kindi_pack.json")
    if os.path.exists(man):
        m = json.load(open(man))
        print("pack: %s - %s, from %s" % (cfg["pack"], m.get("model"), m.get("source_file")))
        if m.get("backend") == "kindi_cascade":
            print("backend: Kindi Cascade (Nemotron Cascade 2 converter/validator; GPU kernels pending)")
        elif m.get("backend") == "kindi_lightning":
            print("backend: Kindi Lightning CUDA (Nemotron-H MoE), exact head")
        elif m.get("backend") == "kindi_nanbeige":
            print("backend: Kindi Nanbeige CUDA (llama K/I-quant), exact head")
        elif m.get("backend") == "kindi_dsv41":
            print("backend: Kindi DeepSeek V4.1 (compiled CPU engine)")
        elif m.get("backend") == "kindi_gemma4":
            print("backend: Kindi Gemma 4 CUDA")
            print("fast head: %s" % ("available" if os.path.exists(os.path.join(cfg["pack"],"head_clusters.npz"))
                                     else "not built (kindi fast-head)"))
        else:
            print("fast head: %s" % ("available" if os.path.exists(os.path.join(cfg["pack"], "head_clusters.npz"))
                                     else "not built (kindi fast-head)"))
    else:
        print("pack: none at %s (kindi convert MODEL.gguf)" % cfg["pack"])
    try:
        import cupy as cp
        d = cp.cuda.Device()
        free, total = cp.cuda.runtime.memGetInfo()
        name = cp.cuda.runtime.getDeviceProperties(d.id)["name"]
        name = name.decode() if isinstance(name, bytes) else name
        print("GPU: %s, compute %s, %.0f of %.0f MiB free" % (name, d.compute_capability, free / 2 ** 20, total / 2 ** 20))
    except Exception as e:  # noqa: BLE001
        print("GPU: not available (%s)" % e)
    return 0


def _malloc_tuning():
    """Checkpoints are dozens of 2 MB host arrays per state copy. glibc raises its mmap threshold as large blocks are
    freed, after which such arrays come from the heap of each thread and are never given back (a 9B server grew by
    ~60 MB per request). A fixed 1 MB threshold keeps every such array in its own mapping, returned when freed."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").mallopt(-3, 1 << 20)       # M_MMAP_THRESHOLD = -3
    except (OSError, AttributeError):
        pass                                               # not glibc: nothing to tune


def _split_at_arg(text):
    from .bench import split_prompts
    try:
        split_prompts({"split_at": text})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    return text


def _tandem_layers_arg(text):
    from .tandem import _parse_layers
    try:
        _parse_layers(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    return text


def _mib_arg(text):
    try:
        value = int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be an integer number of MiB") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def _vram_args(s):
    """A per-card ceiling and free-memory reserve for the dense tandem planner."""
    s.add_argument("--vram-budget-mib", type=_mib_arg, default=None, metavar="N",
                   help="ceiling in MiB for the tandem GPU side (0: no ceiling); the planner uses "
                        "min(free - reserve, this budget), so a 6 GiB P106 can be capped at 6144 "
                        "and a display card can keep a lower barrier")
    s.add_argument("--vram-reserve-mib", type=_mib_arg, default=None, metavar="N",
                   help="free card memory the tandem planner leaves for the display/driver "
                        "(default: 512; KINDI_TANDEM_RESERVE_MB)")


def _tandem_args(s):
    """The dense CPU+GPU split flags, available on the commands that build an engine."""
    s.add_argument("--tandem", choices=("off", "cpu-first", "gpu-first"), default=None,
                   help="split the Bonsai layers between the CPU and one GPU (default: off)")
    s.add_argument("--tandem-cpu-layers", type=_tandem_layers_arg, default=None,
                   help="layers the CPU owns in a tandem split: a count (cpu-first prefix / "
                        "gpu-first suffix), an explicit list such as 0,1,2,4,5,6 / 0-2,4-6, "
                        "gdn:N (the first N non-attention layers), auto[:MiB] (the smallest "
                        "set that makes the context fit, or refuse when no split is needed) "
                        "or rate[:FLOOR] (default: a quarter of the model)")
    s.add_argument("--tandem-gpu", choices=("auto", "plan", "plain"), default=None,
                   help="engine that owns the GPU side of a tandem split: auto (the plan engine, "
                        "which loads only the GPU-owned layers and uses q8_0 K/V), plan or plain "
                        "(also q8_0 K/V; the only owner for a two-card split)")
    _vram_args(s)


def _build_parser():
    """The complete command-line grammar: global options plus every subcommand."""
    p = argparse.ArgumentParser(prog="kindi", description="Kindi %s - native CUDA inference for ternary models and Gemma 4" % __version__)
    p.add_argument("--config")
    p.add_argument("--pack")
    p.add_argument("--state-dir", dest="state_dir")
    p.add_argument("--ctx", type=int)
    p.add_argument("--version", action="version", version="kindi " + __version__)
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("convert", help="build the weight pack from the GGUF")
    s.add_argument("gguf", nargs="?")
    s.add_argument("--out", help="pack directory (default: the configured pack)")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("fast-head", help="build the fast-head cluster file")
    s.add_argument("--clusters", type=int, default=1024)
    s.add_argument("--iters", type=int, default=12)
    s = sub.add_parser("verify", help="check against llama.cpp's reference output")
    _tandem_args(s)
    s.add_argument("--spec", default=None, choices=("ngram", "mtp", "draft2b", "both", "dspine", "auto", "0"),
                   help="verify under this speculation mode (same as KINDI_SPEC; the mode's own variables, "
                        "KINDI_MTP_HEAD or KINDI_SPEC_PACK, are read as usual); the effective mode is printed "
                        "and a drafter that did not attach exits non-zero")
    s = sub.add_parser("bench", help="measure speed")
    _tandem_args(s)
    s.add_argument("--json", default=None, metavar="FILE", help="also record the numbers in a machine-readable JSON file")
    s.add_argument("--sample-vram", action="store_true", help="sample device-wide VRAM and log the peak during the run")
    s.add_argument("--dcache", type=int, default=None, metavar="N",
                   help="MoE decode LRU slots per layer for this run (KINDI_LT_DCACHE; default 10)")
    s.add_argument("--split-at", type=_split_at_arg, default=None, metavar="LIST",
                   help="positive prompt lengths (comma-separated) to time as one whole request and "
                        "report the prefill/decode share of wall clock (default: 4096,8192; "
                        "'none' or 'off' turns it off)")
    s.add_argument("--split-generated", type=int, default=None, metavar="TOKENS",
                   help="tokens generated in each --split-at request (default: 128)")
    s = sub.add_parser("kld", help="teacher-forced KLD quality gate for lossy K/V caches against fp16")
    s.add_argument("--json", default=None, metavar="FILE", help="also record the numbers in a machine-readable JSON file")
    s = sub.add_parser("calibrate", help="measure the per-machine tuning once and persist it")
    s.add_argument("--vram-sample-s", type=float, default=None, help="VRAM sampling interval in seconds (default: 0.5)")
    _tandem_args(s)
    s = sub.add_parser("plan", help="dry-run placement report without loading weights (--assume-* for an absent card)")
    s.add_argument("--ctx", type=int, default=None)
    _tandem_args(s)
    s.add_argument("--assume-vram-gib", type=float, default=None, help="the card's free VRAM, instead of probing it")
    s.add_argument("--assume-ram-gib", type=float, default=None, help="the host RAM, instead of probing it")
    s.add_argument("--assume-cores", type=int, default=None, help="the CPU cores, instead of probing them")
    s.add_argument("--portfolio", action="store_true",
                   help="compare full-GPU, cpu-first and gpu-first dense placements in one dry run "
                        "(no weights loaded; capacity-first recommendation, not a throughput claim)")
    s.add_argument("--usage", default=None, help="an MoE usage file (kindi.moe Usage JSON); defaults to the shipped lightning-experts.json")
    s.add_argument("--io-report", default=None, metavar="FILE",
                   help="reuse a saved `kindi io --json FILE` report instead of measuring the rates "
                        "again (also KINDI_IO_REPORT); a missing report falls back to one probe")
    s.add_argument("--moe-cache-mib", type=int, default=None, help="cap the MoE experts' card budget (planner only; the engine derives it from free VRAM)")
    s.add_argument("--dcache", type=int, default=None, help="MoE decode LRU slots per layer (KINDI_LT_DCACHE; default 10)")
    s.add_argument("--moe-chunk", type=int, default=None, metavar="TOKENS",
                   help="also report what MoeTandem would do for a TOKENS-token sample prompt chunk "
                        "(lane split, distinct copies, bytes); planner only, no bytes moved")
    s.add_argument("--moe-lane", choices=("copy", "cpu"), default=None,
                   help="the tandem lane the --moe-chunk report plans for (default copy)")
    s.add_argument("--json", default=None, metavar="FILE", help="also record the report in a machine-readable JSON file")
    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--api-key", dest="api_key")
    s.add_argument("--head", choices=("exact", "fast"))
    _tandem_args(s)
    s = sub.add_parser("chat", help="chat in the terminal")
    s.add_argument("--system", default="")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    s.add_argument("--no-record", action="store_true", help="do not save this chat as a session (state_dir/sessions)")
    s.add_argument("--resume", metavar="SESSION_ID", help="continue a saved session (kindi sessions list) instead of starting a new one; new turns append back to it")
    _tandem_args(s)
    s = sub.add_parser("sessions", help="list, triage, tag and export chat sessions (state_dir/sessions)")
    s.add_argument("action", choices=("list", "show", "export", "triage", "tag", "status", "import", "prune", "stats", "protect"))
    s.add_argument("--off", action="store_true", help="protect: clear the protection flag instead of setting it")
    s.add_argument("id", nargs="*", help="session id (show/export/tag/status) or file (import, - for stdin)")
    s.add_argument("--out", help="export directory (default: current directory)")
    s.add_argument("--tag", help="tag filter (list) or tag to add (tag)")
    s.add_argument("--status", help="status filter (list) or new status (status)")
    s.add_argument("--days", type=int, default=30, help="prune: delete sessions not updated for this many days")
    s.add_argument("--all", action="store_true", help="prune: delete sessions of any status (default: resolved only)")
    s.add_argument("--dry-run", action="store_true", help="prune: list what would be deleted without deleting")
    s.add_argument("--json", action="store_true", help="stats: print the summary as JSON")
    s = sub.add_parser("head", help="exact or fast output head for a running server")
    s.add_argument("mode", choices=("exact", "fast"))
    sub.add_parser("info", help="show configuration, pack and GPU")
    s = sub.add_parser("caps", help="what each card and this CPU can do, and which Kindi paths that enables")
    s.add_argument("--json", action="store_true", help="print the whole report as JSON")
    s = sub.add_parser("intel-info", help="read-only Intel M1 probe: CPU, display, NPU, Level Zero and OpenVINO")
    s.add_argument("--json", action="store_true", help="print the whole probe as JSON")
    s = sub.add_parser("intel-smoke", help="Level Zero device-memory round trip on an Intel GPU (no kernel)")
    s.add_argument("--mib", type=int, default=1, help="device USM allocation in MiB (default 1, max 256)")
    s.add_argument("--device", type=int, default=0, help="Level Zero device index (default 0)")
    s.add_argument("--loader", default=None, help="Level Zero loader name/path (default: the host loader)")
    s.add_argument("--json", action="store_true", help="print the whole report as JSON")
    s = sub.add_parser("intel-cl", help="compile and run OpenCL vector-add/GEMM on an Intel GPU (no model)")
    s.add_argument("--n", type=int, default=1048576, help="vector-add elements (default 1048576, max 16777216)")
    s.add_argument("--dim", type=int, default=128, help="square GEMM dimension (default 128, 8-512)")
    s.add_argument("--repeats", type=int, default=5, help="timed GEMM launches (default 5, max 50)")
    s.add_argument("--device", type=int, default=0, help="Intel OpenCL GPU index (default 0)")
    s.add_argument("--loader", default=None, help="OpenCL loader name/path (default: the host loader)")
    s.add_argument("--json", action="store_true", help="print the whole report as JSON")
    s = sub.add_parser("loop-info", help="inspect a looped-transformer GGUF map (no tensor data loaded)")
    s.add_argument("gguf")
    s.add_argument("--json", action="store_true", help="print the whole report as JSON")
    s = sub.add_parser("io", help="measure this host's memory-tier rates: RAM copy, disk read, PCIe copies")
    s.add_argument("--mib", type=int, default=128, help="buffer/file size per probe in MiB (default 128)")
    s.add_argument("--disk-dir", default=None, help="where to create the temporary disk-probe file (default: the system temp dir)")
    s.add_argument("--skip", default="", help="comma-separated probes to skip: ram,disk,pcie")
    s.add_argument("--json", nargs="?", const=True, default=False, metavar="FILE",
                   help="print the report as JSON; with FILE, also save it for plan/autotune reuse")
    s = sub.add_parser("autotune", help="measure the placements for this model and rank them (one bench per candidate)")
    _vram_args(s)
    s.add_argument("--max-runs", type=int, default=6, help="how many configurations to measure (default 6)")
    s.add_argument("--dry-run", action="store_true", help="print the candidate list and stop (loads nothing)")
    s.add_argument("--io-report", default=None, metavar="FILE",
                   help="reuse a saved `kindi io --json FILE` report instead of measuring the rates "
                        "again (also KINDI_IO_REPORT); a missing report falls back to one probe")
    s.add_argument("--include-cpu", action="store_true", help="also measure the CPU engine, even with a card present")
    s.add_argument("--timeout-s", type=float, default=600.0, help="per-candidate time limit (default 600 s)")
    s.add_argument("--json", default=None, metavar="FILE", help="record the candidates, every run and the winner")
    s = sub.add_parser("backends", help="list registered backend plugins")
    s.add_argument("--json", action="store_true", help="print registered backends as JSON")
    sub.add_parser("models", help="list the models and what fits this machine")
    s = sub.add_parser("use", help="download a model, build its pack and select it")
    s.add_argument("model")
    s.add_argument("--gguf", help="use this GGUF file instead of downloading it")
    return p


def _apply_overrides(a):
    """Merge the global CLI options into the config; command-only values stay on `a`."""
    ov = {k: getattr(a, k, None) for k in ("pack", "state_dir", "ctx", "host", "port", "api_key", "head")}
    cfg = C.load(a.config, ov)
    if getattr(a, "tandem", None) is not None:
        cfg["tandem"] = a.tandem
    if getattr(a, "tandem_cpu_layers", None) is not None:
        cfg["tandem_cpu_layers"] = a.tandem_cpu_layers
    if getattr(a, "tandem_gpu", None) is not None:
        cfg["tandem_gpu"] = a.tandem_gpu
    if getattr(a, "vram_budget_mib", None) is not None:
        cfg["tandem_vram_mb"] = int(a.vram_budget_mib)
    if getattr(a, "vram_reserve_mib", None) is not None:
        cfg["tandem_reserve_mb"] = int(a.vram_reserve_mib)
    if getattr(a, "json", None) is not None:
        cfg["json"] = a.json
    if getattr(a, "sample_vram", False):
        cfg["sample_vram"] = True
    if getattr(a, "dcache", None) is not None:
        cfg["dcache"] = a.dcache
    if getattr(a, "split_at", None) is not None:
        cfg["split_at"] = a.split_at
    if getattr(a, "split_generated", None) is not None:
        cfg["split_generated"] = a.split_generated
    if getattr(a, "vram_sample_s", None) is not None:
        cfg["vram_sample_s"] = a.vram_sample_s
    return cfg


_COMMANDS = {
    "convert": cmd_convert,
    "fast-head": cmd_fast_head,
    "verify": cmd_verify,
    "bench": cmd_bench,
    "kld": cmd_kld,
    "calibrate": cmd_calibrate,
    "plan": cmd_plan,
    "serve": cmd_serve,
    "chat": cmd_chat,
    "head": cmd_head,
    "info": cmd_info,
    "caps": cmd_caps,
    "intel-info": cmd_intel_info,
    "intel-smoke": cmd_intel_smoke,
    "intel-cl": cmd_intel_cl,
    "loop-info": cmd_loop_info,
    "io": cmd_io,
    "autotune": cmd_autotune,
    "backends": cmd_backends,
    "models": cmd_models,
    "use": cmd_use,
    "sessions": cmd_sessions,
}


def _dispatch(cfg, a):
    """Call the selected command through the module global, so it stays replaceable."""
    return getattr(sys.modules[__name__], _COMMANDS[a.cmd].__name__)(cfg, a)


def main(argv=None):
    _malloc_tuning()
    p = _build_parser()
    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1
    return _dispatch(_apply_overrides(a), a)


if __name__ == "__main__":
    sys.exit(main())
