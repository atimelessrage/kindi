#!/usr/bin/env python3
"""Real tandem prompt-cache check over the HTTP server path.

The in-process harness (run_check.py) proves the cache inside one process. This one exercises
the same joint cache through `kindi serve`: a warm system prompt, a branch request that writes
an opening to disk, a server restart, and a request that must load that opening. A separate
server with KINDI_PROMPT_CACHE=0 on a fresh state directory generates the reference reply.

Every server is a real process. The saved opening is read from its metadata on disk, and the
reply cached-token count comes from the response's usage.prompt_tokens_details.cached_tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEFAULT_PACK = Path.home() / ".kindi" / "packs" / "bonsai-2b"
DEFAULT_CPU_CACHE = HERE / "cpu-cache"
def toolchain(clang, llvm):
    """The clang/LLVM the CPU engine compiles with: the flags first, then KINDI_CC/KINDI_LLVM.

    The run that produced these results named its own toolchain (`--clang`/`--llvm`, which the
    result file records); no machine's install path is baked into the script.
    """
    clang = clang or os.environ.get("KINDI_CC")
    llvm = llvm or os.environ.get("KINDI_LLVM")
    if not clang or not llvm:
        raise SystemExit("pass --clang and --llvm (or set KINDI_CC and KINDI_LLVM) to name the CPU engine's toolchain")
    return Path(clang).expanduser(), Path(llvm).expanduser()


SYSTEM = ("The archive contains carefully indexed notes about weather, rivers, tools, "
          "and quiet places. " * 20).strip()
USERS = [
    "Answer briefly: name one color.",
    "Choose a different color.",
    "Name a third color.",
]
MODES = ("cpu-first", "gpu-first")


def prompt_token_info(pack: Path) -> dict:
    """Token lengths and the first system-end position, without loading the engine."""
    from kindi import chat
    from kindi.tokenizer import Tok

    vocab = json.load(open(pack / "vocab.json", encoding="utf-8"))
    tok = Tok(vocab, str(pack / "merges.txt"))
    out = []
    for user in USERS:
        body, gp = chat.render(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            variant="qwen35",
        )
        ids = tok.encode(body) + tok.encode(gp)
        opening_end = ids.index(248046) + 1
        out.append({"tokens": len(ids), "opening_end": opening_end,
                    "first_token": ids[0]})
    return {"prompts": out}


def server_env(pack: Path, cpu_cache: Path, clang: Path, llvm: Path, prompt_cache: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KINDI_")}
    env.update({
        "KINDI_DEVICE": "gpu",
        "KINDI_SPEC": "0",
        "KINDI_CARDS": "1",
        "KINDI_HEAD": "exact",
        "KINDI_PROMPT_CACHE": "1" if prompt_cache else "0",
        "KINDI_CPU_ISA": "avx2-fma-f16c",
        "KINDI_CPU_CACHE": str(cpu_cache),
        "KINDI_CC": str(clang),
        "KINDI_LLVM": str(llvm),
        "PYTHONUNBUFFERED": "1",
    })
    return env


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write_config(state_dir: Path, ctx: int) -> Path:
    cfg = state_dir.parent / (state_dir.name + ".conf")
    cfg.write_text(
        "[kindi]\n"
        "state_dir = %s\n"
        "ctx = %d\n"
        "prefix_max = 1\n"
        "prefix_min = 256\n"
        "park_max = 0\n"
        "checkpoint_every = 32\n"
        "checkpoint_budget_mb = 512\n"
        "head = exact\n"
        "host = 127.0.0.1\n"
        % (state_dir.as_posix(), ctx),
        encoding="utf-8",
    )
    return cfg


def tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_server(pack: Path, state_dir: Path, mode: str, cpu_layers: int, ctx: int,
                 prompt_cache: bool, clang: Path, llvm: Path, cpu_cache: Path,
                 phase: str, timeout: float):
    state_dir.mkdir(parents=True, exist_ok=True)
    cfg = write_config(state_dir, ctx)
    port = free_port()
    log_path = state_dir.parent / (state_dir.name + "-" + phase + ".log")
    handle = open(log_path, "w", encoding="utf-8", buffering=1)
    cmd = [
        sys.executable, "-m", "kindi",
        "--config", str(cfg),
        "--pack", str(pack),
        "--ctx", str(ctx),
        "serve",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tandem", mode,
        "--tandem-cpu-layers", str(cpu_layers),
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=server_env(pack, cpu_cache, clang, llvm, prompt_cache),
        stdout=handle, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            last_error = "server exited with %s" % proc.returncode
            break
        if health(port):
            return proc, handle, log_path, port
        time.sleep(0.5)
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    handle.close()
    detail = tail(log_path)
    raise RuntimeError("server did not become ready (%s):\n%s" % (last_error or "timeout", detail))


def stop_server(proc, handle) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        try:
            handle.close()
        except Exception:
            pass


def chat(port: int, user: str, tokens: int) -> dict:
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        data = json.load(r)
    choice = data["choices"][0]
    usage = data.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    return {
        "content": choice.get("message", {}).get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "raw_usage": usage,
    }


def opening_entries(state_dir: Path) -> list:
    out = []
    for p in sorted((state_dir / "openings").glob("*/*/meta.json")):
        try:
            meta = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"path": str(p.relative_to(state_dir)), "p": meta.get("p"),
                    "engine": meta.get("engine")})
    return out


def compare(a: dict, b: dict) -> dict:
    keys = ("content", "finish_reason", "completion_tokens")
    diffs = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
    return {"exact": not diffs, "diffs": diffs}


def run_mode(mode: str, args) -> dict:
    stamp = "%s-%d" % (mode, os.getpid())
    base = Path(args.run_dir) / stamp
    cached_state = base / "cached-state"
    off_state = base / "off-state"
    base.mkdir(parents=True, exist_ok=True)

    warm = {"phase": "cached-warm"}
    proc = handle = None
    try:
        proc, handle, log_path, port = start_server(
            args.pack, cached_state, mode, args.cpu_layers, args.ctx, True,
            args.clang, args.llvm, args.cpu_cache, "warm", args.timeout)
        warm["log"] = str(log_path)
        try:
            warm["a"] = chat(port, USERS[0], args.tokens)
            warm["b"] = chat(port, USERS[1], args.tokens)
        finally:
            stop_server(proc, handle)
            proc = handle = None
        warm["entries"] = opening_entries(cached_state)
    finally:
        if proc is not None:
            stop_server(proc, handle)

    saved_p = None
    for entry in warm["entries"]:
        if isinstance(entry.get("p"), int):
            saved_p = entry["p"]
            break

    reload_phase = {"phase": "cached-reload"}
    proc = handle = None
    try:
        proc, handle, log_path, port = start_server(
            args.pack, cached_state, mode, args.cpu_layers, args.ctx, True,
            args.clang, args.llvm, args.cpu_cache, "reload", args.timeout)
        reload_phase["log"] = str(log_path)
        try:
            reload_phase["cache_on"] = chat(port, USERS[2], args.tokens)
        finally:
            stop_server(proc, handle)
            proc = handle = None
    finally:
        if proc is not None:
            stop_server(proc, handle)

    off_phase = {"phase": "cache-off"}
    proc = handle = None
    try:
        proc, handle, log_path, port = start_server(
            args.pack, off_state, mode, args.cpu_layers, args.ctx, False,
            args.clang, args.llvm, args.cpu_cache, "off", args.timeout)
        off_phase["log"] = str(log_path)
        try:
            off_phase["cache_off"] = chat(port, USERS[2], args.tokens)
        finally:
            stop_server(proc, handle)
            proc = handle = None
    finally:
        if proc is not None:
            stop_server(proc, handle)

    on = reload_phase["cache_on"]
    off = off_phase["cache_off"]
    diff = compare(on, off)
    loaded = saved_p is not None and on["cached_tokens"] == saved_p
    reused = loaded and on["completion_tokens"] == off["completion_tokens"]
    return {
        "mode": mode,
        "warm": warm,
        "saved_opening_tokens": saved_p,
        "reload": reload_phase,
        "cache_off": off_phase,
        "loaded_saved_opening": loaded,
        "exact": diff["exact"],
        "diff": diff,
        "cache_used": reused,
        "status": "ok" if diff["exact"] and loaded else "mismatch",
    }


def write_results(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    model_name = Path(result["pack"]).name
    lines = [
        "# %s tandem cache — HTTP server path" % model_name,
        "",
        "Generated by `run_server_check.py`; every number below came from a real server process.",
        "",
        "Prompt: one long system message (%d tokens), a branch request with a different user turn, "
        "then a server restart and a third user turn. `prefix_max=1`, `prefix_min=256`, "
        "`checkpoint_every=32`, ctx %d, %d CPU layers, real RTX 4060 Ti + i7-8086K."
        % (result["prompts"]["prompts"][0]["tokens"], result["ctx"], result["cpu_layers"]),
        "",
        "| placement | opening end | saved opening | warm-branch cached | reload cached | cache-off cached | reply exact | status |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for p in result["placements"]:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                p["mode"],
                result["prompts"]["prompts"][0]["opening_end"],
                p["saved_opening_tokens"],
                p["warm"]["b"]["cached_tokens"],
                p["reload"]["cache_on"]["cached_tokens"],
                p["cache_off"]["cache_off"]["cached_tokens"],
                "yes" if p["exact"] else "no",
                p["status"],
            )
        )
    lines += [
        "",
        "The cache-on reply after the restart and the cache-off reply from a fresh state directory "
        "are compared on content, finish reason and completion-token count. The cached count is "
        "`usage.prompt_tokens_details.cached_tokens`; the opening size is the `p` in its on-disk meta.json.",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--run-dir", type=Path, default=HERE / "server-runs")
    ap.add_argument("--out", type=Path, default=HERE / "server-results.json")
    ap.add_argument("--cpu-cache", type=Path, default=DEFAULT_CPU_CACHE)
    ap.add_argument("--clang", type=Path, default=None,
                    help="the clang the CPU engine compiles with (default: KINDI_CC)")
    ap.add_argument("--llvm", type=Path, default=None,
                    help="the LLVM install root (default: KINDI_LLVM)")
    ap.add_argument("--cpu-layers", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=240)
    ap.add_argument("--modes", default="cpu-first,gpu-first")
    args = ap.parse_args()
    args.pack = args.pack.expanduser()
    args.cpu_cache = args.cpu_cache.expanduser()
    args.clang, args.llvm = toolchain(args.clang, args.llvm)

    prompts = prompt_token_info(args.pack)
    print("prompt tokens:", [(p["tokens"], p["opening_end"]) for p in prompts["prompts"]],
          flush=True)
    if not (256 <= prompts["prompts"][0]["opening_end"] <= args.ctx - 16):
        raise SystemExit("system opening is outside the saved-opening/context window: %r"
                         % prompts["prompts"][0])

    result = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "pack": str(args.pack),
        "ctx": args.ctx,
        "cpu_layers": args.cpu_layers,
        "tokens": args.tokens,
        "prompts": prompts,
        "placements": [],
    }
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print("== server %s ==" % mode, flush=True)
        p = run_mode(mode, args)
        result["placements"].append(p)
        print("%s: saved=%s warm_b_cached=%s reload_cached=%s off_cached=%s exact=%s status=%s"
              % (mode, p["saved_opening_tokens"], p["warm"]["b"]["cached_tokens"],
                 p["reload"]["cache_on"]["cached_tokens"],
                 p["cache_off"]["cache_off"]["cached_tokens"], p["exact"], p["status"]),
              flush=True)

    result["exact"] = all(p["exact"] for p in result["placements"])
    result["cache_used"] = all(p["cache_used"] for p in result["placements"])
    result["status"] = "ok" if result["exact"] and result["cache_used"] else "mismatch"
    write_results(result, args.out)
    print("status:", result["status"], "->", args.out, flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
