#!/usr/bin/env python3
"""Compare cached and cache-off continuations through the real 2B tandem engine.

The harness loads one placement, warms the joint cache on a long T1 prompt, then
runs two continuations:

* extension: T1 + suffix, which reuses the end state;
* branch: T1[:branch] + a different suffix, which restores a joint checkpoint
  containing both owners' recurrent state and K/V.

Each cached continuation is repeated with KINDI_PROMPT_CACHE=0 from a reset
engine.  The generated greedy tokens must be identical.  The CPU engine is
built with the clang/libomp install named by --clang/--llvm; no inference code
is changed by this harness.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_PACK = Path(os.path.expanduser("~/.kindi/packs/bonsai-2b"))
DEFAULT_CLANG = Path("C:/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc/bin/clang.exe")
DEFAULT_LLVM = Path("C:/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc")


def configure_environment(clang: Path, llvm: Path) -> None:
    os.environ["KINDI_CC"] = str(clang)
    os.environ["KINDI_LLVM"] = str(llvm)
    os.environ["KINDI_CPU_ISA"] = "avx2-fma-f16c"
    os.environ["KINDI_CPU_CACHE"] = str(Path(__file__).with_name("cpu-cache"))
    os.environ["KINDI_DEVICE"] = "gpu"
    os.environ["KINDI_HEAD"] = "exact"
    os.environ["KINDI_SPEC"] = "0"
    os.environ["KINDI_CARDS"] = "1"


def cache_config(state_dir: Path, checkpoint_every: int = 32, prefix_max: int = 1,
                 prefix_min: int = 256) -> dict:
    return {
        "checkpoint_every": str(checkpoint_every),
        "prefix_max": str(prefix_max),
        "prefix_min": str(prefix_min),
        "park_max": "0",
        "checkpoint_budget_mb": "512",
        "state_dir": str(state_dir),
        "head": "exact",
    }


def sync_gpu() -> None:
    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
    except Exception:
        pass


def prompt_ids(tok, length: int = 320) -> list[int]:
    text = "The quick brown fox jumps over the lazy dog. " * 40
    ids = tok.encode(text)
    while len(ids) < length:
        ids.extend(tok.encode(" The quick brown fox jumps over the lazy dog."))
    return ids[:length]


def prepare_only(runner, ids: list[int]) -> None:
    """Leave the runner's state exactly at ids, without generating."""
    runner.prepare(ids, len(ids), len(ids))


def continuation(runner, ids: list[int], count: int, tok) -> dict:
    sync_gpu()
    started = time.perf_counter()
    nxt, cached = runner.prepare(ids, len(ids), len(ids))
    if nxt is None:
        raise RuntimeError("prepare returned no next token")
    out = [int(nxt)]
    for _ in range(count - 1):
        out.append(int(runner.step(out[-1])))
    sync_gpu()
    return {
        "cached_tokens": int(cached),
        "prompt_tokens": len(ids),
        "output_ids": out,
        "output_text": tok.decode(out),
        "seconds": round(time.perf_counter() - started, 3),
    }


def first_diff(a: list[int], b: list[int]):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def free() -> None:
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def run_placement(mode: str, pack: Path, ctx: int, cpu_layers: int, count: int,
                  t1_tokens: int = 320, branch_at: int = 256, checkpoint_every: int = 32,
                  prefix_max: int = 1, prefix_min: int = 256, skip_saved: bool = False,
                  state_parent: Path | None = None) -> dict:
    from kindi.tandem import TandemEngine, TandemRunner
    from kindi.tokenizer import Tok

    t0 = time.perf_counter()
    eng = TandemEngine(str(pack), ctx=ctx, cpu_layers=cpu_layers, mode=mode, log=print)
    load_s = round(time.perf_counter() - t0, 3)
    tok = Tok(eng.vocab, os.path.join(str(pack), "merges.txt"))
    t1 = prompt_ids(tok, t1_tokens)
    if not 0 < branch_at < len(t1):
        raise ValueError("branch_at must be inside T1: %d not in 1..%d" % (branch_at, len(t1) - 1))
    extension = tok.encode(" The next sentence begins with")
    branch_suffix = tok.encode(" A different branch begins with")
    t2_ext = t1 + extension
    t2_branch = t1[:branch_at] + branch_suffix
    if len(t2_ext) + count > eng.ctx:
        raise ValueError("extension prompt plus %d tokens exceeds context %d" % (count, eng.ctx))
    state_dir_parent = Path(state_parent) if state_parent else Path(__file__).parent
    state_dir_parent.mkdir(parents=True, exist_ok=True)
    state_root = Path(tempfile.mkdtemp(prefix="state-%s-" % mode, dir=str(state_dir_parent)))
    cfg = cache_config(state_root, checkpoint_every, prefix_max, prefix_min)
    runner = TandemRunner(eng, cfg, "tandem-cache-%s" % mode, log=print)
    fresh = None

    try:
        os.environ["KINDI_PROMPT_CACHE"] = "1"
        prepare_only(runner, t1)
        on_ext = continuation(runner, t2_ext, count, tok)
        prepare_only(runner, t1)
        on_branch = continuation(runner, t2_branch, count, tok)

        os.environ["KINDI_PROMPT_CACHE"] = "0"
        off_ext = continuation(runner, t2_ext, count, tok)
        off_branch = continuation(runner, t2_branch, count, tok)

        saved = []
        on_saved = off_saved = None
        if not skip_saved:
            # The branch above shares branch_at tokens with T1 and diverges before the opening end,
            # so the cached branch saved a disk opening there. A new runner is the next session: it
            # must load that opening from disk, not reuse the still-resident checkpoint in the old
            # runner.
            os.environ["KINDI_PROMPT_CACHE"] = "1"
            fresh = TandemRunner(eng, cache_config(state_root, checkpoint_every, prefix_max, prefix_min),
                                 "tandem-cache-%s" % mode, log=print)
            saved = [json.load(open(p)) for p in sorted((state_root / "openings").glob("*/*/meta.json"))]
            on_saved = continuation(fresh, t2_branch, count, tok)
            os.environ["KINDI_PROMPT_CACHE"] = "0"
            off_saved = continuation(fresh, t2_branch, count, tok)
    finally:
        os.environ["KINDI_PROMPT_CACHE"] = "1"

    ext_diff = first_diff(on_ext["output_ids"], off_ext["output_ids"])
    branch_diff = first_diff(on_branch["output_ids"], off_branch["output_ids"])
    saved_diff = None if skip_saved else first_diff(on_saved["output_ids"], off_saved["output_ids"])
    result = {
        "mode": mode,
        "engine": type(eng).__name__,
        "load_s": load_s,
        "ctx": int(eng.ctx),
        "cpu_layers": int(eng.cpu_layers),
        "cpu_isa": getattr(eng.cpu, "isa", None),
        "cpu_threads": getattr(eng.cpu, "threads", None),
        "ranges": {"cpu": list(eng.cpu_range), "gpu": list(eng.gpu_range)},
        "cache": {"checkpoint_every": checkpoint_every, "prefix_max": prefix_max,
                  "prefix_min": prefix_min},
        "prompts": {"t1_tokens": len(t1), "branch_at": branch_at,
                    "extension_tokens": len(t2_ext), "branch_tokens": len(t2_branch)},
        "extension": {"cache_on": on_ext, "cache_off": off_ext,
                      "exact": ext_diff is None, "first_diff": ext_diff},
        "branch": {"cache_on": on_branch, "cache_off": off_branch,
                   "exact": branch_diff is None, "first_diff": branch_diff},
        "saved_opening": ({"skipped": True} if skip_saved else
                          {"entries": saved, "cache_on": on_saved, "cache_off": off_saved,
                           "exact": saved_diff is None, "first_diff": saved_diff}),
        "exact": ext_diff is None and branch_diff is None and (skip_saved or saved_diff is None),
        "cache_used": (on_ext["cached_tokens"] > 0 and on_branch["cached_tokens"] > 0
                       and (skip_saved or on_saved["cached_tokens"] > 0)),
        "status": "ok" if (ext_diff is None and branch_diff is None
                           and (skip_saved or saved_diff is None)) else "mismatch",
    }
    del fresh, runner, eng
    free()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--clang", type=Path, default=DEFAULT_CLANG)
    ap.add_argument("--llvm", type=Path, default=DEFAULT_LLVM)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cpu-layers", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--t1-tokens", type=int, default=320)
    ap.add_argument("--branch-at", type=int, default=256)
    ap.add_argument("--checkpoint-every", type=int, default=32)
    ap.add_argument("--prefix-max", type=int, default=1)
    ap.add_argument("--prefix-min", type=int, default=256)
    ap.add_argument("--skip-saved", action="store_true")
    ap.add_argument("--state-root", type=Path, default=None,
                    help="parent directory for the per-run state dir (defaults to this script's dir)")
    ap.add_argument("--modes", default="cpu-first,gpu-first")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    a = ap.parse_args()

    configure_environment(a.clang, a.llvm)
    result = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pack": str(a.pack),
        "clang": str(a.clang),
        "llvm": str(a.llvm),
        "ctx": a.ctx,
        "cpu_layers": a.cpu_layers,
        "tokens": a.tokens,
        "placements": {},
        "status": "error",
    }
    try:
        for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
            print("\n=== %s ===" % mode, flush=True)
            result["placements"][mode] = run_placement(
                mode, a.pack, a.ctx, a.cpu_layers, a.tokens, a.t1_tokens, a.branch_at,
                a.checkpoint_every, a.prefix_max, a.prefix_min, a.skip_saved, a.state_root)
            p = result["placements"][mode]
            for case in ("extension", "branch"):
                c = p[case]
                print("%s: cached=%d off_cached=%d exact=%s on=%r off=%r" % (
                    case, c["cache_on"]["cached_tokens"], c["cache_off"]["cached_tokens"],
                    c["exact"], c["cache_on"]["output_text"], c["cache_off"]["output_text"]), flush=True)
            c = p["saved_opening"]
            if c.get("skipped"):
                print("saved opening: skipped in this context run", flush=True)
            else:
                print("saved opening: %d entries, cached=%d off_cached=%d exact=%s on=%r off=%r" % (
                    len(c["entries"]), c["cache_on"]["cached_tokens"], c["cache_off"]["cached_tokens"],
                    c["exact"], c["cache_on"]["output_text"], c["cache_off"]["output_text"]), flush=True)
        result["exact_all"] = all(p["exact"] for p in result["placements"].values())
        result["cache_used"] = all(p["cache_used"] for p in result["placements"].values())
        result["status"] = "ok" if result["exact_all"] and result["cache_used"] else "mismatch"
    except BaseException as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % a.out, flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
