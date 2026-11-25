#!/usr/bin/env python3
"""Measure joint-cache prefix reuse and checkpoint cost on the real 2B tandem split.

For each placement this records:

* cold T1 prompt preparation and the end-of-prompt joint checkpoint;
* a warm extension that reuses the T1 end state;
* a warm branch that restores a joint checkpoint at the requested branch position;
* cache-off controls for the same two continuation prompts;
* repeated save/restore timings for the joint recurrent-state checkpoint.

The prompt and branch positions are configurable; ``--checkpoint-only`` skips the continuation and
cache-off controls when only the joint checkpoint cost is wanted.

The checkpoint measured here is the ``cache.Runner`` checkpoint: both owners'
``M``/``conv`` recurrent arrays.  K/V remains in the two engines and is not part
of this checkpoint.  No inference source is changed.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_PACK = Path(os.path.expanduser("~/.kindi/packs/bonsai-2b"))
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



def configure_environment(clang: Path, llvm: Path) -> None:
    os.environ["KINDI_CC"] = str(clang)
    os.environ["KINDI_LLVM"] = str(llvm)
    os.environ["KINDI_CPU_ISA"] = "avx2-fma-f16c"
    os.environ["KINDI_CPU_CACHE"] = str(Path(__file__).with_name("cpu-cache"))
    os.environ["KINDI_DEVICE"] = "gpu"
    os.environ["KINDI_HEAD"] = "exact"
    os.environ["KINDI_SPEC"] = "0"
    os.environ["KINDI_CARDS"] = "1"
    os.environ["KINDI_CACHE_TIMING"] = "1"


def cache_config(state_dir: Path, checkpoint_every: int = 32,
                 checkpoint_budget_mb: int = 512) -> dict:
    return {
        "checkpoint_every": str(checkpoint_every),
        "prefix_max": "0",
        "prefix_min": "0",
        "park_max": "0",
        "checkpoint_budget_mb": str(checkpoint_budget_mb),
        "state_dir": str(state_dir),
        "head": "exact",
    }


def sync_gpu() -> None:
    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
    except Exception:
        pass


def gpu_memory() -> dict | None:
    """Device-wide VRAM from nvidia-smi; this includes the desktop and other processes."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        row = out.stdout.strip().splitlines()[0]
        used, total, free = [int(x.strip()) for x in row.split(",")[:3]]
        return {"used_mib": used, "total_mib": total, "free_mib": free}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class VramMonitor(object):
    """Sample device-wide VRAM while the real engine loads and runs.

    nvidia-smi is used rather than the CuPy pool so the reported peak includes all device
    allocations. The baseline is taken before the engine constructor starts, and every sample
    carries the latest phase marker so the peak can be attributed to load, prompt, or checkpoint
    work. The sampler is opt-in so the existing 2B result files keep their original shape.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = max(0.05, float(interval))
        self.samples = []
        self.markers = []
        self.phase = "startup"
        self.baseline = None
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.perf_counter()

    def _sample(self, phase: str | None = None) -> dict | None:
        mem = gpu_memory()
        if mem is None:
            return None
        rec = dict(mem)
        rec["at_s"] = round(time.perf_counter() - self._t0, 3)
        rec["phase"] = phase or self.phase
        self.samples.append(rec)
        return rec

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self.baseline = self._sample("startup")
        self._thread = threading.Thread(target=self._run, name="vram-monitor", daemon=True)
        self._thread.start()

    def mark(self, phase: str) -> None:
        self.phase = phase
        rec = self._sample(phase)
        if rec is not None:
            self.markers.append(rec)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._sample("stop")

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False, "interval_s": self.interval}
        peak = max(self.samples, key=lambda r: r["used_mib"])
        base = self.baseline or self.samples[0]
        return {
            "available": True,
            "source": "nvidia-smi memory.used (device-wide)",
            "interval_s": self.interval,
            "baseline": base,
            "peak": peak,
            "peak_used_mib": peak["used_mib"],
            "baseline_used_mib": base["used_mib"],
            "peak_delta_mib": peak["used_mib"] - base["used_mib"],
            "peak_free_mib": peak["free_mib"],
            "sample_count": len(self.samples),
            "markers": self.markers,
            "samples": self.samples,
        }


def prompt_ids(tok, length: int = 320) -> list[int]:
    text = "The quick brown fox jumps over the lazy dog. " * 40
    ids = tok.encode(text)
    while len(ids) < length:
        ids.extend(tok.encode(" The quick brown fox jumps over the lazy dog."))
    return ids[:length]


def timed_prepare(runner, ids: list[int]) -> tuple[float, int]:
    sync_gpu()
    started = time.perf_counter()
    nxt, cached = runner.prepare(ids, len(ids), len(ids))
    sync_gpu()
    if nxt is None:
        raise RuntimeError("prepare returned no next token")
    return time.perf_counter() - started, int(cached)


def bytes_of(arrays) -> int:
    return sum(int(a.nbytes) for a in arrays if a is not None)


def checkpoint_bytes(view) -> tuple[int, int, int, int]:
    """(joint recurrent, CPU recurrent, GPU recurrent, K/V) bytes in the cache view."""
    recurrent = list(view.M) + list(view.conv)
    cpu = sum(int(a.nbytes) for a in recurrent if isinstance(a, np.ndarray))
    total = bytes_of(recurrent)
    return total, cpu, total - cpu, bytes_of(view.Kc) + bytes_of(view.Vc)


def timing_summary(values: list[float]) -> dict:
    return {
        "median_s": round(statistics.median(values), 6),
        "min_s": round(min(values), 6),
        "max_s": round(max(values), 6),
        "samples_s": [round(v, 6) for v in values],
    }


def measure_checkpoints(runner, view, save_pos: int, restore_pos: int, reps: int) -> dict:
    """Time repeated joint recurrent-state saves/restores without changing tokens."""
    ckpts = sorted(runner.ckpt)
    if save_pos not in runner.ckpt:
        raise RuntimeError("no %d-token checkpoint (have %s)" % (save_pos, ckpts))
    if restore_pos not in runner.ckpt:
        raise RuntimeError("no %d-token checkpoint (have %s)" % (restore_pos, ckpts))

    # Reserve the state at the end of T1; both checkpoint positions were made
    # while that prompt was read, so the 320 state is valid for save_pos.
    runner.ckpt.pop(save_pos, None)
    runner._save_ckpt(save_pos)                  # warm-up allocation
    save_s = []
    for _ in range(reps):
        runner.ckpt.pop(save_pos, None)
        sync_gpu()
        t0 = time.perf_counter()
        runner._save_ckpt(save_pos)
        sync_gpu()
        save_s.append(time.perf_counter() - t0)
    st_save = runner.ckpt[save_pos]

    def restore(st) -> list[float]:
        sync_gpu()
        runner._restore(st)
        sync_gpu()
        values = []
        for _ in range(reps):
            sync_gpu()
            t0 = time.perf_counter()
            runner._restore(st)
            sync_gpu()
            values.append(time.perf_counter() - t0)
        return values

    restore_branch_s = restore(runner.ckpt[restore_pos])
    restore_end_s = restore(st_save)
    total, cpu, gpu, kv = checkpoint_bytes(view)
    return {
        "save_position": save_pos,
        "restore_position": restore_pos,
        "checkpoint_every_tokens": int(runner.every),
        "checkpoint_keys": sorted(runner.ckpt),
        "state_bytes": total,
        "state_mib": round(total / (1024.0 * 1024.0), 3),
        "cpu_state_bytes": cpu,
        "gpu_state_bytes": gpu,
        "kv_bytes_not_in_checkpoint": kv,
        "save": timing_summary(save_s),
        "restore_branch": timing_summary(restore_branch_s),
        "restore_end": timing_summary(restore_end_s),
    }


def run_placement(mode: str, pack: Path, ctx: int, cpu_layers: int, reps: int,
                  t1_tokens: int = 320, branch_at: int = 256, checkpoint_only: bool = False,
                  measure_vram: bool = False, vram_interval: float = 0.5,
                  checkpoint_every: int = 32, checkpoint_budget_mb: int = 512) -> dict:
    from kindi.tandem import TandemEngine, TandemRunner
    from kindi.tokenizer import Tok

    vram = VramMonitor(vram_interval) if measure_vram else None
    if vram is not None:
        vram.start()
    logs = []

    def log(*args):
        line = " ".join(str(a) for a in args)
        logs.append(line)
        print(line, flush=True)

    t0 = time.perf_counter()
    eng = TandemEngine(str(pack), ctx=ctx, cpu_layers=cpu_layers, mode=mode, log=log)
    load_s = round(time.perf_counter() - t0, 3)
    if vram is not None:
        vram.mark("engine-ready")
    meta = {
        "engine": type(eng).__name__,
        "cpu_isa": getattr(eng.cpu, "isa", None),
        "cpu_threads": getattr(eng.cpu, "threads", None),
        "ranges": {"cpu": list(eng.cpu_range), "gpu": list(eng.gpu_range)},
    }
    tok = Tok(eng.vocab, os.path.join(str(pack), "merges.txt"))
    t1 = prompt_ids(tok, t1_tokens)
    if not 0 < branch_at < len(t1):
        raise ValueError("branch_at must be inside T1: %d not in 1..%d" % (branch_at, len(t1) - 1))
    extension = tok.encode(" The next sentence begins with")
    branch_suffix = tok.encode(" A different branch begins with")
    t2_ext = t1 + extension
    t2_branch = t1[:branch_at] + branch_suffix
    if len(t2_ext) > ctx:
        raise ValueError("extension prompt %d exceeds context %d" % (len(t2_ext), ctx))
    state_root = Path(tempfile.mkdtemp(prefix="prefix-state-%s-" % mode, dir=Path(__file__).parent))
    runner = None

    try:
        runner = TandemRunner(eng, cache_config(state_root, checkpoint_every,
                                             checkpoint_budget_mb), "prefix-%s" % mode, log=log)
        os.environ["KINDI_PROMPT_CACHE"] = "1"
        os.environ["KINDI_CACHE_TIMING"] = "1"
        cold_s, cold_cached = timed_prepare(runner, t1)
        cold_tsave_s = float(getattr(runner, "_tsave", 0.0))
        if vram is not None:
            vram.mark("cold-t1")
        checkpoints = measure_checkpoints(runner, runner.eng, len(t1), branch_at, reps)
        if vram is not None:
            vram.mark("checkpoints")

        ext_s = ext_cached = ext_tsave_s = None
        branch_s = branch_cached = branch_tsave_s = None
        off_ext_s = off_ext_cached = off_branch_s = off_branch_cached = None
        if not checkpoint_only:
            ext_s, ext_cached = timed_prepare(runner, t2_ext)
            ext_tsave_s = float(getattr(runner, "_tsave", 0.0))
            branch_s, branch_cached = timed_prepare(runner, t2_branch)
            branch_tsave_s = float(getattr(runner, "_tsave", 0.0))

            os.environ["KINDI_PROMPT_CACHE"] = "0"
            off_ext_s, off_ext_cached = timed_prepare(runner, t2_ext)
            off_branch_s, off_branch_cached = timed_prepare(runner, t2_branch)
    finally:
        os.environ["KINDI_PROMPT_CACHE"] = "1"
        if vram is not None:
            vram.stop()
        if runner is not None:
            del runner
        del eng
        gc.collect()
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        shutil.rmtree(state_root, ignore_errors=True)

    def prefix(name: str, seconds: float, cached: int, tokens: int, tsave: float = None) -> dict:
        row = {
            "cached_tokens": cached,
            "prompt_tokens": tokens,
            "new_prompt_tokens": tokens - cached,
            "reuse_fraction": round(cached / tokens, 4) if tokens else 0.0,
            "prepare_s": round(seconds, 6),
        }
        if tsave is not None:
            row["checkpoint_save_s"] = round(tsave, 6)
        return row

    expected = {"cold": cold_cached == 0}
    if not checkpoint_only:
        expected.update({
            "extension": ext_cached == len(t1),
            "branch": branch_cached == branch_at,
            "cache_off_extension": off_ext_cached == 0,
            "cache_off_branch": off_branch_cached == 0,
        })
    result = {
        "mode": mode,
        "engine": meta["engine"],
        "load_s": load_s,
        "ctx": int(ctx),
        "cpu_layers": int(cpu_layers),
        "cpu_isa": meta["cpu_isa"],
        "cpu_threads": meta["cpu_threads"],
        "ranges": meta["ranges"],
        "prompts": {"t1_tokens": len(t1), "branch_at": branch_at,
                    "extension_tokens": len(t2_ext), "branch_tokens": len(t2_branch)},
        "cold_t1": prefix("cold_t1", cold_s, cold_cached, len(t1), cold_tsave_s),
        "checkpoint": checkpoints,
        "cache_timing_logs": [line for line in logs if line.startswith("cache timing:")],
        "expected": expected,
        "status": "ok" if all(expected.values()) else "unexpected-reuse",
    }
    if checkpoint_only:
        result["checkpoint_only"] = True
    else:
        result.update({
            "warm_extension": prefix("warm_extension", ext_s, ext_cached, len(t2_ext), ext_tsave_s),
            "warm_branch": prefix("warm_branch", branch_s, branch_cached, len(t2_branch), branch_tsave_s),
            "cache_off": {
                "extension": prefix("cache_off_extension", off_ext_s, off_ext_cached, len(t2_ext)),
                "branch": prefix("cache_off_branch", off_branch_s, off_branch_cached, len(t2_branch)),
            },
        })
    if vram is not None:
        vram.stop()
        result["vram"] = vram.summary()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--clang", type=Path, default=None,
                    help="the clang the CPU engine compiles with (default: KINDI_CC)")
    ap.add_argument("--llvm", type=Path, default=None,
                    help="the LLVM install root (default: KINDI_LLVM)")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cpu-layers", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--t1-tokens", type=int, default=320)
    ap.add_argument("--branch-at", type=int, default=256)
    ap.add_argument("--checkpoint-every", type=int, default=32)
    ap.add_argument("--checkpoint-budget-mb", type=int, default=512)
    ap.add_argument("--checkpoint-only", action="store_true")
    ap.add_argument("--vram", action="store_true", help="sample device-wide VRAM with nvidia-smi")
    ap.add_argument("--vram-sample-s", type=float, default=0.5)
    ap.add_argument("--modes", default="cpu-first,gpu-first")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("prefix-results.json"))
    a = ap.parse_args()

    a.clang, a.llvm = toolchain(a.clang, a.llvm)
    configure_environment(a.clang, a.llvm)
    result = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pack": str(a.pack),
        "clang": str(a.clang),
        "llvm": str(a.llvm),
        "ctx": a.ctx,
        "cpu_layers": a.cpu_layers,
        "reps": a.reps,
        "t1_tokens": a.t1_tokens,
        "branch_at": a.branch_at,
        "checkpoint_only": bool(a.checkpoint_only),
        "vram": {"enabled": bool(a.vram), "sample_interval_s": a.vram_sample_s},
        "modes": [m.strip() for m in a.modes.split(",") if m.strip()],
        "placements": {},
        "status": "error",
    }
    try:
        for mode in result["modes"]:
            print("\n=== %s ===" % mode, flush=True)
            result["placements"][mode] = run_placement(
                mode, a.pack, a.ctx, a.cpu_layers, a.reps, a.t1_tokens, a.branch_at,
                a.checkpoint_only, a.vram, a.vram_sample_s,
                a.checkpoint_every, a.checkpoint_budget_mb)
            p = result["placements"][mode]
            if p.get("checkpoint_only"):
                print("%s: checkpoint-only cold=%d save=%.3fms restore=%.3fms" % (
                    mode, p["cold_t1"]["cached_tokens"],
                    p["checkpoint"]["save"]["median_s"] * 1000.0,
                    p["checkpoint"]["restore_end"]["median_s"] * 1000.0), flush=True)
            else:
                print("%s: cold=%d ext=%d/%d branch=%d/%d save=%.3fms restore=%.3fms" % (
                    mode, p["cold_t1"]["cached_tokens"],
                    p["warm_extension"]["cached_tokens"], p["warm_extension"]["prompt_tokens"],
                    p["warm_branch"]["cached_tokens"], p["warm_branch"]["prompt_tokens"],
                    p["checkpoint"]["save"]["median_s"] * 1000.0,
                    p["checkpoint"]["restore_end"]["median_s"] * 1000.0), flush=True)
        result["status"] = "ok" if all(p["status"] == "ok" for p in result["placements"].values()) else "unexpected-reuse"
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
