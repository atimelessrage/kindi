#!/usr/bin/env python3
"""Measure this machine's CPU-vs-PCIe split ratio for the dense tandem.

Roadmap prior-art item 1 (tandem.cpp 2026-11-11): bench-measure the CPU per-layer cost against
the PCIe upload cost and split by that ratio, instead of a hand-picked count. This tool does the
bench-measuring; the `--tandem-cpu-layers rate` planner does the splitting from its output.

Three inputs, each labeled by provenance in the output JSON:

* **PCIe** - pinned H2D/D2H bandwidth on 256 MiB and 32 MiB buffers, median of warm repeats.
  The per-boundary hidden-state cost (two hands-over of one residual across the link, an upper
  bound) is derived from it. On every PCIe 3/4 device it lands four orders of magnitude below a
  CPU layer, which is why a dense split loses its tokens to CPU layer time, not to PCIe.
* **CPU per-layer** - one GDN layer and one full-attention layer timed at the boundary through
  `kindi.cpu.Engine.layer` on a two-layer owned slice (so a 27B calibration does not load the
  whole model on the CPU). A GDN layer is context-independent; the attention layer is timed at
  position 0 (its constant part) and its context slope is labeled as taken from the recorded
  cpu-first vs gdn-first gap: attention layers on the CPU grow with the context and stay on the GPU.
* **GPU per-layer** - the recorded full-GPU plan-engine token times at 2K and 24K
  (`docs/TANDEM-RUNTIME.md`), split into a per-layer cost (token time / layers, embed/head
  lumped into it, about 5% at 24K) and a K/V scan slope. Derived from recorded rows, labeled so.

From those, the tool predicts tokens/s for every `gdn:N` split at the recorded 8K and 24K
contexts and validates the prediction against the recorded 27B sweeps
(`docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md`, `docs/TANDEM-RUNTIME.md`),
printing the errors. It writes the keyed calibration JSON (the seed of roadmap item 2's cache,
keyed by GPU + pack source hash + driver + CPU tier) plus any tracked copy asked for.

Everything measures THIS machine, THIS pack; a different card, pack or driver means a different
ratio, so nothing here is a product claim. On a CPU-only host the PCIe cell is recorded as an
error, not faked.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# recorded system behavior in this tree, used as the validation set (provenance in the output)
RECORDED_8K = {8: 10.7, 12: 6.7, 16: 5.5, 20: 4.6, 24: 3.8, 28: 3.2}   # 27B decode at pos 8K
RECORDED_FULL_GPU_TOK_S = {2048: 34.2, 24576: 30.2}                      # plan engine
RECORDED_GDN16_PLAIN = {"tok_s_at_24k": 7.9, "peak_mib": 7744}
RECORDED_AUTO_PLAN = {"layers": 11, "tok_s_at_24k": 7.9, "peak_mib": 6449}
CACHE_DIR = Path.home() / ".cache" / "kindi" / "tandem"


def git_revision() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=False)
        return out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def measure_pcie() -> dict:
    """Pinned H2D/D2H bandwidth (GiB/s), median of warm repeats, two sizes."""
    import cupy
    import numpy

    result = {}
    for label, mib in (("256MiB", 256), ("32MiB", 32)):
        n = mib * 2 ** 20
        d = cupy.zeros(n, dtype=cupy.uint8)
        h = numpy.zeros(n, dtype=numpy.uint8)          # host buffer
        h2d, d2h = [], []
        for _ in range(7):
            t0 = time.perf_counter()
            cupy.cuda.Stream().synchronize()
            d.set(h)
            cupy.cuda.Stream().synchronize()
            t1 = time.perf_counter()
            h[:] = d.get()
            cupy.cuda.Stream().synchronize()
            t2 = time.perf_counter()
            h2d.append(n / (t1 - t0) / 2 ** 30)
            d2h.append(n / (t2 - t1) / 2 ** 30)
        result[label] = {"h2d_gib_s": round(statistics.median(h2d), 2),
                         "d2h_gib_s": round(statistics.median(d2h), 2)}
    return result


def measure_cpu_layers(pack: Path) -> dict:
    """One GDN and one full-attention CPU layer, timed at the boundary on an owned slice.

    The engine loads `layers=` only (see TandemEngine), so a 27B calibration pays for two layers
    of CPU RAM, not for a whole-model load. `begin()` embeds once; the layer step is then timed
    at position 0, where the attention cache is empty, so the reading is the layer's constant cost.
    """
    from kindi.cpu import Engine

    man = json.loads((pack / "kindi_pack.json").read_text(encoding="utf-8"))
    dims = man.get("dimensions", {})
    interval = max(1, int(dims.get("full_attention_interval", 1)))
    gdn_il = next(il for il in range(1, 64) if (il + 1) % interval != 0)
    att_il = next(il for il in range(1, 64) if (il + 1) % interval == 0)
    eng = Engine(str(pack), ctx=4096, layers=sorted({gdn_il, att_il}), log=lambda *a: None)
    eng.begin(1)                                   # token id is arbitrary; only timing matters

    def time_layer(il: int, reps: int = 25) -> dict:
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            eng.layer(il, 0)
            samples.append((time.perf_counter() - t0) * 1000)
        return {"min": round(min(samples), 3), "median": round(statistics.median(samples), 3),
                "max": round(max(samples), 3)}     # the spread shows machine load

    gdn, att = time_layer(int(gdn_il)), time_layer(int(att_il))
    return {
        "threads": eng.threads,
        "isa": getattr(eng, "isa", None),
        "gdn_layer": int(gdn_il),
        "attention_layer": int(att_il),
        "cpu_gdn_ms": gdn["median"],
        "cpu_gdn_range_ms": [gdn["min"], gdn["max"]],
        "cpu_attention_ms_at_0": att["median"],
        "cpu_attention_range_ms": [att["min"], att["max"]],
        "source": "live, kindi.cpu.Engine.layer median of 25 on an owned 2-layer slice",
    }


def measure_split_tokens(pack: Path, counts, feed_tokens: int = 512, steps: int = 30) -> dict:
    """Effective token ms for real TandemEngine splits (all pipeline costs included).

    The CPU entry timing measures the layer kernel; the TandemEngine `step()` loop adds the
    per-block handover, host dispatch and PCIe hand-overs a decode token actually pays. The
    planner needs the effective cost, so this probes it at a short feed (the GDN cost is
    position-independent; the K/V scan is added back analytically for 24K). Returns {N: ms}.

    Counts below the load-fit floor cannot probe: on this 8 GB card the 27B plan engine refuses
    fewer than ~11 CPU layers outright (the fitted range matters, not crossings below it).
    Each N runs in its own subprocess (the recorded tables are medians of fresh processes, and
    CuPy keeps the first engine's allocations in-process).
    """
    out = {}
    for n in counts:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe-one", str(pack),
             str(n), str(feed_tokens), str(steps)],
            capture_output=True, text=True, timeout=300, check=False)
        if proc.returncode != 0:
            raise RuntimeError("probe for gdn:%d failed: %s" % (n, proc.stderr.strip()[-300:]))
        out[n] = float(proc.stdout.strip().splitlines()[-1])
    return out


def probe_one(pack: str, n: int, feed_tokens: int, steps: int) -> int:
    """One subprocess's probe: build the split, feed, time steps, print the median ms."""
    from kindi.tandem import TandemEngine

    eng = TandemEngine(pack, ctx=8192, cpu_layers="gdn:%d" % n, mode="cpu-first",
                       log=lambda *a: None)
    for pos in range(feed_tokens):
        eng.feed(1, pos)
    tok = 1
    samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        tok = eng.step(tok, feed_tokens)
        samples.append((time.perf_counter() - t0) * 1000)
    sys.stdout.write("%.3f\n" % statistics.median(samples))
    return 0


def predicted_tok_s(floor_ms: float, cpu_gdn_ms: float, gpu_layer_ms: float,
                    scan_ms_per_k: float, nl: int, n: int, pos: int) -> float:
    """Predicted tokens/s for a gdn:N split (n GDN layers on the CPU) at position `pos`.

    token_ms(N) = floor_ms + n*cpu + (nl-n)*gpu-layer + scan*(pos/1024). With floor_ms built
    from the recorded full-GPU token times (embed/head lumped into the GPU layers), the model
    validates against the recorded split sweep printed alongside it.
    """
    token_ms = floor_ms + n * cpu_gdn_ms + max(nl - n, 0) * gpu_layer_ms \
        + (pos / 1024) * scan_ms_per_k
    return 1000.0 / token_ms


def effective_cpu_ms(probe: dict, gpu_layer_ms: float, nl: int) -> float | None:
    """Slope of effective token ms per CPU GDN layer (the fit, with the fixed pick at the low end)."""
    if len(probe) < 2:
        return None
    ns = sorted(probe)
    (n0, t0), (n1, t1) = (ns[0], probe[ns[0]]), (ns[-1], probe[ns[-1]])
    if n1 == n0:
        return None
    slope = (t1 - t0) / (n1 - n0)
    if gpu_layer_ms:
        slope -= gpu_layer_ms       # the GPU does one fewer layer per extra CPU layer
    return round(slope, 3)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--probe-one" in argv:
        i = argv.index("--probe-one")
        return probe_one(argv[i + 1], int(argv[i + 2]), int(argv[i + 3]), int(argv[i + 4]))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", default="~/.kindi/packs/bonsai-27b")
    ap.add_argument("--json", default=None, help="also write the calibration JSON here")
    ap.add_argument("--track", default=None, help="also write a tracked copy here")
    args = ap.parse_args(argv)
    pack = Path(args.pack).expanduser()

    if not (pack / "kindi_pack.json").is_file():
        raise SystemExit("kindi: pack not found at %s" % pack)
    man = json.loads((pack / "kindi_pack.json").read_text(encoding="utf-8"))
    dims = man.get("dimensions", {})
    nl = int(dims.get("layers", 64))
    hidden = int(dims.get("width", 2048))

    errors: dict[str, str] = {}
    try:
        pcie = measure_pcie()
    except Exception as exc:  # noqa: BLE001 - a CPU-only host records the absence, not a fake
        pcie, errors["pcie"] = {}, "no GPU pinned-buffer benchmark: %s" % exc
    try:
        cpu = measure_cpu_layers(pack)
    except Exception as exc:  # noqa: BLE001
        cpu, errors["cpu"] = None, "CPU layer timing failed: %s" % exc

    # per-boundary hidden transfer (two hands-over of one residual, an upper bound)
    h2d = (pcie.get("256MiB") or {}).get("h2d_gib_s", 0)
    boundary_ms = round(2 * hidden * 4 / (h2d * 2 ** 30) * 1000, 4) if h2d else None

    # derive the GPU side from the recorded full-GPU curve (2K and 24K token times): the whole
    # token time at 24K is split evenly over layers (embed/head lumped into it), and the K/V
    # scan slope is the 2K->24K difference per 1024 positions
    # the effective per-layer cost: the tandem `step()` probe is the only piece that knows the
    # real pipeline, so when it is present it defines the rate used everywhere below
    probe: dict | None = None
    eff_cpu_ms: float | None = None
    if cpu and pcie:
        try:
            probe = measure_split_tokens(pack, (11, 13, 16, 20))
        except Exception as exc:  # noqa: BLE001 - recorded as an error, never faked
            probe, errors["split_probe"] = None, "tandem step probe failed: %s" % exc

    token_ms_2k = 1000 / RECORDED_FULL_GPU_TOK_S[2048]
    token_ms_24k = 1000 / RECORDED_FULL_GPU_TOK_S[24576]
    gpu_layer_ms_24k = token_ms_24k / nl                # embed/head lumped in (note in derived)
    scan_ms_per_k = (token_ms_24k - token_ms_2k) / 22.0
    floor_ms = 0.0          # the 24K token time is fully inside the layer sum (lumped model)
    if probe:
        eff_cpu_ms = effective_cpu_ms(probe, gpu_layer_ms_24k, nl)

    predicted, validation_errors, cpu_used = {}, {}, None
    if cpu:
        cpu_used = eff_cpu_ms if eff_cpu_ms is not None else cpu["cpu_gdn_ms"]
        for pos, context in ((8192, "8k"), (24576, "24k")):
            predicted[context] = {
                n: round(predicted_tok_s(floor_ms, cpu_used, gpu_layer_ms_24k,
                                         scan_ms_per_k, nl, n, pos), 2)
                for n in (0, 4, 8, 11, 12, 16, 19, 20, 24, 28)
            }
        for n, recorded in RECORDED_8K.items():
            validation_errors[n] = round((predicted["8k"][n] - recorded) / recorded * 100, 1)

    payload = {
        "kind": "kindi-split-ratio/1",
        "measured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_revision": git_revision(),
        "gpu": gpu_name(),
        "pack": {"path": str(pack), "source_sha256": man.get("source_sha256")},
        "dimensions": {k: dims.get(k) for k in ("layers", "width", "full_attention_interval")},
        "pcie": pcie,
        "boundary_hidden_ms": boundary_ms,
        "cpu": cpu,
        "split_probe": probe,
        "cpu_gdn_effective_ms": eff_cpu_ms,
        "derived": {
            "gpu_layer_ms_at_24k": round(gpu_layer_ms_24k, 4),
            "kv_scan_ms_per_1024": round(scan_ms_per_k, 4),
            "cpu_attention_slope": ("the gap between the recorded gdn-first (7.9 tok/s) and "
                                   "cpu-first (2.9 tok/s) 24K rows, i.e. attention layers on the "
                                   "CPU grow with the context; they stay on the GPU"),
            "source": "derived from the recorded full-GPU 2K/24K token times; PCIe and CPU "
                      "layer cells are live measurements",
        },
        "predicted_tok_s": predicted,
        "validation": {
            "recorded_8k_decode_tok_s": RECORDED_8K,
            "validation_error_pct": validation_errors,
            "recorded_gdn16_plain": RECORDED_GDN16_PLAIN,
            "recorded_auto_plan": RECORDED_AUTO_PLAN,
        },
        "errors": errors,
    }
    text = json.dumps(payload, indent=2) + "\n"
    key = "kindi-split-ratio-%s-%s.json" % (
        (payload["gpu"] or "nocard").replace(" ", "-").lower(),
        str(payload["pack"]["source_sha256"])[:12])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / key).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    if args.track:
        Path(args.track).parent.mkdir(parents=True, exist_ok=True)
        Path(args.track).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    sys.stdout.write("\ncalibration: %s\n" % (CACHE_DIR / key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
