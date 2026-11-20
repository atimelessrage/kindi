#!/usr/bin/env python3
"""Build the cross-mode benchmark matrix: CPU-only, GPU-only, GPU tensor-core (plan engine),
GPU+CPU tandem.

The CPU-only and GPU+CPU tandem cells come from the 2026-11-19/20 controlled same-context pairs;
the 9B is the median of three fresh-process pairs and the 2B/4B are one pair each. The GPU cells
are the earlier median of three; tensor-core rows come from the 2026-11-13 tc
folder, where the 9B needs `KINDI_TC_HP=1` (see `kindi/spec.py`). The old 2026-11-12 tandem rows
remain in the repository as history. `--check` exits 1 when CROSS-MODE.md or CROSS-MODE.csv is
stale.
"""
import argparse
import csv
import io
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU_RUNS = HERE / "2026-11-10-i7-8086k-windows-cpu" / "runs.csv"
GPU_RUNS = HERE / "2026-11-09-i7-8086k-rtx4060ti" / "runs.csv"
TANDEM_RUNS = HERE / "2026-11-12-i7-8086k-rtx4060ti-tandem" / "runs.csv"
TC_RUNS = HERE / "2026-11-13-i7-8086k-rtx4060ti-tc" / "runs.csv"
CONTROLLED_2B4B = HERE / "2026-11-19-i7-8086k-rtx4060ti-2b-4b-controlled"
CONTROLLED_9B = HERE / "2026-11-19-i7-8086k-rtx4060ti-9b-controlled"
CONTROLLED_RUNS = {
    "2b": [CONTROLLED_2B4B / "runs-2b.json"],
    "4b": [CONTROLLED_2B4B / "runs-4b.json"],
    "9b": [CONTROLLED_9B / "runs.json", CONTROLLED_9B / "runs-2.json",
           CONTROLLED_9B / "runs-3.json"],
}
MD = HERE / "CROSS-MODE.md"
CSV = HERE / "CROSS-MODE.csv"
POSITIONS = (100, 2000, 8000, 16000, 24000)
METRICS = ("read",) + tuple("gen_%d" % p for p in POSITIONS)
MODELS = ("2b", "4b", "9b")
LABELS = {"cpu": "CPU only", "gpu": "GPU only", "tc": "GPU tensor-core (plan)", "tandem": "GPU+CPU tandem"}
TANDEM_SOURCE = "no throughput benchmark (runtime in docs/TANDEM-RUNTIME.md)"


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cpu_rows():
    """The native Windows, 6-thread, one-per-physical-core CPU run; 12 threads is not a mode."""
    out = {}
    for row in _read_csv(CPU_RUNS):
        if row["model"] in MODELS and row["threads"] == "6" and row["binding"] == "close/cores":
            out[row["model"]] = {"read": float(row["prompt"])}
            out[row["model"]].update({k: float(row["gen_%d" % p]) for k, p in zip(METRICS[1:], POSITIONS)})
    missing = [m for m in MODELS if m not in out]
    if missing:
        raise SystemExit("cpu runs are missing %s in %s" % (", ".join(missing), CPU_RUNS))
    return out


def gpu_rows():
    """Median of the three fresh-process GPU runs per cell."""
    vals = {m: {k: [] for k in METRICS} for m in MODELS}
    for row in _read_csv(GPU_RUNS):
        if row["source"] != "current" or row["model"] not in MODELS:
            continue
        vals[row["model"]]["read"].append(float(row["prompt"]))
        for k, p in zip(METRICS[1:], POSITIONS):
            vals[row["model"]][k].append(float(row["gen_%d" % p]))
    out = {}
    for model, cells in vals.items():
        if not cells["read"]:
            raise SystemExit("gpu runs are missing %s in %s" % (model, GPU_RUNS))
        out[model] = {k: statistics.median(v) for k, v in cells.items()}
        out[model]["runs"] = len(cells["read"])
    return out


def tandem_rows():
    """Measured GPU+CPU tandem runs; incomplete or absent data leaves the model unavailable."""
    if not TANDEM_RUNS.is_file():
        return {}
    vals = {m: {k: [] for k in METRICS} for m in MODELS}
    placements = {m: [] for m in MODELS}
    cpu_layers = {m: [] for m in MODELS}
    free = {m: [] for m in MODELS}
    for row in _read_csv(TANDEM_RUNS):
        model = row.get("model")
        if model not in MODELS:
            continue
        try:
            got = {"read": float(row["prompt"])}
            got.update({k: float(row["gen_%d" % p]) for k, p in zip(METRICS[1:], POSITIONS)})
        except (KeyError, TypeError, ValueError):
            continue
        for k, v in got.items():
            vals[model][k].append(v)
        if row.get("placement"):
            placements[model].append(row["placement"])
        if row.get("cpu_layers"):
            cpu_layers[model].append(row["cpu_layers"])
        if row.get("free_mib"):
            try:
                free[model].append(float(row["free_mib"]))
            except ValueError:
                pass
    out = {}
    for model, cells in vals.items():
        if not cells["read"] or any(len(cells[k]) != len(cells["read"]) for k in METRICS):
            continue
        out[model] = {k: statistics.median(v) for k, v in cells.items()}
        out[model]["runs"] = len(cells["read"])
        layer_vals = sorted(set(cpu_layers[model]))
        out[model]["placement"] = "+".join(sorted(set(placements[model]))) if placements[model] else "tandem"
        out[model]["cpu_layers"] = layer_vals[0] if len(layer_vals) == 1 else ("mixed" if layer_vals else "quarter")
        out[model]["free_mib"] = statistics.median(free[model]) if free[model] else None
    return out


def tc_rows():
    """Measured tensor-core (plan-engine) runs; a model without complete data stays unavailable."""
    if not TC_RUNS.is_file():
        return {}
    vals = {m: {k: [] for k in METRICS} for m in MODELS}
    for row in _read_csv(TC_RUNS):
        model = row.get("model")
        if model not in MODELS:
            continue
        try:
            got = {"read": float(row["prompt"])}
            got.update({k: float(row["gen_%d" % p]) for k, p in zip(METRICS[1:], POSITIONS)})
        except (KeyError, TypeError, ValueError):
            continue
        for k, v in got.items():
            vals[model][k].append(v)
    out = {}
    for model, cells in vals.items():
        if not cells["read"] or any(len(cells[k]) != len(cells["read"]) for k in METRICS):
            continue
        out[model] = {k: statistics.median(v) for k, v in cells.items()}
        out[model]["runs"] = len(cells["read"])
    return out


def controlled_rows():
    """The 2026-11-19 same-context CPU-only/tandem pairs at 24,576 tokens.

    A model with more than one pair uses the per-metric median; every file is still checked
    for the same context, placement and 6-layer split.
    """
    out = {}
    for model, paths in CONTROLLED_RUNS.items():
        vals = {mode: {m: [] for m in METRICS} for mode in ("cpu", "tandem")}
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SystemExit("%s: %s" % (path, exc)) from exc
            if (data.get("ctx") != 24576 or data.get("placement") != "cpu-first"
                    or int(data.get("cpu_layers", 0)) != 6):
                raise SystemExit("%s: expected ctx 24576, cpu-first, 6 CPU layers" % path)
            for mode in ("cpu", "tandem"):
                cells = data[mode]
                vals[mode]["read"].append(float(cells["read"]))
                for p in POSITIONS:
                    vals[mode]["gen_%d" % p].append(float(cells[str(p)]))
        out[model] = {}
        for mode in ("cpu", "tandem"):
            out[model][mode] = {m: statistics.median(vals[mode][m]) for m in METRICS}
            out[model][mode]["runs"] = len(paths)
    return out


def gpu_comparison(gpu_lo, gpu_hi):
    """Describe the 24K tandem/GPU ratio without assuming the tandem always loses.

    ``gpu_lo`` and ``gpu_hi`` are GPU-only divided by tandem, so a value above 1 means
    the tandem is slower and a value below 1 means it is faster.
    """
    if gpu_lo > 1.0:
        return ("The same rows are %.1fx-%.1fx slower than GPU-only, so this is a capacity "
                "trade, not a win for a model that fits the card." % (gpu_lo, gpu_hi))
    if gpu_hi < 1.0:
        return ("The same rows are %.1fx-%.1fx faster than GPU-only, so the split is not just "
                "a capacity trade at these measured rows." % (1.0 / gpu_hi, 1.0 / gpu_lo))
    return ("The 24K rows bracket GPU-only: up to %.1fx faster where the split wins and up to "
            "%.1fx slower where it does not." % (1.0 / gpu_lo, gpu_hi))


def matrix():
    """One row per mode and model; incomplete tandem data stays explicitly unavailable."""
    cpu, gpu, tandem, tc = cpu_rows(), gpu_rows(), tandem_rows(), tc_rows()
    controlled = controlled_rows()
    rows = []
    for model in MODELS:
        if model in controlled:
            c = controlled[model]["cpu"]
            runs = c["runs"]
            rows.append(dict(mode="cpu", available=1, model=model, runs=runs,
                             source="2026-11-19 controlled pair (CPU-only, %d run%s)"
                                    % (runs, "" if runs == 1 else "s"),
                             **{k: c[k] for k in METRICS}))
        else:
            rows.append(dict(mode="cpu", available=1, model=model, runs=1,
                             source="2026-11-10-i7-8086k-windows-cpu (6 threads, one run)",
                             **cpu[model]))
        rows.append(dict(mode="gpu", available=1, model=model, runs=gpu[model]["runs"],
                         source="2026-11-09-i7-8086k-rtx4060ti (median of 3)",
                         **{k: gpu[model][k] for k in METRICS}))
        if model in tc:
            rows.append(dict(mode="tc", available=1, model=model, runs=tc[model]["runs"],
                             source="2026-11-13-i7-8086k-rtx4060ti-tc (median of %d; "
                                    "9B with KINDI_TC_HP=1)" % tc[model]["runs"],
                             **{k: tc[model][k] for k in METRICS}))
        if model in controlled:
            t = controlled[model]["tandem"]
            runs = t["runs"]
            rows.append(dict(mode="tandem", available=1, model=model, runs=runs,
                             source="2026-11-19 controlled pair (cpu-first, 6 CPU layers, %d run%s)"
                                    % (runs, "" if runs == 1 else "s"),
                             **{k: t[k] for k in METRICS}))
        elif model in tandem:
            t = tandem[model]
            rows.append(dict(
                mode="tandem", available=1, model=model, runs=t["runs"],
                source="2026-11-12-i7-8086k-rtx4060ti-tandem (%s, %s CPU layer(s), %d run(s))"
                       % (t["placement"], t["cpu_layers"], t["runs"]),
                **{k: t[k] for k in METRICS}))
        else:
            rows.append(dict(mode="tandem", available=0, model=model, runs="", source=TANDEM_SOURCE,
                             **{k: None for k in METRICS}))
    return rows


def render_csv(rows):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(("mode", "available", "model") + METRICS + ("runs", "source"))
    for r in rows:
        w.writerow([r["mode"], r["available"], r["model"]]
                   + ["" if r[k] is None else "%.1f" % r[k] for k in METRICS]
                   + [r["runs"], r["source"]])
    return out.getvalue()


def render_md(rows):
    by = {(r["mode"], r["model"]): r for r in rows}

    measured_tandem = [m for m in MODELS if by[("tandem", m)]["available"] == 1]
    missing_tandem = [m for m in MODELS if m not in measured_tandem]
    if measured_tandem == list(MODELS):
        tandem_intro = ("All three rows are controlled CPU-only/tandem pairs from\n"
                        "  2026-11-19/20; the 2B/4B pairs are one run per placement, the 9B is the median of\n"
                        "  three, so no tandem-to-CPU ratio is cross-run.")
    elif measured_tandem:
        tandem_intro = ("Measured tandem rows in the table below are the median of three fresh-process runs\n"
                        "  per model (2026-11-13) (models: %s); the other tandem rows say `n/a` because they have\n"
                        "  not been measured."
                        % ", ".join(m.upper() for m in measured_tandem))
    else:
        tandem_intro = "No throughput benchmark has been run at these positions; the rows below say `n/a` and no tandem number is claimed."

    def cell(v):
        return "n/a" if v is None else "%.1f" % v

    lines = [
        "# Cross-mode benchmarks: CPU-only, GPU-only and GPU+CPU tandem",
        "",
        "Generated by `python docs/benchmarks/build_cross_mode.py`; `--check` fails on drift.",
        "All measured cells are from one machine: an Intel Core i7-8086K (6 physical cores) and an",
        "NVIDIA GeForce RTX 4060 Ti 8 GB, at a 24,576-token context, exact head, greedy, no speculative",
        "drafting.",
        "",
        "- **CPU only:** `KINDI_DEVICE=cpu` on native Windows 10, 6 threads (one per physical core),",
        "  24,576-token context. One fresh-process run per model for 2B/4B, three for 9B; no WSL. The",
        "  2B/4B/9B rows are the CPU half of the 2026-11-19/20 controlled pairs below.",
        "- **GPU only:** the card runs every layer; the CPU only drives the process (Windows 10, CuPy",
        "  14.2.0). Three fresh-process runs per model; the table uses their median.",
        "- **GPU tensor-core (plan):** `KINDI_TP=1` `KINDI_TC=1` (plan engine, tensor cores) on the",
        "  same card; the 9B runs with `KINDI_TC_HP=1` (4096 Hadamard blocks are refused otherwise).",
        "  Three fresh-process runs per model; the table uses their median.",
        "- **GPU+CPU tandem:** the dense `TandemEngine` exists and is token-exact on the 2B/9B (`docs/TANDEM-RUNTIME.md`). %s" % tandem_intro,
        "",
        "\"Read\" is the benchmark's 1,000-token prompt-reading probe. 100, 2K, 8K, 16K and 24K are",
        "generation at that token position in the allocated 24,576-token context. Tokens/s.",
        "",
        "| Model | Mode | Read | 100 | 2K | 8K | 16K | 24K |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for mode in ("gpu", "tc", "cpu", "tandem"):
            r = by[(mode, model)]
            lines.append("| %s | %s | %s |" % (model.upper(), LABELS[mode],
                                               " | ".join(cell(r[k]) for k in METRICS)))
    lines += [
        "",
        "Sources: [2026-11-09-i7-8086k-rtx4060ti](2026-11-09-i7-8086k-rtx4060ti/README.md) (GPU-only)",
        "and the controlled pair pages below. The CPU-only and tandem cells are one run per placement",
        "for 2B/4B and the median of three for 9B; the GPU and tensor-core cells are medians of three.",
        "The old 2026-11-10 CPU cells and 2026-11-12/13 tandem cells stay in the repository as history.",
        ""]
    if "2b" in measured_tandem and "4b" in measured_tandem:
        lines.append("Controlled 2B/4B pair: [2026-11-19-i7-8086k-rtx4060ti-2b-4b-controlled]"
                     "(2026-11-19-i7-8086k-rtx4060ti-2b-4b-controlled/RESULTS.md).")
    if "9b" in measured_tandem:
        lines.append("Controlled 9B runs (three): [2026-11-19-i7-8086k-rtx4060ti-9b-controlled]"
                     "(2026-11-19-i7-8086k-rtx4060ti-9b-controlled/RESULTS.md).")
    if any(("tc", m) in by and by[("tc", m)]["available"] == 1 for m in MODELS):
        lines.append("Tensor-core source: [2026-11-13-i7-8086k-rtx4060ti-tc](2026-11-13-i7-8086k-rtx4060ti-tc/README.md).")
    lines += [
        "",
        "## GPU-only speedup over CPU-only",
        "",
        "| Model | 100 | 2K | 8K | 16K | 24K |",
        "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        c, g = by[("cpu", model)], by[("gpu", model)]
        lines.append("| %s | %s |" % (model.upper(), " | ".join(
            "%.1fx" % (g[k] / c[k]) for k in METRICS[1:])))
    lines += [
        "",
        "The ratio grows with depth because the CPU attention path slows down faster than the GPU path.",
        "",
        "## Tensor-core (plan engine) speedup over GPU-only",
        "",
        "| Model | 100 | 2K | 8K | 16K | 24K |",
        "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        if ("tc", model) in by and by[("tc", model)]["available"] == 1:
            g, t = by[("gpu", model)], by[("tc", model)]
            lines.append("| %s | %s |" % (model.upper(), " | ".join(
                "%.1fx" % (t[k] / g[k]) for k in METRICS[1:])))
    if measured_tandem == list(MODELS):
        tandem_status = "All tandem cells are from the 2026-11-19/20 controlled pairs (9B: median of three)."
    elif measured_tandem:
        tandem_status = ("The %s tandem cell(s) are recorded from 2026-11-13; the %s cell(s) are not "
                         "recorded because the same single-run throughput benchmark has not been run "
                         "for those models." % (", ".join(m.upper() for m in measured_tandem),
                                                ", ".join(m.upper() for m in missing_tandem)))
    else:
        tandem_status = "No tandem cell is recorded because no throughput benchmark has been run."
    lines += [
        "",
        "## Tandem speedup over CPU-only",
        "",
        "| Model | 100 | 2K | 8K | 16K | 24K |",
        "|---|---:|---:|---:|---:|---:|"]
    tandem_gains = []
    for model in MODELS:
        c, t = by[("cpu", model)], by[("tandem", model)]
        if t["available"] != 1:
            lines.append("| %s | n/a | n/a | n/a | n/a | n/a |" % model.upper())
            continue
        gains = [t[k] / c[k] for k in METRICS[1:]]
        tandem_gains.append(max(gains))
        lines.append("| %s | %s |" % (model.upper(), " | ".join("%.1fx" % g for g in gains)))
    if tandem_gains:
        deep = list(measured_tandem)
        lo = min(by[("tandem", m)]["gen_24000"] / by[("cpu", m)]["gen_24000"] for m in deep)
        hi = max(by[("tandem", m)]["gen_24000"] / by[("cpu", m)]["gen_24000"] for m in deep)
        gpu_lo = min(by[("gpu", m)]["gen_24000"] / by[("tandem", m)]["gen_24000"] for m in deep)
        gpu_hi = max(by[("gpu", m)]["gen_24000"] / by[("tandem", m)]["gen_24000"] for m in deep)
        lines += [
            "",
            "Every measured tandem cell above is faster than CPU-only; at 24K the gain is",
            f"**{lo:.1f}x-{hi:.1f}x**. {gpu_comparison(gpu_lo, gpu_hi)}",
            "The measured capacity-bound case is outside this table: AH-Nanbeige4.3-33B IQ3_XXS",
            "on a 12 GB Quadro K6000 keeps 13",
            "layers in host memory and computes 55% of their FFN rows on the CPU; it writes",
            "3.85/3.55/3.56 tok/s at 480/2,016/8,160 tokens against llama.cpp's 2.19/1.96/1.52, and",
            "4.4-5.0 against 2.5-2.7 at 1,024 (Kindi 0.4.0; [BENCHMARKS.md](../BENCHMARKS.md)). That is a",
            "dense 33B model, not a MoE. A big-MoE CPU-compute row still needs the CPU expert lane in",
            "[MOE-TANDEM.md](../MOE-TANDEM.md), which has no backend yet.",
        ]
        lines += [
            "",
            "The 2B and 4B rows are direct A/B pairs. The 2B verified 4/4 in both placements",
            "(160/160 tokens); the 4B CPU placement verified 4/4, but the tandem placement was 3/4",
            "with one llama.cpp top-2 tie (159/160 tokens), so the 4B row is throughput evidence, not",
            "an exactness claim. The 9B was measured three times: run 3 reproduced run 2's tandem row",
            "within 3-8% and its CPU row within 4-10% at every CPU cell except 24K, where 1.8 -> 2.4",
            "tok/s (+33%). Run 2 is therefore not a run-wide outlier; run 1 is the high side. The three-run",
            "median makes the shallow ratio 3.0x-4.2x (median 4.1x), not a stable 3.0x; at 24K it is",
            "5.8x-8.2x (median 6.2x). Both 9B placements verified 4/4 (133/133) in run 1; runs 2-3 were",
            "throughput only. Tandem sampled peaks were 1,627 / 2,385 MiB device-wide for 2B/4B",
            "(605 / 1,373 MiB over idle), with 6,577 / 5,895 MiB free after load. The 9B's three runs",
            "peaked at 2,864 / 2,968 / 2,959 MiB (1,939 / 1,863 / 1,939 MiB over idle), with 5,086 /",
            "4,982 / 4,991 MiB free at peak and 5,245 MiB free after load in all three. CPU-only placements",
            "have no GPU allocation.",
        ]
    lines += [
        "",
        "## Tandem status and the gate for a number",
        "",
        tandem_status,
        "[TANDEM-STATUS.md](../TANDEM-STATUS.md)",
        "requires, in order:",
        "",
        "1. the CPU gate (supported ISA flags build, actionable fallback);",
        "2. a real layer-contiguous streaming pack;",
        "3. the `TandemEngine` exactness gate (passed on 2B/9B; 4B verify is 3/4 with one llama.cpp tie; 27B `cpu-first` and `gpu-first` passed 2026-11-12 at 8K);",
        "4. decode throughput and peak VRAM recorded for a larger-than-VRAM model, with the CPU/GPU",
        "   split stated; a tandem cell in this table is only a throughput rate, so the larger-than-VRAM",
        "   gate still needs a measured tandem rate plus a sampled peak for a model that does not fit VRAM;",
        "5. a committed replacement with the full CPU suite and the GPU verify passing.",
        "",
        "A tandem row is added to this matrix only after a recorded verify result and a measured rate at",
        "each of the same positions with peak VRAM. The 4B tandem row has a documented llama.cpp top-2",
        "tie, so it is throughput evidence, not a gate-3 exactness claim. Until then every unmeasured",
        "tandem row stays `n/a`.",
        "",
        "## Limits",
        "",
        "- **No 27B, Nemotron or Nanbeige CPU row.** The CPU engine covers the 2B, 4B and 9B only; the",
        "  larger models are GPU targets.",
        "- **CPU is native Windows, not WSL2.** The 2026-11-09 WSL2 page is kept as history; its CPU cells",
        "  are no longer the cross-mode source.",
        "- **One or three runs per CPU/tandem cell.** The 2B/4B pairs are one run per placement; the 9B",
        "  has three. Run 1 is the high side, run 3 tracks run 2 except CPU-only 24K; the row is a",
        "  median, not a stability claim. The third pair followed a passive >7 h cooldown, with the GPU",
        "  at 32 C at start; there was no instrumented thermal control. The GPU and tensor-core cells have",
        "  three runs and a spread of a few percent. The 4B tandem verify is 3/4 with one tie; do not",
        "  read that row as exactness.",
        "- **Tensor-core cells are medians of three fresh-process runs (2026-11-13).** The 9B cell",
        "  runs with `KINDI_TC_HP=1`; without it the engine refuses HB 4096 (see `kindi/spec.py`).",
        "- **24K is the deepest measured position.** The packs advertise 262,144 tokens, but no 32K/64K",
        "  cross-mode run exists.",
        "- This is a speed table, not a correctness test. The token-exact `verify` results are recorded in",
        "  the controlled pair pages.",
        ""]
    return "\n".join(lines)


def build():
    rows = matrix()
    return render_md(rows), render_csv(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the generated files are stale")
    args = ap.parse_args(argv)
    md, csv_text = build()
    if args.check:
        stale = []
        for path, text in ((MD, md), (CSV, csv_text)):
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(str(path))
        if stale:
            print("stale: %s; run `python docs/benchmarks/build_cross_mode.py`" % ", ".join(stale),
                  file=sys.stderr)
            return 1
        return 0
    with MD.open("w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    with CSV.open("w", encoding="utf-8", newline="\n") as f:
        f.write(csv_text)
    print("wrote %s and %s" % (MD, CSV))
    return 0


if __name__ == "__main__":
    sys.exit(main())
