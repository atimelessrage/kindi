"""The cross-mode matrix must keep at least one measured CPU+GPU tandem that gains speed.

This is a documentation/evidence gate, not a GPU run: it reads the same measured CSV cells
``build_cross_mode.py`` uses.  A tandem that only ties the CPU engine is not the case the
project needs, so at least one measured tandem row must beat CPU-only on every throughput
metric.  It does not pin an ordering against GPU-only: a future tandem that also beats the
single card is allowed, and the page keeps the measured capacity caveat either way.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "benchmarks" / "build_cross_mode.py"


def _matrix_module():
    spec = importlib.util.spec_from_file_location("build_cross_mode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _by_cell(rows):
    return {(row["mode"], row["model"]): row for row in rows}


def test_at_least_one_measured_tandem_gains_over_cpu_only():
    module = _matrix_module()
    by = _by_cell(module.matrix())
    measured = [model for model in module.MODELS if by[("tandem", model)]["available"] == 1]
    assert measured, "no measured tandem rows; the cross-mode matrix lost its only gain evidence"

    gains = {}
    for model in measured:
        cpu, tandem = by[("cpu", model)], by[("tandem", model)]
        if all(tandem[metric] > cpu[metric] for metric in module.METRICS):
            gains[model] = tandem["gen_24000"] / cpu["gen_24000"]
    assert gains, "no measured tandem beats CPU-only on every metric; the matrix lost its gain"


def test_generated_page_names_the_gain_and_its_limit():
    module = _matrix_module()
    md, _csv_text = module.build()
    assert "Tandem speedup over CPU-only" in md
    assert "capacity trade" in md
    by = _by_cell(module.matrix())
    measured = [model for model in module.MODELS if by[("tandem", model)]["available"] == 1]
    assert measured, "no measured tandem rows to compare with GPU-only"
    gpu_ratios = [by[("gpu", model)]["gen_24000"] / by[("tandem", model)]["gen_24000"]
                  for model in measured]
    assert module.gpu_comparison(min(gpu_ratios), max(gpu_ratios)) in md
    assert "big-MoE CPU-compute row still needs the CPU expert lane" in md
