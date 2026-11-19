"""The cross-mode matrix must keep CPU-only and GPU-only measured, and mark a tandem cell
available only when the same complete throughput benchmark exists."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "benchmarks" / "build_cross_mode.py"


def load():
    spec = importlib.util.spec_from_file_location("build_cross_mode", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_files_are_current():
    mod = load()
    md, table = mod.build()
    assert (ROOT / "docs" / "benchmarks" / "CROSS-MODE.md").read_text(encoding="utf-8") == md
    assert (ROOT / "docs" / "benchmarks" / "CROSS-MODE.csv").read_text(encoding="utf-8") == table


def test_matrix_shows_all_four_modes():
    rows = load().matrix()
    assert {r["mode"] for r in rows} == {"cpu", "gpu", "tc", "tandem"}
    assert {r["model"] for r in rows} == {"2b", "4b", "9b"}
    for mode in ("cpu", "gpu", "tc"):
        cells = [r for r in rows if r["mode"] == mode]
        assert len(cells) == 3
        assert all(r["available"] == 1 and r["read"] and r["gen_24000"] for r in cells)
    tandem = [r for r in rows if r["mode"] == "tandem"]
    assert len(tandem) == 3
    for r in tandem:
        if r["available"] == 1:
            assert r["read"] and r["gen_24000"]
            assert r["source"].startswith("2026-11-19 controlled pair")
        else:
            assert r["available"] == 0 and r["read"] is None and r["gen_24000"] is None
    for r in rows:
        if r["mode"] in ("cpu", "tandem"):
            assert r["source"].startswith("2026-11-19 controlled pair"), r
    md, _ = load().build()
    if all(r["available"] == 1 for r in tandem):
        assert "All three rows are controlled CPU-only/tandem pairs from" in md
        assert "no tandem-to-CPU ratio is cross-run" in " ".join(md.split())
        assert "the other tandem rows say `n/a`" not in md
    two_b = next(r for r in tandem if r["model"] == "2b")
    assert two_b["available"] == 1 and two_b["source"].startswith(
        "2026-11-19 controlled pair (cpu-first")
    nine_b = next(r for r in tandem if r["model"] == "9b")
    assert nine_b["available"] == 1 and nine_b["runs"] == 2


def test_gpu_comparison_follows_the_measured_ratio():
    module = load()
    slower = module.gpu_comparison(3.9, 5.5)
    assert "slower than GPU-only" in slower
    assert "capacity trade" in slower
    faster = module.gpu_comparison(0.5, 0.8)
    assert "faster than GPU-only" in faster
    assert "capacity trade, not a win" not in faster
    assert "not just a capacity trade" in faster
    mixed = module.gpu_comparison(0.8, 1.4)
    assert "bracket GPU-only" in mixed
