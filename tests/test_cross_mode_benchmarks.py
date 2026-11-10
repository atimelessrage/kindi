"""The cross-mode matrix must keep CPU-only and GPU-only measured and mark tandem explicitly unavailable."""
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


def test_matrix_shows_all_three_modes():
    rows = load().matrix()
    assert {r["mode"] for r in rows} == {"cpu", "gpu", "tandem"}
    assert {r["model"] for r in rows} == {"2b", "4b", "9b"}
    for mode in ("cpu", "gpu"):
        cells = [r for r in rows if r["mode"] == mode]
        assert len(cells) == 3
        assert all(r["available"] == 1 and r["read"] and r["gen_24000"] for r in cells)
    tandem = [r for r in rows if r["mode"] == "tandem"]
    assert len(tandem) == 3
    assert all(r["available"] == 0 and r["read"] is None and r["gen_24000"] is None for r in tandem)
