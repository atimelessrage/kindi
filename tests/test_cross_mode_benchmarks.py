"""The cross-mode matrix must keep CPU-only and GPU-only measured, mark a tandem cell
available only when the same complete throughput benchmark exists, and derive every
published Tandem / GPU ratio from the rates in its table."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "benchmarks" / "build_cross_mode.py"
RATE_PAGES = (
    ("2B", "2026-11-19-i7-8086k-rtx4060ti-2b-4b-controlled/RESULTS.md", "### 2B"),
    ("4B", "2026-11-19-i7-8086k-rtx4060ti-2b-4b-controlled/RESULTS.md", "### 4B"),
    ("9B", "2026-11-19-i7-8086k-rtx4060ti-9b-controlled/RESULTS.md", "## Throughput (tokens/s)"),
)
METRIC_LABELS = ("Read", "100", "2K", "8K", "16K", "24K")


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
    assert nine_b["available"] == 1 and nine_b["runs"] == 3


def _section_table(text, heading):
    lines = text.splitlines()
    assert heading in lines, "missing heading %r" % heading
    start = lines.index(heading)
    table = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break
    assert table, "missing table under %r" % heading
    return table


def _cells(row):
    return [cell.strip().strip("*").strip() for cell in row.strip().strip("|").split("|")]


def _row(table, placement, qualifier=None):
    matches = [row for row in table
               if _cells(row)[0].startswith(placement)
               and (qualifier is None or (_cells(row)[1] == qualifier))]
    assert len(matches) == 1, "expected one %r row%s, found %d" % (
        placement, " with %r" % qualifier if qualifier else "", len(matches))
    return matches[0]


def _rates(row):
    values = []
    for cell in _cells(row)[1:]:
        if re.fullmatch(r"\d+(?:\.\d+)?", cell):
            values.append(float(cell))
    assert len(values) == len(METRIC_LABELS), "%r does not hold six rates: %r" % (row, values)
    return values


def _ratios(row):
    values = [cell for cell in _cells(row)[1:] if re.fullmatch(r"\d+\.\d+x", cell)]
    assert len(values) == len(METRIC_LABELS), "%r does not hold six ratios: %r" % (row, values)
    return values


def test_tandem_gpu_rows_recompute_from_measured_rates():
    """Every published Tandem / GPU cell is tandem throughput divided by GPU-only."""
    for model, relative_path, heading in RATE_PAGES:
        table = _section_table((ROOT / "docs" / "benchmarks" / relative_path).read_text(encoding="utf-8"),
                               heading)
        qualifier = "3-run median" if model == "9B" else None
        gpu = _rates(_row(table, "GPU only", qualifier))
        tandem = _rates(_row(table, "GPU+CPU tandem", qualifier))
        actual = _ratios(_row(table, "Tandem / GPU"))
        expected = ["%.1fx" % (t / g) for t, g in zip(tandem, gpu, strict=True)]
        for metric, shown, derived in zip(METRIC_LABELS, actual, expected, strict=True):
            assert shown == derived, "%s %s: Tandem / GPU says %s, tandem/GPU gives %s" % (
                model, metric, shown, derived)


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
