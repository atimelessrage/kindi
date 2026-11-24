# SPDX-License-Identifier: AGPL-3.0-only
"""The warp-mask gate: a shuffle whose mask names lanes that cannot arrive hangs the kernel.

Upstream 0.7.1 documents ``tools/check_warp_masks.py`` in its changelog but the tag does not ship
it, so the fork keeps it. These tests pin the classification on synthetic kernels (so the gate is
not vacuous) and the real tree's reviewed baseline.
"""
import json
from pathlib import Path

from tools import check_warp_masks as gate

ROOT = Path(__file__).resolve().parents[1]


def classify(text, line_no=None):
    """The classes of the intrinsics on one line of a synthetic .cu body."""
    lines = text.split("\n")
    if line_no is None:
        line_no = [i for i, ln in enumerate(lines) if "__shfl" in ln or "__syncwarp" in ln
                   or "__activemask" in ln or "KSHFL" in ln][0]
    out = []
    for match in gate.INTRINSIC.finditer(lines[line_no]):
        out.append(gate.classify(None, lines, line_no, match)[0])
    return out


def test_full_mask_shuffle_is_clean():
    assert classify("x = __shfl_xor_sync(0xffffffffu, v, o);") == ["full-mask"]


def test_partial_literal_mask_needs_review():
    assert classify("x = __shfl_xor_sync(0x000000FFu, v, 1);") == ["partial-mask"]


def test_activemask_is_its_own_class(tmp_path):
    assert classify("unsigned m = __activemask();") == ["activemask"]
    source = tmp_path / "probe.cu"
    source.write_text("x = __shfl_sync(__activemask(), v, 0);\n", encoding="utf-8")
    rows = gate.scan(source)
    # the mask argument is not a second site
    assert [row[2] for row in rows] == ["activemask"]


def test_syncwarp_without_a_mask_is_full():
    assert classify("__syncwarp();") == ["full-mask"]
    assert classify("__syncwarp(0xffffffffu);") == ["full-mask"]


def test_unguarded_legacy_intrinsic_is_flagged():
    """The bug class of the 0.7.1 hang: a pre-Volta intrinsic in a CUDA-only file."""
    body = "x = __shfl_xor(v, o);"
    assert classify(body) == ["legacy-unguarded"]


def test_arch_guarded_legacy_intrinsic_is_accepted():
    body = "#if __CUDA_ARCH__ < 700\nx = __shfl_xor(v, o);\n#endif"
    assert classify(body) == ["legacy-guarded"]


def test_hip_port_block_legacy_intrinsic_is_accepted():
    """A HIP branch keeps the legacy form on purpose: HIP has no _sync variants."""
    body = "#if KHIP\nx = __shfl_xor(v, o);\n#endif"
    assert classify(body) == ["legacy-hip"]


def test_compat_macro_use_is_accepted(tmp_path):
    source = tmp_path / "probe.cu"
    source.write_text("x = KSHFLX(v, o);\n", encoding="utf-8")
    assert [row[2] for row in gate.scan(source)] == ["compat-macro"]


def test_the_real_tree_passes_its_baseline():
    files = gate.collect(ROOT)
    assert files, "no kindi/*.cu found"
    baseline = gate.load_baseline()
    new, unexplained, stale = gate.compare(files, baseline)
    assert not new, "\n".join(new)
    assert not unexplained, "\n".join(unexplained)
    assert not stale, "\n".join(stale)


def test_the_baseline_has_a_reason_for_every_row():
    baseline = gate.load_baseline()
    rows = [(name, row) for name, rows in baseline.items() for row in rows]
    assert rows, "the baseline is empty"
    for name, row in rows:
        assert (row.get("reason") or "").strip(), "%s:%d has no reason" % (name, row["line"])


def test_the_baseline_counts_are_pinned():
    """A count that moves without a baseline edit would be a silent pass."""
    files = gate.collect(ROOT)
    total = gate.counts(files)
    assert total.get("partial-mask") == 7
    assert total.get("legacy-unguarded", 0) == 0, "an unguarded legacy intrinsic needs a guard"
    reviewed = sum(len(rows) for rows in gate.reviewable(files).values())
    assert reviewed == 7


def test_update_requires_a_reason(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "BASELINE", tmp_path / "baseline.json")
    assert gate.main(["--update"]) == 2
    assert "needs --reason" in capsys.readouterr().err


def test_a_new_site_fails_the_check(tmp_path, monkeypatch, capsys):
    """A new partial mask is found without any baseline edit."""
    repo = tmp_path / "repo"
    (repo / "kindi").mkdir(parents=True)
    (repo / "kindi" / "probe.cu").write_text(
        "__global__ void k(){ x = __shfl_xor_sync(0x0000000Fu, v, 1); }\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", repo)
    monkeypatch.setattr(gate, "BASELINE", tmp_path / "baseline.json")
    assert gate.main([]) == 1
    err = capsys.readouterr().err
    assert "probe.cu:1" in err
    assert "new (not in the baseline)" in err


def test_the_baseline_file_is_valid_json_with_a_note():
    body = json.loads((ROOT / "tools" / "warp_mask_baseline.json").read_text(encoding="utf-8"))
    assert body.get("note")
    assert isinstance(body.get("sites"), dict)
