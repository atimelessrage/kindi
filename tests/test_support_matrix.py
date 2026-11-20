# SPDX-License-Identifier: AGPL-3.0-only
"""The generated support matrix must be current and cover every model, format and mode.

`docs/SUPPORT-MATRIX.md` is rendered from `kindi/catalog.py`, `kindi/convert.py`,
`kindi/tp.py` and `kindi/kcpu.c`, plus the curated verification table in
`kindi/support_matrix.py`. These tests fail when the code changes without regenerating it
(`python -m kindi.support_matrix`), and when a catalog model has no recorded verification row.
"""
from pathlib import Path

import pytest

from kindi import support_matrix as sm


def matrix_text():
    return Path(sm.MATRIX).read_text(encoding="utf-8").replace("\r\n", "\n")


def test_matrix_is_current():
    assert matrix_text() == sm.render(), \
        "docs/SUPPORT-MATRIX.md is stale; run `python -m kindi.support_matrix`"


def test_every_catalog_model_is_listed():
    text = matrix_text()
    missing = [mid for mid in sm.catalog() if "| `%s` |" % mid not in text]
    assert not missing, "catalog models missing from the matrix: %s" % missing


def test_every_pack_format_is_listed_with_its_type():
    text = matrix_text()
    for f in sm.formats():
        assert "| `%s` | %d |" % (f["format"], f["type"]) in text, f


def test_bits_per_weight_matches_the_block_layout():
    by_name = {f["format"]: f for f in sm.formats()}
    assert by_name["PTQ1_0"]["bpw"] == pytest.approx(1.75)
    assert by_name["PQ2_0"]["bpw"] == pytest.approx(2.125)
    assert by_name["Q2_0"]["bpw"] == pytest.approx(2.25)


def test_the_verification_table_covers_every_model():
    assert set(sm.STATUS) == set(sm.catalog())
    assert all(len(row) == len(sm.CAPABILITIES) for row in sm.STATUS.values())


def test_a_new_catalog_model_without_a_row_is_refused():
    models = dict(sm.catalog())
    models["brand-new"] = {"what": "test", "file": "x.gguf", "size": 1, "one": 1, "two": 0}
    with pytest.raises(SystemExit, match="no verification row"):
        sm.render(models=models)


def test_kv_modes_and_the_k8v4_gate_come_from_tp():
    kv = sm.kv_modes()
    assert kv["default"] == "q8_0"
    assert "k8v4" in kv["modes"] and kv["gated"]
    text = matrix_text()
    assert "`KINDI_KV_EXPERIMENTAL=1`" in text


def test_cpu_requirements_are_read_from_kcpu_c():
    isa = sm.cpu_isa()
    assert "AVX" in isa and "AVX2" in isa and "FMA" in isa and "F16C" in isa
    assert "AVX-512" not in isa        # the source has no 512-bit intrinsics
    assert "FMA is optional" in matrix_text()


def test_the_cards_section_lists_the_rig_cards():
    text = matrix_text()
    assert "| NVIDIA P106-100 | Pascal 6.1 | maxwell |" in text
    assert "| Quadro P4000 | Pascal 6.1 | pascal8 |" in text
    assert "| Quadro M4000 | Maxwell 5.2 | maxwell |" in text
    assert "| Quadro M5000 | Maxwell 5.2 | maxwell | not measured | not measured | not measured | not measured | not measured |" in text
    assert "| Quadro RTX 4000 | Turing 7.5 | turing | not measured | not measured | not measured | not measured | not measured |" in text
    assert "| Quadro M4000 | Maxwell 5.2 | maxwell | not measured | not measured | not measured | not measured | not measured |" in text


def test_the_cards_section_lists_the_measured_cards():
    # the already-measured cards are registered alongside the rig cards: the 4060 Ti from its
    # folded verify_matrix run, the GTX 1080 / K1200 with honest not-measured verify cells (only
    # speed benchmarks exist for them, linked from the section note)
    text = matrix_text()
    assert "| NVIDIA RTX 4060 Ti | Ada 8.9 | tc | plain 4/4 @ 193.4 tok/s" in text
    assert "measured 2026-11-10" in text and "2026-11-10-i7-8086k-rtx4060ti/results.json" in text
    assert "| NVIDIA GTX 1080 | Pascal 6.1 | pascal8 | not measured | not measured | not measured | not measured | not measured |" in text
    assert "| Quadro K1200 | Maxwell 5.0 | maxwell | not measured | not measured | not measured | not measured | not measured |" in text


def _payload(card_name="Quadro P4000", gpus=None):
    if gpus is None:
        gpus = [{"name": card_name, "driver": "535.104.05", "memory_mib": 8192, "compute_capability": "6.1"}]
    return {
        "started_utc": "2026-11-16T10:00:00Z", "git_revision": "abc1234",
        "environment": {"gpus": gpus},
        "results": [
            {"model": "2b", "mode": "plain", "status": "pass", "prompts_pass": 4, "prompts_tie": 0,
             "prompts_fail": 0, "tokens_per_second": 30.1},
            {"model": "2b", "mode": "tc", "status": "refused", "prompts_pass": 0, "prompts_tie": 0,
             "prompts_fail": 0, "tokens_per_second": None},
            {"model": "9b", "mode": "plain", "status": "pass", "prompts_pass": 3, "prompts_tie": 1,
             "prompts_fail": 0, "tokens_per_second": 13.5},
            {"model": "27b", "mode": "plain", "status": "fail", "prompts_pass": 2, "prompts_tie": 0,
             "prompts_fail": 2, "tokens_per_second": 4.2, "first_diff_token": 36},
        ],
    }


def test_from_verify_derives_the_cells_from_the_run(tmp_path, monkeypatch):
    # the rig path: no hand-copied number anywhere; the cells are built from the run's rows
    import json as _json
    source = tmp_path / "results.json"
    source.write_text(_json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(sm, "_evidence_link", lambda _src: "benchmarks/2026-11-16-p4000/results.json")
    slug, cells = sm.apply_results(str(source), registry_path=tmp_path / "card-results.json",
                                   matrix_path=tmp_path / "SUPPORT-MATRIX.md")
    assert slug == "p4000"
    assert cells["2b"] == "plain 4/4 @ 30.1 tok/s · tc refused"
    assert cells["9b"] == "plain 3 exact + tie @ 13.5 tok/s"
    assert cells["4b"] == "not run"
    assert cells["27b"] == "plain failed (first diff 36)"
    registry = _json.loads((tmp_path / "card-results.json").read_text(encoding="utf-8"))
    entry = registry["cards"]["p4000"]
    assert entry["measured"] == "2026-11-16T10:00:00Z" and entry["git"] == "abc1234"
    assert entry["evidence"] == "benchmarks/2026-11-16-p4000/results.json"
    text = (tmp_path / "SUPPORT-MATRIX.md").read_text(encoding="utf-8")
    assert "| Quadro P4000 | Pascal 6.1 | pascal8 | plain 4/4 @ 30.1 tok/s · tc refused | not run |" in text
    assert "[results](benchmarks/2026-11-16-p4000/results.json)" in text
    assert "| Quadro M4000 | Maxwell 5.2 | maxwell | not measured | not measured | not measured | not measured | not measured |" in text


def test_from_verify_accepts_m5000_and_rtx4000_slugs(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(sm, "_evidence_link", lambda _src: "benchmarks/t/results.json")
    for name, slug in (("Quadro M5000", "m5000"), ("Quadro RTX 4000", "rtx4000")):
        source = tmp_path / ("%s.json" % slug)
        source.write_text(_json.dumps(_payload(card_name=name)), encoding="utf-8")
        got_slug, _cells = sm.apply_results(str(source), registry_path=tmp_path / "registry.json",
                                            matrix_path=None)
        assert got_slug == slug


def test_from_verify_refuses_a_gitignored_source():
    # a bench-logs link would fail the docs-link guard, so the tool refuses instead of writing it
    with pytest.raises(SystemExit, match="gitignored"):
        sm._evidence_link("bench-logs/verify-matrix-p4000/results.json")
    with pytest.raises(SystemExit, match="not under docs/"):
        sm._evidence_link("/tmp/results.json")


def test_from_verify_refuses_an_unknown_card_and_names_the_known_ones(tmp_path):
    import json as _json
    source = tmp_path / "results.json"
    source.write_text(_json.dumps(_payload(card_name="Mystery 3000")), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a known card") as excinfo:
        sm.apply_results(str(source), registry_path=tmp_path / "registry.json", matrix_path=None)
    assert "p4000" in str(excinfo.value)


def test_from_verify_refuses_a_multi_card_run_without_a_slug(tmp_path):
    import json as _json
    gpus = [{"name": "Quadro P4000"}, {"name": "Quadro M4000"}]
    source = tmp_path / "results.json"
    source.write_text(_json.dumps(_payload(gpus=gpus)), encoding="utf-8")
    with pytest.raises(SystemExit, match="--card"):
        sm.apply_results(str(source), registry_path=tmp_path / "registry.json", matrix_path=None)


def test_from_verify_card_slug_skips_name_matching(tmp_path, monkeypatch):
    # no recorded GPU at all (CPU container) is fine when the operator says which card it was
    import json as _json
    source = tmp_path / "results.json"
    source.write_text(_json.dumps(_payload(gpus=[])), encoding="utf-8")
    monkeypatch.setattr(sm, "_evidence_link", lambda _src: "benchmarks/t/results.json")
    slug, cells = sm.apply_results(str(source), registry_path=tmp_path / "registry.json",
                                   matrix_path=None, card="m4000")
    assert slug == "m4000"
    assert cells["2b"] == "plain 4/4 @ 30.1 tok/s · tc refused"
