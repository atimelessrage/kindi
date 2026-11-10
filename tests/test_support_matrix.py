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
    assert "AVX2" in isa and "FMA" in isa and "F16C" in isa
    assert "AVX-512" not in isa        # the source has no 512-bit intrinsics
