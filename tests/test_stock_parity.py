"""The tracked stock-v0.6.1 greedy goldens are well formed and comparable."""
import json
import subprocess

import pytest

from tools import stock_parity as parity


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def prompts():
    return _load(parity.DEFAULT_PROMPTS)


@pytest.fixture(scope="module")
def goldens():
    return _load(parity.DEFAULT_GOLDENS)


def test_prompts_are_the_fixed_lengths_and_prefixes(prompts):
    assert prompts["format"] == 1
    assert prompts["generated_tokens"] == parity.GENERATED_TOKENS
    assert prompts["context"] == parity.CONTEXT
    by_position = {p["position"]: p for p in prompts["prompts"]}
    assert sorted(by_position) == sorted(parity.POSITIONS)
    for pos in parity.POSITIONS:
        assert len(by_position[pos]["ids"]) == pos
        assert by_position[pos]["sha256"] == parity._sha256_text(
            parity._canonical(by_position[pos]["ids"]))
    assert by_position[100]["ids"] == by_position[2000]["ids"][:100]
    assert by_position[2000]["ids"] == by_position[8000]["ids"][:2000]


def test_prompt_source_hash_is_the_tracked_source(prompts):
    source = parity.SOURCE_TEMPLATE * parity.SOURCE_REPEATS
    assert prompts["source_sha256"] == parity._sha256_text(source)


def test_stock_tag_is_the_pinned_commit():
    got = subprocess.check_output(["git", "-C", str(parity.ROOT), "rev-parse", "v0.6.1^{commit}"],
                                  text=True).strip()
    assert got == parity.STOCK_COMMIT


def test_goldens_cover_every_model_and_position(prompts, goldens):
    assert goldens["format"] == 1
    assert goldens["stock"]["tag"] == parity.STOCK_TAG
    assert goldens["stock"]["commit"] == parity.STOCK_COMMIT
    assert goldens["stock"]["prompts_sha256"] == parity._prompts_sha256(prompts)
    assert goldens["stock"]["source_sha256"] == prompts["source_sha256"]
    assert sorted(goldens["models"]) == list(parity.MODELS)
    for model in parity.MODELS:
        row = goldens["models"][model]
        assert len(row["manifest_sha256"]) == 64
        assert sorted(row["positions"]) == sorted(str(p) for p in parity.POSITIONS)
        for pos in parity.POSITIONS:
            out = row["positions"][str(pos)]
            assert len(out["ids"]) == parity.GENERATED_TOKENS
            assert all(isinstance(token, int) for token in out["ids"])
            assert isinstance(out["text"], str)


def test_exact_comparison_passes(goldens):
    result = {"models": {
        model: {"positions": goldens["models"][model]["positions"]}
        for model in parity.MODELS
    }}
    rows, mismatches = parity._compare(goldens, result, "fork")
    assert not mismatches
    assert len(rows) == len(parity.MODELS) * len(parity.POSITIONS)
    assert {row[3] for row in rows} == {"exact"}


def test_comparison_can_scope_to_one_model(goldens):
    result = {"models": {"2b": {"positions": goldens["models"]["2b"]["positions"]}}}
    rows, mismatches = parity._compare(goldens, result, "fork", models=("2b",))
    assert not mismatches
    assert len(rows) == len(parity.POSITIONS)
    assert {row[0] for row in rows} == {"2b"}


def test_comparison_names_the_first_difference(goldens):
    result = {"models": {
        model: {"positions": {
            pos: {"ids": list(value["ids"])}
            for pos, value in goldens["models"][model]["positions"].items()
        }}
        for model in parity.MODELS
    }}
    result["models"]["2b"]["positions"]["100"]["ids"][3] ^= 1
    rows, mismatches = parity._compare(goldens, result, "fork")
    assert mismatches == [("2b", 100, "fork", 3)]
    assert ("2b", 100, "fork", "MISMATCH", 3) in rows
