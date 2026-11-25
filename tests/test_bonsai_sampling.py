# SPDX-License-Identifier: AGPL-3.0-only
"""The repeatable Bonsai 2.5 sampling benchmark plans the publisher's own settings.

The values are Prism's card for Ternary Bonsai 2.5 27B; the point of these tests is that the
benchmark cannot drift from ``kindi.profiles`` (it asks that module), and that the combos it runs
cover both modes and every supported effort. No engine is needed.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bonsai_sampling_bench as bench  # noqa: E402, I001


THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
            "repeat_last_n": 64, "presence_penalty": 0.0}
INSTRUCT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
            "repeat_last_n": 64, "presence_penalty": 1.5}


def test_the_benchmark_uses_the_publisher_sets():
    assert bench.settings("bonsai2.5-27b", thinking=True) == THINKING
    assert bench.settings("bonsai2.5-27b", thinking=False) == INSTRUCT


def test_the_combos_cover_instruct_and_every_thinking_effort():
    assert bench.combos() == [("instruct", None), ("thinking", "xhigh"),
                              ("thinking", "medium"), ("thinking", "low")]
    assert bench.combos(("xhigh",)) == [("instruct", None), ("thinking", "xhigh")]


def test_the_benchmark_refuses_a_model_without_the_2_5_profile():
    # the tool's own check, without loading an engine: the manifest id must be the 2.5 one
    from kindi.backends.bonsai import model_id_for_manifest

    assert model_id_for_manifest({"model": "Ternary-Bonsai-2.5-27B"}) == "bonsai2.5-27b"
    assert model_id_for_manifest({"model": "Ternary-Bonsai-2-27B"}) != "bonsai2.5-27b"
