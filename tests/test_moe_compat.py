# SPDX-License-Identifier: AGPL-3.0-only
"""The requested small MoE models are recognized names, not Kindi backends.

The classifier is not an engine and makes no inference claim. It exists so that
`kindi convert` can refuse a recognized architecture with the milestone that is
actually missing, instead of the generic "expected a qwen35 ..." message.
"""
import pytest

from kindi.moe_compat import unsupported_reason


class FakeGGUF:
    def __init__(self, arch, name="Model", tensors=(), **kv):
        self.kv = {"general.architecture": arch, "general.name": name}
        self.kv.update(kv)
        self.tensors = list(tensors)


@pytest.mark.parametrize(
    ("arch", "model"),
    [
        ("qwen3moe", "Qwen3-Coder-30B-A3B"),
        ("qwen35moe", "Qwen3.6-35B-A3B"),
        ("cohere2moe", "North Mini Code 1.0"),
        ("flint", "Recursive Flint 40B/2.5B"),
    ],
)
def test_each_requested_small_moe_architecture_is_recognized(arch, model):
    reason = unsupported_reason(FakeGGUF(arch, model))
    assert reason is not None
    assert model in reason
    assert arch in reason
    assert "does not have a native backend" in reason


def test_nemotron_cascade_2_is_handled_by_the_cascade_path_not_the_generic_refusal():
    # convert.py routes Cascade 2 to kindi/cascade.py before this classifier;
    # the generic unsupported-MoE message must not claim it either.
    g = FakeGGUF(
        "nemotron_h_moe",
        "Nemotron Cascade 2 30B-A3B",
        tensors=[("blk.0.ffn_gate_exps.weight", [1856, 2688, 128], 12, 0)],
    )
    assert unsupported_reason(g) is None


def test_ternary_lightning_is_left_to_the_lightning_backend():
    g = FakeGGUF(
        "nemotron_h_moe",
        "Nemotron 3.5 Lightning 30B-A3B",
        tensors=[("blk.0.ffn_gate_exps.weight", [1856, 2688, 128], 42, 0)],
    )
    assert unsupported_reason(g) is None


def test_gemma4_26b_moe_is_recognized_by_expert_tensors_and_shape():
    g = FakeGGUF(
        "gemma4",
        "Gemma 4 26B-A4B",
        tensors=[
            ("blk.0.ffn_gate_up_exps.weight", [2816, 1408, 128], 12, 0),
            ("blk.0.ffn_down_exps.weight", [704, 2816, 128], 7, 0),
        ],
        **{"gemma4.block_count": 30, "gemma4.embedding_length": 2816},
    )
    reason = unsupported_reason(g)
    assert reason is not None
    assert "Gemma 4 26B-A4B" in reason
    assert "E4B-only" in reason


def test_supported_dense_architectures_are_not_labeled_moes():
    assert unsupported_reason(FakeGGUF("qwen35", "Ternary Bonsai")) is None
    assert unsupported_reason(FakeGGUF("llama", "Nanbeige")) is None
