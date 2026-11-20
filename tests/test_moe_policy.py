"""kindi/moe.py: measured-use coverage and the expert placement every MoE engine shares.

The module is model-free on purpose, so these tests pin the numbers `kindi plan` and the Lightning
engine report: the share of routed picks a tier serves, and the top-n coverage the decode-slot line
compares against residency.
"""
import json

import pytest

from kindi.moe import MoESpec, Usage, plan


def _spec(n_exp=8, layers=(0, 1), n_used=2, eb=100):
    return MoESpec(layers, n_exp, n_used, eb, shared=True)


def test_coverage_is_the_share_of_picks_the_top_n_experts_take():
    spec = _spec()
    usage = Usage(spec, {0: [8, 1, 1, 0, 0, 0, 0, 0], 1: [4, 4, 0, 0, 0, 0, 0, 0]})
    assert usage.coverage(0) == 0.0
    assert usage.coverage(1) == pytest.approx(12 / 18.0)      # (8 + 4) / 18
    assert usage.coverage(2) == pytest.approx(17 / 18.0)      # (9 + 8) / 18
    assert usage.coverage(8) == 1.0 and usage.coverage(99) == 1.0
    assert usage.coverage(-3) == 0.0                          # never negative


def test_coverage_makes_no_claim_without_counts():
    spec = _spec()
    assert Usage(spec).coverage(2) is None                    # order only (an older usage file)
    # one layer without counts is enough to withhold the number: has_counts() needs every layer
    assert Usage(spec, {0: [1] * 8}).coverage(2) is None


def test_a_malformed_usage_file_is_refused_with_its_path(tmp_path):
    """A short or non-numeric count row changes the tier split silently; refuse it at load."""
    spec = _spec()
    path = tmp_path / "usage.json"
    for counts in ({0: [1] * 7}, {0: [1, 2, 3, 4, 5, 6, 7, -1]},
                   {0: [1, 2, 3, 4, 5, 6, 7, "x"]}):
        path.write_text(json.dumps({"counts": counts}), encoding="utf-8")
        with pytest.raises(ValueError, match="bad expert usage file"):
            Usage.load(spec, str(path))


def test_a_malformed_order_is_refused_too():
    spec = _spec()
    for order in ({0: [0, 1, 8]}, {0: [0, 0]}, {0: [0, "x"]}, {9: [0]}):
        with pytest.raises(ValueError):
            Usage(spec, order=order)


def test_a_partial_count_map_and_a_partial_order_are_valid():
    """Only the shape is mandatory: an older file may carry one layer's counts or a top-k order."""
    spec = _spec()
    usage = Usage(spec, {0: [3, 2, 1, 0, 0, 0, 0, 0]}, {1: [7, 6]})
    assert usage.counts[0][1] == 2.0
    assert usage.order(1)[:2] == [7, 6] and len(usage.order(1)) == 8


def test_a_usage_file_for_another_model_keeps_only_this_specs_layers(tmp_path):
    """A bundled 128-expert file read against a 32-expert test pack is not corrupt: those rows
    simply do not describe this pack, so they are ignored rather than indexing a short list."""
    spec = _spec()
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"counts": {"0": [1] * 8, "9": [1] * 128}}), encoding="utf-8")
    usage = Usage.load(spec, str(path))
    assert set(usage.counts) == {0}


def test_a_card_budget_that_holds_everything_puts_every_expert_on_the_card():
    spec = _spec()
    p = plan(spec, Usage(spec), 16 * 100)
    assert p.counts("card") == {0: 8, 1: 8}
    assert p.counts("host") == {0: 0, 1: 0}
    assert p.served("card") is None                           # no counts, no share claimed


def test_the_host_budget_moves_the_least_used_experts_to_disk():
    spec = _spec(n_exp=4, layers=(0,), n_used=1)
    usage = Usage(spec, {0: [10, 5, 1, 0]})
    p = plan(spec, usage, 2 * 100, host_bytes=100)
    assert p.tier[0][0] == "card" and p.tier[0][1] == "card"  # the most used, in order
    assert p.tier[0][2] == "host"                             # the used one that still fits
    assert p.tier[0][3] == "disk"                             # the unused one does not
    assert p.served("card") == pytest.approx(15 / 16.0)


def test_a_card_below_one_token_of_experts_is_a_memory_error():
    spec = _spec(n_exp=8, layers=(0, 1), n_used=2)
    with pytest.raises(MemoryError, match="need 2 experts each"):
        plan(spec, Usage(spec), 100)                          # one expert of room, two needed per layer


def test_the_weighted_placement_keeps_each_layer_servable():
    spec = _spec(n_exp=8, layers=(0, 1), n_used=2)
    usage = Usage(spec, {0: [9, 8, 7, 6, 5, 4, 3, 2], 1: [2, 3, 4, 5, 6, 7, 8, 9]})
    p = plan(spec, usage, 8 * 100, weighted=True)
    assert all(p.count(i, "card") >= spec.n_used for i in spec.layers)
    # the share is exactly the picks whose expert landed on the card, summed over both layers
    served = sum(usage.counts[i][e] for i in spec.layers for e in range(spec.n_exp)
                 if p.tier[i][e] == "card")
    assert p.served("card") == pytest.approx(served / (sum(range(2, 10)) * 2))
