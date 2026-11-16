# SPDX-License-Identifier: AGPL-3.0-only
"""Model-free tests for the MoE tandem scheduler.

Every test builds a real Placement and ExpertCache and checks what the plan does
with slots, pins and lanes: no tensors, no GPU, no backend.
"""
import json

import pytest

from kindi.expert_cache import ExpertCache, ExpertCacheFull
from kindi.moe import MoESpec, Usage, plan as moe_plan
from kindi.moe_tandem import CARD, COPY, CPU, MoeTandem

LAYERS = (0, 1)
N_EXP = 4


def placement(card_bytes=600, host_bytes=None):
    """Two MoE layers of 4 experts, 2 used per token, experts of 100 bytes."""
    spec = MoESpec(LAYERS, N_EXP, 2, 100, shared=True)
    usage = Usage(spec, {0: [9.0, 8.0, 1.0, 0.0], 1: [9.0, 8.0, 1.0, 0.0]})
    return spec, moe_plan(spec, usage, card_bytes, host_bytes=host_bytes)


def tiers(p):
    return {(layer, e): p.tier[layer][e] for layer in p.spec.layers for e in range(p.spec.n_exp)}


def test_card_experts_need_no_slot_and_host_experts_do():
    spec, p = placement(card_bytes=600)          # 2 of 4 per layer on the card
    t = tiers(p)
    cache = ExpertCache(8, slot_bytes=100)
    tandem = MoeTandem(p, cache)
    keys = [(0, expert) for expert in range(N_EXP)]

    plan = tandem.forward([(1, keys)])
    by_key = {(pair.layer, pair.expert): pair for pair in plan[1]}

    assert len(plan[1]) == N_EXP
    for key, pair in by_key.items():
        if t[key] == "card":
            assert pair.lane == CARD and pair.slot is None
        else:
            assert pair.lane == COPY and pair.slot is not None
    # only the non-card experts were copied, once each, in router order
    assert plan.copy_keys == tuple(k for k in keys if t[k] != "card")
    assert plan.metrics()["card_pairs"] == sum(1 for k in keys if t[k] == "card")
    assert plan.metrics()["copy_bytes"] == len(plan.copies) * 100
    assert plan.release() == 1
    assert cache.metrics()["active_pins"] == 0


def test_an_expert_shared_by_two_tokens_is_copied_once():
    spec, p = placement(card_bytes=600)
    t = tiers(p)
    shared = next(k for k in tiers(p) if t[k] != "card")
    cache = ExpertCache(8, slot_bytes=100)
    plan = MoeTandem(p, cache).forward([(10, [shared]), (11, [shared])])

    assert plan.copy_keys.count(shared) == 1
    assert plan[10][0].slot is plan[11][0].slot
    assert plan.metrics()["coalesced"] == 1
    assert plan.metrics()["saved_copies"] == 1
    assert plan[11][0].lane == COPY


def test_the_cpu_lane_takes_host_experts_and_leaves_the_cache_empty():
    spec, p = placement(card_bytes=600)
    t = tiers(p)
    cache = ExpertCache(8, slot_bytes=100)
    keys = [(layer, e) for layer in LAYERS for e in range(N_EXP)]
    plan = MoeTandem(p, cache, lane=CPU).forward([(1, keys)])

    for pair in plan[1]:
        key = (pair.layer, pair.expert)
        if t[key] == "card":
            assert pair.lane == CARD
        else:
            assert pair.lane == CPU and pair.slot is None
    assert plan.copies == ()
    assert cache.metrics()["active_pins"] == 0
    assert plan.metrics()["cpu_pairs"] == sum(1 for k in keys if t[k] != "card")


def test_a_disk_expert_is_still_copied_on_the_cpu_lane():
    spec, p = placement(card_bytes=600, host_bytes=100)   # one host expert, one on disk
    t = tiers(p)
    disk = next(k for k in t if t[k] == "disk")
    cache = ExpertCache(8, slot_bytes=100)
    plan = MoeTandem(p, cache, lane=CPU).forward([(1, [disk])])
    assert plan[1][0].lane == COPY                  # its bytes must come through anyway
    assert plan.copy_keys == (disk,)


def test_cpu_share_thins_the_cpu_lane_deterministically():
    spec, p = placement(card_bytes=600)
    t = tiers(p)
    host = [k for k in tiers(p) if t[k] == "host"]
    assert len(host) >= 2

    def lanes(share):
        cache = ExpertCache(8, slot_bytes=100)
        plan = MoeTandem(p, cache, lane=CPU, cpu_share=share).forward([(1, host)])
        return [pair.lane for pair in plan[1]]

    assert lanes(1.0) == [CPU] * len(host)
    assert lanes(0.0) == [COPY] * len(host)
    half = lanes(0.5)
    assert half.count(CPU) == (len(host) + 1) // 2
    assert lanes(0.5) == half                       # deterministic, not sampled


def test_combine_order_is_separate_from_scheduling_order():
    spec, p = placement(card_bytes=600)
    cache = ExpertCache(8, slot_bytes=100)
    router_order = [(0, 3), (1, 2), (0, 0)]
    plan = MoeTandem(p, cache, combine_order="expert").forward([(1, router_order)])
    # ascending expert id (Gemma 4's loop); the layer term only disambiguates a mixed batch
    assert [(x.layer, x.expert) for x in plan[1]] == [(0, 0), (1, 2), (0, 3)]
    assert [x.expert for x in plan.scheduling(1)] == [3, 2, 0]   # router order kept
    assert [x.rank for x in plan[1]] == [2, 1, 0]     # router ranks travel with the pair

    plain = MoeTandem(p, ExpertCache(8, slot_bytes=100)).forward([(1, router_order)])
    assert [(x.layer, x.expert) for x in plain[1]] == router_order


def test_a_token_that_cannot_fit_rolls_the_forward_back():
    spec, p = placement(card_bytes=600)
    t = tiers(p)
    host = [k for k in tiers(p) if t[k] == "host"]
    cache = ExpertCache(1, slot_bytes=100)
    before = [(s["state"], s["key"], s["pins"]) for s in cache.snapshot()]

    with pytest.raises(ExpertCacheFull) as err:
        MoeTandem(p, cache).forward([(1, [host[0]]), (2, [host[1]])])

    assert err.value.token == 2
    after = [(s["state"], s["key"], s["pins"]) for s in cache.snapshot()]
    assert after == before
    assert cache.metrics()["active_pins"] == 0
    assert cache.keys() == ()


def test_the_plan_reports_a_json_safe_summary():
    spec, p = placement(card_bytes=600)
    plan = MoeTandem(p, ExpertCache(8, slot_bytes=100)).forward(
        [(1, [(0, 0), (0, 3)]), (2, [(1, 0)])])
    m = plan.metrics()
    assert json.loads(json.dumps(m)) == m
    assert m["tokens"] == 2 and m["pairs"] == 3
    assert m["card_pairs"] + m["copy_pairs"] + m["cpu_pairs"] == m["pairs"]
    assert m["lane"] == "copy" and m["combine_order"] == "router"


def test_the_scheduler_validates_its_inputs():
    spec, p = placement(card_bytes=600)
    cache = ExpertCache(8, slot_bytes=100)
    with pytest.raises(ValueError, match="lane must be"):
        MoeTandem(p, cache, lane="gpu")
    with pytest.raises(ValueError, match="combine_order"):
        MoeTandem(p, cache, combine_order="sorted")
    with pytest.raises(ValueError, match="cpu_share"):
        MoeTandem(p, cache, cpu_share=1.5)
    with pytest.raises(TypeError, match="Placement"):
        MoeTandem(object(), cache)
    with pytest.raises(TypeError, match="ExpertCache"):
        MoeTandem(p, object())

    tandem = MoeTandem(p, cache)
    with pytest.raises(ValueError, match="not a MoE layer"):
        tandem.forward([(1, [(9, 0)])])
    with pytest.raises(ValueError, match="outside 0"):
        tandem.forward([(1, [(0, N_EXP)])])
    with pytest.raises(ValueError, match="appears twice"):
        tandem.forward([(1, [(0, 0)]), (1, [(0, 1)])])
    with pytest.raises(TypeError, match="must be \\(layer, expert\\)"):
        tandem.forward([(1, ["oops"])])
    assert cache.metrics()["active_tokens"] == 0   # nothing was reserved by the failures
