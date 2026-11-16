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


class _FakeExecutor(object):
    """Stands in for the GPU expert kernel, the CPU worker and the copy stream.

    A copied expert and a resident one run the same kernel, so both return the same value for a
    key; the CPU worker returns the same value too, which is exactly what the CPU lane's parity
    gate would have to prove. The combine is a non-commutative fold, so a wrong order changes it.
    """

    def __init__(self):
        self.copies = []
        self.ran = []

    def copy(self, key):
        self.copies.append(key)

    def gpu(self, pair):
        self.ran.append((pair.layer, pair.expert, "gpu"))
        return pair.layer * 100 + pair.expert

    def cpu(self, pair):
        self.ran.append((pair.layer, pair.expert, "cpu"))
        return pair.layer * 100 + pair.expert

    def combine(self, rows):
        acc = 0
        for expert, value in rows:
            acc = acc * 31 + value + expert
        return acc


def _execute(cache, plan, executor):
    """The backend loop the scheduler documents: copy, mark ready, compute, combine, release."""
    for key in plan.copy_keys:
        executor.copy(key)
        cache.mark_ready(key)
    out = {}
    for token in plan:
        rows = []
        for pair in plan.pairs(token):                 # combine (reference) order
            value = executor.cpu(pair) if pair.lane == CPU else executor.gpu(pair)
            rows.append((pair.expert, value))
        out[token] = executor.combine(rows)
    assert plan.release() == len(plan.tokens)
    return out


def test_an_executed_forward_repeats_exactly_from_cache_hits():
    spec, p = placement(card_bytes=600)
    t = tiers(p)
    host = [k for k in tiers(p) if t[k] == "host"]
    card = [k for k in tiers(p) if t[k] == "card"]
    assert len(host) >= 2 and card
    # one token whose pairs cover all three lanes: card experts, one copied, one CPU-computed
    keys = [card[0], host[0], card[-1], host[1]]
    cache = ExpertCache(len(host) + 2, slot_bytes=100)
    tandem = MoeTandem(p, cache, lane=CPU, cpu_share=0.5)

    first_exec = _FakeExecutor()
    first = tandem.forward([(1, keys), (2, keys[:2])])
    lanes = {pair.lane for pair in first[1]}
    assert lanes == {CARD, COPY, CPU}                  # all three lanes in one forward
    first_out = _execute(cache, first, first_exec)

    assert set(first_exec.copies) == set(first.copy_keys)
    assert first_exec.copies == list(first.copy_keys)  # copies in the plan's order
    # the copy list is exactly the COPY-lane pairs, deduped, in scheduling order
    expected_copies = []
    for token in first.tokens:
        for pair in first.scheduling(token):
            if pair.lane == COPY and (pair.layer, pair.expert) not in expected_copies:
                expected_copies.append((pair.layer, pair.expert))
    assert list(first.copy_keys) == expected_copies
    # every pair ran once, by the lane the plan assigned, once per token
    assert first_exec.ran == [(pair.layer, pair.expert, "cpu" if pair.lane == CPU else "gpu")
                              for token in first.tokens for pair in first.pairs(token)]
    # nothing leaks after the combine
    m = cache.metrics()
    assert m["active_pins"] == 0 and m["active_tokens"] == 0
    assert cache.slot_states()["cache"]["loading"] == 0
    assert cache.slot_states()["transient"]["loading"] == 0

    second_exec = _FakeExecutor()
    second = tandem.forward([(1, keys), (2, keys[:2])])
    second_out = _execute(cache, second, second_exec)

    # the same lanes, every copied expert now a hit: no copy, same arithmetic
    assert {pair.lane for pair in second[1]} == lanes
    assert second.copies == () and second_exec.copies == []
    # every COPY-lane selection (duplicates across tokens included) is now a hit
    copy_selections = sum(1 for token in second.tokens
                          for pair in second.pairs(token) if pair.lane == COPY)
    assert copy_selections > 0
    assert second.metrics()["hits"] == copy_selections
    assert second_out == first_out                       # cold vs warm expert: identical output
    assert len(second_exec.ran) == len(first_exec.ran)
    m = cache.metrics()
    assert m["active_pins"] == 0 and m["cache_states"]["loading"] == 0
    assert set(expected_copies) <= set(cache.keys())     # slots stayed resident for the next token
