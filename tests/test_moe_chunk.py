# SPDX-License-Identifier: AGPL-3.0-only
"""Model-free tests for the multi-token (prefill chunk) expert planner.

The planner only arranges ExpertCache.plan() calls, so every test drives the
real cache and checks slots, pins and counters - no model, no tensor, no GPU.
"""
import json

import pytest

from kindi.expert_cache import ExpertCache, ExpertCacheFull, ExpertKey, LOADING, READY
from kindi.expert_chunk import ChunkPlan, plan_chunk

A, B, C, D, E = [ExpertKey(0, i) for i in range(5)]


def load(cache, key, token):
    """Make key resident and ready, as a previous decode step would have."""
    plan = cache.plan([key], token)
    assert not plan[0].hit
    cache.mark_ready(key)
    cache.release(token)


def test_shared_experts_are_copied_once_for_the_whole_chunk():
    cache = ExpertCache(8, slot_bytes=100)
    chunk = plan_chunk(cache, [(10, [A, B]), (11, [B, C]), (12, [A, C])])

    assert chunk.tokens == (10, 11, 12)
    assert chunk.copy_keys == (A, B, C)            # first-need order, each once
    assert chunk.misses == 3
    assert chunk.selections == 6
    assert chunk.saved_copies == 3                 # three selections shared a copy
    assert chunk.coalesced == 3                    # they found a LOADING slot
    assert chunk.hits == 0
    # one copy per distinct expert, not one per selection
    assert cache.metrics()["copy_bytes"] == 3 * 100
    assert cache.metrics()["admissions"] == 3
    # every token still holds its own pins, in router order
    assert [p.keys for p in chunk] == [(A, B), (B, C), (A, C)]
    for plan in chunk:
        cache.release(plan.token)
    assert cache.metrics()["active_pins"] == 0


def test_the_copy_order_is_first_need_and_repeats_across_runs():
    def run():
        cache = ExpertCache(8, slot_bytes=100)
        chunk = plan_chunk(cache, [(1, [C, A]), (2, [D, A]), (3, [B, C])])
        return chunk.copy_keys

    assert run() == (C, A, D, B)                   # token order, then router order
    assert run() == run()                          # no slot index or set order leaks in


def test_a_resident_expert_is_a_hit_and_is_not_in_the_copy_list():
    cache = ExpertCache(8, slot_bytes=100)
    load(cache, A, token=1)
    chunk = plan_chunk(cache, [(2, [A, B]), (3, [A])])

    assert chunk.hits == 2                         # both selections of A are READY
    assert chunk.coalesced == 0
    assert chunk.copy_keys == (B,)
    assert chunk.misses == 1
    # saved_copies counts selections that shared a copy: two of the three here
    assert chunk.selections == 3 and chunk.saved_copies == 2

    # the resident slot keeps its identity across every plan of the chunk
    assert chunk[0][0].slot is cache.slot_for(A) and chunk[0][0].hit
    assert chunk[1][0].slot is cache.slot_for(A) and chunk[1][0].hit


def test_one_tokens_duplicate_picks_stay_in_router_order():
    cache = ExpertCache(4)
    chunk = plan_chunk(cache, [(5, [A, B, A])])
    assert chunk[0].keys == (A, B, A)              # the plan is not deduped
    assert chunk.copy_keys == (A, B)               # the copy list is
    assert chunk.selections == 3 and chunk.misses == 2 and chunk.coalesced == 1


def test_a_token_that_does_not_fit_rolls_the_whole_chunk_back():
    cache = ExpertCache(2, slot_bytes=100)
    before = cache.snapshot()

    with pytest.raises(ExpertCacheFull) as err:
        plan_chunk(cache, [(1, [A, B]), (2, [C, D])])

    assert err.value.token == 2                    # the token that could not fit
    # (loads is a cumulative counter, so compare the state the caller can observe)
    view = lambda snap: [(s["state"], s["key"], s["pins"]) for s in snap]
    assert view(cache.snapshot()) == view(before)  # no slot changed state or key
    m = cache.metrics()
    assert m["active_pins"] == 0 and m["resident_keys"] == 0
    assert m["cache_states"]["loading"] == 0
    assert m["used_slots"] == 0
    assert cache.keys() == ()


def test_rollback_leaves_an_experts_already_loading_for_another_token_alone():
    cache = ExpertCache(2, slot_bytes=100)
    other = cache.acquire(E, "decode-step")        # outside this chunk
    assert cache.slot_for(E).state == LOADING

    with pytest.raises(ExpertCacheFull):
        plan_chunk(cache, [(1, [A]), (2, [B])])

    # the chunk's own copy was failed, the other token's copy was not touched
    assert cache.slot_for(E) is not None
    assert cache.slot_for(E).state == LOADING
    assert cache.slot_for(E).pins == {"decode-step"}
    assert cache.keys() == (E,)
    assert cache.release("decode-step") is True


def test_the_chunk_reports_a_json_safe_summary():
    cache = ExpertCache(8, slot_bytes=64)
    chunk = plan_chunk(cache, [(1, [A, B]), (2, [A])])
    m = chunk.metrics()
    assert json.loads(json.dumps(m)) == m
    assert m == {"tokens": 2, "selections": 3, "hits": 0, "coalesced": 1,
                 "misses": 2, "saved_copies": 1, "copy_bytes": 128, "slot_bytes": 64}
    for key in chunk.copy_keys:                   # one copy per distinct expert
        cache.mark_ready(key)
    assert all(cache.slot_for(k).state == READY for k in (A, B))


def test_an_empty_chunk_is_a_plan_with_nothing_to_copy():
    cache = ExpertCache(4)
    chunk = plan_chunk(cache, [])
    assert isinstance(chunk, ChunkPlan) and len(chunk) == 0
    assert chunk.copies == () and chunk.metrics()["tokens"] == 0
    empty = plan_chunk(cache, [(1, [])])
    assert len(empty) == 1 and empty[0].keys == () and empty.copies == ()


def test_a_plan_is_per_token_so_one_token_cannot_appear_twice():
    cache = ExpertCache(4)
    with pytest.raises(ValueError, match="appears twice"):
        plan_chunk(cache, [(1, [A]), (1, [B])])
    assert cache.metrics()["active_tokens"] == 0   # nothing was reserved


def test_the_planner_validates_its_input():
    cache = ExpertCache(4)
    with pytest.raises(TypeError, match="must be \\(token, keys\\)"):
        plan_chunk(cache, [(1, [A]), "bare"])
    with pytest.raises(TypeError, match="hashable"):
        plan_chunk(cache, [([A], [A])])            # a list is not a usable token
    with pytest.raises(TypeError, match="ExpertKey"):
        plan_chunk(cache, [(1, ["not-a-key"])])
    with pytest.raises(TypeError, match="ExpertCache"):
        plan_chunk(object(), [(1, [A])])
