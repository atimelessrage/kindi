# SPDX-License-Identifier: AGPL-3.0-only
"""Model-free tests for the copy-only expert cache state machine."""
import json

import pytest

from kindi.expert_cache import (
    ExpertCache,
    ExpertCacheFull,
    ExpertKey,
    FREE,
    IN_USE,
    LOADING,
    READY,
)


def load(cache, key, token):
    """Simulate one complete miss: plan, copy, mark ready, release."""
    plan = cache.plan([key], token)
    assert not plan[0].hit
    assert plan[0].slot.state == LOADING
    cache.mark_ready(key)
    assert plan[0].slot.state == READY
    assert cache.release(token)
    return plan[0].slot


def test_miss_hit_and_metrics():
    cache = ExpertCache(2, slot_bytes=100)
    key = ExpertKey(1, 2)

    first = cache.plan([key], token=1)
    entry = first[0]
    assert not entry.hit
    assert entry.slot.state == LOADING
    assert cache.slot_for(key) is entry.slot
    assert cache.metrics()["misses"] == 1
    assert cache.metrics()["copy_bytes"] == 100

    cache.mark_ready(key)
    assert entry.slot.state == READY
    cache.release(1)

    second = cache.plan([key], token=2)
    assert second[0].hit
    assert second[0].slot is entry.slot
    assert second[0].slot.state == READY
    cache.release(2)

    m = cache.metrics()
    assert m["hits"] == 1
    assert m["misses"] == 1
    assert m["hit_rate"] == 0.5
    assert m["copy_bytes"] == 100
    assert m["bytes_copied"] == 100
    assert m["cache_used_slots"] == 1
    assert m["used_bytes"] == 100
    assert m["active_pins"] == 0
    assert m["active_tokens"] == 0
    assert m["cache_states"][READY] == 1
    assert json.loads(json.dumps(m))


def test_lru_evicts_the_least_recently_used_ready_slot():
    cache = ExpertCache(2, slot_bytes=10)
    a, b, c = ExpertKey(0, 1), ExpertKey(0, 2), ExpertKey(0, 3)

    load(cache, a, 1)
    load(cache, b, 2)

    hit = cache.plan([a], token=3)
    assert hit[0].hit
    cache.release(3)

    miss = cache.plan([c], token=4)
    assert not miss[0].hit
    assert cache.slot_for(a) is not None
    assert cache.slot_for(b) is None
    assert cache.slot_for(c) is miss[0].slot

    m = cache.metrics()
    assert m["evictions"] == 1
    assert m["admissions"] == 3
    assert m["hits"] == 1
    assert m["misses"] == 3
    assert m["cache_used_slots"] == 2
    cache.release(4)


def test_pinned_ready_slot_is_not_evicted():
    cache = ExpertCache(1, slot_bytes=10)
    a, b = ExpertKey(0, 1), ExpertKey(0, 2)

    cache.plan([a], token=1)
    cache.mark_ready(a)
    with pytest.raises(ExpertCacheFull) as exc:
        cache.plan([b], token=2)
    assert exc.value.needed == 1
    assert exc.value.available == 0
    assert cache.slot_for(b) is None
    assert cache.metrics()["active_tokens"] == 1
    assert cache.metrics()["rejected"] == 1

    cache.release(1)
    miss = cache.plan([b], token=2)
    assert not miss[0].hit
    assert cache.slot_for(b) is miss[0].slot
    cache.release(2)


def test_transient_slot_is_used_only_after_the_cache_is_pinned():
    cache = ExpertCache(1, slot_bytes=10, transient_slots=1)
    a, b = ExpertKey(0, 1), ExpertKey(0, 2)

    cache.plan([a], token=1)
    cache.mark_ready(a)

    plan = cache.plan([b], token=2)
    assert not plan[0].hit
    assert plan[0].slot.transient
    assert plan[0].slot.state == LOADING
    assert cache.metrics()["evictions"] == 0
    assert cache.metrics()["transient_admissions"] == 1

    cache.mark_ready(b)
    cache.release(2)
    assert cache.slot_for(b) is None
    assert cache.metrics()["transient_drops"] == 1
    cache.release(1)


def test_evictable_cache_slot_is_preferred_over_a_transient_slot():
    cache = ExpertCache(1, slot_bytes=10, transient_slots=1)
    a, b = ExpertKey(0, 1), ExpertKey(0, 2)

    load(cache, a, 1)
    plan = cache.plan([b], token=2)
    assert not plan[0].slot.transient
    assert cache.metrics()["evictions"] == 1
    assert cache.metrics()["transient_admissions"] == 0
    cache.release(2)


def test_release_does_not_reuse_a_loading_transient_slot():
    cache = ExpertCache(1, slot_bytes=10, transient_slots=1)
    a, b = ExpertKey(0, 1), ExpertKey(0, 2)

    cache.plan([a], token=1)
    cache.mark_ready(a)
    cache.plan([b], token=2)
    assert cache.release(2)
    assert cache.slot_for(b).state == LOADING

    cache.mark_ready(b)
    assert cache.slot_for(b) is None
    assert cache.metrics()["transient_drops"] == 1
    cache.release(1)


def test_cache_full_plan_is_atomic():
    cache = ExpertCache(1, slot_bytes=10)
    a, b, c = ExpertKey(0, 1), ExpertKey(0, 2), ExpertKey(0, 3)

    cache.plan([a], token=1)
    with pytest.raises(ExpertCacheFull) as exc:
        cache.plan([b, c], token=2)
    assert exc.value.needed == 2
    assert exc.value.available == 0
    assert cache.slot_for(b) is None
    assert cache.slot_for(c) is None
    assert cache.metrics()["active_tokens"] == 1
    assert cache.metrics()["rejected_keys"] == 2
    assert cache.slot_for(a).pins == {1}
    cache.release(1)


def test_duplicate_keys_share_one_slot_and_one_pin():
    cache = ExpertCache(2, slot_bytes=10)
    key = ExpertKey(0, 1)

    plan = cache.plan([key, key], token=1)
    assert len(plan) == 2
    assert plan[0].slot is plan[1].slot
    assert plan[0].slot.pins == {1}
    assert cache.metrics()["selected"] == 2
    assert cache.metrics()["misses"] == 1
    assert cache.metrics()["duplicate_keys"] == 1
    cache.release(1)
    assert cache.metrics()["active_pins"] == 0


def test_acquire_is_the_single_key_form():
    cache = ExpertCache(2, slot_bytes=10)
    key = ExpertKey(0, 1)

    slot = cache.acquire(key, token=1)
    assert slot.state == LOADING
    cache.mark_ready(key)
    again = cache.acquire(key, token=1)
    assert again is slot
    assert cache.metrics()["hits"] == 1
    cache.begin_use(key)
    assert slot.state == IN_USE
    cache.release(1)
    assert slot.state == READY
    assert cache.metrics()["uses"] == 1


def test_failed_copy_requires_release_then_discards_the_slot():
    cache = ExpertCache(1, slot_bytes=10)
    key = ExpertKey(0, 1)

    cache.plan([key], token=1)
    cache.release(1)
    assert cache.slot_for(key).state == LOADING
    cache.fail(key)
    assert cache.slot_for(key) is None
    assert cache.metrics()["copy_failures"] == 1
    assert cache.metrics()["aborted_loads"] == 1
    assert cache.slot_for(key) is None
    assert cache.plan([key], token=2)[0].slot.state == LOADING
    cache.release(2)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        ExpertKey(-1, 0)
    with pytest.raises(TypeError):
        ExpertKey(True, 0)
    with pytest.raises(ValueError):
        ExpertCache(1, slot_bytes=0)
    with pytest.raises(TypeError):
        ExpertCache(True)
    with pytest.raises(TypeError):
        ExpertCache(1).plan([(0, 1)], token=[])
