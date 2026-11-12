# SPDX-License-Identifier: AGPL-3.0-only
"""Randomized property traces for the model-free expert cache.

The tests use a fixed seed per trace, so failures are reproducible and no new
test dependency is required. Every operation is followed by structural,
capacity, pin-ownership and metric checks. The traces mix multi-key route
plans, single-key acquires, copy completion, use/release and failed copies.
"""
import random

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

KEYS = tuple(ExpertKey(layer, expert) for layer in range(3) for expert in range(12))


def _state_counts(slots):
    counts = {FREE: 0, LOADING: 0, READY: 0, IN_USE: 0}
    for slot in slots:
        counts[slot.state] += 1
    return counts


def _assert_invariants(cache, active, context):
    """Assert every slot, pin and capacity invariant at a quiescent point."""
    slots = cache.all_slots
    cache_slots = cache._cache
    transient_slots = cache._transient

    assert len(cache_slots) == cache.slot_count, context
    assert len(transient_slots) == cache.transient_count, context
    assert [slot.index for slot in cache_slots] == list(range(cache.slot_count)), context
    assert [slot.index for slot in transient_slots] == [
        -i for i in range(1, cache.transient_count + 1)
    ], context

    used_cache = 0
    used_transient = 0
    resident_keys = []
    for slot in slots:
        if slot.transient:
            assert slot.index < 0, context
            assert slot not in cache_slots, context
        else:
            assert 0 <= slot.index < cache.slot_count, context
            assert slot not in transient_slots, context

        assert slot.loads >= 0, context
        if slot.state == FREE:
            assert slot.key is None, context
            assert not slot.pins, context
            assert slot.last_used == 0, context
            continue

        assert slot.key is not None, context
        assert slot.last_used > 0, context
        assert slot.state in (LOADING, READY, IN_USE), context
        if slot.transient:
            used_transient += 1
        else:
            used_cache += 1
        resident_keys.append(slot.key)
        assert cache._by_key.get(slot.key) is slot, context
        if slot.state == IN_USE:
            assert slot.pins, context

    assert len(resident_keys) == len(set(resident_keys)), context
    assert set(cache._by_key) == set(resident_keys), context
    assert cache._pin_count == sum(len(slot.pins) for slot in slots), context

    assert set(cache._tokens) == set(active), context
    for token, token_slots in cache._tokens.items():
        assert token_slots, context
        assert token_slots <= set(slots), context
        for slot in token_slots:
            assert token in slot.pins, context
    for slot in slots:
        for token in slot.pins:
            assert token in cache._tokens, context
            assert slot in cache._tokens[token], context

    metrics = cache.metrics()
    used = used_cache + used_transient
    assert used_cache <= cache.slot_count, context
    assert used_transient <= cache.transient_count, context
    assert used <= cache.slot_count + cache.transient_count, context
    assert metrics["slot_count"] == cache.slot_count, context
    assert metrics["transient_slots"] == cache.transient_count, context
    assert metrics["slot_bytes"] == cache.slot_bytes, context
    assert metrics["capacity_bytes"] == cache.slot_count * cache.slot_bytes, context
    assert metrics["used_slots"] == used, context
    assert metrics["cache_used_slots"] == used_cache, context
    assert metrics["transient_used_slots"] == used_transient, context
    assert metrics["used_bytes"] == used * cache.slot_bytes, context
    assert metrics["resident_keys"] == len(resident_keys), context
    assert metrics["active_pins"] == cache._pin_count, context
    assert metrics["active_tokens"] == len(cache._tokens), context
    assert metrics["cache_states"] == _state_counts(cache_slots), context
    assert metrics["transient_states"] == _state_counts(transient_slots), context
    assert metrics["copy_bytes"] == metrics["admissions"] * cache.slot_bytes, context
    assert metrics["bytes_copied"] == metrics["copy_bytes"], context
    assert metrics["transient_copy_bytes"] == (
        metrics["transient_admissions"] * cache.slot_bytes
    ), context
    assert metrics["peak_pinned"] >= metrics["active_pins"], context
    assert metrics["peak_used_slots"] >= used, context
    assert cache.slot_states()["cache"] == metrics["cache_states"], context
    assert cache.slot_states()["transient"] == metrics["transient_states"], context

    snapshot = cache.snapshot()
    assert len(snapshot) == len(slots), context
    for record, slot in zip(snapshot, slots):
        assert record["index"] == slot.index, context
        assert record["transient"] is slot.transient, context
        assert record["state"] == slot.state, context
        assert record["key"] == (tuple(slot.key) if slot.key is not None else None), context
        assert record["pins"] == len(slot.pins), context
        assert record["loads"] == slot.loads, context


def _tokens_copy(cache):
    return {token: set(slots) for token, slots in cache._tokens.items()}


def _run_trace(seed, steps=90):
    rng = random.Random(seed)
    cache = ExpertCache(
        rng.randint(1, 8),
        slot_bytes=rng.randint(1, 4096),
        transient_slots=rng.randint(0, 2),
    )
    active = {}
    next_token = 1

    for step in range(steps):
        if len(active) >= 6:
            operation = rng.choice(
                ("plan", "acquire", "advance", "use", "release", "release", "fail")
            )
        else:
            operation = rng.choice(
                ("plan", "plan", "acquire", "advance", "use", "release", "fail")
            )
        context = "seed=%d step=%d operation=%s" % (seed, step, operation)
        _assert_invariants(cache, active, context)

        if operation == "plan":
            token = next_token
            next_token += 1
            keys = [rng.choice(KEYS) for _ in range(rng.randint(1, 8))]
            if len(keys) > 1 and rng.random() < 0.25:
                keys[-1] = keys[0]

            before_snapshot = cache.snapshot()
            before_tokens = _tokens_copy(cache)
            before_metrics = cache.metrics()
            try:
                plan = cache.plan(keys, token)
            except ExpertCacheFull as exc:
                assert cache.snapshot() == before_snapshot, context
                assert _tokens_copy(cache) == before_tokens, context
                assert token not in cache._tokens, context
                assert token not in active, context
                after_metrics = cache.metrics()
                assert after_metrics["rejected"] == before_metrics["rejected"] + 1, context
                assert after_metrics["rejected_keys"] == (
                    before_metrics["rejected_keys"] + exc.needed
                ), context
                assert exc.available >= 0, context
            else:
                active[token] = tuple(plan.keys)
                assert plan.keys == tuple(keys), context
                assert token in cache._tokens, context
                assert len(cache._tokens[token]) == len(set(keys)), context
                for entry in plan:
                    assert entry.slot is cache.slot_for(entry.key), context
                    assert token in entry.slot.pins, context
                    if entry.hit:
                        assert entry.slot.state in (READY, IN_USE), context
                    else:
                        assert entry.slot.state == LOADING, context

        elif operation == "acquire":
            use_active = bool(active) and rng.random() < 0.6
            if use_active:
                token = rng.choice(tuple(active))
                is_new = False
            else:
                token = next_token
                next_token += 1
                is_new = True
            key = rng.choice(KEYS)

            before_snapshot = cache.snapshot()
            before_tokens = _tokens_copy(cache)
            before_metrics = cache.metrics()
            try:
                slot = cache.acquire(key, token)
            except ExpertCacheFull as exc:
                assert cache.snapshot() == before_snapshot, context
                assert _tokens_copy(cache) == before_tokens, context
                if is_new:
                    assert token not in cache._tokens, context
                assert token in active or is_new, context
                after_metrics = cache.metrics()
                assert after_metrics["rejected"] == before_metrics["rejected"] + 1, context
                assert after_metrics["rejected_keys"] == (
                    before_metrics["rejected_keys"] + 1
                ), context
                assert exc.needed == 1, context
            else:
                if is_new:
                    active[token] = ()
                active[token] += (key,)
                assert cache.slot_for(key) is slot, context
                assert token in slot.pins, context
                assert slot in cache._tokens[token], context

        elif operation == "advance":
            candidates = [slot for slot in cache.all_slots if slot.state == LOADING]
            if candidates:
                slot = rng.choice(candidates)
                key = slot.key
                was_transient = slot.transient
                previously_pinned = bool(slot.pins)
                cache.mark_ready(key)
                if was_transient and not previously_pinned:
                    assert cache.slot_for(key) is None, context
                else:
                    assert cache.slot_for(key) is slot, context
                    assert slot.state == READY, context

        elif operation == "use":
            choices = []
            for token, token_slots in cache._tokens.items():
                for slot in token_slots:
                    if slot.pins == {token} and slot.state in (READY, IN_USE):
                        choices.append((token, slot))
            if choices:
                token, slot = rng.choice(choices)
                if slot.state == READY:
                    cache.begin_use(slot.key)
                    assert slot.state == IN_USE, context
                else:
                    cache.end_use(slot.key)
                    assert slot.state == READY, context

        elif operation == "release":
            if active:
                token = rng.choice(tuple(active))
                assert cache.release(token), context
                del active[token]
                assert token not in cache._tokens, context
                for slot in cache.all_slots:
                    if token in slot.pins:
                        assert False, context

        elif operation == "fail":
            candidates = [
                slot for slot in cache.all_slots
                if slot.state == LOADING and not slot.pins
            ]
            if candidates:
                slot = rng.choice(candidates)
                key = slot.key
                cache.fail(key)
                assert cache.slot_for(key) is None, context
                assert slot.state == FREE, context

        _assert_invariants(cache, active, context)

    # Drain live plans and copy-in-flight slots before the final invariant
    # pass. A released LOADING slot must be failed before it is reused.
    for token in tuple(active):
        assert cache.release(token)
        del active[token]
    for slot in tuple(cache.all_slots):
        if slot.state == LOADING:
            assert not slot.pins
            cache.fail(slot.key)
    assert not cache._tokens
    assert cache._pin_count == 0
    assert cache.slot_states()["transient"][FREE] == cache.transient_count
    _assert_invariants(cache, active, "seed=%d final" % seed)


@pytest.mark.parametrize("seed", range(32))
def test_randomized_route_traces_preserve_cache_invariants(seed):
    _run_trace(seed)
