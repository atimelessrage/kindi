# SPDX-License-Identifier: AGPL-3.0-only
"""tools/grammar_cost.py's walk() against the real cache type.

`walk()` used `state in cache`, which silently became a TypeError when the
grammar cache moved from a dict to the bounded `StateMaskCache` LRU: the
harness in `tools/` and the REQ-0030 profile both stopped running while the
server path kept working, so no test caught it. These pin the hit detection
and the returned shape on the real cache class, with a tiny vocabulary.
"""
import os
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import grammar_cost  # noqa: E402

from kindi.token_classes import StateMaskCache  # noqa: E402


class _Tok(object):
    """One id per character: 'abc' -> [1, 2, 3] on the fake vocabulary below."""

    def encode(self, text):
        return [ord(ch) - ord("a") + 1 for ch in text]


class _Grammar(object):
    """A three-state automaton over texts ["a", "b", "c"]: 0 ->1 ->2 ->2."""

    def __init__(self, budget=None):
        self.states = 3
        self._vocab_cache = StateMaskCache(budget if budget is not None else 4096)
        self.calls = 0

    def start(self):
        return 0

    def advance(self, state, text):
        if text not in ("a", "b", "c"):
            return None
        return min(state + 1, 2)

    def vocab_allowed(self, state, texts):
        """The real grammars' shape: consult the cache, compute and store on a miss."""
        import numpy as np

        cache = self._vocab_cache
        if cache is not None:
            cached = cache.get(state)
            if cached is not None:
                return cached
        self.calls += 1
        ids = np.array([i for i, t in enumerate(texts) if t in ("a", "b", "c")], dtype=np.int64)
        if cache is not None:
            cache.put(state, ids)
        return ids


TEXTS = ["<pad>", "a", "b", "c"]


def test_walk_runs_against_the_bounded_mask_cache():
    """The regression: a StateMaskCache is not iterable, so `state in cache` raised."""
    grammar = _Grammar()
    mean, ok, costs, misses = grammar_cost.walk(grammar, _Tok(), TEXTS, "abcab", "tiny")
    assert ok is True
    assert len(costs) == 5
    assert mean >= 0
    # states 1 and 2 are each computed once and then served from the cache
    assert grammar.calls == 3
    assert misses >= 1


def test_walk_marks_cached_steps_as_free_and_counts_misses():
    grammar = _Grammar()
    _, ok, costs, misses = grammar_cost.walk(grammar, _Tok(), TEXTS, "aaaa", "tiny")
    assert ok is True
    # 0 ->1 ->2 ->2 ->2: three real masks (start, state 1, state 2) and two hits, so the
    # last two steps cost nothing and the miss count is 2
    assert grammar.calls == 3
    assert sum(1 for cost in costs if cost == 0.0) == 2
    assert misses == 2


def test_walk_stops_at_a_dead_state_without_raising():
    grammar = _Grammar()

    class _Rejecting(_Tok):
        def encode(self, text):
            return [0]  # the "<pad>" id: in range, and advance() rejects it

    _, ok, costs, misses = grammar_cost.walk(grammar, _Rejecting(), TEXTS, "a", "tiny")
    assert ok is False
    assert costs == []
    assert misses >= 0  # the dead step has no mask of its own; walk() stops there


def test_walk_works_without_any_cache():
    grammar = _Grammar()
    grammar._vocab_cache = None
    _, ok, costs, misses = grammar_cost.walk(grammar, _Tok(), TEXTS, "abc", "tiny")
    assert ok is True
    # without a cache every step recomputes its mask (the start mask plus one per
    # token); walk() reports no misses because hit detection needs the cache's counter
    assert len(costs) == 3
    assert grammar.calls == 4
    assert misses == 0


@pytest.mark.parametrize("budget", (0, 4096))
def test_a_disabled_or_tiny_cache_still_reports_the_walk(budget):
    grammar = _Grammar(budget=budget)
    _, ok, costs, miss_count = grammar_cost.walk(grammar, _Tok(), TEXTS, "abc", "tiny")
    assert ok is True and len(costs) == 3
