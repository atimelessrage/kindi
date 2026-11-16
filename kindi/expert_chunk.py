# SPDX-License-Identifier: AGPL-3.0-only
"""Multi-token (prefill chunk) planning on top of the expert cache.

`ExpertCache.plan()` covers one router token. A prompt chunk routes many
tokens at once, so the same `(layer, expert)` is usually selected by several
tokens of the chunk: copying it once per token would multiply PCIe traffic by
the chunk size for no gain. This module is the chunk-level planner described in
docs/MOE-TANDEM.md: it drives one `ExpertCache.plan()` per token (so each
token keeps the cache's all-or-nothing reservation), then reports the chunk as
one copy list.

Rules, all model-free:

* **Dedupe.** A key selected by several tokens is copied once. The second and
  later selections point at the same slot and are counted as `coalesced`, not
  as hits: the slot is LOADING, not ready, which is the cache's own meaning.
* **Deterministic order.** The copy list is first-need order: tokens in chunk
  order, and inside a token the router's own order. No slot index, dict
  iteration or set order leaks into it, so two runs over the same picks
  produce the same list.
* **Per-token atomicity.** Each token is reserved by `ExpertCache.plan()`,
  which changes nothing when it cannot fit the whole token.
* **Chunk all-or-nothing.** If a later token does not fit, the tokens already
  reserved by this call are released and the copies this call started are
  failed, so a refused chunk leaves no pins, no LOADING slot and no resident
  key behind. Keys that were already resident (or already LOADING for another
  token outside the chunk - prompts and a concurrent decode step) are never
  undone.

The caller copies `chunk.copies` in that order, calls `cache.mark_ready(key)`
for each, runs the expert kernels, then `cache.release(plan.token)` for every
plan in `chunk.plans`. This module does no I/O and owns no stream; it is not
thread-safe, like the cache it plans against.
"""
import json

__all__ = ["ChunkPlan", "plan_chunk"]

from .expert_cache import ExpertCache, ExpertCacheFull, LOADING, _key


class ChunkPlan(object):
    """The result of plan_chunk(): one ExpertPlan per token, one copy list.

    plans: the per-token ExpertPlans, in chunk order.
    copies: distinct (key, slot) pairs to copy, in first-need order.
    hits / coalesced / misses: token-expert selections served by a READY slot,
    selections that found a LOADING slot (copied once for the chunk), and the
    number of distinct keys the chunk must copy.
    """

    __slots__ = ("plans", "copies", "hits", "coalesced", "selections",
                 "slot_bytes")

    def __init__(self, plans, copies, hits, coalesced, selections, slot_bytes):
        self.plans = tuple(plans)
        self.copies = tuple(copies)
        self.hits = int(hits)
        self.coalesced = int(coalesced)
        self.selections = int(selections)
        self.slot_bytes = int(slot_bytes)

    @property
    def misses(self):
        """Distinct experts the chunk must copy (not tokens that missed)."""
        return len(self.copies)

    @property
    def tokens(self):
        return tuple(plan.token for plan in self.plans)

    @property
    def copy_keys(self):
        return tuple(key for key, _slot in self.copies)

    @property
    def copy_bytes(self):
        return self.misses * self.slot_bytes

    @property
    def saved_copies(self):
        """Selections that shared another selection's copy in this chunk."""
        return self.selections - self.misses

    def __len__(self):
        return len(self.plans)

    def __iter__(self):
        return iter(self.plans)

    def __getitem__(self, i):
        return self.plans[i]

    def metrics(self):
        """A JSON-safe summary for logs, /metrics or a bench record."""
        return {
            "tokens": len(self.plans),
            "selections": self.selections,
            "hits": self.hits,
            "coalesced": self.coalesced,
            "misses": self.misses,
            "saved_copies": self.saved_copies,
            "copy_bytes": self.copy_bytes,
            "slot_bytes": self.slot_bytes,
        }

    def __repr__(self):
        return ("ChunkPlan(tokens=%d, selections=%d, hits=%d, coalesced=%d, copies=%d)"
                % (len(self.plans), self.selections, self.hits, self.coalesced, self.misses))


def _normalise(picks):
    """[(token, [ExpertKey, ...]), ...] from [(token, keys), ...], validating tokens."""
    items = []
    seen = set()
    for item in picks:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("each chunk item must be (token, keys)")
        token, keys = item
        try:
            hash(token)
        except TypeError:
            raise TypeError("token must be hashable")
        if token in seen:
            raise ValueError("token %r appears twice in one chunk" % (token,))
        seen.add(token)
        items.append((token, [_key(k) for k in keys]))
    return items


def _rollback(cache, plans, admitted):
    """Undo this call: release its tokens, then fail the copies it started.

    A released LOADING slot stays LOADING (the cache cannot know whether the
    copy is still in flight), so the copies this call admitted are failed
    explicitly. Keys that were resident or loading before the call are kept.
    """
    for plan in reversed(plans):
        cache.release(plan.token)
    for key in reversed(admitted):
        slot = cache.slot_for(key)
        if slot is not None and slot.state == LOADING and not slot.pins:
            cache.fail(key)


def plan_chunk(cache, picks):
    """Reserve every token of one prefill chunk; return its ChunkPlan.

    picks: [(token, keys), ...] - one entry per token in the chunk, keys in
    router order. Tokens must be unique and hashable (the cache keys pins by
    token, so two entries with one token would alias).

    Raises ExpertCacheFull for the first token that does not fit. The chunk is
    then rolled back entirely (see the module docstring); the exception still
    names that token, so a caller falling back to per-token copies can say
    which token refused.
    """
    if not isinstance(cache, ExpertCache):
        raise TypeError("cache must be an ExpertCache")
    items = _normalise(picks)
    plans = []
    admitted = []
    try:
        for token, keys in items:
            before = set(cache.keys())
            plan = cache.plan(keys, token)
            for entry in plan.entries:
                if not entry.hit and entry.key not in before and entry.key not in admitted:
                    admitted.append(entry.key)
            plans.append(plan)
    except BaseException:
        _rollback(cache, plans, admitted)
        raise

    copies = []
    seen = set()
    admitted_set = set(admitted)
    hits = coalesced = selections = 0
    for plan in plans:
        for entry in plan.entries:
            selections += 1
            if entry.hit:
                hits += 1
            elif entry.key in seen or entry.key not in admitted_set:
                # another token of this chunk (or a copy already in flight outside it)
                # started this expert's copy: the selection shares it
                coalesced += 1
            if entry.key in seen:
                continue
            seen.add(entry.key)
            if not entry.hit:
                copies.append((entry.key, entry.slot))
    return ChunkPlan(plans, copies, hits, coalesced, selections, cache.slot_bytes)
