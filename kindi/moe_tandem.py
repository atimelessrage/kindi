# SPDX-License-Identifier: AGPL-3.0-only
"""MoE tandem: one execution unit's expert work, split across card / copy / CPU.

The dense ``TandemEngine`` splits complete layers and cannot run a routed layer.
This is the split a MoE actually has (docs/MOE-TANDEM.md §5-§6): attention, the
router, the shared branch and the combine stay on the GPU, and each selected
``(layer, expert)`` pair is served by one of three lanes

* ``card``  - the expert is resident on the card (``moe.Placement`` tier
  "card"), so the ordinary GPU expert kernel reads it in place;
* ``copy``  - the expert lives in host memory (or on disk): a complete expert is
  copied into an ``ExpertCache`` slot, then the same GPU kernel runs on it;
* ``cpu``   - a CPU worker computes the expert from host-resident weights and
  its output vector comes back to the card. Optional and gated: only a bitwise
  match with the GPU kernel makes it exact, so it is opt-in (``lane="cpu"``),
  never the default.

Scheduling and arithmetic are separate. ``pairs_by_token`` is router order, that
is also the scheduling order; the combine order is the backend's reference order
(``combine_order="router"`` for Raon, ``"expert"`` for Gemma 4). The cache only
ever decides slots, never the order a combination reads.

This module is model-free: it takes ints, tuples, an ``ExpertCache`` and a
``moe.Placement``, and returns a plan. It does not move bytes or run a kernel,
and no backend uses it yet.
"""
from collections import namedtuple

from .expert_cache import ExpertCache
from .expert_chunk import plan_chunk
from .moe import Placement

__all__ = ["MoeTandem", "TandemPlan", "TandemPair", "LANES", "CARD", "COPY", "CPU"]

CARD = "card"
COPY = "copy"
CPU = "cpu"
LANES = (COPY, CPU)
ORDERS = ("router", "expert")

TandemPair = namedtuple("TandemPair", "layer expert lane slot rank")
"""One selected expert of one token.

lane: "card" | "copy" | "cpu"; slot: the ExpertCache slot for a copy (None for
card and cpu); rank: the position in the router's own order (scheduling order).
"""


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer, not %r" % (name, type(value).__name__))
    return value


class TandemPlan(object):
    """One forward's assignments: per-token pairs, the copy list, the counters."""

    __slots__ = ("tokens", "copies", "chunk", "lane", "combine_order",
                 "_pairs", "_cache", "slot_bytes")

    def __init__(self, tokens, pairs, copies, chunk, lane, combine_order, cache):
        self.tokens = tuple(tokens)
        self.copies = tuple(copies)             # (key, slot) in first-need order
        self.chunk = chunk
        self.lane = lane
        self.combine_order = combine_order
        self._pairs = dict(pairs)               # token -> pairs in combine order
        self._cache = cache
        self.slot_bytes = getattr(chunk, "slot_bytes", 0)

    # ---- reading the plan -------------------------------------------
    def pairs(self, token):
        """The token's pairs in combine (reference) order."""
        return self._pairs[token]

    def scheduling(self, token):
        """The token's pairs in router order, the order copies and kernels are issued."""
        return tuple(sorted(self._pairs[token], key=lambda p: p.rank))

    @property
    def copy_keys(self):
        return tuple(key for key, _slot in self.copies)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, token):
        return self._pairs[token]

    def metrics(self):
        """A JSON-safe summary of where the forward's expert work went."""
        card = copy = cpu = 0
        for token in self.tokens:
            for pair in self._pairs[token]:
                if pair.lane == CARD:
                    card += 1
                elif pair.lane == COPY:
                    copy += 1
                else:
                    cpu += 1
        m = self.chunk.metrics()
        return {
            "tokens": len(self.tokens),
            "pairs": card + copy + cpu,
            "card_pairs": card,
            "copy_pairs": copy,
            "cpu_pairs": cpu,
            "distinct_copies": len(self.copies),
            "copy_bytes": len(self.copies) * self.slot_bytes,
            "hits": m["hits"],
            "coalesced": m["coalesced"],
            "saved_copies": m["saved_copies"],
            "lane": self.lane,
            "combine_order": self.combine_order,
        }

    def release(self):
        """Release every token's pins after the combine (the cache keeps READY slots)."""
        released = 0
        for token in self.tokens:
            if self._cache.release(token):
                released += 1
        return released

    def __repr__(self):
        return "TandemPlan(tokens=%d, copies=%d, lane=%r, order=%r)" % (
            len(self.tokens), len(self.copies), self.lane, self.combine_order)


class MoeTandem(object):
    """Assigns one execution unit's routed experts to the card, a copy or the CPU."""

    def __init__(self, placement, cache, lane=COPY, cpu_share=1.0, combine_order="router"):
        if not isinstance(placement, Placement):
            raise TypeError("placement must be a kindi.moe.Placement")
        if not isinstance(cache, ExpertCache):
            raise TypeError("cache must be an ExpertCache")
        lane = str(lane).strip().lower()
        if lane not in LANES:
            raise ValueError("lane must be one of %s, not %r" % (", ".join(LANES), lane))
        order = str(combine_order).strip().lower()
        if order not in ORDERS:
            raise ValueError("combine_order must be one of %s, not %r"
                             % (", ".join(ORDERS), combine_order))
        try:
            share = float(cpu_share)
        except (TypeError, ValueError):
            raise TypeError("cpu_share must be a number in [0, 1]")
        if not 0.0 <= share <= 1.0:
            raise ValueError("cpu_share must be in [0, 1], not %r" % (cpu_share,))
        self.placement, self.cache = placement, cache
        self.lane, self.cpu_share, self.combine_order = lane, share, order

    def _offload(self, k):
        """Deterministic CPU-lane thinning: share=1 all, 0 none, 0.5 alternate.

        The k-th host pair of a token is offloaded when the running share crosses
        an integer, so the choice is even, repeatable and independent of slot
        state - a rate control, not a sample.
        """
        if self.lane != CPU or self.cpu_share <= 0.0:
            return False
        share = self.cpu_share
        return int((k + 1) * share + 1e-9) > int(k * share + 1e-9)

    def _tier(self, layer, expert):
        spec = self.placement.spec
        if layer not in spec.layers:
            raise ValueError("layer %r is not a MoE layer of this model (%s)"
                             % (layer, tuple(spec.layers)))
        if expert < 0 or expert >= spec.n_exp:
            raise ValueError("expert %r is outside 0..%d in layer %d"
                             % (expert, spec.n_exp - 1, layer))
        return self.placement.tier[layer][expert]

    def forward(self, pairs_by_token):
        """Plan one execution unit: [(token, [(layer, expert), ...]), ...].

        The keys are in router order. Every token is reserved through the chunk
        planner, so a token's copies are atomic (all slots or none) and an
        expert shared by several tokens is copied once.
        """
        items, seen = [], set()
        for item in pairs_by_token:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("each forward item must be (token, keys)")
            token, keys = item
            try:
                hash(token)
            except TypeError:
                raise TypeError("token must be hashable")
            if token in seen:
                raise ValueError("token %r appears twice in one forward" % (token,))
            seen.add(token)
            pairs = []
            for key in keys:
                if not isinstance(key, (tuple, list)) or len(key) != 2:
                    raise TypeError("an expert key must be (layer, expert), not %r" % (key,))
                layer = _count(key[0], "layer")
                expert = _count(key[1], "expert")
                pairs.append((layer, expert))
            items.append((token, pairs))

        router_pairs, copy_requests = [], []
        for token, keys in items:
            host_seen = 0
            assigned, copies = [], []
            for rank, (layer, expert) in enumerate(keys):
                tier = self._tier(layer, expert)
                if tier == "card":
                    assigned.append(TandemPair(layer, expert, CARD, None, rank))
                    continue
                lane = COPY
                if tier == "host":
                    lane = CPU if self._offload(host_seen) else COPY
                    host_seen += 1
                if lane == CPU:
                    assigned.append(TandemPair(layer, expert, CPU, None, rank))
                else:
                    assigned.append(TandemPair(layer, expert, COPY, None, rank))
                    copies.append((layer, expert))
            router_pairs.append((token, assigned))
            copy_requests.append((token, copies))

        chunk = plan_chunk(self.cache, copy_requests)

        pairs = {}
        for (token, assigned), plan in zip(router_pairs, chunk.plans, strict=False):
            # plan.entries is the token's copy keys in router order, so it lines up with
            # the COPY pairs of `assigned` (card and cpu pairs hold no slot)
            copy_at = [i for i, pair in enumerate(assigned) if pair.lane == COPY]
            if len(copy_at) != len(plan.entries):
                raise RuntimeError("chunk plan and assignment disagree for token %r" % (token,))
            for i, entry in zip(copy_at, plan.entries, strict=False):
                assigned[i] = assigned[i]._replace(slot=entry.slot)
            if self.combine_order == "expert":
                # Gemma 4's reference loop is one layer per plan, so the layer term only
                # keeps a mixed batch deterministic
                assigned = sorted(assigned, key=lambda pair: (pair.expert, pair.layer, pair.rank))
            pairs[token] = tuple(assigned)
        return TandemPlan([token for token, _ in items], pairs, chunk.copies, chunk,
                          self.lane, self.combine_order, self.cache)
