# SPDX-License-Identifier: AGPL-3.0-only
"""Model-free copy-only expert cache for MoE inference.

This module implements the slot manager described in docs/MOE-TANDEM.md. It
does not know about a model, CUDA, tensors or bytes on a device: a caller plans
the experts selected by one router token, copies each miss into its slot, marks
the slot ready, runs the expert kernel, and releases the token after the
combine. The state machine is deliberately separate from the future MoE
backend so its eviction and pinning rules can be tested without a pack.

A slot is always one complete expert. The cache uses free slots first, then
evicts the least-recently-used unpinned READY slot, and only uses a transient
slot when every cache slot is pinned or in flight. LOADING, IN_USE and pinned
slots are never victims. A plan is all-or-nothing: if it cannot reserve every
missing expert, it raises ExpertCacheFull without changing any slot.

The module is not thread-safe. The future backend owns the copy stream and
calls these methods from its scheduler thread.
"""
import operator
from collections import namedtuple

__all__ = [
    "ExpertCache",
    "ExpertCacheFull",
    "ExpertKey",
    "ExpertPlan",
    "PlanEntry",
    "Slot",
    "FREE",
    "LOADING",
    "READY",
    "IN_USE",
]

FREE = "free"
LOADING = "loading"
READY = "ready"
IN_USE = "in_use"
STATES = (FREE, LOADING, READY, IN_USE)


class ExpertKey(namedtuple("ExpertKey", "layer expert")):
    """The identity of one routed expert: its layer and expert id."""
    __slots__ = ()

    def __new__(cls, layer, expert):
        values = []
        for name, value in (("layer", layer), ("expert", expert)):
            if isinstance(value, bool):
                raise TypeError("%s must be an integer, not bool" % name)
            try:
                value = operator.index(value)
            except TypeError:
                raise TypeError("%s must be an integer" % name)
            if value < 0:
                raise ValueError("%s must be non-negative" % name)
            values.append(value)
        return super(ExpertKey, cls).__new__(cls, values[0], values[1])


class Slot(object):
    """One cache or transient slot.

    index is non-negative for a cache slot and negative for a transient slot.
    key is None while the slot is FREE. pins contains the tokens currently
    using or waiting for the slot; one token pins a slot once.
    """
    __slots__ = ("index", "transient", "key", "state", "last_used", "pins", "loads")

    def __init__(self, index, transient=False):
        self.index = index
        self.transient = bool(transient)
        self.key = None
        self.state = FREE
        self.last_used = 0
        self.pins = set()
        self.loads = 0

    @property
    def free(self):
        return self.state == FREE

    @property
    def ready(self):
        return self.state in (READY, IN_USE)

    def __repr__(self):
        return "Slot(%d, %s, %r, pins=%d)" % (
            self.index, self.state, self.key, len(self.pins))


PlanEntry = namedtuple("PlanEntry", "key slot hit")


class ExpertPlan(object):
    """The ordered result of ExpertCache.plan()."""
    __slots__ = ("token", "entries")

    def __init__(self, token, entries):
        self.token = token
        self.entries = tuple(entries)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]

    @property
    def keys(self):
        return tuple(e.key for e in self.entries)

    @property
    def slots(self):
        return tuple(e.slot for e in self.entries)

    @property
    def misses(self):
        return tuple(e for e in self.entries if not e.hit)

    def __repr__(self):
        return "ExpertPlan(token=%r, entries=%r)" % (self.token, self.entries)


class ExpertCacheFull(RuntimeError):
    """A plan could not reserve every selected expert without evicting a pin."""

    def __init__(self, needed, available, token):
        self.needed = int(needed)
        self.available = int(available)
        self.token = token
        super(ExpertCacheFull, self).__init__(
            "expert cache full: need %d slot(s), %d available (token %r)"
            % (self.needed, self.available, self.token))


def _count(value, name, minimum=0):
    if isinstance(value, bool):
        raise TypeError("%s must be an integer, not bool" % name)
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError("%s must be an integer" % name)
    if value < minimum:
        raise ValueError("%s must be >= %d" % (name, minimum))
    return value


def _key(value):
    if isinstance(value, ExpertKey):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return ExpertKey(*value)
    raise TypeError("an expert key must be ExpertKey(layer, expert) or a 2-tuple")


def _state_counts(slots):
    counts = dict((s, 0) for s in STATES)
    for slot in slots:
        counts[slot.state] += 1
    return counts


class ExpertCache(object):
    """A copy-only, byte-fixed expert slot cache.

    Parameters:
      slots: number of cache slots (not including transient slots).
      slot_bytes: bytes in one complete expert slot. Used only for accounting.
      transient_slots: extra slots used for a miss when no cache slot can be
        reused. A transient slot is released as soon as its last pin is
        released and its copy is complete.
    """
    def __init__(self, slots, slot_bytes=1, transient_slots=0):
        self.slot_count = _count(slots, "slots")
        self.slot_bytes = _count(slot_bytes, "slot_bytes", minimum=1)
        self.transient_count = _count(transient_slots, "transient_slots")
        self._cache = [Slot(i, False) for i in range(self.slot_count)]
        self._transient = [Slot(-(i + 1), True) for i in range(self.transient_count)]
        self._by_key = {}
        self._tokens = {}
        self._clock = 0
        self._pin_count = 0
        self.stats = {
            "plans": 0,
            "selected": 0,
            "hits": 0,
            "misses": 0,
            "coalesced": 0,
            "evictions": 0,
            "admissions": 0,
            "transient_admissions": 0,
            "transient_drops": 0,
            "aborted_loads": 0,
            "copy_failures": 0,
            "copies_completed": 0,
            "uses": 0,
            "pins": 0,
            "releases": 0,
            "rejected": 0,
            "rejected_keys": 0,
            "duplicate_keys": 0,
            "copy_bytes": 0,
            "transient_copy_bytes": 0,
            "peak_pinned": 0,
            "peak_used_slots": 0,
        }

    @property
    def capacity_bytes(self):
        return self.slot_count * self.slot_bytes

    @property
    def hit_rate(self):
        total = self.stats["hits"] + self.stats["misses"]
        return (float(self.stats["hits"]) / total) if total else 0.0

    @property
    def all_slots(self):
        return tuple(self._cache) + tuple(self._transient)

    def __len__(self):
        return self.slot_count

    def __contains__(self, key):
        return _key(key) in self._by_key

    def __repr__(self):
        m = self.metrics()
        return "ExpertCache(slots=%d, transient=%d, used=%d, pins=%d)" % (
            self.slot_count, self.transient_count, m["used_slots"], m["active_pins"])

    def slot_for(self, key):
        """The slot holding key, including a LOADING or transient slot."""
        return self._by_key.get(_key(key))

    def keys(self):
        """The keys currently resident, including LOADING slots."""
        return tuple(self._by_key)

    def slot_states(self):
        """State counts for every cache and transient slot."""
        return {
            "cache": _state_counts(self._cache),
            "transient": _state_counts(self._transient),
        }

    def snapshot(self):
        """A JSON-safe view of every slot, for diagnostics and tests."""
        return [
            {
                "index": slot.index,
                "transient": slot.transient,
                "state": slot.state,
                "key": tuple(slot.key) if slot.key is not None else None,
                "last_used": slot.last_used,
                "pins": len(slot.pins),
                "loads": slot.loads,
            }
            for slot in self.all_slots
        ]

    def metrics(self):
        """Counters plus current occupancy; safe to serialize with json.dumps."""
        cache_states = _state_counts(self._cache)
        transient_states = _state_counts(self._transient)
        cache_used = self.slot_count - cache_states[FREE]
        transient_used = self.transient_count - transient_states[FREE]
        out = dict(self.stats)
        out.update({
            "slot_count": self.slot_count,
            "transient_slots": self.transient_count,
            "slot_bytes": self.slot_bytes,
            "capacity_bytes": self.capacity_bytes,
            "used_slots": cache_used + transient_used,
            "cache_used_slots": cache_used,
            "transient_used_slots": transient_used,
            "used_bytes": (cache_used + transient_used) * self.slot_bytes,
            "resident_keys": len(self._by_key),
            "active_pins": self._pin_count,
            "active_tokens": len(self._tokens),
            "hit_rate": self.hit_rate,
            "bytes_copied": self.stats["copy_bytes"],
            "cache_states": cache_states,
            "transient_states": transient_states,
        })
        return out

    # ---- slot lifecycle -------------------------------------------------
    def _touch(self, slot):
        self._clock += 1
        slot.last_used = self._clock

    def _pin(self, slot, token):
        if token in slot.pins:
            return
        slot.pins.add(token)
        self._pin_count += 1
        self.stats["pins"] += 1
        if self._pin_count > self.stats["peak_pinned"]:
            self.stats["peak_pinned"] = self._pin_count

    def _unpin(self, slot, token):
        if token not in slot.pins:
            return
        slot.pins.remove(token)
        self._pin_count -= 1

    def _free_slot(self, slot, aborted=False):
        if slot.pins:
            raise RuntimeError("cannot free a pinned expert slot")
        old = slot.key
        if old is not None:
            self._by_key.pop(old, None)
        slot.key = None
        slot.state = FREE
        slot.last_used = 0
        if slot.transient:
            self.stats["transient_drops"] += 1
        elif aborted:
            self.stats["aborted_loads"] += 1

    def _assign(self, slot, key, token):
        if slot.state != FREE or slot.key is not None:
            raise RuntimeError("tried to assign a non-free expert slot")
        slot.key = key
        slot.state = LOADING
        slot.loads += 1
        self._by_key[key] = slot
        self._touch(slot)
        self.stats["admissions"] += 1
        if slot.transient:
            self.stats["transient_admissions"] += 1
            self.stats["transient_copy_bytes"] += self.slot_bytes
        self.stats["copy_bytes"] += self.slot_bytes
        self._pin(slot, token)

    def _evict(self):
        """Evict the LRU unpinned READY cache slot, or return None."""
        victim = None
        for slot in self._cache:
            if slot.state != READY or slot.pins:
                continue
            if victim is None or (slot.last_used, slot.index) < (victim.last_used, victim.index):
                victim = slot
        if victim is None:
            return None
        self.stats["evictions"] += 1
        self._free_slot(victim)
        return victim

    def _take_slot(self):
        for slot in self._cache:
            if slot.state == FREE:
                return slot
        slot = self._evict()
        if slot is not None:
            return slot
        for slot in self._transient:
            if slot.state == FREE:
                return slot
        return None

    # ---- public request path -------------------------------------------
    def plan(self, keys, token):
        """Pin every selected expert for token, preserving route order.

        A key already resident is a hit. A new key is a miss and its slot
        starts LOADING; the caller must call mark_ready() after the copy. A
        key already LOADING is coalesced and is not copied again. The plan is
        atomic: if capacity is insufficient, no slot or pin is changed and
        ExpertCacheFull is raised.
        """
        try:
            hash(token)
        except TypeError:
            raise TypeError("token must be hashable")
        if token in self._tokens:
            raise ValueError("token %r already has an active expert plan" % (token,))

        original = [_key(k) for k in keys]
        unique = []
        seen = set()
        for key in original:
            if key not in seen:
                seen.add(key)
                unique.append(key)

        existing = {}
        missing = []
        for key in unique:
            slot = self._by_key.get(key)
            if slot is None:
                missing.append(key)
            else:
                if slot.state == FREE:
                    raise RuntimeError("a FREE slot is still indexed by key %r" % (key,))
                existing[key] = slot

        protected = set(id(slot) for slot in existing.values())
        free_cache = sum(1 for slot in self._cache if slot.state == FREE)
        evictable = sum(1 for slot in self._cache
                        if slot.state == READY and not slot.pins and id(slot) not in protected)
        free_transient = sum(1 for slot in self._transient if slot.state == FREE)
        available = free_cache + evictable + free_transient
        if len(missing) > available:
            self.stats["rejected"] += 1
            self.stats["rejected_keys"] += len(missing)
            raise ExpertCacheFull(len(missing), available, token)

        slots = set()
        hits = {}
        for key in unique:
            slot = existing.get(key)
            if slot is None:
                continue
            self._pin(slot, token)
            slots.add(slot)
            self._touch(slot)
            if slot.state == LOADING:
                self.stats["coalesced"] += 1
                hits[key] = False
            else:
                self.stats["hits"] += 1
                hits[key] = True
        for key in missing:
            slot = self._take_slot()
            if slot is None:
                # The preflight above makes this unreachable without a
                # concurrent caller, which this module does not support.
                raise RuntimeError("expert cache capacity changed during plan")
            self._assign(slot, key, token)
            slots.add(slot)
            self.stats["misses"] += 1
            hits[key] = False

        self._tokens[token] = slots
        self.stats["plans"] += 1
        self.stats["selected"] += len(original)
        self.stats["duplicate_keys"] += len(original) - len(unique)
        used = sum(1 for slot in self.all_slots if slot.state != FREE)
        if used > self.stats["peak_used_slots"]:
            self.stats["peak_used_slots"] = used
        entries = [PlanEntry(key, self._by_key[key], hits[key]) for key in original]
        return ExpertPlan(token, entries)

    def acquire(self, key, token):
        """Pin one expert for token, without a multi-key plan.

        This is the single-key form for a backend that does not use plan().
        The token may acquire several keys before release(token). The caller
        must mark_ready() after each miss is copied.
        """
        try:
            hash(token)
        except TypeError:
            raise TypeError("token must be hashable")
        key = _key(key)
        slot = self._by_key.get(key)
        made_token = token not in self._tokens
        if made_token:
            self._tokens[token] = set()
        if slot is None:
            slot = self._take_slot()
            if slot is None:
                if made_token:
                    self._tokens.pop(token, None)
                self.stats["rejected"] += 1
                self.stats["rejected_keys"] += 1
                raise ExpertCacheFull(1, 0, token)
            self._assign(slot, key, token)
            self.stats["misses"] += 1
        else:
            self._pin(slot, token)
            self._touch(slot)
            if slot.state == LOADING:
                self.stats["coalesced"] += 1
            else:
                self.stats["hits"] += 1
        self._tokens[token].add(slot)
        self.stats["selected"] += 1
        used = sum(1 for s in self.all_slots if s.state != FREE)
        if used > self.stats["peak_used_slots"]:
            self.stats["peak_used_slots"] = used
        return slot

    def mark_ready(self, key):
        """Mark a LOADING slot's complete copy as READY."""
        key = _key(key)
        slot = self._by_key.get(key)
        if slot is None:
            raise KeyError("expert %r is not resident" % (key,))
        if slot.state != LOADING:
            raise RuntimeError("expert %r is %s, not LOADING" % (key, slot.state))
        slot.state = READY
        self.stats["copies_completed"] += 1
        self._touch(slot)
        if slot.transient and not slot.pins:
            self._free_slot(slot)

    def begin_use(self, key):
        """Mark a pinned READY slot as IN_USE by an expert kernel."""
        key = _key(key)
        slot = self._by_key.get(key)
        if slot is None:
            raise KeyError("expert %r is not resident" % (key,))
        if slot.state != READY:
            raise RuntimeError("expert %r is %s, not READY" % (key, slot.state))
        if not slot.pins:
            raise RuntimeError("cannot use an expert slot that has no pin")
        slot.state = IN_USE
        self.stats["uses"] += 1
        self._touch(slot)

    def end_use(self, key):
        """Return an IN_USE slot to READY after its expert kernel completes."""
        key = _key(key)
        slot = self._by_key.get(key)
        if slot is None:
            raise KeyError("expert %r is not resident" % (key,))
        if slot.state != IN_USE:
            raise RuntimeError("expert %r is %s, not IN_USE" % (key, slot.state))
        slot.state = READY
        self._touch(slot)
        if slot.transient and not slot.pins:
            self._free_slot(slot)

    def fail(self, key):
        """Discard a LOADING slot after a copy failed.

        The caller must release the token first. This is a hard error path:
        there is no fallback slot and no partial expert.
        """
        key = _key(key)
        slot = self._by_key.get(key)
        if slot is None:
            raise KeyError("expert %r is not resident" % (key,))
        if slot.state != LOADING:
            raise RuntimeError("expert %r is %s, not LOADING" % (key, slot.state))
        if slot.pins:
            raise RuntimeError("release the expert plan before failing its copy")
        self.stats["copy_failures"] += 1
        self._free_slot(slot, aborted=True)

    def release(self, token):
        """Release every pin held by token.

        A LOADING slot stays LOADING until mark_ready()/fail(); this prevents a
        cancelled copy from being reused before its event completes. A READY
        cache slot stays resident. A transient slot is freed once its last pin
        is gone and its copy is complete.
        """
        slots = self._tokens.pop(token, None)
        if slots is None:
            return False
        for slot in sorted(slots, key=lambda s: s.index):
            self._unpin(slot, token)
            if slot.pins:
                continue
            if slot.transient:
                if slot.state in (READY, IN_USE):
                    self._free_slot(slot)
            elif slot.state == IN_USE:
                slot.state = READY
                self._touch(slot)
        self.stats["releases"] += 1
        return True
