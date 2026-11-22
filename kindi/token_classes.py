# SPDX-License-Identifier: AGPL-3.0-only
"""Token-class table for constrained decoding (REQ-0027).

The old ``vocab_allowed`` advanced the automaton through every one of a pack's
token texts for every new automaton state.  This table keeps the per-state
operation exact without that full sweep:

* token ids are indexed by their first character and by the set of characters
  their decoded text contains;
* a state that can prove a universal character loop (for example the body of a
  JSON string) can add every token whose text is drawn only from that loop in
  one slice;
* a small character trie handles the remaining multi-character entries, and
  prunes a whole subtree only when the same proof covers every character below
  it.

The proof comes from the grammar, never from a shape heuristic.  A bulk set is
used only when a productive automaton loop accepts every character in the set;
otherwise the table falls back to the exact per-token advance.  The resulting
ids are sorted, like the old implementation.
"""
import collections
import threading

import numpy as np

# The large part of the table (the character buckets, safe-id sets and tries)
# depends only on the vocabulary.  Keep a few vocabularies' indexes alive so a
# new grammar in the same process does not rebuild them.  The cache is bounded:
# a server normally has one pack, while tests create many small vocabularies.
_INDEX_CACHE_LIMIT = 2
# A first-character exact scan is cheaper than building the trie up to this many
# candidate ids.  The 2B vocabulary's JSON start state keeps ~107K and scans in
# ~70 ms.  The trie is the default for both grammars; this limit applies only to
# a table built with prefer_trie=False.
_FIRST_CHAR_LIMIT = 150000
_INDEX_CACHE: "collections.OrderedDict[int, tuple[object, _VocabularyIndex]]" = collections.OrderedDict()
_INDEX_LOCK = threading.Lock()


class _VocabularyIndex(object):
    """The vocabulary-only part of a token-class table."""

    __slots__ = ("texts", "by_first", "char_masks", "char_bits", "all_mask", "alphabet",
                 "all_ids", "safe_ids", "outside_ids", "outside_tries", "root", "lock")

    def __init__(self, texts, by_first, char_masks, char_bits, alphabet):
        self.texts = texts
        self.by_first = by_first
        self.char_masks = char_masks
        self.char_bits = char_bits
        self.all_mask = (1 << len(char_bits)) - 1
        self.alphabet = alphabet
        self.all_ids = np.arange(len(texts), dtype=np.int64)
        self.safe_ids = {}
        self.outside_ids = {}
        self.outside_tries = {}
        self.root: _Node | None = None
        self.lock = threading.RLock()


def _build_index(texts):
    """Build the vocabulary-only indexes for ``texts``."""
    by_first = collections.defaultdict(list)
    char_masks = []
    char_bits = {}
    alphabet = set()
    for token_id, text in enumerate(texts):
        by_first[text[:1]].append(token_id)
        mask = 0
        for ch in set(text):
            bit = char_bits.get(ch)
            if bit is None:
                bit = 1 << len(char_bits)          # one bit per character in the alphabet
                char_bits[ch] = bit
                alphabet.add(ch)
            mask |= bit
        char_masks.append(mask)
    by_first = {k: np.asarray(v, dtype=np.int64) for k, v in by_first.items()}
    return _VocabularyIndex(texts, by_first, char_masks, char_bits, frozenset(alphabet))


def _index_for(texts):
    """The shared index for one texts sequence (keyed by object identity)."""
    key = id(texts)
    with _INDEX_LOCK:
        entry = _INDEX_CACHE.get(key)
        if entry is not None and entry[0] is texts:
            _INDEX_CACHE.move_to_end(key)
            return entry[1]
        index = _build_index(texts)
        _INDEX_CACHE[key] = (texts, index)
        _INDEX_CACHE.move_to_end(key)
        while len(_INDEX_CACHE) > _INDEX_CACHE_LIMIT:
            _INDEX_CACHE.popitem(last=False)
        return index


class _Node(object):
    __slots__ = ("children", "ids", "all_ids", "chars")

    def __init__(self):
        self.children = {}
        self.ids = []
        self.all_ids = None
        self.chars = None


class TokenClassTable(object):
    """A grammar-owned view over a process-shared vocabulary index.

    ``step(state, ch)`` advances one character and ``safe_sets(state, alphabet)``
    returns zero or more character sets for which the state has a productive
    universal loop.  Empty ``safe_sets`` is always safe: the exact path is used.

    The vocabulary-dependent indexes are reused across grammar instances that
    pass the same ``texts`` object, so a new structured request does not rebuild
    them.  Each grammar still owns its own step function, safe-set proof and
    per-state mask cache.  ``texts`` must not be mutated while the index is in
    use; the server and the backends already build it once and keep it.
    """

    def __init__(self, texts, step, safe_sets, prefer_trie=True):
        self.texts = texts
        self._step = step
        self._safe_sets = safe_sets
        # The trie is the default for both grammars.  The exact first-character
        # path remains available for a caller that explicitly turns the trie off.
        self._prefer_trie = prefer_trie
        self._index = _index_for(texts)
        self._by_first = self._index.by_first
        self._char_masks = self._index.char_masks
        self._char_bits = self._index.char_bits
        self._all_mask = self._index.all_mask
        self._alphabet = self._index.alphabet
        self._all_ids = self._index.all_ids
        self._safe_ids = self._index.safe_ids
        self._outside_ids = self._index.outside_ids
        self._outside_tries = self._index.outside_tries

    def allowed(self, state):
        """The exact ids allowed at ``state``; the caller owns the result cache."""
        safe = [s for s in self._safe_sets(state, self._alphabet) if s]
        if safe:
            # A broad class is the useful one; do not price every narrow
            # alternative just to select it.  The broad class makes its
            # complement small, so only tokens containing a non-loop character
            # need the exact check.  Narrow loops go to the trie.
            chosen = max(safe, key=len)
            # A narrow class (whitespace before a JSON value, digits, a GBNF
            # character class) is cheaper through the prewarmed trie than a
            # full-vocabulary safe-set scan.  Broad classes (a JSON string
            # body) still take the one-slice fast path below.
            if self._prefer_trie and len(chosen) * 8 < len(self._alphabet):
                return self._trie_allowed(state, safe)
            ids = self._safe_ids_for(chosen)
            if len(ids) * 2 >= len(self.texts):
                outside = self._outside_ids_for(chosen)
                if outside.size:
                    kept = self._subset_allowed(state, outside, chosen)
                    ids = np.union1d(ids, kept)
                return ids
            # A narrow safe set (for example the whitespace before a JSON
            # value) should not pay for the whole trie when the first-character
            # buckets are cheap to check exactly.
            if not self._prefer_trie:
                exact = self._first_char_allowed(state)
                if exact is not None:
                    return exact
            return self._trie_allowed(state, safe)
        if not self._prefer_trie:
            ids = self._first_char_allowed(state)
            if ids is not None:
                return ids
        return self._trie_allowed(state, safe)

    def _advance(self, state, text):
        for ch in text:
            state = self._step(state, ch)
            if state is None:
                return None
        return state

    def _safe_ids_for(self, chars):
        with self._index.lock:
            cached = self._safe_ids.get(chars)
            if cached is not None:
                return cached
            safe_bits = 0
            for ch in chars:
                safe_bits |= self._char_bits.get(ch, 0)
            outside = self._all_mask & ~safe_bits
            if outside:
                ids = np.fromiter(
                    (i for i, mask in enumerate(self._char_masks) if not (mask & outside)),
                    dtype=np.int64)
            else:
                ids = self._all_ids
            self._safe_ids[chars] = ids
            return ids

    def _outside_ids_for(self, chars):
        with self._index.lock:
            outside = self._outside_ids.get(chars)
            if outside is None:
                outside = np.setdiff1d(self._all_ids, self._safe_ids_for(chars))
                self._outside_ids[chars] = outside
            return outside

    def _exact_candidates(self, state, pieces):
        if not pieces:
            return np.empty(0, dtype=np.int64)
        candidates = np.concatenate(pieces)
        accepted = np.fromiter(
            (int(i) for i in candidates
             if self._advance(state, self.texts[int(i)]) is not None),
            dtype=np.int64)
        return np.unique(accepted)

    def _first_char_allowed(self, state):
        """Exact mask from the first-character buckets, when that is cheap."""
        pieces = []
        kept = 0
        for ch, ids in self._by_first.items():
            if ch == "" or self._step(state, ch) is not None:
                pieces.append(ids)
                kept += len(ids)
        if kept > _FIRST_CHAR_LIMIT:
            return None
        return self._exact_candidates(state, pieces)


    def _subset_trie(self, ids, key):
        with self._index.lock:
            tree = self._outside_tries.get(key)
            if tree is not None:
                return tree
            root = _Node()
            for token_id in ids:
                node = root
                for ch in self.texts[int(token_id)]:
                    node = node.children.setdefault(ch, _Node())
                node.ids.append(int(token_id))
            self._outside_tries[key] = root
            return root

    def _subset_allowed(self, state, ids, key):
        root = self._subset_trie(ids, key)
        out = []
        stack = [(root, state)]
        while stack:
            node, node_state = stack.pop()
            if node_state is None:
                continue
            if node.ids:
                out.extend(node.ids)
            for ch, child in node.children.items():
                next_state = self._step(node_state, ch)
                if next_state is not None:
                    stack.append((child, next_state))
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted(out), dtype=np.int64)

    def _build_trie(self):
        return _trie_for(self._index)

    def _chars_under(self, node):
        """The characters below ``node``, computed once and cached."""
        cached = node.chars
        if cached is not None:
            return cached
        with self._index.lock:
            stack = [(node, False)]
            while stack:
                current, seen = stack.pop()
                if current.chars is not None:
                    continue
                if seen:
                    chars = set()
                    for ch, child in current.children.items():
                        chars.add(ch)
                        chars.update(child.chars)
                    current.chars = frozenset(chars)
                else:
                    stack.append((current, True))
                    stack.extend((child, False) for child in current.children.values()
                                 if child.chars is None)
            return node.chars

    def _ids_under(self, node):
        """All ids below ``node``, collected once on demand and cached."""
        cached = node.all_ids
        if cached is not None:
            return cached
        with self._index.lock:
            if node.all_ids is not None:
                return node.all_ids
            out = []
            stack = [node]
            while stack:
                current = stack.pop()
                out.extend(current.ids)
                stack.extend(current.children.values())
            node.all_ids = np.asarray(out, dtype=np.int64)
            return node.all_ids

    def _trie_allowed(self, state, safe):
        """Walk the character trie, pruning subtrees covered by ``safe``."""
        root = self._build_trie()
        pieces = []
        stack = [(root, state)]
        while stack:
            node, node_state = stack.pop()
            if node_state is None:
                continue
            if any(self._chars_under(node) <= chars for chars in safe):
                pieces.append(self._ids_under(node))
                continue
            if node.ids:
                pieces.append(np.asarray(node.ids, dtype=np.int64))
            for ch, child in node.children.items():
                next_state = self._step(node_state, ch)
                if next_state is not None:
                    stack.append((child, next_state))
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(pieces))


def _trie_for(index):
    """Build (once) and return the character trie for a shared vocabulary index."""
    with index.lock:
        if index.root is not None:
            return index.root
        root = _Node()
        for token_id, text in enumerate(index.texts):
            node = root
            for ch in text:
                node = node.children.setdefault(ch, _Node())
            node.ids.append(token_id)
        index.root = root
        return root


def _chars_for_all(root):
    """Fill every node's character set bottom-up (prewarm only)."""
    stack = [(root, False)]
    while stack:
        node, seen = stack.pop()
        if node.chars is not None:
            continue
        if seen:
            chars = set()
            for ch, child in node.children.items():
                chars.add(ch)
                chars.update(child.chars)
            node.chars = frozenset(chars)
        else:
            stack.append((node, True))
            stack.extend((child, False) for child in node.children.values()
                         if child.chars is None)


def _ids_for_all(root):
    """Fill every node's id tuple bottom-up (prewarm only)."""
    stack = [(root, False)]
    while stack:
        node, seen = stack.pop()
        if node.all_ids is not None:
            continue
        if seen:
            ids = list(node.ids)
            for child in node.children.values():
                ids.extend(child.all_ids)
            node.all_ids = tuple(ids)
        else:
            stack.append((node, True))
            stack.extend((child, False) for child in node.children.values()
                         if child.all_ids is None)


def prewarm(texts):
    """Build the vocabulary index, trie, character sets and id sets before the first request.

    The trie is grammar-independent, so a server can build it at startup (in the
    background) and a later structured request only pays its per-state mask.
    """
    index = _index_for(texts)
    root = _trie_for(index)
    with index.lock:
        _chars_for_all(root)
        _ids_for_all(root)
    return root


__all__ = ["TokenClassTable", "prewarm"]
