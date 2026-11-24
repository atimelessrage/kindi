# Edge-derived mask proofs: REQ-0027 rows after the 2026-11-24 change

Host: i7-8086K, Windows 10. Packs: `bonsai-2b` (248,320 vocabulary entries) and `bonsai-9b` (same
tokenizer). Greedy, the 22-token document and the 200-token long document from
`tools/grammar_cost.py`. Every row is the median of three fresh processes. This is a CPU filter
measurement: no GPU was available on this host, so no GPU decode, exactness or token rate is
claimed here.

Machine-readable artifacts: [grammar-mask-short.json](grammar-mask-short.json) (the short document,
the harness's row order, index cold), [grammar-mask-long.json](grammar-mask-long.json) (the 200-token
document in its own process, as `--doc long` runs) and [grammar-mask-9b.json](grammar-mask-9b.json)
(the short document on the 9B pack). All three are written by `tools/grammar_mask_record.py`, which
drives the same `grammar_cost` walks, so every number below is reproducible rather than transcribed.
Each row also keeps its three raw `ms_per_token` values under `runs_ms_per_token`.

## What changed

Two costs dominated a mask miss, both found by profiling the 22-token GBNF walk on the real pack:

1. **`Grammar._token_safe_sets` probed the whole alphabet per NFA state.** To prove a state's
   universal character loop it called `_step` once for every character in the vocabulary's alphabet —
   **8,228 characters** on this pack, each a full epsilon closure — and cached the result per state.
   That is a literal re-derivation of `{c | step({q}, c) is not None and q ∈ step({q}, c)}`. The same
   set can be read off the NFA: `step({q}, c)`'s closure contains `q` exactly when one of `q`'s
   character edges lands on a state that can epsilon-reach `q`. `_self_loop_chars_for` computes it from
   the character edges plus a reverse-epsilon closure built once at grammar load. No proof was
   weakened: a state without such an edge still contributes no safe set, and the table still falls
   back to the exact per-token check.
2. **The mask union sorted a whole-vocabulary array per miss.** `np.union1d(ids, kept)` in the broad
   safe-set path, and `np.unique(np.concatenate(pieces))` in the trie path, both sorted up to 248,320
   int64s every time. The safe ids and the accepted "outside" ids are disjoint and each sorted, so a
   cached boolean mask over the vocabulary plus `np.flatnonzero` gives the same ascending ids
   linearly (`token_classes._sorted_ids`, `_safe_mask_for`).

`Grammar._step` also stopped re-scanning epsilon edges and re-dispatching on the label kind for every
member of a state closure: the character edges of each automaton state are flattened once
(`_flat_edges`) and the label test is inlined with `_accepts`' exact semantics.

Rejected alternatives, for the record: memoizing `_step` (85,978 of 105,336 calls on the short
document were distinct state/character pairs, so the cache would have bought little) and a
character→edge index per state (a state has 0.27 character edges on average, at most 1).

## REQ-0027 gate rows, 2B

Commands:

```sh
python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b
python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b --doc long
```

| document | constraint | tokens | misses | filter total | per token | 2026-11-22 | gate |
|---|---|---:|---:|---:|---:|---:|---:|
| 22-token | `json_schema` | 22 | 17 | 221 ms | **10.0 ms** | 15 ms | <= 23 ms |
| 22-token | unrolled `json.gbnf` depth 1 (1,737 states) | 22 | 18 | 1,255 ms | **57.1 ms** | 124 ms | <= 272 ms |
| 22-token | unrolled `json.gbnf` depth 2 (3,195 states) | 22 | 18 | 209 ms | **9.5 ms** | 83 ms | <= 272 ms |
| 22-token | unrolled `json.gbnf` depth 3 (9,483 states) | 22 | 18 | 212 ms | **9.7 ms** | 84 ms | <= 272 ms |
| 22-token | ordered-record GBNF (1,061 states) | 22 | 19 | 184 ms | **8.4 ms** | 70 ms | <= 272 ms |
| 200-token | `json_schema` | 200 | 43 | 349 ms | **1.7 ms** | 3 ms | <= 6 ms |
| 200-token | ordered-record GBNF (831 states) | 200 | 55 | 1,505 ms | **7.5 ms** | 20 ms | <= 46 ms |

Every row clears the 10x gate, and the GBNF rows move most: depth 1 **2.2x**, depth 2/3 **8.7x/8.7x**,
the record GBNF **8.3x** short and **2.7x** long, `json_schema` **1.5x** short and **1.8x** long. The
depth-1 row stays the slowest because its start-state mask builds the shared vocabulary index in a
cold process (about 1,250 ms of the 1,255 ms total — this harness does not prewarm; `kindi serve`
does, at startup). Once that build is out of the way the depth-1 grammar computes the same 10 ms mask
as depth 2/3, whose grammars are larger but whose start-state mask reuses the index built three rows
earlier.

The 9B pack has the same 248,320-entry vocabulary, so its rows track the 2B rather than scaling with
the model (median of three, [grammar-mask-9b.json](grammar-mask-9b.json)):

| document | constraint | per token |
|---|---|---:|
| 22-token | `json_schema` | **10.3 ms** |
| 22-token | unrolled `json.gbnf` depth 1 / 2 / 3 | **58.2 / 9.7 / 10.2 ms** |
| 22-token | ordered-record GBNF | **8.6 ms** |
| 200-token | `json_schema` / ordered-record GBNF | **1.8 / 7.6 ms** |

## Language unchanged

The safe-set proof decides which token ids are legal, so a wrong set would silently change the
accepted language. It is checked two ways on the real vocabularies, not just on toy ones:

- The pre-token-class sweep (`Grammar._vocab_allowed_scan`, one `advance` per vocabulary entry) is
  kept as the oracle. On the 248,320-entry vocabulary it was compared state by state against the new
  mask: **0 mismatches** over 6 states of the 22-token `json.gbnf` walk, 4 states of the `json_schema`
  walk (including the 245,896-id string-body state) and 4 states of the record GBNF walk. The oracle is
  a full `advance` per entry, so this is a spot check over 14 real states, not every state.
- `tests/test_token_classes.py` compares the new mask to that oracle on adversarial small vocabularies
  and on every small reachable state, and now also asserts `_self_loop_chars_for(q, alphabet)` equals
  the probing definition for every member of every reached closure — the epsilon-reached state is
  exactly where a naive edge read would differ.

## Scope and limits

- No decoding change: `JsonGrammar.vocab_allowed` and `Grammar.vocab_allowed` compute the same masks;
  only the cost changed.
- Still open from REQ-0027: A4 (GPU structured-output exactness and same-document generation) and A5
  (unconstrained decode within 1%, per-request host memory). This run claims neither.
- The vocabulary build itself is unchanged (about 2.4 s on the 2B, moved to server startup by the
  prewarm); this work does not touch it.
