# Token-class constrained mask (REQ-0027), 2B, 2026-11-20

Host: i7-8086K + RTX 4060 Ti 8 GB. Pack: `bonsai-2b`, **248,320 vocabulary
entries**. Greedy, one 22-token document and the 200-token long document from
`tools/grammar_cost.py`. This is a CPU filter measurement: the GPU was not
used and no decode rate is claimed.

Machine-readable artifact: [grammar-mask.json](grammar-mask.json).

## What changed

`Grammar.vocab_allowed` and `JsonGrammar.vocab_allowed` now use
`kindi.token_classes.TokenClassTable`:

- ids are indexed by their first character and by the set of characters their
  decoded text contains;
- a productive universal character loop in the current automaton state accepts
  every token whose text stays inside that loop in one slice;
- the remaining tokens are checked exactly, with a token trie pruning a whole
  subtree only when the same loop proof covers it; otherwise the old per-token
  advance is still the fallback.

The table is grammar-owned and state-cached. The language is unchanged: the
CPU set-equality tests compare every vocabulary entry against
`_vocab_allowed_scan`, the pre-change implementation.

## Gate rows

Commands:

```sh
python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b
python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b --doc long
```

| document | constraint | tokens | misses | filter total | per token | gate |
|---|---|---:|---:|---:|---:|---:|
| 22-token | `json_schema` | 22 | 17 | 329 ms | **15 ms** | ≤ 23 ms |
| 22-token | unrolled `json.gbnf` depth 1 (1,737 states) | 22 | 18 | 4,670 ms | **212 ms** | ≤ 272 ms |
| 200-token | `json_schema` | 200 | 43 | 674 ms | **3 ms** | ≤ 6 ms |
| 200-token | ordered-record GBNF (831 states) | 200 | 55 | 5,945 ms | **30 ms** | ≤ 46 ms |

The short-document baselines were 231 and 2,720 ms/token; the long-document
baselines were 57 and 463 ms/token. All four rows improve by more than the
required 10x. The unrolled depth-2 and depth-3 diagnostics were 324 and
303 ms/token; they are not part of the A2 gate, which names the depth-1
unrolled grammar measured by the baseline. The tables remain a diagnostic for
recursive GBNF and are not a serving recommendation.

The same harness reports the first start-state mask separately: 7,023 ms for
the schema and 856 ms for GBNF depth 1. That is the one-time table/trie build;
the gate's per-token rows above exclude it. It is recorded rather than hidden,
but is not claimed as a decode latency.

## Semantics

- `tests/test_token_classes.py` compares the token-class mask with the old
  sweep for every entry in adversarial token lists, including quotes,
  backslashes, control characters, empty pieces and duplicate texts.
- The same comparison on the real 2B vocabulary matched on all 18 short
  schema states and all 19 short GBNF depth-1 states.
- A 17-state sample of each 200-token walk matched with no mismatches.
- Focused grammar/token tests: 54 passed.

## Not covered

- No GPU structured-output exactness run (§39) and no same-document
  temperature-0 generation comparison; A4 remains open.
- No unconstrained decode A/B was run, so the 1% regression condition in A5
  is not claimed.
- The first start-state build is several seconds for `json_schema`; only the
  per-token gate is claimed.
- The whitespace stall (A6) is unchanged and was not measured here.
