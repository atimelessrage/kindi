# What constrained decoding costs per token: unrolled GBNF vs json_schema (2B, 2026-11-17)

Host: i7-8086K + RTX 4060 Ti 8 GB (8,188 MiB). Pack: `bonsai-2b`
(`~/.kindi/packs/bonsai-2b`), **248,320 vocabulary entries**. Greedy (temperature 0).
Machine-readable artifact: [grammar-cost.json](grammar-cost.json).

The question was whether the unrolled recursive grammar is affordable: the canonical grammar is
said to be "4,000 states at depth 1" and the vocabulary filter walks the states every step. Two
things turned out to be different from that description, both measured below.

## 1. End to end through the server, same prompt and same 49-character answer

One 2B server (dense plan engine, `spec: 0`), `max_tokens 20`, the request body switched only in
`response_format`:

| response_format | wall | generation | tok/s | content |
|---|---|---|---|---|
| none | 0.34 s | 0.09 s | 234.8 | same 49 chars |
| `json_schema` (3 properties) | 4.49 s | 4.12 s | 4.9 | same 49 chars |
| `{"type": "grammar"}` — llama.cpp `json.gbnf`, `KINDI_GBNF_MAX_DEPTH=1` | **47.41 s** | 47.24 s | **0.42** | same 49 chars |

So the unrolled grammar costs **2.36 s per generated token** here against 2.8 ms of decode: the
2B's 235 tok/s becomes 0.42 tok/s. `json_schema` costs ~0.21 s per token on the same vocabulary,
which is 12x cheaper than the grammar and still ~50x the unconstrained decoder. A first
`json_schema` run in the same session measured 3.78 s wall / 3.18 s generation (6.3 tok/s), so the
schema path itself varies with the states a document reaches; the grammar row reproduces within a
few percent of its CPU prediction (below).

Without `KINDI_GBNF_MAX_DEPTH` the recursive grammar is refused with an enveloped 400, which is the
documented behaviour and not a cost result.

## 2. Where the time goes — the states are not the multiplier

The filter is `Grammar.vocab_allowed(state, texts)`: it enumerates **every vocabulary entry**,
advances the automaton over that entry's text, and caches the surviving ids **per automaton state**.
So a request pays one vocabulary sweep per *new* state it reaches — about 0.8 of those per token in
this document — and a sweep of 248,320 entries is the unit of cost. Total automaton size barely
matters:

| grammar | states | misses / 22 tokens | filter total | per token |
|---|---|---|---|---|
| `json.gbnf` depth 1 | 1,737 | 18 | 59.8 s | 2,720 ms |
| `json.gbnf` depth 2 | 3,195 | 18 | 62.6 s | 2,847 ms |
| `json.gbnf` depth 3 | 9,483 | 18 | 62.0 s | 2,820 ms |
| record grammar, 8 enumerated keys | 1,061 | 19 | 45.3 s | 2,058 ms |
| `json_schema`, 3 properties | — | 17 | 5.1 s | 231 ms |

**Run-to-run:** re-running both harnesses after they moved into `tools/` reproduced the state counts
and miss counts exactly and the grammar row within 5% (2,567 vs 2,720 ms/token; 2.38 vs 2.36 s per
server token, 0.4 tok/s), while the schema row is noisier — 198 vs 231 ms/token and 6.0 vs 4.9
tok/s across the two server sessions — because it pays fewer, smaller sweeps. The grammar/schema
ratio held at 11-13x in every run, which is the decision-relevant part.

The 1,737 → 9,483 state range moves the per-token cost by 3%, so a grammar at 4,000 states (between
the depth-2 and depth-3 rows) lands at **~2.8 s per token** as well: the premise's 4,000 states
neither explains nor bounds the cost. What does explain it is the per-sweep price: a GBNF sweep over
a JSON-shaped grammar is **~3.5 s** (median miss 3,491 ms at depth 1, max 5,126 ms), because its
`string`/`ws` classes accept almost every byte, so most vocabulary entries advance several
characters before failing; the schema automaton rejects most entries on their first character and
sweeps in **~250 ms**. The start-of-document sweep is the cheap one (0.2-0.6 s) — the expensive
states are the open-string and whitespace states in the middle of a document.

Compilation is not the issue: 3-17 ms per grammar (1,737-9,483 states), against the multi-second
runtime cost.

## 3. At a realistic reply length: 200 tokens, both constraints accept

The 22-token row above is a floor from one short document. The extension uses a **200-token document**
(measured with the 2B tokenizer) and a *pair* of constraints that describe exactly it: an ordered flat
record as GBNF (keys, types and order fixed, 831 states) and the same shape as a JSON schema. Both
accept the document.

| constraint | tokens | cache misses | filter total | per token | quarters |
|---|---|---|---|---|---|
| GBNF ordered record (831 states) | 200 | 55 | 92.6 s | **463 ms** | 573 / 575 / 178 / 526 |
| `json_schema`, same document | 200 | 43 | 11.5 s | **57 ms** | 94 / 49 / 28 / 58 |

**The cost does not grow with the document; it falls.** At 22 tokens the grammar cost 2,720 ms per
token and the schema 231 ms; at 200 tokens they cost 463 ms and 57 ms, an **8x** ratio instead of
12x. The mechanism is the same cache that sets the short-document price: 145 of the 200 grammar
tokens and 157 of the 200 schema tokens landed on automaton states already swept, because a record's
repeated array elements revisit the same state sets. The quarters show it directly — the repetitive
`items`/`tags` stretch runs at 178 ms per token while the unique `value` sentence returns to ~520 ms.

What does not fall is the absolute price of the *unique* part of a reply: the end-to-end
whitespace-free run below measures 909-959 ms per token on 79 tokens of fresh content.

### End to end at `max_tokens 200`, and the whitespace trap

| response_format | tokens | generation | chars | outcome |
|---|---|---|---|---|
| none | 148 | 0.65 s (226.8 tok/s) | 354 | normal reply |
| `json_schema` (whitespace permitted) | 200 | 3.37 s | 9 | **stalled**: all budget on whitespace after the first colon |
| GBNF ordered record (whitespace permitted) | 200 | 6.60 s | 9 | **same stall** |
| GBNF ordered record, **whitespace-free** | 79 | 75.76 s (1.0 tok/s) | 223 | progresses, valid compact JSON |

The stall is not prompt-side: asking for compact JSON with no spaces changed nothing. Both
automata permit `[ \t\n\r]*` between tokens, and the 2B's greedy path spends the entire budget emitting
whitespace — 200 tokens for 9 characters. Note what that does to a throughput number: the stalled
runs look *cheap* per token (17 and 33 ms) only because the repeated whitespace state is cached after
the first sweep, so "tok/s" is high while nothing is produced. A long constrained reply needs a
constraint that bounds or omits whitespace; with that, the same grammar produces 223 characters of
valid compact JSON at 1.0 tok/s.

## Verdict

Do not turn the unrolled grammar on for this path: even after the per-token price falls with length
(2,720 → 463 ms/token on a 200-token document), it is ~8x the schema path and ~165x the 2B's 2.8 ms
decode, and a whitespace-permitting version does not finish a long reply at all. `json_schema`
remains the correct constraint for recursive JSON, but it is not free either: 231 ms/token at 22
tokens and 57 ms/token at 200 on a 248k vocabulary, with the same whitespace trap. Both want the same
fix, which is **not** measured here: compute the mask per state once on the card (or from a
token-class table) instead of sweeping the vocabulary in Python per new state, and bound whitespace in
the constraints.

## Reproduce

The two harnesses are tracked, so this page's numbers can be re-derived from a clean checkout with
the pack as the only local input:

```sh
python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b                    # 22-token rows (2-3 min)
python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b --doc long          # 200-token rows (~2 min)

KINDI_GBNF_MAX_DEPTH=1 python -m kindi --pack ~/.kindi/packs/bonsai-2b serve \
    --host 127.0.0.1 --port 8102                                                 # server rows
python tools/grammar_cost_serve.py http://127.0.0.1:8102                         # 20-token reply (~1 min)
python tools/grammar_cost_serve.py http://127.0.0.1:8102 --doc long              # the whitespace stall (~10 s)
python tools/grammar_cost_serve.py http://127.0.0.1:8102 --doc long --tight      # 79 tokens at 1.0 tok/s (~80 s)
```

`tools/grammar_cost.py` prints one row per grammar (states, compile time, misses, filter total,
ms/token, per-sweep median and max, and per-quarter ms/token for a long document) and exits non-zero
if a grammar refuses the document, so a rejected document can never be reported as a number.
`--doc long` walks the 200-token document through the ordered-record GBNF and its equivalent schema;
`tools/grammar_cost_serve.py --doc long --tight` is the whitespace-free variant that makes a long
constrained reply progress.

## What this does not cover

- One prompt family and one document shape: an ordered flat record. A recursive document adds states
  and depth, and the depth-1 unroll is exactly where a deeper document is refused; the length
  question answered here is the *flat* one. The 200-token CPU rows use the same document for both
  constraints, and the end-to-end rows use the model's own output, which is shorter (79 tokens) under
  the whitespace-free grammar.
- The tensor-core plan engine (`KINDI_TC=1`) was not measured: constrained decoding is CPU-side
  vocabulary filtering, so the engine choice changes only the 2.8 ms decode term it is swamped by.
- No other pack, no other card, no sampled peak (the filter allocates nothing on the card).
