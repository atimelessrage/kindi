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

## Verdict

Do not turn the unrolled grammar on for this path: at ~2.4 s per token it is 560x the 2B's decode
cost, and ~12x the schema path. `json_schema` is the correct constraint for recursive JSON here, but
its own ~0.21 s per token on a 248k vocabulary is a second, separate problem worth a road map entry
of its own (a per-state mask computed once on the card / a token-class table, instead of a Python
sweep per new state, is the obvious direction and is **not** measured here).

## Reproduce

```sh
python bench-logs/gbnf-cost.py                       # CPU: per-state sweeps, both grammars
/python -m kindi --pack ~/.kindi/packs/bonsai-2b serve --host 127.0.0.1 --port 8102   # with KINDI_GBNF_MAX_DEPTH=1
python bench-logs/gbnf-serve.py http://127.0.0.1:8102
```

Both scripts live in the gitignored `bench-logs/` (local evidence). Their content is reproducible
from the commands above; only this page and `grammar-cost.json` are tracked.

## What this does not cover

- The **long-document** rows of the same harness were rejected by the constraints — the schema
  refuses keys it does not declare and the depth-1 grammar refuses deeper nesting — so every number
  above comes from one 22-token document (CPU) and one 20-token reply (server). Longer documents
  would add states and therefore sweeps, so these numbers are a floor for a longer reply, not a
  bound.
- The tensor-core plan engine (`KINDI_TC=1`) was not measured: constrained decoding is CPU-side
  vocabulary filtering, so the engine choice changes only the 2.8 ms decode term it is swamped by.
- No other pack, no other card, no sampled peak (the filter allocates nothing on the card).
