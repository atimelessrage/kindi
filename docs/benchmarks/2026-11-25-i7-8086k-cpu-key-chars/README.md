# The constrained mask: an exact accepted-character skip, and the memo it replaced

Host: i7-8086K, Windows, Python 3.13, no GPU. Pack: `bonsai-2b` (248,320-entry vocabulary).
Date: 2026-11-25. Grammar: the 22-token `json_schema` document from `tools/grammar_cost.py`.
No model is loaded for the mask rows; the served rows run the CPU engine.

## Why

The 2026-11-25 trie-walk round cut the walk with a per-class `character -> next state` memo and
measured 1.486x on a mask miss. That memo was withdrawn the same day: the states it stored embed the
schema stack, so once warm, a state of one class answered with another state's stack. On this
vocabulary, document states returned masks that the full sweep disagreed with - some tokens dropped,
some accepted that the automaton rejects. A mask is only useful if it is exact, so the memo is gone
and the walk needed a different cut.

## The cut: skip the characters the state cannot accept

`JsonGrammar._accepted_chars(state)` returns a **superset** of the characters the state can accept,
read off the same fields `_step` reads: the schema stack for a key (`okey`), the surviving terminals
for a keyword (`kw`), the target node for a value slot, and fixed sets for the structural modes
(`objk*`, `otail`, `atail`, `ocolon`, whitespace, number and string-escape characters). Where the
answer is not analytic (`str`/`akey` string bodies), it returns `None` and the walk keeps its old
path - those states are already covered by the safe-set pruning from the 2026-11-24 round.

`TokenClassTable` takes the hook as `state_chars` and, at a trie node, looks the accepted characters
up in the node's children (`children[ch]` for each accepted `ch`) instead of asking the automaton
about every distinct first character in the vocabulary. Over-reporting is harmless - an extra
character costs one step - so the hook can be a cheap local computation. The walk's pruning is
unchanged; this only removes questions.

## Results

In-process A/B (`mask_ab.py`), the same table with and without the hook, over the document's 23
states, three rounds, alternating, with the grammar's per-state mask cache off:

| quantity | without the hook | with the skip |
|---|---:|---:|
| median mask miss | 4.923 ms | 0.119 ms |
| automaton steps per miss | 8,303 | 38 |
| all 23 first-visit masks | 121.1 ms | 24.9 ms |

The median miss is **41.37x** cheaper and the whole document's first-visit masks **4.863x**
(121.1 -> 24.9 ms). Every mask is identical to the table without the hook, and both are identical
to the pre-token-class `_vocab_allowed_scan` over all 248,320 entries on all 23 states
(`masks_identical_to_exact` and `full_scan_identical` in `mask.json`).

Served A/B (`ab_summary.py` over `ab-old*.json`/`ab-new*.json`): three interleaved pairs of fresh CPU
servers, the `old` arm running the modules from the parent commit, ten runs per shape, 8-token
replies, temperature 0:

| arm | structured ms/token | constrained gap | unconstrained control |
|---|---:|---:|---:|
| old (median of 3) | 23.11 | 3.83 | 19.18 |
| new (median of 3) | 20.82 | 1.65 | 19.17 |

Structured replies are **9.9%** cheaper and the constrained gap **56.9%** smaller, while the
unconstrained control is flat (-0.1%). The gap - what the requirement owns - is now 1.65 ms/token.

## What this shows

- The remaining per-token mask cost was one question per distinct first character per trie node, and
  it is removable exactly, in Python, without a class memo: 8,303 -> 38 automaton steps per miss,
  41.37x on the median miss.
- The mask is exact where it matters most: equal to the pre-token-class sweep on every state of the
  document, not only on first visits.
- End to end, a structured 2B CPU reply gets 9.9% cheaper and the constrained overhead 56.9%
  smaller, with the unconstrained control flat.

## What this does not show

- **REQ-0030 A6's end-to-end half is now met by the Python path, so no native variant was started.**
  The mask half (41.37x against a 1.25x bar) and the end-to-end half (9.9% against 5%) both pass,
  with the unconstrained control flat (-0.1%, inside the 1% allowance). A native mask would have to
  beat 1.65 ms/token of a structured reply to clear the same 5% bar; nothing was built or measured
  for one.
- GBNF is untouched: its states are position sets, `state_chars` is a JSON-schema hook, and the GBNF
  rows of the 2026-11-25 walk page stand as measured.
- No GPU path (no CUDA device here), one context size, greedy decoding only, and each served arm
  runs the server's usual warm-up requests before measurement.
- The `_accepted_chars` sets are *supersets*: they skip only characters the automaton would reject.
  The property is enforced by `tests/test_token_classes.py` (superset check against `_step` on every
  reachable state, mask equality with and without the hook, and a warm-table sweep comparison that
  fails on the commit that carried the memo), not by construction.

## Reproduce

```sh
python docs/benchmarks/2026-11-25-i7-8086k-cpu-key-chars/mask_ab.py --out mask.json --rounds 3 --scan-states 23
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-2b serve --host 127.0.0.1 --port 8099
python docs/benchmarks/2026-11-24-i7-8086k-cpu-mask-fast/measure.py --url http://127.0.0.1:8099/v1 \
    --label new --runs 10 --max-tokens 8 --skip-filter
python docs/benchmarks/2026-11-25-i7-8086k-cpu-key-chars/ab_summary.py
```

For the `old` arm, restore `kindi/token_classes.py` and `kindi/jsongrammar.py` from the parent commit
before starting that server; `ab_summary.py` reads whichever `ab-old*.json`/`ab-new*.json` are
present. `mask.json` is the mask A/B, `ab-summary.json` the served medians.
