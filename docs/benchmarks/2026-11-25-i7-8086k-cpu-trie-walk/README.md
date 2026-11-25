# The constrained mask's trie walk: what it spends, and an exact Python cut

Host: i7-8086K, Windows, Python 3.13, no GPU. Pack: `bonsai-2b` (248,320-entry vocabulary).
Date: 2026-11-25. Grammar: the 22-token `json_schema` document and the ordered-record GBNF from
`tools/grammar_cost.py`. No model is loaded: this measures the mask alone.

## Why

The 2026-11-24 refinement left `TokenClassTable._trie_allowed` as the largest part of a mask miss.
"Trie walk" was an assumption, not a measurement, so this run attributes the time first and only
then changes code.

## What the walk spends

`profile_walk.py` replays the document through the schema, times every state's first-visit mask with
the grammar's per-state cache off, and counts what the walk asks the automaton. Medians over five
passes, three runs:

| quantity (`json_schema`, 22-token document) | value |
|---|---|
| automaton steps per miss | 8,303 |
| distinct states the walk steps from, per miss | 2 |
| of the root fan-out, characters rejected | 91% |
| trie nodes visited, end-of-document state | 761 |
| share of a miss in `jsongrammar._step` + `_key_char` (cProfile tottime) | ~60% |

The walk asks one automaton question per distinct first character of the vocabulary - 8,219 of them
on this pack - and keeps a handful: for the `trail` state 8,219 root edges are stepped and 760 nodes
survive. The cost is that transition count, not the trie, the id sort or the id mask.

## The cut: a per-class edge memo

`TokenClassTable` now keeps a `character -> next state` memo per *character class*: a key a grammar
returns only when every state of that class answers every character identically
(`kindi/jsongrammar.py:_char_class`). A class is memoized from its **second** state, so a request
that never repeats a class pays exactly what it paid before. `json_schema` returns a key for the
mode-stable classes (`trail`, `str`, `akey`, `sx`, `su`, `ocolon`, `atail`, `objka`, `objkb`) and for
`val`/`oval`/`aval` keyed by their target schema node. `okey`, `objk`, `otail`, `num`/`int` and `kw`
are deliberately excluded: their key would have to carry the key buffer, the digit buffer or the
stack, so a walk makes a fresh class per accepted character - churn, not reuse (measured: memoizing
them cost about 6% of a first pass and saved nothing on the next). The memos are an LRU of
`_ACCEPT_MAP_LIMIT` (16) classes plus a 64-key "seen once" set, so REQ-0027 A5's memory bound still
holds. GBNF gets no classes (a state is a position set), so its row is unchanged.

## Results

In-process A/B, alternating the memo table and a table built without `char_class` over the same
states, five passes per run, three runs:

| grammar | median miss, exact | median miss, memo | speedup | automaton steps/miss |
|---|---:|---:|---:|---:|
| `json_schema` | 4.02 ms | 2.70 ms | 1.486x | 8,303 → 309 |
| `gbnf` record | 4.90 ms | 4.92 ms | 0.995x | 8,424 → 8,424 |

Per 22-token document, summing every first-visit mask: warm passes cost **109.1 ms** exact against
**89.2 ms** with the memo (1.223x). The first pass is slower (109.1 ms exact against 124.8 ms with the memo filling) because that is
where the memos are built; later passes over the same classes are the 89.2 ms.

Served A/B, three interleaved pairs of fresh CPU servers (the `old` arm runs the modules restored
from `HEAD`; 10 runs per shape, 8-token replies, temperature 0, `measure.py` from the 2026-11-24
run, `--skip-filter`):

| arm | structured ms/token | unconstrained control | constrained gap |
|---|---:|---:|---:|
| old (median of 3) | 24.42 | 19.73 | 4.57 |
| new (median of 3) | 23.50 | 19.42 | 4.08 |

Structured replies got **3.8%** cheaper and the constrained gap **10.7%** smaller; the unconstrained
control also moved -1.6%, so the gap is the part this change owns. One `new` run (29.15 / 5.69) is an
outlier on both shapes and stays in the median.

## What this shows

- The walk's cost is the automaton transition count, not the trie, the sort or the id mask.
- An exact Python cut removes 96% of those transitions on the schema path (8,303 → 309 per miss),
  cuts the median mask miss by 1.486x, and leaves every mask identical: the oracle is the same table
  built without `char_class`, compared on document states, with the class-agreement invariant tested
  separately in `tests/test_token_classes.py`.
- End to end it is worth about 4% of a structured reply on this host.

## What this does not show

- **REQ-0030 A6 is not met.** The mask half passes (1.486x median miss against a 1.25x bar); the
  end-to-end half does not (3.8% against a 5% bar). No native variant was started, and no end-to-end
  speedup beyond this page is claimed.
- The GBNF path is unchanged (0.995x): its states are position sets, so no class is provable.
- The remaining cost is `okey`/`int`-style states, where acceptance depends on the key or digit
  buffer and every character still needs its own transition. A native transition function would have
  to cover that, which is a grammar port rather than a mask port.
- No GPU path (no CUDA device here), one context size, greedy decoding only, and every served arm
  includes three warm-up requests of both shapes before measurement, exactly as the 2026-11-24 page
  did.

## Reproduce

```sh
python docs/benchmarks/2026-11-25-i7-8086k-cpu-trie-walk/profile_walk.py --out walk.json --rounds 5
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-2b serve --host 127.0.0.1 --port 8099
python docs/benchmarks/2026-11-24-i7-8086k-cpu-mask-fast/measure.py --url http://127.0.0.1:8099/v1 \
    --label new --runs 10 --max-tokens 8 --skip-filter
```

`walk-1.json` to `walk-3.json` are the three profile runs, `summary.json` their medians, and
`ab-summary.json` the six served runs behind the served table.
