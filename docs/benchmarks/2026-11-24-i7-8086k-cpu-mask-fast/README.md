# The constrained mask and the constrained pick get cheaper (2B, CPU engine)

2026-11-24, i7-8086K, Windows 10, Python 3.13.6, `KINDI_DEVICE=cpu`, pack `bonsai-2b`
(248,320-token vocabulary). Before = `5f44622`; after = the change this page records
(`kindi/token_classes.py` and `kindi/sampling.py`).

Two defects, both exactness-preserving, both in the per-token constrained path:

1. **`_outside_ids_for` sorted the whole vocabulary twice per safe char set.** `np.setdiff1d` on
   `arange(vocab)` costs two `np.unique` sorts of 248,320 int64s. `all_ids` *is* `arange`, so the
   complement of the already-cached safe mask is the same array in one linear scan.
2. **`_safe_ids_for` scanned the vocabulary in Python.** The per-token character masks were a list
   of 248,320 Python ints, and every new safe char set ran a generator over all of them plus
   `np.fromiter`. The index now keeps, per character, the ids whose text contains it, so a safe
   mask is a handful of slices: clear the ids containing each character that is *not* safe. That
   removes the per-token mask objects too.
3. **`Sampler._constrained` built an id vector it did not need.** On the CPU engine the candidate
   ids are `arange(vocab)` with no `ylog_map`, so the old path built that arange, cast it to
   int64, and ran `np.isin` over the whole vocabulary to select the allowed logits. A boolean mask
   over `ylog` both selects the values and yields the ids (`flatnonzero`). The generic path stays
   for a mapped head and for a request with `frequency_penalty`/`presence_penalty`
   (`_masked_logits` applies no penalty, so those requests still go through `_logits`).

## What it cost before and after

`measure.py` posts the same prompt twice to a freshly started server on each arm - once with a
`json_schema` `response_format`, once without - and reports milliseconds per *content* token, so
the two arms have the same reply length (8 tokens, `temperature` 0). Three warm requests per shape,
then ten interleaved runs per shape.

| row | before (`5f44622`) | after | change |
|---|---|---|---|
| structured reply | 26.35 ms/token | **24.13 ms/token** | **-8.4%** |
| unconstrained reply | 20.03 ms/token | 19.79 ms/token | -1.2% (noise; unchanged code path) |
| constrained overhead (the gap) | 6.32 ms/token | **4.34 ms/token** | **-31%** |
| 22-token schema filter (in process) | 245.6 ms | **106.1 ms** | **-57%** |
| slowest single mask miss in that document | 154.6 ms | **14.9 ms** | -90% |

The filter row is the same document `tools/grammar_cost.py` walks (`--pack ~/.kindi/packs/bonsai-2b`,
22 tokens, 17 mask misses, 23 states); `measure.py --skip-server` reproduces it. The run-to-run spread
of the structured arm was 23.1-25.9 ms/token after and 25.3-31.0 before, so the medians above are the
honest comparison on this host, not a best-case pair.

## What this does not show

- Only the CPU engine is measured. On the GPU engines the same three functions run, but the mask was
  already the smaller half of a much faster token, and no GPU run happened here (no CUDA device on
  this host).
- `_trie_allowed` (the per-character automaton walk over the trie) is now the largest single part of
  a mask miss: 5.2-5.7 ms per call, 68 calls in a 24-token structured reply. Nothing in this change
  touches it, so a further 2x would have to come from there - which is the native-variant question
  REQ-0030 keeps open, not a free Python win.
- The 9B CPU path was not run; it shares every function changed here.
- Absolute milliseconds are this host's. The comparison is between two arms measured minutes apart
  under the same conditions, with the unconstrained arm as the control; that control moved 1.2%
  while the structured arm moved 8.4%.

## Reproduce

```sh
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-2b serve --host 127.0.0.1 --port 8099
python docs/benchmarks/2026-11-24-i7-8086k-cpu-mask-fast/measure.py --label after --url http://127.0.0.1:8099/v1
# the before arm: git checkout 5f44622 -- kindi/token_classes.py kindi/sampling.py, restart, rerun
```

`measure.json` holds both arms' raw per-run rows.
