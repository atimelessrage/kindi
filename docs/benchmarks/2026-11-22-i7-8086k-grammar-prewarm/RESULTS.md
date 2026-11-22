# Structured-output prewarm and token-class mask (REQ-0027), 2B, 2026-11-22

Host: i7-8086K, Windows 10. Pack: `bonsai-2b`, **248,320 vocabulary entries**. Greedy, one
22-token document and the 200-token long document from `tools/grammar_cost.py`. This is a CPU
filter measurement: CUDA was unavailable and no GPU decode or exactness is claimed.

Machine-readable artifact: [grammar-mask.json](grammar-mask.json).

## What changed

- `kindi/token_classes.prewarm(texts)` builds the process-shared vocabulary index, the character
  trie, every node's character set and every node's id set in one call. `kindi serve` starts it in
  a daemon thread after the engine loads (`KINDI_PREWARM_GRAMMAR=0` disables it).
- The trie is now the default path for narrow safe character sets (the whitespace before a JSON
  value, digits, GBNF classes); broad sets such as a JSON string body still use the one-slice safe
  set. The vocabulary index keeps per-token integer character masks and builds the trie's
  descendant id/character fields lazily, so a cold request that is not prewarmed no longer copies
  every descendant into every ancestor during construction.
- The `JsonGrammar` and `Grammar` paths now both use the trie; no language heuristic changed.

## REQ-0027 gate rows

Commands:

```sh
python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b
python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b --doc long
```

| document | constraint | tokens | misses | filter total | per token | gate |
|---|---|---:|---:|---:|---:|---:|
| 22-token | `json_schema` | 22 | 17 | 340 ms | **15 ms** | <= 23 ms |
| 22-token | unrolled `json.gbnf` depth 1 (1,737 states) | 22 | 18 | 2,724 ms | **124 ms** | <= 272 ms |
| 200-token | `json_schema` | 200 | 43 | 632 ms | **3 ms** | <= 6 ms |
| 200-token | ordered-record GBNF (831 states) | 200 | 55 | 3,951 ms | **20 ms** | <= 46 ms |

All four rows clear the 10x gate. The short-document baselines were 231 and 2,720 ms/token; the
long-document baselines were 57 and 463 ms/token. The previous token-class run (2026-11-20)
measured 15/212 ms/token short and 3/30 long; this run's 124/20 is the current tree.

The short GBNF start row is 1,532 ms because it includes the one-time trie build; with the server
prewarm that build is moved to startup. The gate's per-token rows exclude it.

## Cold start

`tools/grammar_cost.py --repeat` builds the token-text list once, then runs three grammar
instances. The prewarm timings are three fresh processes; values are medians.

| case | first mask |
|---|---:|
| first structured request, no prewarm | 2,350 ms |
| repeat request, shared index | 2.5 ms |
| after `prewarm(texts)` | **20.0 ms** |

`prewarm(texts)` itself costs **2,475 ms** (runs: 2,473 / 2,475 / 2,571 ms), because it builds the
index, trie, character sets and id sets. The first mask after it is 20.0 ms (19.6 / 20.0 / 20.0).
The total is therefore not smaller than the lazy build; the win is that `kindi serve` runs it while
the server is starting, not inside the first structured request. A request that arrives before the
thread finishes still waits on the same index lock.

The prewarm covers the dense `qwen35`/`bonsai27` and `gemma4` paths. Lightning and Nanbeige build
their own token-text lists and are unchanged. The server cold first-token latency was not
re-measured: the earlier 2B/9B server profile measured 4.6-4.7 s before this change, and CUDA is
not available on the current host. The background thread's interaction with an immediate first
request is therefore reasoned about, not measured end to end.

## Semantics and tests

- `tests/test_server.py` starts the prewarm on a stub tokenizer and checks the shared trie exists;
  it also checks `KINDI_PREWARM_GRAMMAR=0` skips it.
- `tests/test_token_classes.py` checks that prewarm fills the lazy fields and that the forced exact
  first-character path still matches the sweep; the existing adversarial set-equality tests cover
  the trie path.
- Focused grammar/server run after the change: **149 passed**. Full CPU suite and quality gates are
  recorded in the commit.

## Not verified

- No GPU structured-output exactness (REQ-0027 A4) or unconstrained-decode A/B (A5).
- No server end-to-end cold TTFT on this host.
- The prewarm's memory footprint was not sampled; it retains the character sets and id sets for the
  active vocabulary, bounded by the two-vocabulary index cache.
