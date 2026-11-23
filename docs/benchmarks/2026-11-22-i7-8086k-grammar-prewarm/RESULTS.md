# Structured-output prewarm and token-class mask (REQ-0027), 2B, 2026-11-22

Host: i7-8086K, Windows 10. Pack: `bonsai-2b`, **248,320 vocabulary entries**. Greedy, one
22-token document and the 200-token long document from `tools/grammar_cost.py`. This is a CPU
filter measurement: CUDA was unavailable and no GPU decode or exactness is claimed.

Machine-readable artifacts: [grammar-mask.json](grammar-mask.json) (mask rows) and
[server-ttft-cpu-20261123.json](server-ttft-cpu-20261123.json) (server cold TTFT).

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
their own token-text lists and are unchanged. The server cold first-token latency is measured on
the CPU engine below; the earlier 2B/9B server profile's 4.6-4.7 s is the pre-change GPU number,
and CUDA is still unavailable on this host.

## Server cold TTFT (2026-11-23, CPU)

CUDA is still unavailable here (`cudaErrorNoDevice`), so the end-to-end request path was measured
on the CPU engine with the same 2B pack and the short `json_schema` request from
`tools/grammar_cost.py` (49 prompt tokens, `max_tokens` 2, `stream: true`). Each run is a fresh
`kindi serve` process; TTFT is the time from sending the request to the first content delta, and
readiness is when `/health` first answered.

```sh
python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 4096 serve --port PORT   # prewarm on (default)
KINDI_PREWARM_GRAMMAR=0 python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 4096 serve --port PORT
```

All runs used `KINDI_DEVICE=cpu`.

| case (3 fresh processes each) | readiness, median | first structured request TTFT | range |
|---|---:|---:|---:|
| prewarm on, request sent when the server is ready | 26.41 s | **0.989 s** | 0.915-1.134 s |
| prewarm on, request sent at the first `/health` answer | 26.33 s | 0.927 s | 0.909-1.035 s |
| `KINDI_PREWARM_GRAMMAR=0` | 23.21 s | **3.603 s** | 3.603-3.637 s |

The request path is **2.61 s faster (72.6%)** with the prewarm ready. The prewarm logged
`structured-output vocabulary ready in 3.1s` (3.2 s once) in all six prewarm runs, and readiness
moved from 23.21 s to 26.41 s: the startup absorbs the build, so the total from process start to
first token is 27.5 s with the prewarm and 26.8 s without, matching the "total is not smaller"
conclusion above. On this CPU host the ready line preceded the first successful `/health` poll in
every prewarm run, so the "request arrives while the thread is building" lock wait did not occur
here; the two prewarm rows differ by less than their ranges. No GPU server was measured.

The per-run evidence is [server-ttft-cpu-20261123.json](server-ttft-cpu-20261123.json) (command,
request shape, runs, medians); the server logs stay in the gitignored `bench-logs/prewarm-ttft/`.

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
- No GPU server end-to-end cold TTFT (CUDA unavailable); the CPU server measurement above is the
  only end-to-end number.
- The prewarm's memory footprint was not sampled; it retains the character sets and id sets for the
  active vocabulary, bounded by the two-vocabulary index cache.
