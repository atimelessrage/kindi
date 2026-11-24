# REQ-0030 native-CPU-path profile (2026-11-24)

REQ-0030 A1 asked for a profile of real request work on the 2B/9B GPU paths and
the native CPU path before any native port. The GPU half landed 2026-11-22; the
CPU half was still unmeasured. This page is that measurement, on the 2B with
`KINDI_DEVICE=cpu` (i7-8086K, CPU engine, pack `bonsai-2b`, ctx 24,576).
Run script: [profile.py](profile.py); raw numbers: [profile.json](profile.json).

The server run (`kindi serve`, prewarm ready, streaming with usage,
`max_tokens=24`, `temperature=0`, a 35-token prompt — the messages render
shorter than the GPU profile's 49, recorded as `prompt_tokens: 35`) gives:

| request (CPU engine) | TTFT | content | wall |
|---|---|---|---|
| unconstrained, warm median (3 runs) | 0.140 s | 0.154 s / 8 chunks | 0.321 s |
| `json_schema`, first | 0.175 s | 0.735 s / 24 chunks | 0.910 s |
| `json_schema`, warm median (3 runs) | 0.147 s | 0.729 s | 0.878 s |

Unconstrained decode here is 8 content chunks (the model stops early), so the
per-token comparison is structured 0.729/24 ≈ 30 ms/token against
unconstrained 0.154/8 ≈ 19 ms/token: about **11 ms/token of mask cost**, which
is what the in-process mask rows below predict.

## Components (real pack, 248,320-entry vocabulary)

Tokenizer (`kindi/tokenizer.py`, `chat.render` prompts):

| prompt | tokens | warm encode |
|---|---|---|
| API request | 28 | 0.034 ms |
| repeated prose | 451 | 0.62 ms |
| repeated prose | 1,351 | 1.87 ms |
| repeated prose | 3,001 | 4.15 ms |

`decode_bytes` over 2,000 ids: 1.27 ms (0.0006 ms/token). Streaming framing
(`json.dumps` of one chunk): 0.0027 ms/chunk.

Constrained mask (steady state, `tools/grammar_cost.py` walk over the
22-token and 200-token documents):

| document | first mask | miss median | miss max | mean/token |
|---|---|---|---|---|
| short (22 tokens) | 1,988 ms | 6.0 ms | 157 ms | 11.2 ms |
| long (200 tokens) | 2.2 ms | 4.8 ms | 12.4 ms | 1.1 ms |

The 1,988 ms first mask is the one-time table/trie build in a fresh process;
the server prewarms it at startup (this run's server logged `structured-output
vocabulary ready in 3.1s`), so it does not land on the first structured
request — the server's structured TTFT (0.147-0.175 s) confirms that.

## Verdict (REQ-0030 candidate list, CPU path)

- **Constrained mask: keep as the only qualifying candidate.** It clears the
  A1 1 ms/token threshold on the CPU path as it does on the GPU path
  (11.2 ms/token short, 1.1 ms/token long), and the server row shows the same
  ~11 ms/token end to end. Any native work must cover the steady mask and the
  startup build; A6 is still not run.
- **BPE tokenizer: remove from the first native set** (0.034 ms for the API
  request; under 1 ms/token at every measured length).
- **Stream framing / JSON: remove** (0.0027 ms/chunk).

Also fixed in passing: `tools/grammar_cost.py`'s `walk()` used `state in
cache`, which broke when the cache became the bounded `StateMaskCache` LRU —
hit detection now reads the cache's miss counter, and the harness runs again
(short rows: 219-237 ms filter over 22 tokens, documents accepted).
