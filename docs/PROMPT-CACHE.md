# The prompt cache

Kindi keeps the work it did for one request and uses it again for the next request. When a new request starts with
the same tokens as an earlier one, Kindi does not read those tokens again. It reads only the new part.

The cache does not change what the model sees. With the Bonsai models, the replies in the examples below were the
same as the replies from a server with the cache off. With the cache, Kindi reads the new part of a prompt in other
batch sizes, so the arithmetic can round a little differently. On a near tie between two tokens, a reply can then
take the other token. In 295 turns of real agent conversations with Nemotron at temperature 0, 283 replies were the
same with the cache on and off, and each of the other 12 took another reasonable path.

## When the cache helps

| Workload | What stays the same | What Kindi reads again |
|---|---|---|
| A chat or an agent | the system prompt and the earlier turns | only the new user message |
| Regenerate | the full prompt | only the last token |
| RAG: many questions about one document | the document | only the question |
| Two or more agents that take turns | each agent's own conversation | only each agent's new message |

## Results: Nemotron 3.5 Bolt E4B on a GTX 1080

These results come from `examples/prompt_cache_bench.py`. We ran it two times against one server: one time with the
cache off (`KINDI_G4_PROMPT_CACHE=0`) and one time with the cache on (the default). The replies were the same in
all 12 requests.

| Workload | Request | Prompt tokens | Tokens from the cache | Cache off | Cache on | Faster by |
|---|---|---|---|---|---|---|
| Agent, 2.2K-token system prompt | turn 1 | 2,197 | 0 | 4.45 s | 4.42 s | 1.0x |
| | turn 2 | 2,236 | 2,223 | 2.76 s | 0.67 s | 4.1x |
| | turn 3 | 2,276 | 2,259 | 2.96 s | 0.86 s | 3.4x |
| Regenerate a 600-token answer | first | 2,203 | 0 | 2.66 s | 2.75 s | 1.0x |
| | again | 2,203 | 2,202 | 2.69 s | 0.43 s | 6.3x |
| RAG, a 9.4K-token document | question 1 | 9,390 | 0 | 10.70 s | 10.89 s | 1.0x |
| | question 2 | 9,389 | 9,375 | 11.03 s | 0.70 s | 15.8x |
| | question 3 | 9,392 | 9,375 | 10.71 s | 0.42 s | 25.5x |
| Two agents that take turns | A, turn 1 | 2,197 | 0 | 2.56 s | 2.77 s | 0.9x |
| | B, turn 1 | 2,199 | 0 | 2.58 s | 2.69 s | 1.0x |
| | A, turn 2 | 2,219 | 2,207 | 2.58 s | 0.58 s | 4.4x |
| | B, turn 2 | 2,223 | 2,211 | 2.57 s | 0.53 s | 4.8x |

The times are for the full request: reading the prompt and writing up to 60 tokens (400 for RAG, 600 for
regenerate). A first request is up to 8% slower with the cache on, because Kindi copies the cache to host memory
while it reads the prompt.

## Results: Ternary-Bonsai-2 27B on a GTX 1080

The same script and the same method, with `KINDI_PROMPT_CACHE=0` for the run without the cache. The replies were the
same in all 12 requests.

| Workload | Request | Prompt tokens | Tokens from the cache | Cache off | Cache on | Faster by |
|---|---|---|---|---|---|---|
| Agent, 2.2K-token system prompt | turn 1 | 2,205 | 0 | 14.53 s | 14.78 s | 1.0x |
| | turn 2 | 2,255 | 2,234 | 15.33 s | 2.52 s | 6.1x |
| | turn 3 | 2,309 | 2,284 | 15.65 s | 2.93 s | 5.3x |
| Regenerate a 600-token answer | first | 2,212 | 0 | 15.28 s | 16.01 s | 1.0x |
| | again | 2,212 | 2,205 | 15.47 s | 1.85 s | 8.4x |
| RAG, a 9K-token document | question 1 | 8,957 | 0 | 47.28 s | 48.05 s | 1.0x |
| | question 2 | 8,956 | 6,912 | 49.42 s | 14.53 s | 3.4x |
| | question 3 | 8,958 | 6,912 | 47.19 s | 11.46 s | 4.1x |
| Two agents that take turns | A, turn 1 | 2,204 | 0 | 14.17 s | 14.93 s | 0.9x |
| | B, turn 1 | 2,207 | 0 | 14.60 s | 15.27 s | 1.0x |
| | A, turn 2 | 2,229 | 2,208 | 14.68 s | 2.61 s | 5.6x |
| | B, turn 2 | 2,244 | 2,224 | 14.80 s | 2.79 s | 5.3x |

In the RAG rows, the 27B continues from its checkpoint at token 6,912, so it reads about 2,000 tokens of the
document again. Its checkpoints are further apart deep in a long prompt, because each one holds the full recurrent
state.

## How it works

Kindi keeps three kinds of saved state.

1. **The conversation on the card.** The keys and values of every token stay on the GPU. A new request that starts
   with the same tokens continues from the end of the shared part.
2. **Checkpoints.** Some attention layers keep only the last part of the text (a sliding window). When a request
   goes back further than that window, Kindi uses a checkpoint: a copy of those layers that it made in host memory
   every 512 tokens of a prompt and at the end of each prompt. The request continues from the last checkpoint
   before the point where it is different.
3. **Parked conversations.** When a request shares less than half of the conversation on the card, that
   conversation moves to host memory. When its next turn comes, it moves back to the card. Thus agents that take
   turns each keep their own cache.

The Bonsai models use the same three kinds of saved state. Their checkpoints also hold the recurrent state of the
Gated-DeltaNet layers, and they can save the start of a conversation to disk, so that it is available after a
restart. See [HOW-IT-WORKS.md](HOW-IT-WORKS.md).

A tandem CPU+GPU split uses the same cache through `TandemRunner`: a checkpoint contains both owners' recurrent
state; each engine keeps its K/V arrays in place. Saved openings also store both owners' K/V and are fingerprinted by the placement, layer ranges and
CPU ISA/thread count, so a state made by one split is never shared with another.

On 2026-11-12, a real 2B tandem check used a 320-token prefix request, a 256-token branch, and a
new runner that loaded the branch's 256-token opening from disk in both placements. All cached and
`KINDI_PROMPT_CACHE=0` continuations were token-identical. The same check through `kindi serve`
saved a 344-token opening, restarted the server, and loaded it again: the response reported 344
cached tokens and matched the cache-off reply. The same harness then passed at 4K and 8K context
with 1,024- and 2,048-token checkpoints, reusing 4,080/8,176 tokens for extensions and restoring
3,072/6,144-token branches. The 9B pack then passed the 512-context check in both placements with
6 CPU layers: extension 320, branch and saved opening 256, all exact; see
[RESULTS-9B.md](../bench-logs/tandem-cache/RESULTS-9B.md). This run also fixed the CuPy K/V restore
in `Openings.load` (it now uses the existing `_to_dev` copy instead of a plain assignment). See
[bench-logs/tandem-cache/RESULTS.md](../bench-logs/tandem-cache/RESULTS.md) and
[LONG-CONTEXT-RESULTS.md](../bench-logs/tandem-cache/LONG-CONTEXT-RESULTS.md).

## Invariants

These rules are why a resumed request is safe; break one and the cache can return wrong output.

- A saved opening is reused only when its engine fingerprint matches. The fingerprint covers the model path,
  K/V type, chunk/attention/tensor-core/speculation settings, head mode and the code of every compiled kernel
  (all `*.cu`/`*.c` files in `kindi/`) plus the explicit Python arithmetic files.
- A checkpoint is a copy of the complete recurrent and attention state at a token boundary. Resume starts at
  the last checkpoint at or before the first differing token; the position-indexed K/V suffix is then valid.
- Parked conversations are moved whole (tokens, checkpoints, end state and K/V). They are not mixed with another
  conversation, and `park_max`/`checkpoint_budget_mb` may evict them but never truncate a live state.
- A damaged or foreign opening is ignored, not repaired in place. The cache must never resume a state made by a
  different engine configuration.
- Eviction drops the least-recently-used opening or the oldest checkpoint that is not needed by a live request.
  The final state after a request is never removed while that request is active.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `KINDI_G4_PROMPT_CACHE` | 1 | 0 turns the Nemotron prompt cache off |
| `KINDI_PROMPT_CACHE` | 1 | 0 turns the Bonsai prompt cache off |
| `KINDI_G4_CKPT_MAX` | 16 | the number of checkpoints kept for each conversation |
| `KINDI_G4_PARK_MAX` | 4 | the number of conversations kept in host memory |

Each parked conversation uses about 16 KB of host memory for each token, plus about 42 MB for each checkpoint.

The Gemma 4 host cache also enforces a 1 GiB pinned-memory budget (`G4Cache.DEFAULT_BUDGET_BYTES`). When a new
checkpoint or parked conversation would exceed it, the oldest parked conversation is dropped first, then the
oldest active checkpoint; an individual snapshot that cannot fit is not created. The cache logs the freed bytes
and exposes `budget_bytes`, `used_bytes`, `peak_bytes`, `evictions`, `evicted_bytes`, `checkpoints_dropped` and
`parked_dropped` counters through `G4Cache.metrics()` and the server's `/metrics` endpoint. The budget is separate
from the Bonsai `checkpoint_budget_mb` setting.
