# 27B CPU MTP head: the variant ladder (2026-11-25)

The head-side follow-up to the [head vs 2B drafter page](../2026-11-25-i7-8086k-27b-cpu-mtp/RESULTS.md):
which of the candidate fixes to `kindi/mtp_cpu.py` actually move the `blk.64` head's acceptance.
The page it follows measured the head as committed at 7/108 (6.5%); the variants below start
there and end at 15/108 (13.9%).

## Method

Loading the 27B costs about six minutes, so the ladder is one probe load per batch:
`tools/mtp_cpu_probe.py --pack PACK --gguf <the -mtp file> --commands cmds.jsonl --out out.jsonl`
loads the pack and the head once and runs every attempt in that process. One JSON attempt per
line: `{"variant": ..., "k": 4, "tokens": 16, "prompts": [...]}`. Five loads, 14 attempts:

| load | attempts | load time |
|---|---|---|
| `probe` | `eh` k=4, `he` k=4, `eh` k=1 | 369.9 s |
| `probe2` | `eh-normed` k=4, `eh-normed` k=1 | 415.0 s |
| `probe3` | `eh-normed-folded`, `eh-normed-rope1`, `eh-folded` (k=4) | 330.2 s |
| `probe4` | `eh-normed-seeded` k=4, `eh-normed` k=1 (second prompt only) | 362.0 s |
| `mtp-seedfull` | `eh-normed-seeded`, `eh-normed-seeded-full`, and the long prompt at window 64 and 132 | 380.9 s |

Same 27B pack (`bonsai25-27b`), same two short prompts (`In one sentence: what is a ternary
weight?`, 11 tokens; `2 + 2 = ? Answer with the number only.`, 12 tokens) unless noted, `k=4`,
16 tokens, ctx 2048, 6 threads, i7-8086K (`avx2-fma-f16c`). The long prompt in the last load
is a 128-token ternary-weights paragraph. A variant ending in `-full` also seeds the prompt's
last row; `"window": "prompt"` grows the head's K/V ring from the prompt; `--trace` writes
every head row's `u`/`out` (748 rows in the seedfull load: 388 seed, 360 draft).

Drafts are scored against the model's own greedy tokens and never substituted, so every
attempt is token-identical to plain greedy decode (`exact: true` on all 14 attempts in
`results.json`). The CPU engine verifies one row per pass, so a drafted run is strictly
slower: this measures acceptance only and makes no speedup claim. `results.json` is the
sanitized record of all five loads (machine-local paths removed; per-prompt draft and model
token ids kept where the probe recorded them).

## Result

The k=4 ladder on the two short prompts:

| variant | accepted | rate | depth [1..4] | what it changes |
|---|---|---|---|---|
| `eh` (as committed) | 9/108 | 8.3% | [1, 1, 4, 3] | raw residual into `eh_proj` |
| `he` | 0/108 | 0.0% | [0, 0, 0, 0] | concat order swapped, `[hidden, emb]` |
| `eh-normed` | 11/108 | 10.2% | [4, 3, 3, 1] | `rms(residual) * output_norm` - what the GPU head is fed |
| `eh-normed-folded` | 11/108 | 10.2% | [6, 2, 3, 0] | embedding row folded with the pack's signs (BTQ1/PTQ1_0 layout) |
| `eh-normed-rope1` | 11/108 | 10.2% | [4, 3, 3, 1] | RoPE at `p + 1` (the reference's MTP position for row `p`) |
| `eh-normed-seeded` | 15/108 | 13.9% | [3, 4, 4, 4] | window seeded with every **verified** row, not just the head's own |
| `eh-normed-seeded-full` | 15/108 | 13.9% | [3, 4, 4, 4] | also seeds the prompt's last row - the row the first draft recomputes, bit for bit |

Two switches move the number. The hidden is the output-normed one - what `kindi/mtp.py`'s
`XN` and llama.cpp's `graph_mtp` both read, and what this head now uses - and seeding the
window with the verified rows, what the reference's MTP cache holds, lifts 10.2% -> 13.9%
overall, 16.7% -> 22.2% on the second prompt (9/54 -> 12/54), with depth [2, 3, 4, 3].
Concat order collapses to zero, and the folded-embedding and RoPE-position variants match
`eh-normed` exactly, so neither was the bug.

The k=1 side: `eh` k=1 is 1/30 (3.3%) and `eh-normed` k=1 is 4/30 (13.3%) on both prompts
(4/15, 26.7%, on the second prompt alone in `probe4`). Includes `eh-folded` k=4 at 11/108
(10.2%), not quoted in `docs/MTP.md`.

## The whole prompt, and what the rows say

The seedfull load ran the two remaining candidates on the same prompts: seeding the prompt's
last row as well (`-full`) changed nothing - 15/108 again (3/54 + 12/54) - and the trace
shows why it cannot. With that row seeded, its `u` and `out` are bit-for-bit the first draft
row's (`max|du| = 0`, same prediction 628): the draft recomputes the same position from the
same token and hidden immediately after the seed, so the seed is overwritten before anything
attends to it. A window grown from 64 to the whole prompt changed nothing either - on the
128-token prompt, 4/54 (7.4%) with the ring at 64 and 4/54 with it at 132 - so the 64-row
horizon is not the limit.

The trace says where the gap is. Comparing the head's own first, self-chained row at a
position with the verified row at that position (`results.json` `trace_summary`): cos(`u`)
median 0.116 on the two short prompts and 0.204-0.220 on the long one (minimum 0.041),
cos(`out`) median 0.158 and 0.333-0.336, and the chained row predicts the model's token at
0, 1, 1 and 1 of the 17, 17, 15 and 15 positions that carry both rows. The head's recurrence
does not reproduce the model's hidden state at all - the rows are nearly orthogonal - so no
seeding or window choice can lift the acceptance: what `blk.64` computes is not this model's
next-token hidden.

## What it decided

The head's best measured acceptance is 13.9%, still less than half the 2B pack drafter's
30.6% (33/108), and the trace shows the gap is not a seeding or window defect. CP1 should be
built against the 2B pack, not the head, and no CPU acceptance or speedup figure for this
head should be repeated without this page behind it.

## What was not verified

No GPU run; the Arc toolchain is still probe-only. Sampling beyond greedy, tool calling and
long context (>2048 tokens) are unmeasured. The trace summary is derived from the seedfull
load's 25 MB `.npz`, which is not tracked: `results.json` carries the derived figures, and
the script that recomputes them from the trace is `bench-logs/check_seedfull.py` (the
sanitizer is `bench-logs/sanitize_mtp_variants.py`), both untracked. Two prompts short and
one long is three prompts on one host, not a distribution.
