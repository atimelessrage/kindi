# The batched T-row verify pass on the CPU: exact, and a measured no-go (2026-11-26)

Roadmap gate **CP1** (`docs/ROADMAP.md`) asked for one thing `kindi/cpu.py` could not do: confirm k
drafted tokens in **one** weight pass instead of k, then show 4/4 token-exact prompts **and >=1.25x**
tokens/s at k=4 on the 2.5 27B. The pass exists now and is exact. The speed bar is missed, and the
reason is in the kernel's arithmetic rather than in the drafter - so this page records a no-go, not a
speedup.

- Host: Intel i7-8086K, 6 threads, ctx 2048, AVX2/FMA/F16C, no AVX-512; Windows 11, clang 18.1.8.
- Verifier: the Ternary-Bonsai-2.5-27B pack (64 layers, `PTQ1_0`). Drafter: the 2B pack, incremental.
- Commands (`results.json` is their sanitized output):
  - `python tools/cpu_batch_check.py --pack <27B> --rows 15 --batch 5 --keep 2 --out check.json`
  - `python tools/spec_cpu_bench.py --pack <27B> --drafter-pack <2B> --tokens 16 --k 4 --prompts 4 --out gate.json`

## What the pass is

`kindi/kcpu.c`'s `kc_step_batch` feeds T rows through **one** stream of packed ternary weights: each
weight word is loaded once, the per-row pair tables are permuted out of it (the single-row kernel's
two `vpermps` become 2T), attention runs per row under the batch's causal mask, the GDN scan runs in
row order, and the head produces T rows at once. `Engine.step_batch` in `kindi/cpu.py` exposes it.
`kc_batch_rollback` then puts the GDN state back at the last accepted row by replaying the scan from
the per-row `QKV`/`XN` vectors the pass stored - no weights are re-read - and `kc_state_len/save/load`
give a drafter a snapshot to rewind to, which is what makes an *incremental* drafter (one that feeds
only new prefix tokens) possible at all.

`kindi.mtp_cpu.speculative_decode` is the loop: each round is `[pending] + drafts[:k]` at consecutive
positions, the leading matches are accepted, the model's next argmax becomes the next pending token,
and the state is rolled back to the accepted rows. The drafter sees `pending`, because row 0's argmax
is the token after it.

## The pass is plain decode, row for row

15 rows through `kc_step_batch` in batches of 5, against the same 15 rows decoded one at a time:

| | rows | s | rows/s | tokens equal | max abs logit delta |
|---|---|---|---|---|---|
| plain decode | 16 | 6.39 | 2.50 | - | - |
| 5-row pass | 15 | 5.27 | 2.84 | yes | 0.000e+00 |

The rows are **bit-identical**, not merely as accurate: every row's 248,320 logits equal what plain
decode produced at that position. After `kc_batch_rollback(2)`, 13 more plain rows equal the
uninterrupted decode too, so the replay lands the state exactly.

## One pass costs 4.4 rows, not 1

That is the whole result. A 5-row pass runs at 1.14x plain rows/s, i.e. it costs **4.4 plain rows**,
not the ~1 a bandwidth-bound kernel would cost. Inside the speculative loop the same accounting holds
and is slightly better (the T-row head and the fixed per-call costs are amortized): the batch call
costs 0.53-0.68 plain tokens per row, ~0.59 on average, so a round of 5 rows costs **2.95 plain rows**.

The pair-code GEMV is *permute-bound*, not memory-bound. Every row needs its own two `vpermps` per
weight word; only the load and the variable shift are shared between rows. So the pass scales with T
on the arithmetic and barely at all in the traffic it saves: a single row here is already some 1.5
weight-streams-per-second away from the wall, and the 6.75 GB it reads in half a second is well under
what this host can stream.

## The gate

27B, greedy, k=4, 16 tokens per prompt, the 2B pack drafting; plain and speculative timed on the same
engine and their token lists compared:

| prompt | plain tok/s | spec tok/s | speedup | accepted | accepted per depth |
|---|---|---|---|---|---|
| In one sentence: what is a ternary weight? | 1.57 | 0.80 | 0.51x | 10/32 | [5, 2, 2, 1] |
| 2 + 2 = ? Answer with the number only. | 1.54 | 0.67 | 0.43x | 6/40 | [6, 0, 0, 0] |
| Name three colours and stop. | 1.97 | 0.81 | 0.41x | 7/36 | [5, 1, 1, 0] |
| Write a four-word sentence about the sea. | 1.71 | 1.19 | 0.69x | 13/20 | [4, 4, 3, 2] |
| **all four** | | | **0.51x mean** | **36/128 (28.1%)** | |

**4/4 prompts token-exact** - the first half of the gate - and mean **0.51x**. The acceptance is the
rate `score_drafts` measured for this drafter (30.6% on the same packs), so the loop agrees with the
drafter study; the round's shape is not the problem.

## Why it misses, in one line of arithmetic

A round costs 5 rows x 0.59 = **2.95 plain tokens** and yields 1 + m tokens, where m is what the
drafter lands of its k=4 drafts. Break-even (1x) needs **m >= 2.0**; the CP1 bar (1.25x) needs
**m >= 2.7**, i.e. 60-70% of k=4 drafts at every depth. The best drafter on this model accepts 30.6%
per draft, and 28.1% here. Per prompt that is what the table shows: 0.41-0.69x.

The gate is therefore a no-go **on this host with this kernel**, and CP2 (the 2B drafter as a
shipped CPU mode) stays blocked: a better drafter would not close a 3x gap that comes from the pass
costing 3 rows instead of 1.

## What would reopen it

- A verify kernel at ~0.2 plain rows per row: fewer permutes per weight (a byte-lane formulation that
  looks up 32 codes per `vpshufb`, or AVX-512's 32-lane permute), or a kernel that is genuinely
  bandwidth-bound.
- A host where a single row already saturates memory bandwidth, so the same kernel's shared load is
  the win it was expected to be here.

## What was not verified

- One host, one microarchitecture (AVX2, no AVX-512), one model pack, four short prompts.
- No GPU run and no speedup claim: spec decode was slower on every prompt.
- Batch sizes above the k=4 gate (T=8, k=7) were not measured; the arithmetic above predicts a smaller
  gain per row, not a larger one, because permutes grow as 2T.
