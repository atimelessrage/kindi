# MTP: multi-token prediction

## What it is

An MTP head is a small extra layer that predicts the token after the next one. Kindi uses it to draft tokens.
The main model then checks all the drafted tokens in one pass. Kindi keeps the drafted tokens that equal the
main model's own tokens, and adds one token of its own.

The output is exactly the same as without MTP. MTP changes only the speed.

Kindi has one head: `kindi-mtp-2b-v1.npz` (61 MB), for Ternary-Bonsai-2-2B. It is our own work.

- Download: https://huggingface.co/ahtx/kindi-mtp-heads (the installer downloads it and checks its sha256).
- Base model: Ternary-Bonsai-2-2B by Prism ML (Apache-2.0).
- The head contains weights derived from Qwen3.5-2B by the Qwen team at Alibaba Cloud (Apache-2.0).
- The manifest is `kindi/mtp_heads.json`.

## The modes

| Mode | Setting | Drafts |
|---|---|---|
| A | `KINDI_SPEC=mtp`, `KINDI_MTP_HEAD=<file>` | The model's own head drafts k tokens (`KINDI_SPEC_K`). |
| B0 | `KINDI_SPEC=draft2b`, `KINDI_SPEC_PACK=<2B pack>` | The 2B model drafts k tokens for the 27B. |
| B2 | B0 and `KINDI_SPEC_DRAFT_HEAD=<file>` | The 2B does one step; then the 2B's head drafts the other k - 1 tokens. |
| auto | `KINDI_SPEC=auto` | Kindi selects the mode for each step (see below). |

## Which cards

MTP runs only on the tensor-core path: NVIDIA sm_80 and later, with `KINDI_TC=1`.

On cards before sm_80 (for example the Quadro K1200 and the GTX 1080), Kindi does not use MTP. On these cards, a pass over T tokens costs almost T single steps, so a
draft cannot save time. These are the measurements for T = 2, 3 and 4 tokens per pass (pass time / step time):

| Card | T = 2 | T = 3 | T = 4 | Mode A, predicted |
|---|---|---|---|---|
| Quadro K1200 (sm_50), 2B | 1.80 | 2.61 | 3.42 | 0.80x (k=1) |
| GTX 1080 (sm_61), 27B | 1.82 | 2.81 | 3.65 | 0.95x (k=1) |

## Measured speed

These measurements used greedy decoding, 128 tokens (2B: 512 tokens, the median of 3 runs), and the same text on
each card. The cells give tok/s in auto mode (the default). On the RTX 3080, the 27B now drafts with DSpine:
115.5 tok/s at 32K ([DSPINE.md](DSPINE.md)).

| Card | Model | 512 tokens of context | 8K | 24K |
|---|---|---|---|---|
| RTX 3060 | 27B (the 2B drafter) | 60.6 | 73.5 | 66.7 |
| RTX 3080 (190 W) | 2B (its own head up to 8K) | 409.0 | 395.5 | 349.0 |
| RTX 3060 | 2B (its own head up to 8K) | 350.7 | 340.3 | 287.0 |

The output of every mode was the same as without speculation.

### Measured 2026-11-17: mode A was silently dead, and costs on the 2B here

`k_rmsfold_qT` gained a trailing `hp` argument in 0.5.0; `kindi/mtp.py` never passed it, so the
head's CUDA graph did not build (`cuGraphAddKernelNode failed (1)`) and the engine logged
`speculation off: the MTP head ... did not load` — mode A did not run at all on the tensor-core
layout for four releases. Fixed upstream in 0.6.1 and ported here; `tests/test_kernel_arity.py`
now checks every call site's argument tuple against the kernel's declaration.

With the head building again, on the i7-8086K + RTX 4060 Ti (`KINDI_TC=1`, ctx 24576, one 400-token
greedy request repeated, temperature 0):

| run | generation | accepted |
|---|---|---|
| no speculation (`KINDI_SPEC=0`) | 361.2, 363.7 tok/s | — |
| mode A (`KINDI_SPEC=mtp`) | 236.9, 239.6 tok/s | 378 of 1,680 drafted (22.5%) |
| mode A, guard from position 0 (`KINDI_SPEC_GUARD_FROM=0`) | 230.9, 247.2, 246.6 tok/s | 115 of 520 drafted |

Mode A costs **0.66x** on this card at these contexts, and the guard cannot rescue it inside a
400-token reply: its first windows are spec windows, so the loss is paid before it can pause. Do not
set `KINDI_MTP_HEAD` on this rig; the drafter modes further down (a smaller model drafting a bigger
one) are where the measured win is.

### Where a drafter pays, and where it does not

Measured on the i7-8086K + RTX 4060 Ti, 2026-11-14, greedy, 8K context, one prompt per pair
(the 27B and the 9B both drafting with the same 2B pack, k = 3):

| target | no speculation | 2B drafter, k=3 | accepted | the target pass costs |
|---|---|---|---|---|
| 27B, 200 tokens | 40.0 tok/s | **62.5 tok/s (1.56x)** | 112/261 = 43% | ~13x a drafter step |
| 9B, 400 tokens | 115 tok/s | 113 tok/s (0.98x) | 221/534 = 41% | ~5x a drafter step |

The same drafter and almost the same acceptance give opposite answers. A draft pays only when the target
pass is far more expensive than a drafter step *and* the accepted run is long enough to cover the verify
pass (about 2.3 plain steps at T = 4 here). At 41-43% acceptance that holds for the 27B and not for the
9B, so the installer offers the 2B drafter to the 27B only. Do not add it to the 9B without measuring
acceptance first. Output was byte-identical in every run: speculation changes the speed, never a token.

**At 24K on the 8 GB card the drafter does not fit** (2026-11-14, same host; evidence in
`docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-drafter-24k/`): with the drafter the loader reduces
the 27B context from 24,576 to **12,032** tokens (323 MiB free without, 115 MiB with) - the drafter
costs ~1.6 GB. Where it fits it still pays: at 5,012 tokens of context, 136 tokens completed in
5.00 s without and 3.25 s with (27.2 -> 41.8 tok/s, 1.54x), consistent with the 8K row above. So on
8 GB you choose 24K without the drafter or 12K with it; 24K *with* the drafter needs a tandem slice.

**The other pairs, same host and method** (evidence in the same directory): a **4B drafter next to the
27B does not fit at all** - it loads (0.92 GB of weights, plans ready) and then the 27B cannot even
stream its 5.60 GB of weights: the loader refuses with "not enough GPU memory for a 1024-token
context" and the server exits. And a **2B drafter next to the 4B fits fully** (24,576 unreduced) with
good acceptance (73/160 = 45.6%) **and still loses**: 111 warm tokens in 1.30 s without (85.6 tok/s)
vs 1.47 s with (75.5 tok/s, 0.88x). The 4B target pass is too cheap next to a drafter step + T=4 verify
for 45% acceptance to pay - the same law as the 9B row, one size smaller. Fits does not imply helps.

## How auto mode selects

- A model with its own head (the 2B with `KINDI_MTP_HEAD`): the head, greedy requests only.
  - Up to 8K tokens of context (`KINDI_SPEC_MTP_K2_MAX`, default 8192): k=2.
  - Above that: no speculation.
- A model with the 2B drafter (the 27B): B0.
  - Below 1K tokens of context (`KINDI_SPEC_B0_K2_MAX`, default 1024): k=2.
  - At 1K and above: k=3.
  - Sampled requests use B0 with speculative sampling.
  - B2 is off (`KINDI_SPEC_B2_MAX`, default 0). Set it to a context length to use B2 below that length.
- Every other case: no speculation.

The speculation guard (`KINDI_SPEC_GUARD`, on by default) pauses speculation when it is slower than plain
decoding. It works from 32K tokens of context (`KINDI_SPEC_GUARD_FROM`, default 32768). Below 32K, its pauses cost
9-12% on the RTX 3060.

## When a head or drafter file is missing

An MTP head, a draft head and the 2B drafter are optional. If one is missing or does not load, Kindi
prints one line (`speculation off: ...`) and serves without it - the same fallback the installer
makes below. The reply is unchanged; only the speed is. (Before 2026-11-14 a stale path in
`kindi.conf` stopped the server from starting at all.)

## How to turn it off

Set `KINDI_SPEC=0` to turn off all speculation. To keep the 2B drafter and stop only the head, set
`KINDI_SPEC=draft2b` and remove `KINDI_SPEC_DRAFT_HEAD`. On the 2B, remove `KINDI_MTP_HEAD`.

The installer turns on `KINDI_SPEC=auto` only on tensor-core profiles. For the 27B, auto needs only the 2B
drafter. For the 2B, the installer downloads the head file and checks its sha256. If that fails, it prints one
line, and the 2B runs without MTP.
