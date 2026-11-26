# MTP: multi-token prediction

## What it is

An MTP head is a small extra layer that predicts the token after the next one. Kindi uses it to draft tokens.
The main model then checks all the drafted tokens in one pass. Kindi keeps the drafted tokens that equal the
main model's own tokens, and adds one token of its own.

The output is exactly the same as without MTP. MTP changes only the speed.

Kindi has one head: `kindi-mtp-2b-v1.npz` (61 MB), for Ternary-Bonsai-2-2B. It is our own work.

The Bonsai 2.5 27B `-mtp` GGUF carries its own draft head as a 65th block. `kindi convert` accepts
that file and builds the 64-layer base model's pack, recording the head block in `kindi_pack.json`.

### The 2.5 27B head on the CPU engine (2026-11-25)

`kindi/mtp_cpu.py` runs that head on the CPU engine. The block is one Qwen3.5 decoder layer plus the
four `nextn` tensors (`enorm`, `hnorm`, `eh_proj`, `shared_head_norm`), stored Q4_0 in the `-mtp`
file; the head predicts with the main model's own `output.weight`, so it needs no second vocabulary
and no second head. Its row for position `q` reads the main model's residual at `q` and the token at
`q + 1` and predicts the token at `q + 2`, the same convention as `kindi/mtp.py`. The eight Q4_0
matrices run through `kc_q4gemv` in `kindi/kcpu.c`; the norms, RoPE and the gated attention over the
head's own 64-position window are NumPy at 5,120/24 heads, and the argmax is the pack's folded
output head through `Engine.gemv_rows` (the input is folded with the pack's own signs and Hadamard
block, which is what makes the pack's folded matrix and the head's float32 output agree).

**On the CPU this drafts and scores, it does not yet accelerate.** `kindi/cpu.py` decodes one row per
forward pass, so confirming a draft costs exactly the pass plain decoding would have run; there is
nothing for a speculative loop to skip. `kindi.mtp_cpu.score_drafts` therefore decodes exactly
`Engine.generate` and compares the head's drafts with the model's own tokens: that is the
acceptance measurement, and the tokens are asserted identical to plain greedy decoding. The road map
carries the missing piece (a batched T-row verify in `kcpu.c`: one weight pass for T rows, per-row
attention and GDN scan, T-row head) with the gate it has to clear. Until then, use the head for
acceptance studies; `tools/mtp_cpu_bench.py --pack PACK --gguf <the -mtp file>` is the tool. The
2026-11-25 run is a tracked evidence page ([docs/benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp](benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp/RESULTS.md),
claim `mtp-cpu-20261125`): it records the head's cost and its acceptance, and the acceptance is not
yet good enough to wire the head as a win (road map CP1).

- Download: https://huggingface.co/ahtx/kindi-mtp-heads (the installer downloads it and checks its sha256).
- Base model: Ternary-Bonsai-2-2B by Prism ML (Apache-2.0).
- The head contains weights derived from Qwen3.5-2B by the Qwen team at Alibaba Cloud (Apache-2.0).
- The manifest is `kindi/mtp_heads.json`.

### Verified facts about the 2.5 head (2026-11-25)

Read from the two files themselves - `Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf`, 6,232,818,944 B, and
its base `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf`, 5,993,834,848 B, both GGUF v3 - by parsing their
headers with `kindi.gguf.GGUF` and pricing every tensor with `kindi.mmproj.tensor_bytes`; no weight was
run for this. The base file has 851 tensors and the `-mtp` file 866, and the difference is exactly
`blk.64`; the `-mtp` file also carries `qwen35.nextn_predict_layers = 1`, which the base file does not
have.

- **`blk.64` is exactly 15 tensors**: `attn_q.weight` (5120x12288), `attn_k.weight` and
  `attn_v.weight` (5120x1024), `attn_output.weight` (6144x5120), `attn_q_norm.weight` and
  `attn_k_norm.weight` (256), `attn_norm.weight`, `post_attention_norm.weight`, `ffn_gate.weight`
  and `ffn_up.weight` (5120x17408), `ffn_down.weight` (17408x5120), and `nextn.eh_proj.weight`
  (10240x5120), `nextn.enorm.weight`, `nextn.hnorm.weight`, `nextn.shared_head_norm.weight` (5120).
  There is no `nextn.shared_head_head.weight`, and neither file carries a second `token_embd.weight`
  or `output.weight`: both files' three non-block tensors are `token_embd.weight`, `output.weight`
  and `output_norm.weight`, so the head borrows the main model's embedding table and its output
  projection.
- **Eight Q4_0 matrices and seven f32 vectors, 238,983,168 B in total (239.0 MB)**: 238,878,720 B of
  Q4_0 (type 2, 18 bytes per 32 weights) plus 104,448 B of f32 (`attn_q_norm` and `attn_k_norm` at
  1,024 B each, the five 5120-wide norms at 20,480 B each). The `-mtp` file is 238,984,096 B larger
  than the base - that table plus 928 B of header and alignment - which is why a build that reads
  only PTQ1_0 has to read `blk.64` as Q4_0 (the CPU path does, through `kc_q4gemv`).
- **It is a full gated-attention block, not a GDN block.** `blk.64` has `attn_q/k/v`, the q/k norms
  and no `ssm_*` tensor at all; `blk.62`, a GDN layer, has 14 tensors, including `ssm_a` (no suffix)
  and `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_conv1d.weight`, `ssm_dt.bias` (a bias, not a
  weight), `ssm_norm.weight`, `ssm_out.weight`, plus `attn_qkv.weight` and `attn_gate.weight`. The
  head therefore needs a 64-position K/V ring and **no recurrent state**, so a prompt-cache
  checkpoint never has to carry GDN state for it.
- **The output projection is the pack's folded `output.weight`.** Feeding it the head's float32
  vector needs the pack's fold first: on a real residual, `fold(x)` + `gemv_rows` reproduces the
  engine's own head argmax (both 0), while the unfolded input returns 141747. HB is 1024
  (`prism.hadamard.block_size`) and RD 64 (`qwen35.rope.dimension_count`), both in the header.
- **The embedding lookup is raw.** The language blocks are folded at conversion, so the engine folds
  their activations; `blk.64`'s Q4_0 matrices are stored unfolded, so the head reads the raw PTQ1_0
  embedding row. The 128-trits-per-block layout in that row is the converter's (`kindi/convert.py`),
  not something the header states: the first version of `mtp_cpu.trits()` decoded only 80 of the 128
  trits per block, and that alone produced nonsense.

**A figure that is not ours.** A "1.12-1.17x on CPU with 83-88% drafted-token acceptance" number for
this head appears in notes written about Kindi. It is **not in this tree and not measured here**. The
1.12x in this file is the **2B** `draft2b` result on the 4060 Ti at 512 tokens, with 69% acceptance
(the tables above); what the 2.5 head measured on this host so far is worse: **9 accepted of 108
drafted (8.3%), up to 15/108 (13.9%) once the hidden is output-normed and the window seeded**, from
`tools/mtp_cpu_probe.py` on the real 27B, tracked at the [variant-ladder evidence page](benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp-variants/RESULTS.md) (claim `mtp-cpu-variants-20261125`).
Do not repeat a CPU acceptance or speedup figure for this head without a tracked page behind it.

### Which drafter, on the 27B CPU (2026-11-25)

`kindi.mtp_cpu.score_drafts` takes either drafter through one interface (`reset()` and
`draft(prefix, hin, k)`): `Head` for the `-mtp` file's own `blk.64`, `PackDrafter` for a second,
smaller converted pack. `tools/mtp_cpu_bench.py` picks between them with `--gguf` or
`--drafter-pack` (giving both is an error). Both runs below used the same 27B pack, the same two
prompts, `k=4`, 16 tokens and 6 threads on this host:

| drafter | accepted | rate | depth [1..4] | plain tok/s | drafted tok/s |
|---|---|---|---|---|---|
| integrated `blk.64` head | 7/108 | 6.5% | [3, 1, 2] | 1.69 / 1.51 | 1.11 / 1.02 |
| 2B pack (`~/.kindi/packs/bonsai-2b`) | 33/108 | 30.6% | [15, 8, 7, 3] | 2.03 / 2.12 | 1.01 / 1.16 |

The external 2B drafter is the better draft source by about five times the acceptance, and **still
slower than plain decode**: every round replays the known prefix through the 2B engine (O(n) forwards)
and the 27B verifies one row per pass, so no draft removes work. Neither route is a speedup until
CP1's batched T-row verify exists; what the measurement settles is which drafter CP1 should be built
against - the 2B pack, not the head. Per prompt the 2B accepted 23/54 (43%) on the first and 10/54
(19%) on the second. Both numbers come from the tracked [2026-11-25 evidence page](benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp/RESULTS.md)
(claim `mtp-cpu-20261125`, sanitized from the two `bench-logs/` runs), and both asserted their tokens
equal to plain greedy decode, so acceptance is measured against the real model and nothing else.

#### The head's acceptance ladder (2026-11-25, one load per batch)

Loading the 27B costs six minutes, so the head-side question was turned into a probe that loads once
and runs several variants in that process: `tools/mtp_cpu_probe.py --pack PACK --gguf <the -mtp file>
--commands cmds.jsonl --out out.jsonl`, one JSON attempt per line
(`{"variant": ..., "k": 4, "tokens": 16, "prompts": [...]}`). It records the acceptance, the depth
histogram and the actual draft token ids next to the model's own, so a variant is judged by data, not
by a compile. A variant ending in `-full` also seeds the prompt's last row, `"window": "prompt"`
grows the head's K/V ring from the prompt (or an integer sets it), and `--trace out.npz` writes every
head row's `u` and `out` with its position, prediction, attempt and phase. Every attempt below
asserted its tokens equal to plain greedy decode. The five loads behind the table and the whole-prompt
section are tracked at the [variant-ladder evidence page](benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp-variants/RESULTS.md)
(claim `mtp-cpu-variants-20261125`, sanitized from the `bench-logs/` probe JSONL and the seedfull trace).

| variant | accepted | rate | what it changes |
|---|---|---|---|
| `eh` (as committed) | 9/108 | 8.3% | raw residual into `eh_proj` |
| `he` | 0/108 | 0.0% | concat order swapped, `[hidden, emb]` |
| `eh-normed` | 11/108 | 10.2% | `rms(residual) * output_norm` - what the GPU head is fed |
| `eh-normed-folded` | 11/108 | 10.2% | embedding row folded with the pack's signs (BTQ1/PTQ1_0 layout) |
| `eh-normed-rope1` | 11/108 | 10.2% | RoPE at `p + 1` (the reference's MTP position for row `p`) |
| `eh-normed-seeded` | 15/108 | 13.9% | window seeded with every **verified** row, not just the head's own |
| `eh-normed-seeded-full` | 15/108 | 13.9% | also seeds the prompt's last row - the row the first draft recomputes, bit for bit |

The two switches that matter are in the table's last rows: the hidden is the output-normed one (what
`kindi/mtp.py`'s `XN` and llama.cpp's `graph_mtp` both read, and what this head now uses), and
seeding the window with the verified rows - what the reference's MTP cache holds - lifts 10.2% ->
13.9% overall, 16.7% -> 22.2% on the second prompt, with depth `[2, 3, 4, 3]`. The remaining gap is
not one of these switches.

#### The whole prompt, and what the rows say (one more load)

The two remaining candidates ran in one load, on the same two prompts and settings: seeding the
prompt's last row as well (`-full`) changed nothing - 15/108 again (3/54 + 12/54) - and the trace
shows why it cannot. With that row seeded, its `u` and `out` are bit-for-bit the first draft row's
(`max|du| = 0`, same prediction): the draft recomputes the same position from the same token and
hidden immediately after the seed, so the seed is overwritten before anything attends to it. A window
grown from 64 to the whole prompt changed nothing either - on a 128-token prompt, 4/54 (7.4%) with
the ring at 64 and 4/54 with it at 132 - so the 64-row horizon is not the limit.

The per-row dump (`--trace`, 748 rows: 388 seed, 360 draft) says where the gap is. Comparing the
head's own first, self-chained row at a position with the verified row at that position: cos(`u`)
median 0.12 on the two short prompts and 0.20-0.22 on the long one (minimum 0.04), cos(`out`) median
0.16 and 0.33, and the chained row predicts the model's token at 0, 1, 1 and 1 of the 17, 17, 15 and
15 positions that carry both rows. The head's recurrence does not reproduce the model's hidden state
at all - the rows are nearly orthogonal - so no seeding or window choice can lift the acceptance:
what `blk.64` computes is not this model's next-token hidden. **The 2B pack is the better drafter
(30.6% vs 8.3-13.9%), and neither is a speedup before CP1** (tracked at the [variant-ladder evidence page](benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp-variants/RESULTS.md), claim `mtp-cpu-variants-20261125`).

## The modes

| Mode | Setting | Drafts |
|---|---|---|
| A | `KINDI_SPEC=mtp`, `KINDI_MTP_HEAD=<file>` | The model's own head drafts k tokens (`KINDI_SPEC_K`). |
| B0 | `KINDI_SPEC=draft2b`, `KINDI_SPEC_PACK=<2B pack>` | The 2B model drafts k tokens for the 27B. |
| B2 | B0 and `KINDI_SPEC_DRAFT_HEAD=<file>` | The 2B does one step; then the 2B's head drafts the other k - 1 tokens. |
| auto | `KINDI_SPEC=auto` | Kindi selects the mode for each step (see below). |

For explicit mode A, `KINDI_SPEC_K` now defaults to **1**; the model drafter modes keep the
previous default of 4. An explicit `KINDI_SPEC_K` still wins. The depth run below is why: k=1
measured 1.12-1.13x at 512/24K and 1.01x at 8K, while the old k=4 default measured 0.66-0.89x.

A fresh 2026-11-19 A/B with `KINDI_SPEC_K` unset confirmed it on the current tree: 512 was
341.8 -> 378.2 tok/s (**1.11x**, 52/75 accepted) and 24K was 301.7 -> 343.2 (**1.14x**, 54/74),
medians of 2 and 4 runs. The current `docs/*.md` prompt had grown to 306,836 tokens, so this is a
confirmation rather than an identical replay of the 281,805-token 2026-11-17 prompt.
[Results](benchmarks/2026-11-19-i7-8086k-rtx4060ti-mtp-k1-confirm/RESULTS.md).

## Which cards

MTP runs only on the tensor-core path: NVIDIA sm_80 and later, with `KINDI_TC=1`. Without that switch the server logs `warning: speculation off: KINDI_SPEC=... requires the tensor-core plan engine: set KINDI_TC=1 (and do not set KINDI_TP=0)` at startup, and its `/health` `drafting` block reports `enabled: false` with the same reason, not a null one.

On cards before sm_80 (for example the Quadro K1200 and the GTX 1080), Kindi does not use MTP. On these cards, a pass over T tokens costs almost T single steps, so a
draft cannot save time. These are the measurements for T = 2, 3 and 4 tokens per pass (pass time / step time):

| Card | T = 2 | T = 3 | T = 4 | Mode A, predicted |
|---|---|---|---|---|
| Quadro K1200 (sm_50), 2B | 1.80 | 2.61 | 3.42 | 0.80x (k=1) |
| GTX 1080 (sm_61), 27B | 1.82 | 2.81 | 3.65 | 0.95x (k=1) |

## Smoke test it (opt-in, GPU)

`tools/mtp_smoke.py` is the server-level check for this mode. It starts a real `kindi serve`
process with the registered head and `KINDI_SPEC=mtp`, waits for `/health`, sends one greedy chat
completion, and reads the result back from both `/health` and `/metrics`. It exits non-zero unless
`drafting.enabled` is true, the effective method is `mtp`, `drafting.last.method` is `mtp`,
`drafting.last.drafted` is above zero, and `kindi_spec_drafted_tokens_total` is above zero. The engine is not a stand-in: this is the same
server path a client uses.

```bash
python tools/mtp_smoke.py \
  --pack ~/.kindi/packs/bonsai-2b \
  --head ~/.kindi/models/kindi-mtp-2b-v1.npz
```

The defaults are `--spec-k 1` (the measured win), `--ctx 4096`, 16 greedy tokens, and a free
loopback port. The script uses a temporary state directory and an empty config, so a host config
cannot change authentication, rate limits, or the engine; `--log FILE` keeps the server log. It
needs a tensor-core GPU and the registered head, and fails before starting the server when either
is missing. It is a `tools/` script, not a `tests/test_*.py` file, so the CPU-only pytest suite does
not collect it.

### Live result: attached is not the same as drafted (2026-11-18)

A live 2B server run with the head attached reported `drafting.method: "mtp"` and
`drafting.last` `passes: 3, drafted: 12, accepted: 8`. The method is the attached state; the
non-zero drafted count is the part that says the head actually produced draft tokens for the
recorded request rather than `KINDI_SPEC=mtp` being echoed back, and the eight accepted drafts say
the target verified them.

The run's context, `KINDI_SPEC_K`, pack, head and exact launch command were not captured, so the
record is the counts from the server's last-request block, not a fully specified reproduction. The
shape is consistent with the then-engine default `KINDI_SPEC_K=4` (four drafts per pass) but not with
the smoke tool's default `--spec-k 1`, which can draft at most one token per pass. The number of
requests the session served was also not recorded: `drafting.last` describes one request, not a
one-request session. This is not a throughput or token-exactness measurement.

`tools/mtp_smoke.py` now closes that gap for future runs: it prints and writes a run record at the
top of its `--log` with the pack, the head and its digest, the context, `KINDI_SPEC_K`, the GPU,
the exact serve command and its `KINDI_*` environment, and the request count.

A record-bearing run on 2026-11-19 used the settings the earlier counts imply
(`KINDI_SPEC_K=4`) and the tool's default 4,096-token context:

```bash
python tools/mtp_smoke.py \
  --pack ~/.kindi/packs/bonsai-2b \
  --head ~/.kindi/models/kindi-mtp-2b-v1.npz \
  --ctx 4096 --spec-k 4 --max-tokens 16 --timeout 300 \
  --log bench-logs/mtp-smoke/runrecord-k4-20261119.log
```

It exited 0 on the RTX 4060 Ti (sm_89). The record (printed before the server started and kept at
the top of the log) recorded the following; paths are shown with `~` and `<tmp>` here, while the
log has the host-absolute forms:

| field | value |
|---|---|
| pack | `~/.kindi/packs/bonsai-2b` (`Ternary-Bonsai-2-2B`; served as `bonsai2-2b`) |
| head | `~/.kindi/models/kindi-mtp-2b-v1.npz`, registered, 60,960,552 bytes |
| head sha256 | `1b5c0c790a594af368c3646370c527c319cdb3bb0592dd3e523e752f721a5520` |
| ctx | 4,096 |
| `KINDI_SPEC_K` | 4 |
| GPU | NVIDIA GeForce RTX 4060 Ti (sm_89) |
| requests | 1 |

The serve command carried the run's full `KINDI_*` environment, including `KINDI_DEVICE=gpu`,
`KINDI_HEAD=exact`, `KINDI_TC=1`, `KINDI_TP=1`, `KINDI_SPEC=mtp`, `KINDI_SPEC_K=4`,
`KINDI_CTX=4096`, an empty API key and zero rate-limit/queue limits. Normalized only for the
home directory and the throwaway temporary config/state (the log has the host paths), it was:

```bash
KINDI_API_KEY='' KINDI_CTX=4096 KINDI_DEVICE=gpu KINDI_HEAD=exact KINDI_LOG_JSON=0 \
KINDI_MTP_HEAD="$HOME/.kindi/models/kindi-mtp-2b-v1.npz" \
KINDI_PACK="$HOME/.kindi/packs/bonsai-2b" KINDI_PARK_MAX=0 KINDI_PREFIX_MAX=0 \
KINDI_QUEUE_MAX=0 KINDI_RATE_LIMIT_PER_MIN=0 KINDI_SPEC=mtp KINDI_SPEC_K=4 \
KINDI_STATE_DIR='<tmp>/state' KINDI_TC=1 KINDI_TP=1 KINDI_TURN_SAVE=0 \
python -m kindi --config '<tmp>/kindi.conf' serve --host 127.0.0.1 --port 52176
```

The run printed `drafting.last: method=mtp passes=2 drafted=8 accepted=3` (3 of 8 drafted
tokens accepted), with `kindi_spec_drafted_tokens_total=8` and
`kindi_spec_accepted_tokens_total=3`. The completion was five tokens ("Red Blue Green") from 21
prompt tokens. The tool sends exactly one completion request, and its post-request method check
passed. This reproduces the attached-and-drafting path, not the earlier request's 3/12/8 counts;
it is not a throughput or token-exactness measurement.

### Record-bearing run at the recommended k=1 (2026-11-19)

The default `--spec-k 1` - the setting the engine now selects for explicit mode A - has its own
record-bearing run:

```bash
python tools/mtp_smoke.py \
  --pack ~/.kindi/packs/bonsai-2b \
  --head ~/.kindi/models/kindi-mtp-2b-v1.npz \
  --ctx 4096 --spec-k 1 --max-tokens 16 --timeout 300 \
  --log bench-logs/mtp-smoke/runrecord-k1-20261119.log
```

It exited 0 on the same RTX 4060 Ti (sm_89). The record at the top of the log names the pack
`~/.kindi/packs/bonsai-2b` (`Ternary-Bonsai-2-2B`; served as `bonsai2-2b`), the registered head
`~/.kindi/models/kindi-mtp-2b-v1.npz` (60,960,552 bytes, sha256
`1b5c0c790a594af368c3646370c527c319cdb3bb0592dd3e523e752f721a5520`), ctx 4,096,
`KINDI_SPEC_K` 1, the GPU and `requests: 1`. Its `KINDI_*` environment and serve command are the
k=4 record's above with `KINDI_SPEC_K=1` and loopback port 60,585; the startup log confirms
`MTP head attached: kindi-mtp-2b-v1.npz (61 MB, window 64, sha256 1b5c0c790a59)` and
`speculation: requested mtp, effective mtp`.

The run printed `drafting.last: method=mtp passes=3 drafted=3 accepted=2` (2 of 3 drafted tokens
accepted), with `kindi_spec_drafted_tokens_total=3` and
`kindi_spec_accepted_tokens_total=2`. The completion was five tokens ("Red Blue Green"), and the
tool's post-request method check passed. At most one draft per pass is the signature of
`KINDI_SPEC_K=1`, so 3/3 drafted is the expected shape and distinguishes this record from the
k=4 run's 8 drafted over 2 passes. This is not a throughput or token-exactness measurement.

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

That is now a gate, not a claim: `kindi verify --spec <mode>` runs the same reference matrix under the mode
and prints the effective mode on its first line (a drafter that did not attach exits non-zero instead of
verifying the plain engine). Measured 2026-11-17 on the 2B / RTX 4060 Ti, ctx 1024, 4 prompts, 40 greedy
tokens each, `KINDI_TC=1`, pack `bonsai-2b`:

| mode | command | result | tok/s | drafts accepted |
|---|---|---|---|---|
| off | `kindi verify` | 4/4 prompts, 160/160 tokens | 279.0 | - |
| ngram | `kindi verify --spec ngram` | 4/4 | 435.3 | 51/152 |
| A (`mtp`) | `--spec mtp` with `KINDI_MTP_HEAD=~/.kindi/models/kindi-mtp-2b-v1.npz` | 4/4 | 203.5 | 98/248 |
| B0 (`draft2b`) | `--spec draft2b` with `KINDI_SPEC_PACK=<2B pack>` | 4/4 | 311.1 | 128/128 |
| B2 (`both`) | `--spec both` with both variables | 4/4 | 276.2 | 102/242 |

Every mode is token-exact with llama.cpp here, so the sentence above holds for the modes that can run on
this host. Two caveats: B0 and B2 draft with the same 2B that is the target, so their path is exercised and
exact but the 128/128 and 102/242 acceptance numbers say nothing about the 27B; and `dspine` needs the
DSpine weights, which are not on this host, so it has no cell yet (owner action).

`tools/verify_matrix.py` runs the same cells as `<base>-<spec>` (`tc-ngram`, `tc-mtp`, `tc-draft2b`,
`tc-both`): `--spec-modes` defaults to all four, `--spec-models` to `2b` and `--spec-base` to `tc`;
its rows and CSV/JSON carry the requested/effective mode and accepted/drafted counts, a fallback is
`refused`, and a cell whose optional head or pack is absent is printed as a skip. A baseline saved
with those files therefore makes `--fail-on-regression` gate them, and the exact 2B baseline and
gate commands are in [CARD-SUPPORT.md](CARD-SUPPORT.md#2b-speculation-baseline); a host without
them still runs the plain matrix.

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

At the then-default (`KINDI_SPEC_K=4`) mode A costs **0.66x** on this card, and the guard cannot rescue
it inside a 400-token reply: its first windows are spec windows, so the loss is paid before it can
pause. The depth measurement below changes the conclusion for one setting: with `KINDI_SPEC_K=1`
mode A is a win at 512 and 24K tokens of context, and a wash at 8K.

### Mode A at depth, and the k that matters (2026-11-17)

Same host, same pack and head, `kindi/mtp.py` fixed: `tools/mtp_context_bench.py` builds the engine
once and runs plain and mode A alternately on it, 128 greedy tokens from a real-docs prompt at each
position, prompt reading excluded from the clock ([RESULTS](benchmarks/2026-11-17-i7-8086k-rtx4060ti-mtp-context/RESULTS.md)):

| context | `KINDI_SPEC_K` | plain tok/s | mode A tok/s | ratio | drafts accepted |
|---:|---:|---:|---:|---:|---|
| 512 | 1 | 335.4 | **375.6** | **1.12x** | 52/75 (69%) |
| 512 | 4 (old default) | 347.6 | 307.9 | 0.89x | 77/200 (39%) |
| 8K | 1 | 327.4 | 329.1 | 1.01x | 43/84 (51%) |
| 8K | 4 (old default) | 330.1 | 218.3 | 0.66x | 60/272 (22%) |
| 24K | 1 | 303.6 | **344.4** | **1.13x** | 53/74 (72%) |
| 24K | 2 | 292.2 | 302.2 | 1.03x | 64/126 (51%) |
| 24K | 3 | 293.1 | 279.2 | 0.95x | 71/168 (42%) |
| 24K | 4 (old default) | 292.1 | 254.4 | 0.87x | 75/208 (36%) |
| 24K | 6 | 292.5 | 194.1 | 0.66x | 75/312 (24%) |

Acceptance does rise with depth - 72% at k=1 and 24K against 22% at k=4 and 8K - but the extra
drafts only pay when each one is cheap: `k=1` turns one draft into one verified token almost every
step, while k=4 pays a 5-token verify for 1-2 accepted tokens. The 24K/k=1 row is four runs a side
(plain 301.8-304.8, mode A 338.0-345.4) and `KINDI_SPEC_K=1 kindi verify --spec mtp` is still
**4/4 token-exact**.

A 2026-11-19 check of the new default (no `KINDI_SPEC_K`; `KINDI_TC=1`, `KINDI_TP=1`,
`KINDI_SPEC_GUARD=0`, ctx 1024, `verify --spec mtp` on the same 2B / RTX 4060 Ti) passed all four
prompts, **160/160 tokens**, with **65/94 drafts accepted** and **403.3 tok/s** over the run. It is
an exactness/smoke check, not a new A/B speed row.

So on this card: **the engine now defaults explicit mode A to `KINDI_SPEC_K=1`** (2026-11-19), the
measured win at 512 and 24K and a wash at 8K. An explicit `KINDI_SPEC_K=4` still loses
0.66-0.89x; do not raise the default without a new measurement. The `auto` mode cannot reach that
win: it caps mode A at k=2 below 8K and switches it off above
(`AUTO_MTP_K2_MAX`), so at 24K it runs the plain engine. The drafter modes further down (a smaller
model drafting a bigger one) are still where the large measured win is.

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
acceptance first. A direct A/B on 2026-11-18 compared the 27B plain output with `ngram`,
`draft2b` and `both` from the same engine and prompt, byte for byte: **12/12 prompt/mode pairs were
identical** (4 prompts x 3 modes, 40 greedy tokens each), with matching token ids and decoded byte
hashes. The harness is `tools/spec_ab.py`; the command and per-prompt hashes are in
[RESULTS](benchmarks/2026-11-18-i7-8086k-rtx4060ti-27b-spec-ab/RESULTS.md). Speculation changes
the speed, never a token.
The recorded [27B drafter exactness matrix](benchmarks/2026-11-17-i7-8086k-rtx4060ti-27b-drafter/RESULTS.md)
backs that for `tc-ngram`, `tc-draft2b` and `tc-both`: each matches the no-speculation cell's
verdict (3 exact + 1 accepted llama.cpp tie, 0 fail).

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

The engine checks the file again before it loads it: `kindi/mtp_heads.json` records the sha256 and the size,
and a head that disagrees (or is not registered) is refused with `warning: speculation off: the mtp head ...`
naming both digests, so a hand-patched file never reaches the card. A head you built yourself is allowed with
`KINDI_MTP_HEAD_ALLOW_UNREGISTERED=1`. `/health`'s `drafting` block and the startup line
`speculation: requested ..., effective ...` answer what attached, not what `KINDI_SPEC` asked for.
