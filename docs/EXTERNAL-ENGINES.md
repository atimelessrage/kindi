# Prior art: what llama.cpp v1.0.0, Strata 0.2.13 and tandem.cpp (2026-11-11) change for Kindi

Reviewed 2026-11-14. Three external engines moved in one week; this page records what each one
changed, what we take from it, and what we deliberately leave. The work items it creates are in
[ROADMAP.md](ROADMAP.md) under "Prior-art follow-ups (2026-11-14)"; this page is the evidence.

Sources, as read on 2026-11-14 (numbers in this page are **theirs**, not ours):

| project | release | what it is |
|---|---|---|
| [llama.cpp v1.0.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0) | 2026-11-13 (ggml 1.0.0) | the reference implementation of the formats we convert from; v1.0.0 prunes the deprecated API and lands FP8 K/V, a MoE expert cache with a CPU share, and MTP/NextN in the public header |
| [Strata 0.2.13](https://github.com/Niko1221/Strata/releases/tag/v0.2.13) | 2026-11-09 | a packaged multi-engine distribution (CUDA/HIP/SYCL/Metal) with per-card tuned defaults; runs the same Bonsai-class ternary models we do |
| [tandem.cpp](https://github.com/ewhacc/tandem.cpp) | update 2026-11-11, base llama.cpp `f5222f25c` | a patch set that keeps MoE experts in host RAM and splits *missed* experts between CPU compute and GPU upload by a **measured** CPU/PCIe ratio; this is the design our tandem-scheduler goal names |

## What we take

| # | learning | source | verdict for Kindi |
|---|---|---|---|
| 1 | The CPU/GPU split follows **measured** CPU speed vs PCIe upload speed (tandem.cpp uses factor 2.25 versus Strata's 1.85; +2.7% on one CPU), not a fixed byte budget | tandem.cpp | **adopt** — our `auto` planner is capacity-only and `gdn:16` is hand-picked |
| 2 | Tuning is **calibrated once per machine+model+build and cached** (`~/.cache/tandem/<key>.json`; 1-2 min on first start, reused after; `--calibrate` redoes it) | tandem.cpp | **adopt** — we re-tune 5-trit kernel shapes at every start and throw the result away |
| 3 | Quality of a lossy change is measured as **teacher-forced KL divergence + same-top-token share + PPL ratio** against the same build with the feature off, at 8-token batches | tandem.cpp quality table | **adopt** — our only evidence for `k8v4` is "2 of 4 prompts differ", which cannot rank two lossy formats |
| 4 | A same prompt must give the **same answer on every run**: Strata replaced oneMKL's bf16 matmul on Arc with a fixed-order kernel because results varied between launches (50/50 identical afterwards, against 3-5 differences in 5 runs before) | Strata 0.2.13 | **adopt as a test** — we assert exactness against llama.cpp but never run-to-run determinism |
| 5 | A launcher that prints **the plan, one reason per choice, a dry run and `--assume-vram-gib/--assume-ram-gib/--assume-cores`** for hardware you do not own | tandem.cpp | **done 2026-11-14** (`kindi plan`; ROADMAP prior-art #5 carries the gate record) |
| 6 | Checkpoints are **reference-counted instead of deep-copied** (server-side context slots) | llama.cpp v1.0.0 | **partly already done** — our checkpoints copy only the recurrent state (K/V is position-indexed and stays live). The copies that remain are parked conversations and saved openings, which *do* duplicate a shared prefix per session |
| 7 | MoE cache as a first-class engine feature: `moe_cache_size`, `moe_cache_cpu_share`, LRFU/ClassLRU replacement, missed experts computed on the CPU, no CPU wait on cache hits | llama.cpp v1.0.0 (+ tandem.cpp's split on top) | **adopt the semantics, not the code** — our MoE row is design-only; this is the reference to match when a backend lands |
| 8 | FP8 E4M3 K/V cache with optional **static per-tensor scales** (`-ctk/-ctv f8_e4m3`); 8 bits per value against q8_0's 8.5, wider dynamic range, hardware conversion on Ada+ | llama.cpp v1.0.0 | **measure** — 6% smaller than our q8_0 for a possibly better distribution; our FP8 kernels and `KINDI_KV` plumbing already exist |
| 9 | Thread defaults count **physical cores inside the process's CPU set**, not `os.cpu_count()` (a 6c/12t CPU or an E-core host otherwise gets ~half or double) | tandem.cpp | **adopt** — our CPU half defaults to `os.cpu_count()` |
| 10 | Host-side copies use **huge pages when RAM allows** (a fragmented host lost up to 30%) | tandem.cpp | **note for the MoE/expert-cache work** — nothing in the dense tandem copies weights per token |
| 11 | Windows gets an **unbuffered model load** mirroring the Linux DirectIO path | llama.cpp v1.0.0 | **measure later** — we already drop the page cache per chunk on Linux (`posix_fadvise`), nothing on Windows |

## What we leave

- **llama.cpp's multimodal, TTS, web-UI, MCP and server-security work.** We are a single-model
  inference engine, not a multi-modal server; the SSRF/CORS/DNS-rebinding items target a server
  that proxies URLs, which ours does not.
- **NextN/`llama.h` API churn.** Our speculative path is our own (`kindi/spec.py`, `kindi/mtp.py`);
  there is no ABI to follow.
- **Approximate expert skipping** (Strata's `STRATA_ROUTE_TAIL_SKIP`, tandem.cpp's
  `LLAMA_MOE_TAIL_SKIP=7`: 127 tok/s and 123 tok/s against 97 exact on their benchmark). Faster,
  and explicitly approximate. Kindi's contract is token-exactness against the pinned reference, so
  this stays out of the tree; if it is ever offered it is an off-by-default flag with a KLD number
  attached (item 3 above is what would measure it).
- **Their throughput numbers as targets for us.** They run MoE models on 16-128 GB cards; our gates
  are our own rows on the hardware in `docs/benchmarks/`.

## What this changes in our expectations

Two of our recorded results now have external corroboration, and one is qualified:

- *A tandem only pays when capacity forces the offload.* tandem.cpp's own summary is that models
  which nearly fit gain least (1.04x for a fitting Qwen3.6-35B against 1.4-1.9x for a budget-bound
  one), and their launcher leaves the expert cache out when the model fits. Our measured
  "tandem does not beat GPU-only when it fits" is the same finding, so the ≥25% scheduler gate
  should be tested against a *capacity-bound* row, not the fitting one.
- *Quantised K/V costs accuracy, but far less than 4-bit weights.* Our q8_0 K/V is token-exact at
  verify length on 2B/9B/27B; their published q8_0 perplexity cost is +0.03% while q4_0 is +3.31%.
  That is consistent with `k8v4`'s measured failures and argues for a graded, measured decision
  (item 3) rather than more spot checks.
- *Speed work on our list is the same shape as theirs.* Their biggest single wins came from
  scheduling and staging (measured split, read-ahead, pinned/huge-page copies, kept graphs), not
  from new math. Our plan-engine work was the same kind of win (peak 6,004 MiB against 7,721 MiB
  at the same placement). The overlap scheduler remains the open half.

## Strata, KTransformers and BigMoeOnEdge (2026-11-17)

A user-supplied comparison of Kindi 0.6.0 with **Strata**, **KTransformers**,
**BigMoeOnEdge** and **tandem.cpp** was checked against the tree, not taken at face value. The
public sources read for it are BigMoeOnEdge's repository and its `docs/` pages (`moe-streaming`,
`hardware-planning`, `benchmarks`, `comparison`, `roadmap`); Strata is a closed-source desktop
engine and KTransformers' CPU path is documented in its own repository. Every number below is
theirs, on their hardware; none is a Kindi measurement.

The comparison's shape is right on its three big items, but two of its premises are stale for this
tree: **the host tier is already pinned** (`kindi/lightning.py` stages experts into pinned RAM, which
is why it survives the 8 GB cards), and **the slot count is not a hardcoded 10** — 10 is the default
of `KINDI_LT_DCACHE`, floored at `n_used` and weighted by `kindi/ltfit.py`'s slot cost. `kindi plan`,
`kindi caps` and `kindi autotune` already print hardware, reasons and measured candidates, so the
`--auto` gap is narrower than "no planner": the missing input was this host's I/O rates, now
measured by `kindi io`.

| # | their mechanism | what we have | verdict and gate |
|---|---|---|---|
| 1 | **NVMe cold-expert streaming** (BigMoeOnEdge): the dense core and attention stay in RAM, routed experts come from the pack file with `O_DIRECT` and a read-ahead queue; hot experts sit in a GPU cache. `--no-mmap` forces the path, the benchmark separates flash bytes from compute, and it is explicitly **n=1 decode only** — batching and speculation are incompatible. | `kindi/lightning.py` now has the opt-in reader: `KINDI_LT_COLD=1` keeps the hottest host experts pinned under `KINDI_LT_HOST_MB` and makes only the overflow file-backed, unbuffered where possible, coalesced with a bounded lookahead prefetch queue and byte-accounted (`kindi/cold_experts.py`, `Engine.cold_stats()`); a routed cold expert keeps its pinned slot across steps (only an unrouted owner can be evicted), so a warm slot takes no read. `kindi io` measures the disk rate (2,509.8 MiB/s unbuffered here on 2026-11-17). | **started, gate open** (ROADMAP memory-tier #1): a Lightning pack larger than RAM, `kindi verify` 4/4, decode tok/s against the same model's recorded mmap-fault path, and flash MiB/token with the compute/I/O split. Must run at n=1; our drafters stay off for that row. No tok/s is claimed from the CPU tests. |
| 2 | **Compute misses on the CPU** instead of shipping weights (Strata's hybrid mode keeps most experts on the CPU; tandem.cpp splits by the measured CPU/PCIe ratio; KTransformers tiles CPU experts for AVX-512/AMX). | More than the comparison credits: `kindi/moe_tandem.py` already schedules `card`/`copy`/`cpu` lanes with a deterministic `cpu_share`, and prior-art row 1 below is our measured CPU/PCIe ratio (8.0 vs 7.9). What is missing is still the real thing: CPU expert **kernels** and a backend that uses them. | **expert gate passed 2026-11-19** (ROADMAP #2): `kindi/moe_expert_lane.py` is bit-identical to `lt_mv_q2`/`lt_mv_q2_down` for the Lightning Q2_0 format (0 differing elements on both matrices with device-resident or host-mapped weights), but the CPU lane is 3.06x slower than the 0.783 ms copy lane and 3.62x slower than the engine's direct host-mapped GPU read (2.395 vs 0.662 ms per expert); the host-mapped read is also 1.18x faster than the copy lane, so the CPU lane is off by default; the mixed card/CPU combine contract now exists and is gated (`Q2Matrix.partials`, `lt_mv_q2_down_parts`, `lt_q2_shared_parts`, `lt_q2_combine`), but Engine wiring and the capacity-bound model row remain blocked by the absent Lightning pack. |
| 3 | **Measured `--auto` planner**: probe the machine and the model in a separate process, choose cache/threads/device, print the reason for each choice; re-rank for asymmetric cards. | `kindi caps` (CPU/GPU/pack facts), `kindi plan` (placement with reasons and `--assume-*`), `kindi autotune` (measured candidates incl. two-card and `dcache-*`). The gap was a measured **I/O** input, not a planner. | **partly adopted 2026-11-17**: `kindi io` measures RAM/disk/PCIe, and `kindi plan`/`kindi autotune` now consume those rates for the decode-slot count. The gate - predicted slots within 10% of the measured best on a Lightning pack - is still open because no Lightning pack is on this host. |
| 4 | **Any MoE GGUF**: BigMoeOnEdge is a llama.cpp patch, so it inherits every MoE GGUF and expert layout llama.cpp loads, plus OpenAI and Anthropic servers. | Custom per-family CUDA kernels; `kindi/moe_compat.py` recognizes and refuses what it cannot run; T4.4's proxy is undecided; the server is OpenAI-shaped (REQ-0028 proposes the Anthropic surface). | **already tracked** (T4.4, REQ-0028); no new row here. Nothing to port: their generic path is llama.cpp's, and we do not ship llama.cpp. |
| 5 | **Mixed-precision tiers**: Strata's native FP4/hybrid quants on newer cards; KTransformers repacks CPU experts into cache-line-aligned INT4/INT8 tiles. | Both tiers use the same format. `MoESpec` already carries per-layer expert bytes and Q2_0/PQ2_0/FP8 codecs exist, so the plumbing for a different format per tier is partly there. | **parked** (ROADMAP #4): gate on bytes/token for the same served share, exactness where the format is exact, KLD where it is not. No codec work before a measured memory wall. |
| 6 | **Cache-aware routing** (BigMoeOnEdge): re-rank the top-k toward experts already cached, reported as +62% decode at 1-4% perplexity cost. | Kindi's contract is token-exactness against the pinned reference. Approximate skipping/substitution is already in [What we leave](#what-we-leave), with the reason. | **rejected as a default**, and not a roadmap row. If ever offered it is off by default with a KLD gate, like `k8v4` was. |

What this changes in our expectations: nothing about our numbers, and one thing about our plans. The
disk tier is no longer blocked on "we do not know what flash costs here" — it is blocked on the
engine work in ROADMAP #1, whose gate is written above. The comparison's "full model in RAM" limit is
real for the engine as it stands (pinned host tier), and it is the one external claim in the list
that matches a Kindi limitation exactly.
