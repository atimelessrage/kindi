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
| 5 | A launcher that prints **the plan, one reason per choice, a dry run and `--assume-vram-gib/--assume-ram-gib/--assume-cores`** for hardware you do not own | tandem.cpp | **adopt** — this is how we test placement rules for the P106/M4000 before the cards exist |
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
