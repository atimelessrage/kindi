# AI inventory scan ledger

What has been read from the external knowledge base
the folder named by `$KINDI_AI_INVENTORY` (the path is deliberately not written down here: a
synced-folder path is personal data), what was considered, and what was decided. The
procedure is the [ai-inventory skill](../../.agents/skills/ai-inventory/SKILL.md); this file is its
memory. One row per scan, one line per verdict.

The knowledge base is read-only and its numbers are not our evidence: `[M]` there means measured on
their Ollama/llama.cpp rig, not on Kindi, and `[P]`/`[E]`/`[?]` are vendor claims, their arithmetic
and untested items. Nothing in this file is a Kindi measurement.

## Scan 2026-11-17 (baseline)

**Last checked:** 2026-11-17. **Knowledge base HEAD:** `1135eab` ("Record the 2026-10-25 GPU
benchmark run and repair the row it malformed"), with uncommitted work in
`INVENTORY_LOCAL_OLLAMA.md`, `scripts/benchmark_models.py`, `-layout`, `check_ps.py` and
`external_research/AI_inventory_todos.txt`, and files touched up to 2026-11-15 — its git view lags
its working tree, so the next scan must compare mtimes too, not just commits.

Files read: `README.md`, `HERMES_CRON_TASK.md`, `docs/KINDI.md` (opening), `docs/LOCAL_ENGINE_TUNING.md`
(opening), `docs/KINDI_CHANGELOG.md` (opening).

| candidate | what the source says | what Kindi already has | verdict |
|---|---|---|---|
| KV cache below q8_0 (q4_0) | Ollama 0.34.4, `qwen2.5-coder:1.5b`, 32K ctx: `f16` 2.113 GB → `q8_0` 1.702 GB → `q4_0` 1.482 GB, with **no** speed cost `[M]` (`docs/LOCAL_ENGINE_TUNING.md` §2) | `q8_0`, `k8v4` (half-width V) and `fp8` K/V already exist: `kindi/engine.py` (q8 store kernels), `kindi/chunk.py` (`k8v4`/`fp8` chunk gate), `kindi/config.py` (`kv_cache` auto/f16/q8_0) | **parked.** A V-side q4 store is only worth a kernel when a measured memory wall demands it; it is approximate, so its gate is perplexity/`verify` tolerance plus a non-exact label, and `docs/EXTERNAL-ENGINES.md` already records llama.cpp's own q4_0 cost (+3.31% vs +0.03% for q8_0). No roadmap step yet. |
| prefill dominates wall clock | commit-review measurement: 67% of wall clock is prefill; `qwen2.5-coder:14b` reads at 10.1 tok/s but decodes at 2.75 tok/s `[M]` (`docs/COMMIT_REVIEW_MODEL.md`, summarised in `LOCAL_ENGINE_TUNING.md` §1) | `kindi bench` prints `prompt reading: N tokens/s` (`kindi/bench.py`) | **parked, cheap.** Kindi reports the two rates but never their share of a request mix, which is exactly what decides whether to tune read or decode. Gate: our own measured split on the 2B/27B packs at 4K and 8K (`kindi bench`), recorded in a `docs/benchmarks/` results file. |
| "Flash Attention is a mandatory prerequisite for KV cache quantization" | the popular write-up claim did not survive their measurement on Ollama 0.34.4 `[M]` | irrelevant: Kindi's q8 cache is its own store kernel (`kindi/engine.py`), with no Flash Attention coupling | **rejected (not applicable).** Recorded so the next scan does not re-raise it. |
| CPU engine anchors | Ryzen 9 7940HS: 64.8 vs llama.cpp 23.6 tok/s (2.7×); Core i7-8700 1.6× `[P]`, their box | our own CPU/llama.cpp ratio is the measured 8.0 vs 7.9 tok/s in `docs/EXTERNAL-ENGINES.md` (beat clause not met) | **reference only.** The i7-8700 is the closest published anchor to the i7-8086K here; it is their number, not a claim, and it is not comparable to a different pack and context. |
| the knowledge base as a source for upstream Kindi state | `docs/KINDI_CHANGELOG.md` tracks upstream `v0.3.0` (feed check 2026-10-30); `docs/KINDI.md` is explicitly research-only, `[?]`, "never run here" | this tree contains upstream `v0.6.0` (`CURRENT-VS-KINDI-0.7.md`) | **rejected as a source.** Three minor versions stale; read upstream `CHANGELOG.md`/releases directly when upstream state matters. |

Nothing was adopted in this scan, so no roadmap or requirement step was created and nothing was
implemented — which is the expected outcome of a baseline scan.

## Check attempt 2026-11-17 (path unavailable)

A roadmap session (the `docs/ROADMAP.md` "Engine-liveness follow-ups") asked for a scan.
`$KINDI_AI_INVENTORY` was unset in the shell and empty in both the Windows User and Machine scopes,
so nothing was read this time and no candidate is recorded — the previous section is still the
whole record. The path is deliberately not written into the repository, so it has to be supplied
per session rather than recovered from here.

Gate for the next scan: set `$KINDI_AI_INVENTORY`, then follow the incremental procedure (last seen
commit `1135eab` in that section, its HEAD since, and file mtimes newer than 2026-11-17 because its
working tree lags its git view). The roadmap work in this round did not depend on it: the adopted
items come from this repository's own measurements, and none of them cite the knowledge base.

## Scan 2026-11-17 (incremental: tandem.cpp, Strata, BigMoeOnEdge)

**Last checked:** 2026-11-17. **Knowledge base HEAD:** `1135eab` in its main checkout (no commit
since the baseline there), but the delta this scan read lives in the base's own Freebuff worktree:
`390cf16` and `32ca6f5` ("Record the latest benchmarks and the week's model, engine, and hardware
knowledge"), i.e. two commits past `1135eab`, files touched up to 2026-11-17 09:08. The main
checkout's tree is dirty in the same five paths the baseline recorded, so both views were needed.

Files read: `docs/TANDEM_CPP.md` (verdict, §4–5, §9), `docs/STRATA.md` (verdict, §4–5),
`docs/BIGMOEONEDGE.md` (verdict, §4–5), `docs/KINDI.md` §"The CPU+GPU tandem paste — checked
2026-11-11", `docs/KINDI_CHANGELOG.md` (v0.6.0 opening), `models/raon_1_mini.md`,
`architecture/hardware_optimization.md` (diff: a stale-pointer note only).

| candidate | what the source says | what Kindi already has | verdict |
|---|---|---|---|
| Anthropic `/v1/messages` surface | BigMoeOnEdge: "the sleeper feature is the API surface, not the speed" — its server speaks Anthropic `/v1/messages`, which Claude Code and the harness rows expect; Ollama/llama.cpp give only OpenAI shapes `[P]` (`docs/BIGMOEONEDGE.md` §5) | nothing: no `v1/messages`, `x-api-key` or Anthropic content-block translation anywhere in `kindi/` or `docs/` (grep, 2026-11-17) | **adopted as a proposal — [REQ-0028](requirements/REQ-0028.md)**, roadmap bullet in the Agent API section. Translation onto the existing chat path, tested app-level with a Claude-Code-shaped request; no decoding change, so no GPU gate |
| MoE expert cache with a CPU share | tandem.cpp / llama.cpp v1.0.0 add `moe_cache_size` + `moe_cache_cpu_share`, LRFU/ClassLRU and no CPU wait on cached experts `[P]` (`docs/TANDEM_CPP.md` §1) | `kindi/lightning.py` keeps the least-used experts of each layer in pinned host memory, with decode slots (10/layer) and LRU staging; `kindi/moe.py` owns placement; `kindi plan --moe-cache-mib/--dcache` reports the tier | **already-have** (`kindi/lightning.py` decode slots, `kindi/moe.py`, `kindi/plan.py`) |
| slot concurrency on a MoE | Strata `parallel` 2/4 on a 5070: 61.0/61.2 tok/s total against 70.7 one at a time (−13/−15%), first token 11.2 s → 1.8 s, because a MoE window over four conversations reads ~24 experts/layer against ~8 `[P]` (`docs/STRATA.md` §4) | measured from the other direction and NO-GO: continuous batching B=4 = 131.8 tok/s = 0.69x of single decode (gate 1.5x); the server serves one active request under a bounded queue | **already-measured / NO-GO** (`docs/ROADMAP.md` batching rows, `docs/CONTINUOUS-BATCHING.md`). Strata's finding argues for the queue, not for a new spike |
| an 8 GB/31 GB MoE-streaming row | Qwen3.6-35B-A3B (21 GB) on 8 GB + 31 GB RAM: 37–44 tok/s, 33–37 at 16k context, Windows installer, no card floor; v0.43.x is a prerelease `[P]` (`docs/BIGMOEONEDGE.md` §4–5) | the model is already a recognized-and-refused roadmap row (Qwen 3.6-35B-A3B; its Q0 gate is a pinned GGUF plus llama.cpp reference) | **reference only**: a target for that existing Q0 gate, not a new step; their number was measured by them on rented hardware and does not transfer |
| Raon-1 mini | new card (preview 2026-10-21): 28.8 B / 4.3 B active hybrid MoE, fork-only runtime, gated weights, 8 GB cards only with experts on the CPU `[P]` (`models/raon_1_mini.md`) | roadmap already carries Raon/Gemma 26B backends as an open gate | **already tracked** (roadmap Cards row); no action |
| intra-layer FFN row split | the KINDI.md tandem paste proposes per-matrix CPU/GPU slicing and a CPU drafter; the base's own check shows upstream Kindi already puts 55% of a host layer's FFN rows on the CPU with the card busy, and the paste's K6000 table disagrees with itself 2x (`docs/KINDI.md` §tandem paste) | our dense tandem splits at layer granularity; overlap is designed and measured (worker thread not justified; head work 17.6 ms/token and double-buffered staging first), and a hybrid FFN-on-CPU option is listed in `CPU_GPU_TANDEM_DESIGN.md` phase 4.1 | **parked, already in design**: the gate is the existing 24K ≥12.4 tok/s tandem gate with `kindi verify` 4/4; no new step |
| the base's view of Kindi's install path | v0.6.0 adds a Windows PowerShell → WSL2 installer and states "native Windows is not in 0.6.0"; BigMoeOnEdge's table calls Kindi Linux-only `[P]` | this fork runs natively on Windows (CPU engine compiled here, `.cu` through CuPy; `kindi verify`/`serve` ran on this host) | **rejected as a source**: the base tracks upstream v0.6.0, not this tree's v0.6.1+ state; read this repository's `CHANGELOG.md` for install claims |

One candidate was adopted as a proposal (REQ-0028); nothing was implemented in this pass, which is
the expected outcome of a scan. No number above is a Kindi measurement.

## Follow-up 2026-11-17 (the prefill-share row, closed for the 2B)

The baseline row "prefill dominates wall clock" (**parked, cheap**, gate: our own measured split on
the 2B/27B packs at 4K and 8K) is measured for the 2B. `kindi bench` now times one whole request
(`--split-at`, default 4096,8192; `--split-generated`, default 128) and reports each phase's share of
wall clock, so the two rates are no longer left for the reader to combine.

| prompt + reply | prefill | decode | share (prefill / decode) |
|---|---|---|---|
| 512 + 128 | 0.08 s (6,189 tok/s) | 0.52 s (244.5 tok/s) | **13.6% / 86.4%** |
| 4,096 + 128 | 0.94 s (4,349 tok/s) | 0.53 s (242.7 tok/s) | **64.1% / 35.9%** |
| 8,192 + 128 | 2.55 s (3,215 tok/s) | 0.54 s (233.2 tok/s) | **82.2% / 17.8%** |

Verdict: **adopted and measured** (2B, i7-8086K + RTX 4060 Ti, two runs per context, medians;
[results](../../docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-bench-split/RESULTS.md)). What the
measurement changed:

- the candidate's claim holds only above roughly 1K prompt tokens: at 512 the request is 86% decode,
  at 4K 64% prefill, at 8K 82% prefill. A fixed "prefill dominates" reading would be wrong for short
  chat turns;
- the split is *not* derivable from the two numbers `kindi bench` already printed. The `prompt
  reading` line times one token per `feed` call (300.6-304.5 tok/s warm, 63.0 on a cold first run),
  while a real chunked prefill runs at 4,349 tok/s at 4K and 3,215 at 8K - 11-14x faster. From 302
  tok/s a 4K request would look like 13.6 s of prefill; it spends 0.94 s.

The 27B half of the original gate stays open (no 27B run on this card in this pass), and the share is
one-request, greedy, 128-token, default chunk size; a second concurrent request or a different chunk
size would move it.

## How a later scan appends

Add a `## Scan <date>` section with **Last checked**, the knowledge base **HEAD**, the files read and
a table like the one above. Verdicts are `adopted` (with the commit), `parked` (with the gate that
would close it), `rejected` (with the reason) or `already-have` (with the path that proves it). Keep
earlier sections as written; a reversed verdict goes in a new row that names the row it supersedes.

## Check attempt 2026-11-17 (knowledge base unavailable; public comparison triaged)

`$KINDI_AI_INVENTORY` was unset in this session's shell and in both Windows scopes (User and
Machine), and the path is deliberately not stored in the repository, so the base itself was not
read. Nothing was taken from it and no `[M]` number was adopted.

The user supplied a written comparison of Kindi 0.6.0 with Strata, KTransformers, BigMoeOnEdge and
`tandem.cpp`. The three projects were read from their public sources (read-only pages; no scripts and
no models), and the claims were checked against the tree:

| source | read | verdict |
|---|---|---|
| BigMoeOnEdge (`Helldez/BigMoeOnEdge`) | README and `docs/moe-streaming.md`, `hardware-planning.md`, `benchmarks.md`, `comparison.md`, `roadmap.md` | NVMe expert streaming (`O_DIRECT`, n=1 decode only), a measured `--auto` planner, any-MoE-GGUF via a llama.cpp patch, Anthropic+OpenAI servers. Triaged in [EXTERNAL-ENGINES.md](../../docs/EXTERNAL-ENGINES.md) "Strata, KTransformers and BigMoeOnEdge (2026-11-17)" and gated in [ROADMAP.md](../../docs/ROADMAP.md) "Memory-tier follow-ups"; the measurement half, `kindi io`, is implemented this round |
| Strata | the same comparison, and its row in BigMoeOnEdge's comparison page | hybrid CPU/VRAM expert mode and SSD reads for a 500B-class MoE; their numbers are their rig's. No new step: the CPU lane is memory-tier follow-up #2 |
| KTransformers | the AVX-512/AMX `kt-kernel` CPU expert path named in the comparison | same CPU-lane gate; its cache-line-aligned INT4/INT8 packing is follow-up #4 (mixed-precision tiers) |
| `tandem.cpp` (already in this ledger) | re-checked only through the comparison; no new file | its measured CPU/PCIe split is already prior-art row 1, met by `--tandem-cpu-layers rate` |

Adopted result: the I/O probe (`kindi/io_probe.py`, `kindi io`) and the gates in the two docs above.
No speed claim was adopted, and no other engine's number became ours. The knowledge base still needs
its path supplied before the next real incremental scan.

## Check attempt 2026-11-19 (path unavailable; repository-only ranking)

**Last checked:** 2026-11-19. `$KINDI_AI_INVENTORY` is unset in this session's shell, so the base was
not read and no knowledge-base candidate is recorded. The roadmap review therefore uses only this
repository's measurements and the tracked Intel fork; no `[M]` or `[P]` number was adopted. Gate for
the next scan: set the variable, then follow the incremental procedure from the last recorded commit
`1135eab` (or the base's then-current HEAD), comparing mtimes because its working tree lags git.

## Check attempt 2026-11-20 (path unavailable; repository-only ranking)

**Last checked:** 2026-11-20. `$KINDI_AI_INVENTORY` is unset in this session's shell and in both the
Windows User and Machine scopes, so the base was not read and no knowledge-base candidate or number
is recorded. The ranking in [ROADMAP.md](../../docs/ROADMAP.md) "Follow-ups from the 2026-11-20
review" is repository-derived: the current v0.6.1 comparison, the 9B default-vs-TC smoke page, the
grammar-cost page, the handoff tests and the release checklist. Files reviewed for this pass:
`docs/ROADMAP.md`, `CURRENT-VS-KINDI-0.7.md`, `docs/development/AI-INVENTORY-SCAN.md`,
`docs/benchmarks/2026-11-20-i7-8086k-rtx4060ti-9b-default-vs-tc/RESULTS.md`,
`docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-grammar-cost/RESULTS.md`, `docs/API.md`, and
`tests/test_agents_handoff.py`. Nothing was adopted from the knowledge base; the roadmap items are
this repository's own gaps and each carries a gate.

Gate for the next real scan: set `$KINDI_AI_INVENTORY`, then follow the incremental procedure from the
last recorded commit `1135eab` (or the base's then-current HEAD), comparing mtimes because its working
tree lags git. The path is deliberately not stored in this repository.
