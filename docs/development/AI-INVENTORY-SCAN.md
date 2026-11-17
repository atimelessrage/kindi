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
| the knowledge base as a source for upstream Kindi state | `docs/KINDI_CHANGELOG.md` tracks upstream `v0.3.0` (feed check 2026-10-30); `docs/KINDI.md` is explicitly research-only, `[?]`, "never run here" | this tree contains upstream `v0.6.0` (`CURRENT-VS-KINDI-0.6.md`) | **rejected as a source.** Three minor versions stale; read upstream `CHANGELOG.md`/releases directly when upstream state matters. |

Nothing was adopted in this scan, so no roadmap or requirement step was created and nothing was
implemented — which is the expected outcome of a baseline scan.

## How a later scan appends

Add a `## Scan <date>` section with **Last checked**, the knowledge base **HEAD**, the files read and
a table like the one above. Verdicts are `adopted` (with the commit), `parked` (with the gate that
would close it), `rejected` (with the reason) or `already-have` (with the path that proves it). Keep
earlier sections as written; a reversed verdict goes in a new row that names the row it supersedes.
