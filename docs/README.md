# docs/

Documentation index. Where two pages overlap, the one named here as authoritative wins; the
others are kept as history, design notes or evidence and are not maintained in parallel.

| subject | authoritative page | the rest are |
|---|---|---|
| Cards and engines | [SUPPORT-MATRIX.md](SUPPORT-MATRIX.md) (generated, per-card rows) and [CARDS.md](CARDS.md) (model/settings) | [CARD-SUPPORT.md](CARD-SUPPORT.md), [P106.md](P106.md), [P4000.md](P4000.md), [M4000.md](M4000.md), [BONSAI-2-SUPPORT.md](BONSAI-2-SUPPORT.md), [CPU-SUPPORT.md](CPU-SUPPORT.md), [AMD.md](AMD.md), [INTEL-ARC.md](INTEL-ARC.md) are per-card notes, [OLD-CARD-RIG.md](OLD-CARD-RIG.md) the session script for the unmeasured P106/P4000/M4000 |
| Hardware identification and tuning | [AUTOTUNE.md](AUTOTUNE.md) (`kindi caps`, `kindi autotune`) | per-card notes in [CARDS.md](CARDS.md), generated rows in [SUPPORT-MATRIX.md](SUPPORT-MATRIX.md), [REQ-0022](development/requirements/REQ-0022.md) |
| Tandem (CPU+GPU split) | [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md) | [TANDEM-STATUS.md](TANDEM-STATUS.md) is the dated status snapshot, [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md) the design, [MOE-TANDEM.md](MOE-TANDEM.md) design-only, [TANDEM-PROPOSAL-REVIEW.md](TANDEM-PROPOSAL-REVIEW.md) the review |
| Measurements | [BENCHMARKS.md](BENCHMARKS.md) and [benchmarks/](benchmarks/) | raw dumps stay local under `bench-logs/` (gitignored); every file a doc links to must be tracked in `benchmarks/` |
| Context growth | [CONTEXT-GROWTH.md](CONTEXT-GROWTH.md) (per-token KV and fixed-state formulas, with 4K/16K/64K examples for Bonsai, Nanbeige, Lightning, DeepSeek V4.1 and Gemma 4) | [BONSAI-2-SUPPORT.md](BONSAI-2-SUPPORT.md), [DEEPSEEK.md](DEEPSEEK.md) and [PROMPT-CACHE.md](PROMPT-CACHE.md) keep family and cache-policy detail; they do not duplicate the comparison |
| Server API | [AGENTS.md](AGENTS.md) | `docker-compose` and server behavior are in [DOCKER.md](DOCKER.md) |
| Installer and setup | [INSTALL.md](INSTALL.md); client setup in [WINDOWS.md](WINDOWS.md) | [DOCKER.md](DOCKER.md) for containers |
| Operations and safety | [SECURITY.md](SECURITY.md) (secrets, history redaction, push policy) and [BACKUP.md](BACKUP.md) (verified offsite bundle) | [RELEASE-CHECKLIST.md](RELEASE-CHECKLIST.md) is the release procedure |
| Claims | [claims/README.md](claims/README.md) (one `claims.json` per measured gate, checked by `tools/check_claims.py`), [ROADMAP.md](ROADMAP.md) status cells, [VERIFICATION.md](VERIFICATION.md) for how a claim is checked | [CURRENT-VS-KINDI-0.7.md](../CURRENT-VS-KINDI-0.7.md) is the dated comparison (§25 of the 0.5 file is the first 0.6.0 pass) |
| Ternary-MoE census | [TERNARY_MOE.md](TERNARY_MOE.md) (canonical rows and counts, checked by `scripts/check_ternary_moe.py`) | [ROADMAP.md](ROADMAP.md) keeps the Qwen3.6 backend gates and a link, not a second copy |
| What we take from other engines | [EXTERNAL-ENGINES.md](EXTERNAL-ENGINES.md) (llama.cpp, Strata, tandem.cpp: learning → verdict → roadmap item) | the roadmap rows it creates are dated "Prior-art follow-ups (2026-11-14)" |

Tracked evidence promoted from `bench-logs/` on 2026-11-14: `blackwell-fp8-ada-refusal-20261112/`,
`blackwell-fp8-dispatch-20261112/`, `dory-d0/`, `raon-r0/`, `tandem-cache/`. The local `bench-logs/`
copies are unchanged; history in `docs/development/AGENTS-HISTORY.md` still names the original
local locations.
