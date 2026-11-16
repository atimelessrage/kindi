# Decision log

Append-only record of choices that are expensive to reverse or that define a support
claim. A decision is not a requirement; requirements say **what** must be true, decisions
explain **why this approach** was chosen.

Use this shape:

```markdown
## DEC-NNNN: Title

- Status: accepted | superseded by DEC-NNNN
- Date: YYYY-MM-DD
- Requirements: REQ-NNNN, ...
- Context: ...
- Decision: ...
- Consequences: ...
```

## DEC-0001: Exactness outranks speculative speed

- Status: accepted
- Date: 2026-11-11
- Requirements: REQ-0001, REQ-0007
- Context: Some fast paths (split decode attention, reduced-precision K/V, tensor-core
  plans) can change the first divergent token even when aggregate quality looks fine.
- Decision: Keep the exact head and the verified greedy path as the default contract.
  Experimental speed paths must have an explicit opt-in or narrow guard and a recorded
  verification result.
- Consequences: A path that measures faster but is not token-exact cannot silently become
  the default. `kindi verify` and per-card records are release gates.

## DEC-0002: Tensor-core plans refuse unsupported Hadamard widths

- Status: superseded by DEC-0008
- Date: 2026-11-11
- Requirements: REQ-0003
- Context: The plan engine was built for 512- and 1024-point Hadamard blocks. The 9B pack
  uses 4096, which the current plans cannot represent.
- Decision: Refuse the 9B before graph construction with an actionable message. Keep the
  plain engine as the fallback. Do not pad or approximate the block to make the plan run.
- Consequences: 9B tensor-core support remains an explicit open gap. A future
  implementation must add kernels and verification, not just remove the guard.

## DEC-0003: Guard the 4B split-decode path instead of changing the kernel

- Status: accepted
- Date: 2026-11-11
- Requirements: REQ-0001, REQ-0003
- Context: With `KINDI_FD=1`, the 4B's 2560-wide plan diverged at prompt 2 token 6 and
  prompt 3 token 37. With FD disabled, it was 3 exact prompts plus an accepted llama.cpp
  tie.
- Decision: Disable split flash-decode automatically for non-1024-multiple widths and log
  the fallback. Keep the standard decode attention. Do not alter the tensor-core arithmetic
  without a separate exactness investigation.
- Consequences: The 4B tc path has a conditional pass, not a claim that split decode is
  correct for that geometry. The guard is covered by a CPU-only regression test.
- Re-measured 2026-11-16 (RTX 4060 Ti, current tree): with the guard bypassed,
  `KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1` verify on the 4B is 2 exact, 2 fails —
  prompt 2 diverges at token 6 (gap 0.0447, not a tie), prompt 3 at token 37,
  exactly the 2026-11-11 points. Guard stays; no speed gain was recorded
  (a gain on a diverging path is not shippable). Evidence:
  `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-fd-4b/RESULTS.md`.

## DEC-0004: A streamed g5 cache must match the current tensor slices

- Status: accepted
- Date: 2026-11-11
- Requirements: REQ-0003, REQ-0006
- Context: The tied-head alias changed the head key from `output.weight` to
  `token_embd.weight` for the 2B/4B, but the streamed `g5` cache path trusted an older key
  set and built a plan that read a missing tensor.
- Decision: `load_stream()` refuses a cache whose keys do not cover the current
  `_slices()` result. `Shard._load()` may rebuild it; the cache is a performance artifact,
  never an authority.
- Consequences: First start after a layout/alias change can be slower while the cache is
  rebuilt. Correctness wins. Cache fingerprints and key sets are part of the engine
  contract.

## DEC-0005: Generated documentation is checked, not trusted

- Status: accepted
- Date: 2026-11-11
- Requirements: REQ-0009, REQ-0010
- Context: Configuration and support facts had drifted between code, docs and hand-written
  tables.
- Decision: Generate the configuration reference and support matrix from source, and fail
  tests on drift. Keep hand-written architecture and user explanations focused on meaning,
  not duplicated inventories.
- Consequences: A code edit that changes a generated row requires regeneration. The
  generator command and its `--check` mode are part of the handoff.

## DEC-0006: Requirements are one file per stable ID

- Status: accepted
- Date: 2026-11-11
- Requirements: REQ-0010
- Context: Requirements change in both directions; a single monolithic list makes it hard
  to recover the prior wording and the commit that changed it.
- Decision: Store one requirement per `REQ-NNNN.md` file. Use git history per file for
  added and last-modified commits; enumerate deleted files for the deleted section.
- Consequences: Requirement IDs are permanent. Renaming an ID is a removal plus an
  addition. The generated history is checked by `tests/test_project_memory.py`.

## DEC-0007: The tandem's GPU half is the plan engine by default

- Status: accepted
- Date: 2026-11-14
- Requirements: REQ-0013
- Context: The tandem's GPU owner was the plain engine, which uploads the whole f32 file, always
  keeps fp16 K/V and cannot load the full 27B at a 24K context on an 8 GB card. An offloaded layer
  therefore freed only its weights, and on the 27B the split saved ~250 MiB out of 7.8 GiB.
- Decision: `TPEngine` accepts a layer subset and steps one prebuilt plan per block, so it can be
  the owner (`--tandem-gpu auto|plan|plain`; `auto` = plan on one card, `KINDI_TP=0` and two-card
  hosts keep the plain engine). The planner charges each owner its real allocation. Each block's
  residual add is resolved inside its plan with `k_padd`, because a CuPy call per block cost 17% of
  a 2B token.
- Consequences: Same 27B placement at 24K drops from 7,721 to 6,004 MiB peak (1,985 MiB free after load), which is what makes a 49,152-token context fit on this card; exactness and the joint prompt
  cache are re-verified with the plan owner. A plan owner is one card, so a split across two GPUs
  still uses the plain engine. The plan engine's 1024-token context floor applies only to full
  engines now (a partial engine has no chunked or batched prompt path and needs 256).

## DEC-0008: 9B tensor-core support via the 4096 Hadamard fold + high-precision activations

- Status: accepted
- Date: 2026-11-15
- Requirements: REQ-0001, REQ-0003
- Context: DEC-0002 refused 4096-point Hadamard blocks (the 9B) before graph construction.
  Since then the fold was implemented as 512-point chunks plus `k_fwht_hi` to finish a block
  above 512 (`kindi/engine.py: fold_args`, HB = 4096 on the 9B), and the tensor-core verify
  was measured on the RTX 4060 Ti: with the default ~8-bit activations one borderline prompt
  flips at token 36 (3/4), but with high-precision activations (~14 bits, same weight traffic)
  the 9B passes 4/4 at 122.6 tok/s against 74.6-80.0 on the plain engine. `kindi/engine.py`
  therefore defaults `KINDI_TC_HP` on for HB-4096 packs (`KINDI_TC_HP=0` asks for the default
  activations, measured 3/4) and `kindi/spec.py` keeps refusing HB-4096 without them.
- Decision: Supersede DEC-0002. The 9B runs on tensor-core plans with HP activations as the
  measured-exact default; the plain engine remains the fallback (`KINDI_TP=0`). No padding
  or approximation of the block was needed — the exact fold + verified activations is the
  support.
- Consequences: The open gap DEC-0002 recorded is closed. Revision alone gains no new
  performance (the kernels and the HP default already shipped; this entry corrects the log
  so the supported path is not read as refused). DEC-0003's 4B flash-decode guard is
  untouched: no 4B FD-on measurement shows a gain worth the exactness risk, and the guard
  stays.
