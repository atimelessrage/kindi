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

- Status: accepted
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
- Consequences: Same 27B placement at 24K drops from 7,721 to 6,004 MiB peak (1,985 MiB free after
  load), which is what makes a 49,152-token context fit on this card; exactness and the joint prompt
  cache are re-verified with the plan owner. A plan owner is one card, so a split across two GPUs
  still uses the plain engine. The plan engine's 1024-token context floor applies only to full
  engines now (a partial engine has no chunked or batched prompt path and needs 256).
