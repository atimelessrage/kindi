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
- Re-measured and bisected 2026-11-16 (RTX 4060 Ti, current tree): the gate is unchanged
  (guard bypassed, `KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1`: 2 exact, 2 fails at the
  2026-11-11 points — prompt 2 token 6 with llama.cpp's 0.0447 gap, prompt 3 token 37), and
  the bisect corrected the *reason*, not the outcome. Split decode is arithmetically
  equivalent to the standard decode attention on this geometry — ≤2e-6 on synthetic inputs,
  on multi-query masking and on real model state at the failing step — so there is no
  masking, head-group or chunk-merge defect. The divergence is float association: the
  per-chunk softmax merge orders its additions differently, the residual stream amplifies
  the ~6e-8 median difference to ~1e-1 in the logits, and that crosses the 0.0447 gap.
  Failing needs FD in both the feed and the decode phase (either alone is 40/40).
  `width % 1024` keys on a quantity the attention never reads, so it is not why the 4B
  fails: the 4B and 9B share 16-head/4-KV/256 geometry, and the 9B passes because its
  prompts are not on a 0.04-logit tie. Guard stays; no speed gain was recorded (a gain on a
  diverging path is not shippable). Evidence:
  `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-fd-4b/FD-BISECT.md` (analysis) and
  `…/RESULTS.md` (gate).

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

## DEC-0009: The server keeps its own backend booleans until a registry migration round

- Requirements: REQ-0019 (backend plugin interface), REQ-0018 (structured output)
- Context: `Kindi.__init__` still dispatches on `man["backend"]` and builds
  `nanbeige_gen.NanbeigeBackend`, `lightning_gen.LightningBackend` and `kindi.gemma4.Engine`
  directly, while `kindi/backends/*` holds a registry-driven implementation of the same
  contract (`BackendCapabilities`, `resolve`). The 2026-11-16 audit (F11) called that the
  largest structural weakness, and F3 was its concrete damage: the model id a client saw came
  from a second, incomplete list in `config.py`. The adapter stack is not a drop-in
  replacement yet — the `{x}_gen` loops own their family's structured-output constraint,
  draft policy and stop set, and the dense path owns the prompt cache.
- Decision: fix the observable split now (`kindi/modelids.py` is the one resolver for a
  pack's model id, used by both `config` and the engine gateway) and keep the dispatch
  where it is until a migration round can move the families behind
  `BackendCapabilities` with their GPU gates re-run. A half migration would be worse than
  the split: two code paths answering capability questions differently under one flag is
  exactly what F11 describes.
- Consequences: capability changes still have to be made in both places until then. Every
  new family must register a `model_id_for_manifest` hook so the reported id cannot drift
  again, and the migration is a named round, not a side effect of a feature commit.

## DEC-0010: One constant-time bearer check for every HTTP server

- Status: accepted
- Date: 2026-11-22
- Requirements: none (2026-11-22 audit follow-up)
- Context: `kindi/server.py` compared `Authorization` with `hmac.compare_digest`, but
  `kindi/dsv41_server.py` compared the same header with `==`. Two servers, two answers to one
  security question; a future server could copy either one.
- Decision: `kindi.api.auth.bearer_matches(header, key)` is the only implementation. It treats an
  empty key as "no authentication" (the loopback default) and compares the full `Bearer <key>`
  value in constant time. Both servers call it.
- Consequences: a new HTTP surface must import the helper, and `tests/test_server_auth.py` fails
  if a server contains a literal `== "Bearer "`. The helper has no FastAPI or HTTP dependency, so
  it is unit-tested on a CPU-only host and stays Python 3.6-safe. The audit's engine/spec findings
  (broad `except` in `_is_turing`, the HB-4096 refusal after the repack) were left for the
  in-flight engine round that owns those uncommitted files.

## DEC-0011: LoRA adapters are a dense-GPU-only path in this port

- Status: accepted
- Date: 2026-11-25
- Requirements: none (official Kindi 0.7.2 intake: LoRA adapters)
- Context: official 0.7.2 adds PEFT LoRA adapters to the dense engine (`kindi/lora.py`, `k_lora`/
  `k_lora2` in `kindi.cu`, the chunked prompt path, the cache fingerprint, `--lora`/`KINDI_LORA`).
  This fork's default engine for the 2B/4B/9B is the plan engine (`kindi.tp.TPEngine`, `KINDI_TP`
  unset), whose per-token GEMV path and mixed layout do not expose the `gemv` hook the adapter
  rides on; the tandem engine and the CPU engine have their own GEMV paths too. Upstream's own
  port refuses the plan engine as well (it defaults to the plain engine, this fork does not).
- Decision: port the feature where it is coherent - the plain single-card GPU engine - and make
  every other engine refuse an adapter with an actionable message (`KINDI_TP=0` for the plan
  engine, the tandem and CPU refusals) instead of silently ignoring it. The converted adapter
  (`PACK/lora/<hash>/`) and the fingerprint entry are engine-independent, so a later plan-engine
  implementation does not have to reconvert anything.
- Consequences: on a one-card default run (`KINDI_TP` unset) `--lora` stops with the message;
  documenting that is part of the feature. The kernels are compiled but unrun on this host (no
  CUDA device), so no speed or exactness claim is made here; the device-free tests cover the
  fold, the readers, conversion and the plumbing. The port also fixes an upstream reader defect:
  an `.npz` adapter's `lora_alpha` scalar matched the `lora_` tensor filter and aborted the
  conversion (`tests/test_lora.py::test_read_adapter_reads_npz_pairs_and_refuses_half_a_pair`).

## DEC-0012: a Bonsai `-mtp` file converts as its base model; its head is recorded, not drafted

- Status: accepted
- Date: 2026-11-25
- Requirements: none (Bonsai 2.5 27B `-mtp` intake; docs/BONSAI-2-SUPPORT.md section 1a)
- Context: Prism ML publishes the 2.5 27B draft head inside the model GGUF as a 65th block
  (`blk.64`): one decoder layer plus `nextn.{eh_proj,enorm,hnorm,shared_head_norm}`, stored as
  Q4_0 while the language model is PTQ1_0. Kindi's `kindi convert` keyed the model profile on
  `qwen35.block_count`, so it refused the file on both the shape and the tensor-type gate. Kindi's
  own drafter (`kindi/mtp.py:MTPDraft`) is written for the registered 2B head: a separate `.npz`
  with int8 rows, head dimension 256, on the tensor-core plan engine, and it reuses the main
  model's output head. The 27B head is one 256-wide attention block in the same file and has never
  been converted, wired or measured; this host has no CUDA device to test a GPU path.
- Decision: treat a file whose block count is one past a known profile, and whose extra block
  carries `nextn.eh_proj`, as that profile plus a head. Convert the language-model blocks,
  exclude the head block from the tensor-type gate, and record the block index and tensor names
  in `kindi_pack.json` under `mtp_head`. Do not extract, quantize or load the head for drafting,
  and do not imply a speedup: `KINDI_SPEC=mtp` still only drafts with the 2B `.npz` head.
- Consequences: the `-mtp` file is a supported model (catalog id `bonsai-2.5-27b-mtp`, the same
  pack as the base file) and switching between Bonsai 2, 2.5 and the 2.5 `-mtp` file is a
  `kindi use`/`--pack` choice. Drafting with a 27B head remains open work: it needs a head
  extractor, an engine path that accepts head dimension 128 and a GPU run to verify exactness
  and speed. The catalog entry therefore carries the base fit estimate, not a speculative one.
