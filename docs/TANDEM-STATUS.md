# Tandem/CPU recovery status

Status: **dense runtime implemented and token-exact on the 2B, 2026-11-11. The CPU engine's
per-layer entry point is wired to a partial GPU engine in both dense placements; the MoE
expert-cache approach is designed but not implemented. No model larger than one GPU has been
measured, and no tandem support beyond the 2B gate is claimed.**
This page records what `main` actually contains, why the interrupted WIP cannot be merged as-is, and
the gate for deleting it. Archived with the other no-go work in [ROADMAP.md](ROADMAP.md).

Benchmark coverage: [benchmarks/CROSS-MODE.md](benchmarks/CROSS-MODE.md) records the measured CPU-only and
GPU-only 2B/4B/9B cells; its GPU+CPU tandem rows remain `n/a`. The dense runtime's real 2B exactness and
one-run timings are in [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md).

## Answer

At `main` = `2abf9ad` (the recovery audit), the repository had:

- the **host-RAM preflight** in `kindi/hostram.py` (37 lines, used by `kindi/tp.py`);
- the existing **AVX2/FMA/F16C CPU engine** (`kindi/cpu.py`, `kindi/kcpu.c`);
- the existing **GEMV weight layout** (`kindi/layout.py:repack`, `reorder_head`).

The 2026-11-11 slices add:

- `kc_embed`, `kc_layer` and `kc_head` in `kindi/kcpu.c`;
- `Engine.begin`, `layer`, `head`, `feed_layers`, `step_layers` and an in-place `hidden` view in
  `kindi/cpu.py`;
- `kindi/tandem.py`: `TandemEngine` in `cpu-first` and `gpu-first` placements, plus a cache-free
  `TandemRunner` for the server;
- `engine.Engine(layers=..., embed=..., head=...)`: a partial GPU engine that uploads only the
  owned layer tensors and keeps only their GDN/KV state.

It still does **not** have:

- a MoE expert cache, because there is no Bonsai MoE backend or router path to attach it to;
- joint prompt-cache checkpoints for the two ownership domains, so the tandem server reads the
  prompt from position 0 on each request;
- a measured 27B-class (larger-than-VRAM) split or a peak-VRAM comparison;
- a layer-contiguous streaming pack format or a converter that emits one;
- CPU ISA detection or the WIP's CPU dispatch; the AVX-only fallback is a separate CPU task.

So the dense runtime is no longer the missing piece, but **do not delete the WIP branches yet**:
the MoE design, the joint cache and the larger-model gate remain open.

## First slice: per-layer CPU entry point (2026-11-11)

`kindi/kcpu.c` now exports three calls beside the whole-model `kc_step`:

- `kc_embed(Model*, tok)` puts one token's embedding in `m->h`;
- `kc_layer(Model*, il, pos)` runs exactly one block. It treats `m->h` as the complete residual
  entering the block and leaves the complete residual leaving it in `m->h`; `m->y` is scratch and
  is not carried between calls;
- `kc_head(Model*)` runs the final norm and exact head on that residual and returns the greedy id.

`kindi/cpu.py` exposes `begin(tok)`, `layer(il, pos)`, `head()`, `feed_layers(tok, pos)` and
`step_layers(tok, pos)`, plus `hidden`, `get_hidden()` and `set_hidden()` for a future backend
handoff. `step()` and `feed()` still use the original single-OpenMP-region path and are unchanged.

Verification on the native Windows i7-8086K (6 threads, clang 18.1.8): a five-token sequence
through `step_layers` matches `step` token-for-token and leaves the same hidden state on the 2B, 4B
and 9B packs (`pytest tests/test_cpu_layers.py -q` with `KINDI_CPU_PACK`; 5 passed for each pack).
With no pack the wrapper tests are 4 passed, 1 skipped; the compiled CPU smoke group is 16 passed.

This is no longer only a CPU-side primitive. On the i7-8086K + RTX 4060 Ti, the real 2B pack is
4/4 prompts token-exact (40/40 tokens each) in the full-GPU baseline and in both tandem placements:
CPU 0..5 / GPU 6..23 with the GPU head, and GPU 0..17 / CPU 18..23 with the CPU head. One run each
measured 189.9 tok/s full GPU, 70.2 tok/s CPU-first and 70.9 tok/s GPU-first. The HTTP path also
answered a real chat request in both head-ownership directions. These are functional measurements,
not a speed or support claim; see [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md).

## WIP inventory

The interrupted worktree is on `wip/main-interrupted-snapshot-20261109-045352` (`81cf295`), with a
backup on `wip/main-interrupted-backup-20261109-045352` (`4449327`). It is based on the older
`0fc024f` commit and contains unrelated reversions, so a branch merge would revert later audit,
server and test work. Any salvage must be selective.

| Item | WIP state | Recovery verdict |
|---|---|---|
| `docs/CPU_GPU_TANDEM_DESIGN.md` | 647 lines, intact; committed Git-blob SHA-256 `8d1cb927ad3543a4895c427699e697be68f3107310943cde6c8d9a211cdd9f92` (working copy may be CRLF) | **Restored in this branch.** It is a design, not an implementation. |
| `kindi/tandem.py` | 280 lines; only historical blob `ccb8841748`; parse fails at line 131 | **Corrupt; rewrite from the design.** It contains duplicated classes, diff markers, broken indentation and placeholder methods. |
| `kindi/tandem_new.py` | 4 bytes: `test` | **Delete from any salvage; no value.** |
| `kindi/hostram.py` WIP additions | 334 added lines, parses; pinned allocator, mmap loader, double buffer, LRU cache | **Candidate foundation**, but unintegrated and untested. Main keeps only the preflight for now. |
| `kindi/layout.py` WIP streaming functions | 214 added lines, parses | **Placeholder metadata only.** They compute offsets but do not reorganize pack data into layer-contiguous files; not a streaming layout yet. |
| `kindi/kcpu_cpuid.py`, `kindi/kcpu_cpuid.c` | 153 + 165 lines, Python parses | **Candidate CPU work**, but no compiler is available on this Windows host to build or validate it. |
| `kindi/cpu.py`, `kindi/kcpu.c`, `kindi/nanbeige_cpu.py` WIP changes | Parse; add ISA flags, AVX-only fallback and placeholder tandem worker | **Unverified.** The WIP test only checks imports/feature flags, not token exactness or a compiled fallback. |
| `kindi/tp.py` WIP changes | Parse; asymmetric split and per-card tensor-core backend | **Not tandem/CPU recovery**; incomplete (comments admit kernel splits are missing), and out of scope here. |
| `kindi/tandem_kernels.cu` | 224 lines | **Unbuilt.** No CUDA compile or doorbell test was run. |
| helper scripts (`fix_tandem*.py`, `test_brace*.sh`, `update_cpu.py`) | Parse or shell text | **Scratch artifacts; do not merge.** |

The only historical `kindi/tandem.py` blob is the corrupt one; there is no earlier valid
implementation to restore.

## Why the runtime is still unfinished

The native Windows build works when `KINDI_CC`/`KINDI_LLVM` select clang 18.1.8 and libomp; the dense
runtime and its two-backend exactness check now pass on this host. The remaining work is the
larger-model gate: a 27B-class model through both placements, peak-VRAM evidence, and a decision on
joint cache checkpoints. The WIP's CUDA doorbell kernel has still not been compiled or exercised,
and it is not used by the dense implementation. A Linux host with gcc/OpenMP, or CI, is still required
for the AVX-only and portable CPU gates.

## Cleanup gate

Do not remove either WIP branch until a replacement on `main` passes all of these:

1. **CPU gate:** the supported CPU flags build; an AVX-only host fails with an actionable message or
   uses a real fallback; `kindi verify` passes on 2B and 4B; CPU tests pass. The per-layer entry point
   and its 2B/4B/9B exactness check are now present; the AVX-only fallback remains open.
2. **Streaming gate:** `convert` emits a real layer-contiguous pack (or an explicit compatible
   alternative), and existing packs still load unchanged.
3. **Tandem gate:** a clean `TandemEngine` runs layer-by-layer with CPU/GPU placement, and greedy
   output is token-exact against the full-GPU engine on at least 2B and 9B. **2B: passed on
   2026-11-11 in both placements; 9B is still open.**
4. **Benchmark gate:** decode throughput and peak VRAM are recorded for a model that does not fit
   VRAM, with the CPU/GPU split stated.
5. **Cleanup gate:** the replacement is committed, the full CPU suite and the GPU verify pass, and
   `git status` is clean. Only then delete the WIP branches or the backup snapshot.

Until then, the WIP branches are the only copy of the CPU ISA experiments and the tandem design
sketches. Keep them.

## Next implementation slice

The dense placement and handoff are done. The next steps are:

1. Run the 9B tandem gate against the full-GPU engine, then a 27B-class model that does not fit
   the card; record peak VRAM for full GPU, CPU-first and GPU-first.
2. Decide the joint cache design: checkpoint both engines' state or keep the tandem path
   cache-free explicitly.
3. Measure and tune the CPU layer boundary (`--tandem-cpu-layers`) and prompt reading on the
   target small card.
4. Add a MoE expert-cache policy only after a MoE backend with per-expert tensors and router logits
   exists; it is a different algorithm, not a flag on this one.
5. Build `hostram` pinned/mmap primitives and a layer-contiguous pack format if the measurements
   justify them.

The design is restored at [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md). The current CPU
roadmap remains [CPU-SUPPORT.md](CPU-SUPPORT.md).
