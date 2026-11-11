# Tandem/CPU recovery status

Status: **first CPU slice, 2026-11-11. The CPU engine now exposes per-layer entry points; no
`TandemEngine`, layer placement or measured CPU+GPU split is present, and no tandem support is claimed.**
This page records what `main` actually contains, why the interrupted WIP cannot be merged as-is, and
the gate for deleting it. Archived with the other no-go work in [ROADMAP.md](ROADMAP.md).

Benchmark coverage: [benchmarks/CROSS-MODE.md](benchmarks/CROSS-MODE.md) records the measured CPU-only and
GPU-only 2B/4B/9B cells at positions 100-24,000 on the i7-8086K + RTX 4060 Ti. Its GPU+CPU tandem rows are
`n/a` until gates 3 and 4 below pass; no tandem number is claimed.

## Answer

At `main` = `2abf9ad` (the recovery audit), the repository had:

- the **host-RAM preflight** in `kindi/hostram.py` (37 lines, used by `kindi/tp.py`);
- the existing **AVX2/FMA/F16C CPU engine** (`kindi/cpu.py`, `kindi/kcpu.c`);
- the existing **GEMV weight layout** (`kindi/layout.py:repack`, `reorder_head`).

The 2026-11-11 slice adds:

- `kc_embed`, `kc_layer` and `kc_head` in `kindi/kcpu.c`;
- `Engine.begin`, `layer`, `head`, `feed_layers`, `step_layers` and an in-place `hidden` view in
  `kindi/cpu.py`.

It still does **not** have:

- `kindi/tandem.py`, `kindi/tandem_kernels.cu`, or a `TandemEngine`;
- a layer-contiguous streaming pack format or a converter that emits one;
- CPU ISA detection, an AVX-only fallback, or the WIP's CPU dispatch;
- any working CPU+GPU layer split, layer placement or GPU/CPU state handoff.

So there is no uncorrupted tandem/CPU implementation in `main` that makes the WIP branches
disposable. **Do not delete the WIP branches yet.**

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

This is a CPU-side primitive, not a tandem runtime: no GPU layer is handed off, no layer placement
or state ownership exists, and the RTX 4060 Ti is not used by this slice. No tandem benchmark is
claimed.

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

The native Windows build works when `KINDI_CC`/`KINDI_LLVM` select clang 18.1.8 and libomp; the CPU
per-layer gate above was rebuilt and run on this host. The remaining work is the GPU-side handoff:
layer placement, ownership of K/V and GDN state, hidden-state copies, and a two-backend exactness
test. The WIP's CUDA doorbell kernel has still not been compiled or exercised. A Linux host with
gcc/OpenMP, or CI, is still required for the AVX-only and portable CPU gates.

## Cleanup gate

Do not remove either WIP branch until a replacement on `main` passes all of these:

1. **CPU gate:** the supported CPU flags build; an AVX-only host fails with an actionable message or
   uses a real fallback; `kindi verify` passes on 2B and 4B; CPU tests pass. The per-layer entry point
   and its 2B/4B/9B exactness check are now present; the AVX-only fallback remains open.
2. **Streaming gate:** `convert` emits a real layer-contiguous pack (or an explicit compatible
   alternative), and existing packs still load unchanged.
3. **Tandem gate:** a clean `TandemEngine` runs layer-by-layer with CPU/GPU placement, and greedy
   output is token-exact against the full-GPU engine on at least 2B and 9B.
4. **Benchmark gate:** decode throughput and peak VRAM are recorded for a model that does not fit
   VRAM, with the CPU/GPU split stated.
5. **Cleanup gate:** the replacement is committed, the full CPU suite and the GPU verify pass, and
   `git status` is clean. Only then delete the WIP branches or the backup snapshot.

Until then, the WIP branches are the only copy of the CPU ISA experiments and the tandem design
sketches. Keep them.

## Next implementation slice

Start from the design, not the corrupt file. The CPU entry point is done; the next steps are:

1. Add a clean `TandemConfig` and layer-placement planner, with one explicit CPU layer and the rest
   on the GPU.
2. Add the hidden-state handoff and keep K/V and GDN state on the backend that owns each layer.
3. Run the per-layer gate against the full-GPU engine on 2B and 9B before any speed claim.
4. Build `hostram` pinned/mmap primitives and a layer-contiguous pack format only after the
   placement path is exact.
5. Benchmark only after the exactness gate passes.

The design is restored at [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md). The current CPU
roadmap remains [CPU-SUPPORT.md](CPU-SUPPORT.md).
