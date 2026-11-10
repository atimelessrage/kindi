# Tandem/CPU recovery status

Status: **recovery audit, 2026-11-09. No tandem runtime is present and no tandem support is claimed.**
This page records what `main` actually contains, why the interrupted WIP cannot be merged as-is, and
the gate for deleting it. Archived with the other no-go work in [ROADMAP.md](ROADMAP.md).

## Answer

At `main` = `2abf9ad`, the repository has:

- the **host-RAM preflight** in `kindi/hostram.py` (37 lines, used by `kindi/tp.py`);
- the existing **AVX2/FMA/F16C CPU engine** (`kindi/cpu.py`, `kindi/kcpu.c`);
- the existing **GEMV weight layout** (`kindi/layout.py:repack`, `reorder_head`).

It does **not** have:

- `kindi/tandem.py`, `kindi/tandem_kernels.cu`, or a `TandemEngine`;
- a layer-contiguous streaming pack format or a converter that emits one;
- CPU ISA detection, an AVX-only fallback, or the WIP's CPU dispatch;
- any working CPU+GPU layer split.

So there is no uncorrupted tandem/CPU implementation in `main` that makes the WIP branches
disposable. **Do not delete the WIP branches yet.**

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

## Why this host cannot finish the runtime

This Windows machine has the RTX 4060 Ti and a working native CUDA driver, but no `gcc`, `clang` or
`cl` on `PATH`. The CPU engine cannot be rebuilt or tested here, and the tandem CPU worker depends on
that engine. The WIP's CUDA doorbell kernel has also not been compiled or exercised. A Linux host
with gcc/OpenMP, or CI, is required for the CPU acceptance gate.

## Cleanup gate

Do not remove either WIP branch until a replacement on `main` passes all of these:

1. **CPU gate:** the supported CPU flags build; an AVX-only host fails with an actionable message or
   uses a real fallback; `kindi verify` passes on 2B and 4B; CPU tests pass.
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

Start from the design, not the corrupt file:

1. Build `hostram` pinned/mmap primitives with CPU-only unit tests and a fake CuPy layer.
2. Implement a real layer-contiguous pack writer plus a compatibility check for existing packs.
3. Add a clean `TandemConfig` and layer-placement planner before touching the engine.
4. Integrate one layer at a time behind a disabled-by-default flag; run token-exact verification
   after each step.
5. Benchmark only after the exactness gate passes.

The design is restored at [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md). The current CPU
roadmap remains [CPU-SUPPORT.md](CPU-SUPPORT.md).
