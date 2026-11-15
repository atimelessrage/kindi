# Tandem/CPU recovery status

Where this sits: [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md) is authoritative for the runtime behaviour
and its measured rows; this page is the dated recovery snapshot and
[CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md) the design.

Status: **dense runtime implemented and token-exact on the 2B and 9B, 2026-11-12; real joint-cache
checks pass in both placements, including saved-opening reloads. The CPU engine's
per-layer entry point is wired to a partial GPU engine in both dense placements; the MoE
expert-cache path is designed in [MOE-TANDEM.md](MOE-TANDEM.md) and has a model-free state-machine
prototype plus a [Gemma 4 26B adaptation](GEMMA4-26B-MOE.md), but no backend or integration. The
27B `cpu-first` and `gpu-first` splits are token-exact at 8,192 context (peaks 7,005 / 7,017 MiB
vs the 7,238 MiB full-GPU baseline, 2026-11-12). On 2026-11-13 the 27B rate and sampled peak were
measured at 8K and 24K in both placements, the joint checkpoint cost was measured, and the 2B
CPU-layer boundary was swept: the full-GPU 27B also fits this card at 24K (7,783 MiB peak), so the
split trades 4-6x throughput for 446 MiB (8K) / 193 MiB (24K) of peak.**
This page records what this branch actually contains, why the interrupted WIP cannot be merged as-is,
and the gate for deleting it. Archived with the other no-go work in [ROADMAP.md](ROADMAP.md).

Benchmark coverage: [benchmarks/CROSS-MODE.md](benchmarks/CROSS-MODE.md) records the measured CPU-only and
GPU-only 2B/4B/9B cells and the measured 2B/4B/9B GPU+CPU tandem rows. This page is the tandem
status source; other pages link here rather than restating tandem details.
The dense runtime's real 2B/9B exactness and one-run timings are in
[TANDEM-RUNTIME.md](TANDEM-RUNTIME.md); the 9B cache evidence is in
`docs/benchmarks/tandem-cache/RESULTS-9B.md`.

## Answer

At the recovery audit (`2abf9ad`), the repository had:

- the **host-RAM preflight** in `kindi/hostram.py` (37 lines, used by `kindi/tp.py`);
- the existing **AVX2/FMA/F16C CPU engine** (`kindi/cpu.py`, `kindi/kcpu.c`);
- the existing **GEMV weight layout** (`kindi/layout.py:repack`, `reorder_head`).

The 2026-11-11/12 slices in this branch add:

- `kc_embed`, `kc_layer` and `kc_head` in `kindi/kcpu.c`;
- `Engine.begin`, `layer`, `head`, `feed_layers`, `step_layers` and an in-place `hidden` view in
  `kindi/cpu.py`;
- `kindi/tandem.py`: `TandemEngine` in `cpu-first` and `gpu-first` placements, plus a
  `TandemRunner` that reuses the normal prompt cache through joint checkpoints;
- a real 2B joint-cache check in both placements: end-state reuse (320 tokens), a 256-token
  joint-checkpoint restore and a 256-token saved-opening reload matched `KINDI_PROMPT_CACHE=0`
  token-for-token; an HTTP server saved and reloaded a 344-token opening after a restart, reporting
  344 cached tokens versus 0 with the cache off; the same check passed at 4K/8K context with
  1,024/2,048-token checkpoints (4,080/8,176 extension tokens and 3,072/6,144 branch tokens); see
  `docs/benchmarks/tandem-cache/RESULTS.md` and `LONG-CONTEXT-RESULTS.md`;
- the 9B pack passed the 512-context check in both placements with 6 CPU layers: extension 320,
  branch 256 and saved opening 256, all exact against cache-off; CPU load 105-108 s, GPU load
  5.8-6.0 s; see `docs/benchmarks/tandem-cache/RESULTS-9B.md`; saved-opening reloads also pass at
  4K/8K in both placements (4,080/8,176 extension, 3,072/6,144 branch and saved opening, exact;
  33.3-39.5 s owned-slice loads; `saved-9b-4k.json`, `saved-9b-8k.json`);
- a prepare-only 2B measurement in both placements: 320/325 and 256/261 tokens reused, a 19.27 MiB
  joint recurrent checkpoint, 8.7-8.9 ms median save and 3.3-3.5 ms median restore; see
  `docs/benchmarks/tandem-cache/PREFIX-RESULTS.md`;
- `engine.Engine(layers=..., embed=..., head=...)`: a partial GPU engine that uploads only the
  owned layer tensors and keeps only their GDN/KV state;
- `cpu.Engine(layers=..., embed=..., head=...)`: the CPU engine now does the same, so a
  tandem start converts only the CPU-owned layers (16 of 64 on the 27B splits) plus the
  embedding/head it owns, instead of the whole pack. The old ~365 s CPU load was a
  full-model load; the new owned-slice load time is unmeasured.

It still does **not** have:

- a MoE backend and GPU copy path; the copy-only slot manager is in `kindi/expert_cache.py` and the
  design is in [MOE-TANDEM.md](MOE-TANDEM.md), but there is no router, pack or engine integration;
- a device where the full-GPU engine cannot run at a usable context yet a split can. On the 4060 Ti
  the 27B full-GPU caps at ~37K tokens and the tandem at ~30K after auto-reduction, so the split is
  a headroom trade here, not a capacity requirement ([TANDEM-RUNTIME.md](TANDEM-RUNTIME.md));
- a layer-contiguous streaming pack format or a converter that emits one (the CPU backend
  now skips unowned layers at load, but the pack on disk is unchanged);
- native Sandy Bridge or no-FMA hardware validation; the five-tier CPU dispatch and AVX-only
  fallback are implemented and forced-tier tested, but only on an AVX2 host.

So the dense runtime is no longer the missing piece, but **do not delete the WIP branches yet**:
the MoE backend (the model-free cache prototype exists) and the larger-model gate remain open.

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
larger-model stage: a 27B-class model through both placements and peak-VRAM evidence. The joint
prompt-cache checkpoints are implemented for both ownership domains and covered by unit tests; the
real 2B exactness check and a prepare-only prefix/checkpoint measurement now pass on this host. The WIP's CUDA doorbell kernel has still not been compiled or exercised,
and it is not used by the dense implementation. A Linux host with gcc/OpenMP, or CI, is still required
for the AVX-only and portable CPU gates.

## Cleanup gate

Do not remove either WIP branch until a replacement on `main` passes all of these:

1. **CPU gate:** the supported CPU flags build; an AVX-only host fails with an actionable message or
   uses a real fallback; `kindi verify` passes on 2B and 4B; CPU tests pass. The per-layer entry point
   and its 2B/4B/9B exactness check are now present; the AVX-only fallback is implemented, but
   native Sandy Bridge hardware is unverified.
2. **Streaming gate:** `convert` emits a real layer-contiguous pack (or an explicit compatible
   alternative), and existing packs still load unchanged.
3. **Tandem gate:** a clean `TandemEngine` runs layer-by-layer with CPU/GPU placement, and greedy
   output is token-exact against the full-GPU engine on at least 2B and 9B. **2B: passed on
   2026-11-11 in both placements; 9B: passed on 2026-11-12 in both placements at 512 context;
   27B: both placements passed on 2026-11-12 (4/4 prompts; peaks 7,005 / 7,017 MiB).**
4. **Benchmark gate:** decode throughput and peak VRAM are recorded for a model that does not fit
   VRAM, with the CPU/GPU split stated.
5. **Cleanup gate:** the replacement is committed, the full CPU suite and the GPU verify pass, and
   `git status` is clean. Only then delete the WIP branches or the backup snapshot.

Keep the WIP branches as historical references: this branch now has its own five-tier CPU
dispatch and dense tandem runtime, but the WIP still contains unmerged exploration and the original
design sketches.

## Next implementation slice

The dense placement and handoff are done. The next steps are:

1. The capacity measurement is done on this card: full-GPU loads 24K; requesting 65,536 or
   131,072 tokens reduces to 37,120 (tandem: 30,464), so the split does not gain context.
   A genuinely larger-than-VRAM case needs a smaller card where neither fits.
2. Saved-opening reloads and checkpoint cost are measured at 4K/8K in both placements for the 9B
   and the 27B (27B: 54.5-61.5 ms save, 27.0-28.5 ms restore, 149.6 MiB state). Server cache
   metrics at those contexts remain open.
3. The 2B (knee 4-6 layers) and 27B (24K needs >=16 CPU layers) boundaries are swept; prompt
   reading on the target small card and any faster split remain open.
4. Implement the MoE expert cache from [MOE-TANDEM.md](MOE-TANDEM.md) only after a MoE backend with
   per-expert tensors and router logits exists; it is a different algorithm, not a flag on this one.
5. Build `hostram` pinned/mmap primitives and a layer-contiguous pack format if the measurements
   justify them.

## Decode dispatch cost (2026-11-15)

`TandemEngine._run` now iterates a per-layer owner table resolved once at
construction, instead of rebuilding a `frozenset` and re-resolving the owner of
every layer on every token. Routing is unchanged (the fake-backend call-order
tests and the probe-parity test pin it). Dispatch overhead measured on the
fake backends (Python overhead only, no GPU claim): 19.34 µs → 17.57 µs per
token (×0.91). Against the measured real-engine budget (27B @ 24K: CPU
74.7 ms + GPU 8.1 + transfer 3.09 + head 17.6 ms/token) the saving is noise;
the decode cost is the kernels, and the step-0 verdict (CPU already at the
raw-kernel rate) still stands.

The design is restored at [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md). The current CPU
roadmap remains [CPU-SUPPORT.md](CPU-SUPPORT.md).
