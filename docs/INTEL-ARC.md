# Intel Arc A770 support: goal and implementation plan

Status: **planning and architecture audit, 2026-11-10; updated 2026-11-24.** There is no Intel backend
in Kindi yet, and no Intel support claim. The A770 16 GB is now **installed in the Windows test rig**,
and M1 hardware qualification has been measured there: the card is visible at 16 GiB, its display
driver registers a Level Zero driver, and the loader enumerates one driver and one device
([run](benchmarks/2026-11-22-i7-8086k-arc-a770/README.md)). Both M2 halves now run on the same host
without oneAPI: `kindi intel-smoke` round-trips 16 MiB through Level Zero device/shared USM, and
`kindi intel-cl` compiles and runs exact vector-add and GEMM kernels with the driver's OpenCL C
compiler, and `kindi intel-ternary` runs a packed ternary GEMV over the pack's 4-trit/byte layout
with a bit-exact host check, split over the 128-weight phases (5.75x-14.14x the one-work-item
shape at layer-sized matrices, [run](benchmarks/2026-11-24-i7-8086k-arc-a770-parallel-gemv/README.md))
and now also four rows per work-item with 4-byte vector loads (1.357x that kernel at 4096x8192,
[run](benchmarks/2026-11-25-i7-8086k-arc-a770-gemv-rows4/README.md)),
and `kindi intel-gemm` runs the engine's q8 arithmetic (`k_q8x_T2` port feeding a `k_mm_g5q`-shaped
GEMM) bit-exact at 4.2-36.9 GFLOP/s ([run](benchmarks/2026-11-24-i7-8086k-arc-a770-ternary-gemm/README.md)).
There is still no SYCL backend, Kindi engine path or model run. The plan's first support
target remains Linux; the Windows run qualifies the hardware, the memory path and the device kernel
path, not Kindi support (tracked as [REQ-0020](development/requirements/REQ-0020.md)).

## Goal

Kindi should support the Intel Arc A770 16 GB as a first-class Kindi platform:

- one A770, without a CPU layer worker;
- two A770 cards in one process;
- one or two A770 cards with the CPU in tandem (layer split), not merely a separate CPU server.

The CPU-only engine remains supported and unchanged. The target is the same contract as the NVIDIA
cards in [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here): the installer offers the setup,
the advertised model loads and serves, `kindi verify` passes, the prompt cache works, two-card and
tandem modes pass the same checks, and the result is documented with measured numbers.

A working llama.cpp or OpenVINO installation on the same card is useful as a baseline, but it is not
Kindi support. The Kindi engine, pack format, prompt cache and server contract must be the ones that
run.

## Scope

| configuration | target |
|---|---|
| one Arc A770 16 GB | first support target |
| two Arc A770 16 GB | second target, two independent cards; this is not SLI/CrossFire |
| CPU + one A770 | tandem layer split; first for the models the CPU engine already supports |
| CPU + two A770 | last target; more exchange and scheduling surface |
| Arc A770 8 GB, A750, integrated Arc | not in the first support claim; integrated graphics are a separate investigation ([INTEL-IGPU-TANDEM.md](INTEL-IGPU-TANDEM.md), [REQ-0021](development/requirements/REQ-0021.md)), sharing this SYCL backend as a retarget |
| Windows | not the first target; the installer, Docker images and current GPU stack are Linux-first |
| mixed Arc + NVIDIA/AMD | not supported; the two-card path requires two identical Arc devices |

The A770 16 GB is the target because it leaves room for the 27B weights, its K/V cache and a useful
context. The official specification lists 16 GB GDDR6, 32 Xe cores, a 256-bit memory interface and
oneAPI/OpenVINO/OpenCL/Vulkan support. The relevant question for Kindi is not the marketing TOPS
number; it is whether the custom ternary kernels and the exact output can be reproduced.

## Host prerequisites

The first environment is Linux x86-64:

- an Intel Arc A770 16 GB with the kernel driver loaded and a render node under `/dev/dri`;
- a supported Ubuntu release and a pinned oneAPI DPC++/SYCL release with Level Zero and the Intel
  compute runtime (do not assume the CUDA image's Ubuntu 18.04 userland is supported);
- Docker with `/dev/dri` passed into the container and the user in the `video`/`render` groups;
- Resizable BAR/Above 4G Decoding enabled in UEFI for an advertised performance result;
- enough PCIe slots and power for two cards; there is no NVLink on Arc.

Intel's guidance is that Arc can work without Resizable BAR but loses expected performance. Kindi
will therefore record ReBAR, PCIe link width/generation and the UEFI state for every benchmark. A
no-ReBAR machine may be functionally correct, but it is not a performance-supported configuration.

## Why this is a new backend

No existing Kindi path can run on Arc:

| path | why it does not work for Arc |
|---|---|
| CUDA/CuPy | CuPy targets CUDA; Arc needs Level Zero/SYCL |
| ROCm | the ROCm/HIP path is for AMD GPUs |
| CPU engine | runs on the CPU only; it has no GPU kernels |
| llama.cpp SYCL | a useful reference and baseline, but replacing Kindi's engine would lose the pack format, cache, server and exactness contract |
| OpenVINO | a separate model runtime; it would not run Kindi's ternary kernels or `kindi verify` |
| ZLUDA/CUDA translation | unsupported, driver-fragile and not something to advertise as a Kindi backend |

The chosen route is **SYCL/oneAPI with Level Zero**, implemented as a new Kindi backend. The
`llama.cpp` SYCL backend shows that the hardware and toolchain are viable; it is not the
implementation.

## Current code reality (audit, 2026-11-10)

| area | current behaviour | impact |
|---|---|---|
| engine selection | `kindi/engine.py` selects the CPU engine for `KINDI_DEVICE=cpu` (or no CuPy); otherwise it uses CuPy | a third `intel` branch is needed; the default GPU path must stay NVIDIA/AMD |
| kernels | 22 `.cu` files with CUDA/HIP preambles; runtime compilation through CuPy/NVRTC or hipRTC | no SYCL kernels or host launcher exists |
| plan engine | `kindi/spec.py`, `kindi/tp.py`, `kindi/tcf.cu` and `kindi/tc*.cu` use CUDA driver functions, CUDA graphs and tensor-core plans | the first Arc path must use the plain engine; graphs and TC plans come later |
| prompt cache | `kindi/cache.py` uses `cp.cuda` streams, events and pinned memory | needs a backend-neutral host/device-copy layer or an Arc implementation |
| server | `kindi/server.py` and `kindi/catalog.py` call CuPy for device counts, memory and names | health, metrics, device selection and model fit need an Intel branch |
| installer | `install.sh` detects NVIDIA with `nvidia-smi` and AMD through `/dev/kfd`; there is no Intel path | detection and a container image are release work, not the first correctness milestone |
| CPU engine | whole-model C engine; verified for 2B/4B/9B, not 27B; per-layer `kc_embed`/`kc_layer`/`kc_head` entry point added 2026-11-11, but no placement or GPU handoff | CPU+Arc tandem now has the CPU primitive; it still needs layer placement, state ownership and an Arc backend |
| CPU+GPU tandem | dense Bonsai `TandemEngine` exists (token-exact on 2B/9B); CPU+Arc still needs an Arc backend and layer placement/ownership — the NVIDIA tandem does not transfer | CPU+Arc tandem is the last milestone, not an offload flag |

The 22 kernel files include the core Bonsai path (`kindi.cu`), prompt/chunk GEMM
(`chunk.cu`, `chunk1.cu`), speculative/plan kernels (`spec*.cu`, `tcf.cu`, `tcl.cu`, `tcp.cu`),
tensor-core variants (`tc*.cu`, `tca*.cu`), Gemma 4 kernels and Nanbeige kernels. A single-card 2B
path does not need all 22; it does need the core ternary matvec/GEMM, attention, GDN/recurrent
state, norms and the output head.

## Backend design

### Selection and packaging

- `KINDI_DEVICE=intel` selects the new backend; the default `gpu` stays NVIDIA/AMD until Arc is
  verified. Auto-detection is added only after the single-card path passes.
- `KINDI_INTEL_DEVICES=0` or `0,1` selects the visible Arc devices. `KINDI_CARDS=2` keeps the
  existing two-card meaning.
- `kindi/info` must report the device name, PCI/Level Zero identity, oneAPI/Level Zero versions,
  driver version, VRAM, ReBAR state and the fitted context.
- `docker/Dockerfile.intel` is a separate image with oneAPI/SYCL, Level Zero, the Intel compute
  runtime, Python and the existing server dependencies. It must not inherit the CUDA 10.2 image.
- `install.sh` gets an Intel branch only when the backend can pass the acceptance gate; a detection
  message alone is not support.

### Python/C++ boundary

The engine will use a small C++/SYCL library with a C ABI, called through `ctypes` like
`kindi/kcpu.c`, rather than requiring a new Python array library:

```
kindi/intel.py          Engine, backend selection, pack upload, exact-head path
kindi/sycl/ksycl.cpp    SYCL kernels and USM allocation/copy/launch entry points
kindi/sycl/CMakeLists.txt or a documented icpx command
libksycl.so             built for the pinned oneAPI image and DG2 target
```

The library owns `sycl::queue`, USM allocations, kernel launches and events. Python owns the pack
layout, tokenizer, cache policy and server contract, as it does for the CPU and CUDA engines. Start
with an AOT build for the DG2 target so the image does not depend on a runtime compile on the user's
machine.

Do not use XMX as the first correctness step. The ternary weights are not a native XMX format; the
first port should preserve Kindi's arithmetic with plain SYCL work-items/subgroups, then measure
whether INT8/INT4 packing or XMX improves the memory-bound decode path without changing the output.

### Kernel order

1. Pack upload and the exact output head; a synthetic vector-add/GEMM smoke test proves the device,
   USM and launch path before a model is loaded.
2. The 2B plain path: embedding, RMS norm, ternary matvec/GEMM, attention, GDN/recurrent state and
   the exact head. No fast head, plan engine, tensor cores or speculative drafting.
3. Chunked prompt/prompt cache, then the server's non-stream and stream paths.
4. 4B and 9B; then 27B and its context fit.
5. The two-card engine and CPU+Arc tandem only after the single-card 2B verify is stable.

The first port keeps the existing pack layout. Re-packing for a different memory layout is a
performance step, not a prerequisite for correctness.

### Two Arc cards

Two cards are independent devices. Start with a layer split:

- contiguous layer ranges per card, with one worker/queue per device;
- hidden-state exchange through host-visible USM unless measured peer access is available;
- K/V state stays with the layers that own it;
- a context reserve derived from the smaller card;
- exact output and no race under repeated runs.

Tensor parallel/sharded weights can follow if the layer split does not meet the performance gate.
The CUDA `kindi/tp.py` is a design reference, not reusable code; it depends on CUDA graphs and
CuPy device handles.

### CPU + Arc tandem

This is not "run an Arc server and a CPU server at the same time". It means one model whose layers
are split between the CPU engine and one/two Arc cards:

- add a per-layer CPU execution API to the CPU engine (the current C engine runs a whole model);
- define a layer scheduler and pinned/host-visible staging path that works with SYCL USM;
- keep the CPU and Arc copies of the model state consistent at every boundary;
- preserve exact output;
- measure against Arc-only and CPU-only on the same model and context.

The dense Bonsai CPU+GPU tandem runtime exists on the measured NVIDIA host ([TANDEM-RUNTIME.md](TANDEM-RUNTIME.md)),
and the CPU engine has a per-layer entry point. No CPU+Arc tandem runtime exists: Intel Arc still needs
its own backend, so CPU+Arc tandem starts with 2B/4B/9B once that backend exists and waits for the
relevant models, not merely an Arc port.

### Configuration

Provisional names (not implemented):

| name | meaning |
|---|---|
| `KINDI_DEVICE=intel` | select the Arc backend |
| `KINDI_INTEL_DEVICES=0` / `0,1` | visible Arc devices |
| `KINDI_CARDS=2` | two-card layer split |
| existing `KINDI_CTX` | context request; fit to the card/available memory |
| existing `KINDI_HEAD=exact` | required for the first support claim |

Unsupported acceleration switches must fail clearly on Intel, not silently fall back to a different
arithmetic path: no `KINDI_TC`, `KINDI_SPEC`, `KINDI_FD`, fast head or `k8v4` in the first claim.

## Milestones

| milestone | deliverable | gate |
|---|---|---|
| **M0** Architecture audit | this page; 22-kernel inventory; backend boundary and scope | this page names every affected path and the unsupported options |
| **M1** Hardware qualification | one A770 16 GB on Linux; oneAPI/SYCL/Level Zero; `sycl-ls`; ReBAR and PCIe state | device is visible with 16 GB; ReBAR state recorded; CPU and llama.cpp SYCL baselines run |
| **M2** Backend smoke test | memory path, runtime-compiled OpenCL vector-add/GEMM and the phase-parallel pack-layout ternary GEMV (partial); still needs `libksycl.so` and Kindi device selection | the smoke binary runs on the A770 with no CPU fallback |
| **M3** Single-card 2B exact | plain engine, exact head, `kindi verify` | 4/4 Bonsai prompts token-exact; no fast/spec/TC |
| **M4** Server and cache | `kindi serve`, non-stream/stream, prompt cache, health/metrics | same outputs as M3; cache save/restore works |
| **M5** 4B/9B/27B | model matrix and context fit | exact verify for each advertised model; 8K/24K stability measured |
| **M6** Two Arc cards | layer split, two queues, host exchange | 27B exact; dual is at least 1.3x single-card at the same context; no exchange race |
| **M7** CPU + Arc tandem | per-layer CPU API, scheduler, 1/2 Arc | exact; at least 1.15x Arc-only, or it enables a model/context that Arc-only cannot fit |
| **M8** Installer and support claim | Intel container, detection, docs, CI/manual matrix | installed setup passes the same checks as the NVIDIA rows; no hidden environment override |

M1 is **partly measured on the Windows host (2026-11-22)**. The card is installed and enumerable:

| evidence | value |
|---|---|
| device | Intel Arc A770, PCI `8086:56A0`, 16,384 MiB |
| driver | 32.0.101.9175; the display class also registers `ze_intel_gpu64.dll` 32.0.101.9175 |
| Level Zero | loader 1.33.1, `zeInit` 0, 1 driver / 1 device |
| CPU baseline | 2B/4B/9B `verify` 4/4 token-exact, 22.2/11.1/6.5 tok/s at ctx 1,024 |
| open | oneAPI/SYCL (`sycl-ls`), ReBAR, PCIe link state, llama.cpp SYCL baseline |

That is hardware qualification, **not M1 passed**: the gate also asks for the pinned oneAPI/SYCL
toolchain, the ReBAR/PCIe record and the llama.cpp baseline, and none of those exist on this host.
Windows is not the first support target, so this run does not move the support claim either. M2's
memory and device-kernel halves are now measured (Level Zero copies and runtime-compiled OpenCL);
`libksycl.so`, a SYCL kernel and Kindi device selection remain open. The next step is the SYCL
vector-add smoke planned below (oneAPI on Windows; a Linux host is the plan's target); record every
future run in
`docs/benchmarks/<date>-<host>-arc-a770/`.

### M2 memory path (partial, 2026-11-22)

`python -m kindi intel-smoke --mib 16 --json` passes on the Windows A770 through the installed Level
Zero loader. It creates a context, allocates one device-USM buffer and two shared-USM staging
buffers, fills a known byte pattern into the source, copies host -> device -> host through a command
list and a synchronous command queue, and compares all 16 MiB. Measured: **pass, 0 mismatches,
11.6 ms** ([l0-smoke.json](benchmarks/2026-11-22-i7-8086k-arc-a770/l0-smoke.json)). The command is
implemented with `ctypes` in `kindi/intel_l0.py`; there is no compiled kernel and no CPU fallback.

Three driver findings are recorded because each one made a successful-looking call produce zeros:

- `zeMemCopy` is absent from loader 1.33.1; copies must go through a command list and queue.
- Queue group 0 advertises `COMPUTE|COPY|COOPERATIVE` (`flags=0x7`) and accepts copy commands, but the
  copy does not happen. Group 1 is the copy-only engine (`flags=0x2`) and executes it. Selection now
  prefers a copy-only group instead of assuming ordinal 0.
- A host-USM destination produced zeros after the copy; shared USM is the working host-side staging
  buffer on this driver.

This is the memory/queue half of M2 only. It does not run a kernel, GEMM, model or Kindi engine, and
it is **not M2 passed** and not an Intel support claim.

### M2 kernel path (partial, 2026-11-22)

`python -m kindi intel-cl --json` compiles two OpenCL C kernels with the driver's own runtime
compiler and runs them on the A770, with no oneAPI/icpx and no CPU fallback: a 1,048,576-element
float32 vector-add and a 128x128 float32 GEMM (5 launches timed). Inputs are small integers, so every
partial sum is exactly representable and the host reference must match bit for bit. Measured:
**pass**, 0 mismatches on both, vector-add 6.5 ms, GEMM 0.666 ms for 5 launches at 31.5 GFLOP/s wall
([cl-smoke.json](benchmarks/2026-11-22-i7-8086k-arc-a770/cl-smoke.json)). The driver reports 512
compute units, 15,932 MiB, OpenCL C 1.2 and driver 32.0.101.9175.

Three API findings from the first run are recorded because each one made the probe fail or mislabel
the device:

- `CL_PLATFORM_NAME` and `CL_PLATFORM_VENDOR` are `0x0902`/`0x0903`; the `0x0900`/`0x0901` pair is
  profile/version, which labelled the platform `FULL_PROFILE`.
- `clCreateBuffer` returns `CL_INVALID_HOST_PTR` (-37) for a non-NULL host pointer unless
  `CL_MEM_COPY_HOST_PTR` is set; output buffers must pass NULL.
- `CL_DEVICE_MAX_COMPUTE_UNITS` is `0x1002`; `0x1003` is the work-item dimension count and read 3.

This is the kernel/GEMM half of M2 measured through OpenCL, not SYCL: it proves device compute and
exactness on the card, but there is no `libksycl.so`, no Kindi backend or engine path, no model and
no support claim. `kindi intel-cl` is a qualification probe, not a backend entry point.

### M2 packed ternary GEMV probe (2026-11-23)

`kindi intel-ternary` is the first probe over the pack's actual ternary layout instead of dense
float kernels. It builds a matrix with `convert.phase_bytes` (4 trits per byte, byte = t0 + 3*t1 +
9*t2 + 27*t3, stored `[column/4][row]`) and one fp16 scale per 128 weights (`[column/128][row]`,
widened to float32 on the host), uploads those two buffers unchanged, and runs one OpenCL C GEMV
kernel on the A770. The host reference decodes the same packed bytes; the activations are small
integers and the scales are 1.0/0.5/2.0/1.5, so every result must match bit for bit rather than
within a tolerance.

Measured on the A770 (driver 32.0.101.9175), one work-item per row, one warm-up and five timed
launches:

| rows x cols | mismatches | warm-up | 5 launches | GFLOP/s (wall) |
|---|---|---|---|---|
| 64 x 1024 | 0 | 4.989 ms | 0.944 ms | 0.694 |
| 128 x 4096 | 0 | 4.637 ms | 2.470 ms | 2.123 |
| 512 x 4096 | 0 | 4.807 ms | 2.437 ms | 8.604 |

Reports: `docs/benchmarks/2026-11-23-i7-8086k-arc-a770/`. This is the on-disk pack layout, not the
engine's load-time 5-trit lane repack (`kindi/layout.py`); the activation is not quantized and one
work-item per row is not a throughput design, so the rates are observations, not a claim. It adds no
backend, model, SYCL path or support row: M2 stays partial. (The 2026-11-24 kernel below keeps this
layout and splits the phase axis instead of running one work-item per row.)

### M2 packed ternary GEMV: phase-parallel kernel (2026-11-24)

One work-item per row left most of the card idle (512 work-items on 512 EUs), so the probe now splits
the 128-weight phase axis: work-item `(row, part)` handles the phases whose `ph % parts == part` and
writes one partial per row, and a second kernel `ternary_gemv_reduce` adds the parts. `--parts`
defaults to one part per phase (capped at 64) because that measured fastest at every shape tried;
`--parts 1` is the 2026-11-23 one-work-item shape. Rows stay the fastest-varying axis, so the loads
that were coalesced before still are.

The split cannot change the result: a part sums at most 64 exactly representable products and the
reduction at most 64 such floats, all integer-valued below 2^24. The probe also checks that on the
host before it launches — `kindi.intel_ternary.partials` computes the same decomposition and the run
refuses a shape where summing the parts differs from the reference — and the device result is still
compared bit for bit.

Measured on the same Windows A770 (driver 32.0.101.9175), 20 launches, three runs per shape, median
run quoted ([page](benchmarks/2026-11-24-i7-8086k-arc-a770-parallel-gemv/README.md)):

| rows x cols | parts | mismatches | 20 launches | GFLOP/s | `--parts 1` | speedup |
|---|---|---|---|---|---|---|
| 64 x 1024 | 8 | 0 | 1.054 ms | 2.49 | 2.825 ms | 2.68x |
| 128 x 4096 | 32 | 0 | 1.081 ms | 19.40 | 8.870 ms | 8.21x |
| 512 x 4096 | 32 | 0 | 1.111 ms | 75.54 | 8.880 ms | 7.99x |
| 1024 x 8192 | 64 | 0 | 1.224 ms | 274.12 | 17.306 ms | 14.14x |
| 2048 x 4096 | 32 | 0 | 1.152 ms | 291.15 | 9.001 ms | 7.81x |
| 4096 x 4096 | 32 | 0 | 1.577 ms | 425.60 | 9.064 ms | 5.75x |

This is still the on-disk pack layout with an unquantized activation, not the engine's 5-trit lane
repack or its Q8 GEMV, and a launch costs about 1 ms of fixed overhead at these sizes. No backend,
model, engine or support claim is added; M2 stays partial. What it does establish is that the layout
decodes exactly at a useful rate with the parallelism spread over phases, which is the shape the
backend's kernel order needs.

### M2 packed ternary GEMV: four rows per work-item (2026-11-25)

The phase split above still left each work-item doing 1,024 single-byte loads with a row-sized
stride. The layout stores a slot's bytes as consecutive **rows**, so `ternary_gemv_part4` loads four
rows of a slot with one 4-byte `vload4`, decodes the trits with `(b * 171) >> 9` instead of integer
division, and keeps the same per-phase int32 accumulation, so its partials are bit-identical to the
scalar kernel. `--kernel auto` (the default) takes it when four rows fit, `--kernel scalar` keeps the
2026-11-24 shape, and a row count that is not a multiple of four takes a guarded path.

Three runs of 20 launches per arm, medians, on the same Windows A770
([page](benchmarks/2026-11-25-i7-8086k-arc-a770-gemv-rows4/README.md)):

| rows x cols | scalar per call | rows4 per call | speedup | effective GB/s (rows4) |
|---|---|---|---|---|
| 1024 x 2048 | 0.0461 ms | 0.0433 ms | 1.065x | 13.13 |
| 2048 x 4096 | 0.0571 ms | 0.0467 ms | 1.223x | 48.29 |
| 4096 x 8192 | 0.0936 ms | 0.0690 ms | 1.357x | 129.88 |
| 4096 x 128 (floor) | 0.0432 ms | 0.0410 ms | 1.054x | 3.80 |

Every arm reports 0 mismatches against the host reference. The 4096x128 shape moves 128 KB and still
costs 0.041 ms per call, so two enqueued kernels cost about 40 microseconds on this driver: below
roughly 2 MiB of packed weights a GEMV is dispatch-bound, and that - not the decode arithmetic - is
what a backend would have to remove next (a single-launch design has to give up either the coalesced
row-major reads or the exact float sum order). No backend, model or support claim; M2 stays partial.

### M2 ternary GEMM in the engine's q8 shape (2026-11-24)

`kindi intel-gemm` runs the arithmetic the engine's prompt path does, at the CUDA kernels' tile,
bit-exact. `q8x` ports `k_q8x_T2` (one 256-wide group per 128-column group and 64-token band, one
threshold pair per token, `rint` round-half-even, four int8 per word) and its XQ/XS output feeds
`gemm_tile` directly; `gemm_tile` mirrors `k_mm_g5q`: 128 tokens x 128 rows per 256-work-item
block, int32 dots per 128-column group, one `f * (xs * dj)` per group, fp32 accumulate in group
order. The weight bytes are still the pack's on-disk 4-trit layout (the engine repacks to 5-trit
lanes at load), there is no local-memory staging, and OpenCL C 1.2 has no dot-product instruction,
so the int32 dot is computed byte by byte - exactly the integers `dp4a` computes.

Measured on the same Windows A770, 10 launches per shape:

| rows x tokens x cols | mismatches | 10 launches | GFLOP/s |
|---|---|---|---|
| 128 x 128 x 512 | 0 | 40.204 ms | 4.173 |
| 256 x 128 x 1024 | 0 | 85.832 ms | 7.819 |
| 256 x 256 x 2048 | 0 | 283.782 ms | 9.459 |
| 512 x 512 x 4096 | 0 | 581.610 ms | 36.923 |

The quantizer is bit-exact with the host model at every shape (XQ 0, XS 0), so the GEMM input is
the proven one. The rate rises with tile count because a launch is ~4 ms of fixed overhead at
these sizes (the 128-token quantizer alone took 3.766 ms); the 36.9 GFLOP/s at the large shape is
the un-tiled, global-load, byte-dot shape - the baseline a local-memory-staged variant gets
measured against. Exactness is by construction (activations bounded by 127 with every group's
maximum forced to 127, so the quantizer passes through at xs = 1.0; fp16-exact scales), and the
host reference is checked against an independent float64 dot before the device is launched.

No backend, model, engine or support claim is added; M2 stays partial. The local-memory-staged
variant and the 5-trit-lane unpack were named as follow-ups; the first is measured below, the second
is still open.

([page](benchmarks/2026-11-24-i7-8086k-arc-a770-ternary-gemm/README.md))

### M2 next: pinned work-group (+1.71x) and local staging (+1.02-1.05x) (2026-11-24)

Two changes followed, and the larger one was not the tiles. First, the GEMM now always launches one
256-wide work-group per 128-token x 128-row tile (`(256,1)` local size). The first page left the
work-group shape to the runtime, which cost **1.71x** at 512 x 512 x 4096 (555.1 -> 323.4 ms, three
runs per arm, bit-identical output) and - worse - let a work-group span two tiles once the tiles
were staged in local memory, which made the result wrong at more than one tile. Pinning is therefore
a correctness fix first. Second, `--staging local` (the default) stages each group's activation
words and decoded weight words in local memory as `k_mm_g5q` does; it is exact and a further
1.02-1.05x over the pinned global load (67.77 vs 66.17 GFLOP/s at the large shape), small because
the bottleneck is the byte-wise dot and the 4-trit decode, not global traffic. **A paired re-check
later the same day (arms alternated, three runs each) holds 1.05x and 1.06x at the two smaller
shapes but measures 0.97x at 512 x 512 x 4096**, so staging is a small win, not a consistent one:
the 1.02x above is that session's number, not a property of the shape. Either way the bottleneck is
the byte-wise dot and the 4-trit decode, so `dp4a`-shaped packing, not more staging, is the lever.

OpenCL's device order is **not stable on this host**: the driver has returned the integrated UHD 630
first and the Arc A770 second, and the reverse on other runs, so `--device 0` can silently benchmark
the iGPU. All three OpenCL probes now take `--device-name SUBSTR` and put a warning in the report
(and the printed output) when the selected device is not an Arc; every number below was measured
with `--device-name Arc` and records `Intel(R) Arc(TM) A770 Graphics`.

([page](benchmarks/2026-11-24-i7-8086k-arc-a770-gemm-tiled/README.md)) No backend, model, engine or
support claim: M2 stays partial.

### M2 step: SYCL vector-add smoke (ran 2026-11-24 on Windows; M2 still needs the Linux target)

The OpenCL probe proved exact device compute, but the M2 gate names the SYCL runtime (`libksycl.so`)
and SYCL device selection, and OpenCL C is not that path. The smoke is now implemented and has run
on this Windows A770 through oneAPI DPC++ on the **Level Zero** backend: `kindi intel-sycl` builds
`tools/intel_sycl_vecadd.cpp` with the installed `icpx`, runs it with
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, and validates the child's report. Result: `pass`,
0 mismatches and `max_abs_error = 0` at 1,048,576, 1 and 1,000,003 elements, backend
`ext_oneapi_level_zero`, device `Intel(R) Arc(TM) A770 Graphics`, driver `1.17.39894+6`, all ten
acceptance checks below met ([page](benchmarks/2026-11-24-i7-8086k-arc-a770-sycl/README.md)).

Two findings from the run are in the code. The Level Zero adapter is only loaded from the oneAPI
`bin` directories, so a `sycl-ls` run without them on `PATH` lists OpenCL devices and no Level Zero
device at all - which is what made the earlier session record "no L0" on a host that has one;
`kindi/intel.py` now locates the install and prepends those directories. And the current `sycl-ls`
prints `[backend:kind][backend:index]`, not the older `[backend:kind:index]`; the parser dropped
every line of the new shape, so `intel-info` reported no Level Zero device while `sycl-ls` listed
the Arc. Both layouts parse now.

This is a host-local precursor, not M2: the plan's Linux target still needs the same smoke, and one
exact vector-add is not a model, engine or Kindi kernel. No support claim, and
`docs/SUPPORT-MATRIX.md` and the installer are unchanged.

**Toolchain (oneAPI first).** Intel oneAPI Base Toolkit for Windows, pinned to the release that is
actually installed; record `icx-cl --version` and `sycl-ls --version` rather than writing a version
that was not observed. It supplies the DPC++ compiler driver (`icx-cl` on Windows), the SYCL runtime
(`sycl*.dll`), the Level Zero plugin and `sycl-ls.exe`. What the host has today:

| part | state on this host, 2026-11-23 | check |
|---|---|---|
| graphics driver + Level Zero device | present, driver 32.0.101.9175 | `python -m kindi intel-info --json` |
| Level Zero loader and SDK | `ze_loader.dll`; SDK 1.33.1 under `%ProgramFiles%/LevelZeroSDK/1.33.1` | `python -m kindi intel-smoke --mib 16` |
| Visual Studio C++ toolset | VS Community 2026 18.9.1, MSVC 14.51 (record its install path) | `vcvars64.bat`, then compile a hello-world with `cl` |
| Windows SDK | not at the default `%ProgramFiles(x86)%/Windows Kits/10` path; `vcvars64.bat` must find it | the same hello-world build |
| oneAPI Base Toolkit | **installed 2026-11-24**: 2026.2 (`icpx`, `icx-cl`, `dpcpp`, `sycl-ls`); `intel-info` finds it through `KINDI_ONEAPI_ROOT` or the default root | `icx-cl --version`; `sycl-ls --verbose` |
| SPIR-V tools (fallback only) | absent (`llvm-spirv`, `spirv-val`) | `llvm-spirv --version` |

Installation is an owner action (`docs/MANUAL-HELP-NEEDED.md` section 4a), not part of this planning
change. Check the installed oneAPI release's supported Visual Studio version before choosing the
compiler: this host has VS 2026, so if that release is validated only against VS 2022, the owner
decides between the 2022 Build Tools and a newer oneAPI; record the decision instead of working
around it.

**Code.** One C++ file and a thin Python driver, mirroring `intel-cl`'s report surface:

| file | content |
|---|---|
| `tools/intel_sycl_vecadd.cpp` | enumerate the GPUs, keep only devices whose backend is `ext_oneapi_level_zero`, and refuse (exit 1, naming what the runtime offered) when there is none; assert the device name contains `A770`, allocate USM, launch the vector-add, compare with an exact host reference, print one JSON report with `--compiler-version` and `--selector` echoed |
| `kindi/intel_sycl.py` | `smoke(n=..., repeats=..., compiler=None, builder=None, runner=None)`: locate the toolchain, build with `icpx -fsycl -O2` after `vcvars64.bat`, run under `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, validate the child report (a child that says pass but names another backend is a failure), `render()`/`dumps()`; builder and runner injectable so the tests need no toolkit or GPU |
| `kindi/__main__.py` | `kindi intel-sycl [--n N] [--repeats K] [--compiler PATH] [--selector S] [--json]`, exit 0 on pass and 1 on fail (the `intel-smoke`/`intel-cl` convention) |
| `tests/test_intel_sycl.py` | fake builder/runner: compiler absent, build failure, no `level_zero:gpu` device, child report with mismatches, a child claiming another backend, and pass; no GPU |

**Kernel and inputs.** One `nd_range<1>` vector-add with local size 64 and an `i < n` bounds guard.
Inputs are integer-valued float32 (`a[i] = (31*i) % 1000`, `b[i] = (17*i) % 1000`), so every sum is
exactly representable; the host reference is computed independently and compared bit for bit after
the copy back. Sizes: 1,048,576 elements (five timed repeats), 1 element, and 1,000,003 (not a
multiple of the local size, so the guard runs).

**Acceptance checks** (all must hold; a failure records `result: fail` and the stage):

| # | check | exact condition |
|---|---|---|
| A1 | the runtime sees the card | `sycl-ls --verbose` exits 0 with one `[level_zero:gpu:N]` line naming `Intel(R) Arc(TM) A770 Graphics`; the whole line is recorded |
| A2 | selection is explicit | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls` lists only the A770 (no `host`, no `opencl` device) |
| A3 | build | after `vcvars64.bat`, `icpx -fsycl -O2 -o intel_sycl_vecadd.exe tools/intel_sycl_vecadd.cpp` exits 0; binary sha256 and build log recorded |
| A4 | the child asserts its device | backend `ext_oneapi_level_zero`, `is_gpu()` true, name contains `A770`, at least 15,000 MiB of global memory (SYCL reports 15,932 MiB of the card's 16,384); driver version recorded |
| A5 | exactness | 0 mismatches and `max_abs_error = 0` for every size and every repeat, over all 1,048,576 elements, not a sample |
| A6 | bounds guard | 1,000,003 elements pass with 0 mismatches; no out-of-range element is read or written |
| A7 | no CPU/OpenCL fallback | with `ONEAPI_DEVICE_SELECTOR=opencl:gpu` the child exits 1 naming the missing `level_zero:gpu` device and does not run on OpenCL or the CPU; a child report whose backend is not Level Zero is a failure even when the sums match |
| A8 | failure is fail | a build, module or launch error gives `result: fail`, exit 1 and the first error string; a failed launch never prints the host reference as the result |
| A9 | report | `--json` prints `probe: "intel-sycl-smoke"`, `schema: 1`, `support_claim: false`, `scope`, `toolchain`, `device`, `build`, `vector_add` (sizes, repeats, mismatches, max_abs_error, elapsed_ms), `result`, `error` |
| A10 | host tests | `python -m pytest -q tests/test_intel_sycl.py` passes with no oneAPI, compiler or GPU; the full CPU suite still passes |

**Evidence and scope.** The pass is recorded in
[`benchmarks/2026-11-24-i7-8086k-arc-a770-sycl/`](benchmarks/2026-11-24-i7-8086k-arc-a770-sycl/README.md)
as `sycl-smoke.json` and a text copy, with the `sycl-ls` listing, the selector-restricted listing and
the OpenCL-selector refusal, plus an M2 evidence bullet in REQ-0020. It changes neither
`docs/SUPPORT-MATRIX.md` nor the installer and adds no support claim: Windows is not the M2 target,
this kernel is not Kindi's kernel, and one exact vector-add is not a model, engine or GEMM result.
Timings are observations, not a throughput claim. The Linux target still has to run the same smoke
before M2 passes.

The OpenCL fallback below stays on record as a diagnostic and is no longer needed on this host.

**Stop conditions** (what a failing host must do). No oneAPI (`sycl-ls` absent) or no
`level_zero:gpu` line: record the toolchain failure and stop; do not fall back to OpenCL, the CPU or
a host device. On this host the Level Zero line needs the oneAPI `bin` directories on `PATH`, so
check that before recording "no device". Build failure: keep the log,
record the compiler and VS versions, stop. Exactness failure: record the first mismatch index and
both values, stop. Do not weaken the kernel or the acceptance values to get a pass.

**Fallback if oneAPI cannot be installed (raw Level Zero SPIR-V).** The Level Zero SDK 1.33.1
headers/libs and the clang 18.1.8 install the CPU engine already uses are on the host; the missing
piece is a SPIR-V producer: `clang -target spir64 -x cl -cl-std=CL2.0 -O0 -emit-llvm -c vecadd.cl -o
vecadd.bc`, then `llvm-spirv vecadd.bc -o vecadd.spv` (SPIRV-LLVM-Translator matching LLVM 18) and
`spirv-val --target-env spv1.2 vecadd.spv`; launch through `zeModuleCreate` with
`ZE_MODULE_FORMAT_IL_SPIRV`, `zeKernelCreate`, group size 64 and `zeCommandListAppendLaunchKernel`
on the existing USM/copy path in `kindi/intel_l0.py`. It must meet A3-A10 with the module build log
and SPIR-V hash recorded. It proves a kernel through the loader, not the SYCL runtime or SYCL device
selection M2 gates, so it is a diagnostic and does not close M2; when oneAPI is installed this path
is unnecessary.

### Arrival checklist (M1, run when the card is installed)

Host assumptions: Linux x86-64, UEFI with Resizable BAR / Above 4G Decoding enabled, the card in a
PCIe x16 slot. Record the output of every step; do not skip the ReBAR/PCIe state — a no-ReBAR result
is functionally possible but is not a performance-supported configuration.

```sh
# 1. Device visible with the kernel driver loaded
ls -l /dev/dri/
lscpu                              # CPU/feature baseline for intel-info and the CPU comparison
lspci -nnk | grep -A3 -i 'vga\|display'   # BDF, device id and driver in use
ls -l /dev/dri/by-path/            # render node -> PCI slot mapping
lspci -vv -s <bdf> | grep -Ei 'resizable|lnkcap|lnksta'  # ReBAR and link width/generation

# 2. Toolchain: pin oneAPI/SYCL/Level Zero + Intel compute runtime (Open Decision 1);
#    verify against Intel's current oneAPI system-requirements matrix, not the CUDA image.
sycl-ls --verbose
clinfo
#    Optional OpenVINO baseline: read-only, no model load. Record absence explicitly.
python - <<'PY'
try:
    import openvino as ov
except ImportError:
    print("OpenVINO: not installed (optional)")
else:
    print("OpenVINO:", ov.__version__)
    print("Available devices:", ", ".join(ov.Core().available_devices))
PY

# 3. CPU baseline (same host, same packs): the 2B/4B/9B CPU verify rows.
KINDI_DEVICE=cpu python -m kindi --pack <pack> verify

# 4. llama.cpp SYCL baseline (same model, context, token count) — a reference, never a
#    substitute for a Kindi result.
#    (record build commit + backend flags alongside the numbers)
```

M1 passes when: the device is visible with 16 GB, the ReBAR state is recorded, and both baselines
run. The M2 memory and device-kernel halves are already measured on Windows; a Linux host still has
to repeat the kernel/GEMM check through `libksycl.so` before M2 can pass.

### CPU tuning hypotheses from `cputune.py` (record during M1)

Two host-tuning hypotheses survive the `cputune.py` review. They are experiments, not adopted
settings, and neither changes the M1 pass condition. Run them on the A770 host with the same pack,
context, token stream and exact head; keep every `--json` report. A result inside run-to-run noise
is a useful no-go, not a failure.

**THP.** Kindi disables transparent huge pages for ROCm processes today (`engine._no_thp()`), not
for Intel. First record the host policy (`cat /sys/kernel/mm/transparent_hugepage/enabled`) and
screen it on the M1 CPU baseline: three `KINDI_DEVICE=cpu ... bench` passes with THP left at the
host default (`KINDI_THP=1`), and three with `PR_SET_THP_DISABLE` set by a launcher before Kindi
starts. The launcher must check the `prctl` return and the process flag; if either check fails, it
stops before the bench. Then repeat both arms on the M7 CPU+Arc tandem, where host compaction is
the actual concern. Keep `thp-host.state`, `thp-off.state` and every `--json` report.

```sh
cat /sys/kernel/mm/transparent_hugepage/enabled > thp-host.state

# Control: host THP policy (KINDI_THP=1 prevents Kindi's own disable on ROCm).
KINDI_DEVICE=cpu KINDI_THP=1 python -m kindi --pack <pack> --ctx 8192 bench --json thp-on.json

# Experiment: set THP off, verify both calls, then run the bench in the same process.
KINDI_DEVICE=cpu python - <<'PY' > thp-off.state 2>&1
import ctypes, runpy, sys

libc = ctypes.CDLL(None, use_errno=True)
rc = libc.prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE
if rc != 0:
    raise SystemExit("PR_SET_THP_DISABLE failed: rc=%d errno=%d" % (rc, ctypes.get_errno()))
state = libc.prctl(42, 0, 0, 0, 0)  # PR_GET_THP_DISABLE
print("THP state before bench: PR_SET_THP_DISABLE=0, PR_GET_THP_DISABLE=%d" % state, flush=True)
if state != 1:
    raise SystemExit("THP disable did not take effect: PR_GET_THP_DISABLE=%d" % state)
sys.argv = ["kindi", "--pack", "<pack>", "--ctx", "8192", "bench", "--json", "thp-off.json"]
runpy.run_module("kindi.__main__", run_name="__main__")
PY
```

Decide on the M7 tandem result: add the Intel THP call only if the three-pass 8K decode median
improves by at least 3% over THP-on, the three-pass ranges do not overlap, and `kindi verify` stays
exact. If the M1 CPU screen is within noise, record that and do not add an Intel hook on CPU-only
evidence. If the CPU screen wins but the M7 tandem does not, leave THP enabled on Intel.

**P-core pinning.** If `lscpu` and `cpu_capacity` show a hybrid part, compare Kindi's all-core
default with a pin to the CPUs whose `cpu_capacity` is maximal. Choose one logical CPU from each
max-capacity physical core (deduplicate by `thread_siblings_list`), set `OMP_NUM_THREADS` to the
P-core count, and run the same three-pass M1 CPU `bench` at 8K. Record the capacity map, chosen CPU
list, thread count and raw reports.

```sh
# Show capacity per logical CPU, highest first (skip this arm if cpu_capacity is absent).
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  printf '%s %s\n' "${c##*/}" "$(cat "$c/cpu_capacity" 2>/dev/null)"
done | sort -k2,2nr

# After choosing the deduplicated P-core list and count:
KINDI_DEVICE=cpu taskset -c "$PCORES" env OMP_NUM_THREADS="$PCORE_COUNT" \
  python -m kindi --pack <pack> --ctx 8192 bench --json pcore.json
```

Adopt a `cpu_capacity`-based default only if the 8K decode median improves by at least 3% over the
all-core default, the three-pass ranges do not overlap, and neither prompt-read nor exactness
regresses; otherwise keep the current thread default and record the no-go. If the CPU arm wins,
repeat the comparison on the M7 tandem before making it the Arc-host default.

## Acceptance matrix

All cells are pending until a measured, exact result is recorded. "P" refers to the milestone that
must pass; it is not a support claim.

| setup | 2B | 4B | 9B | 27B | 33B (Nanbeige) |
|---|---:|---:|---:|---:|---:|
| one A770 16 GB | P3 | P5 | P5 | P5 | out of first claim |
| two A770 16 GB | P3 | P5 | P5 | P6 | P6, after the Nanbeige backend |
| CPU + one A770 | P7 | P7 | P7 | after CPU per-layer 27B | out of first claim |
| CPU + two A770 | P7 | P7 | P7 | after CPU per-layer 27B | out of first claim |

A checked cell needs the installer option, the model loaded, `kindi verify` exact, the server/cache
check and a measured benchmark row. Do not claim two cards from a one-card run, 27B from a 2B run,
or tandem from two independent servers.

## Performance gates

Correctness comes first; a fast wrong result is not support. The following gates are fixed before
the backend work starts:

- **Single card, 2B/4B/9B:** at 2K context, generation must be at least 1.0x the CPU engine on the
  same host; the target is 2x. Record prompt read, generation and 8K/24K behavior.
- **Single card, 27B:** must be stable and useful at the documented context; the initial floor is
  5 tokens/s at 2K. Record the exact context and memory headroom.
- **Two cards:** 27B generation at the same context must be at least 1.3x one card, or the second
  card is not a support target.
- **CPU + Arc tandem:** must be at least 1.15x Arc-only, or enable a model/context that Arc-only
  cannot run. If it does neither, it is a no-go like the existing CPU+GPU tandem experiment.
- **Comparison:** record a llama.cpp SYCL baseline at the same model, context and token count, but
  do not substitute it for a Kindi result. A speed win is reported only from three-run medians.

## Risks

- **Hardware is present, the toolchain is not.** The Windows host has an enumerable A770 and no
  SYCL compiler; a Linux host with pinned oneAPI is still the first support target. No amount of
  code review substitutes for `sycl-ls` and a token-exact run.
- **Kernel-port effort.** The core Bonsai path is custom CUDA; a thin CuPy replacement does not
  exist. The 22 kernels, plan engine and tensor-core paths are not a configuration change.
- **Numerical exactness.** Subgroup sizes, FP16/FMA contraction and reduction order can change
  logits. The same reference prompts and accumulation policy are mandatory.
- **CUDA graphs and launch-bound plans.** `kindi/spec.py` and `kindi/tp.py` depend on CUDA driver
  graphs; the first Arc path must be the plain engine, which will be slower until an event/queue
  plan exists.
- **Driver/toolchain churn.** Pin oneAPI, Level Zero, compute runtime and the kernel driver; record
  all versions with every benchmark.
- **Two-card exchange.** Arc has no NVLink; host-mediated exchange can dominate token time. The
  1.3x gate exists to reject a second card that only adds synchronization.
- **CPU tandem.** The CPU engine has a per-layer entry point and the dense NVIDIA-host tandem runtime
  exists, but there is no Arc backend and therefore no CPU+Arc tandem runtime.
- **ReBAR and power.** Without ReBAR, performance results are not representative. Two cards need
  two suitable slots and power; measure before advertising a dual setup.

## Open decisions

1. Which Ubuntu/oneAPI/Level Zero/compute-runtime combination is pinned? Verify Intel's current
   system-requirements matrix; do not infer it from the CUDA image.
2. C ABI/`ctypes` (favoured for consistency with `kcpu.c`) or pybind11 for the SYCL library?
3. AOT for DG2 at image build (favoured) or JIT at first use? The CUDA image's runtime-compile
   pattern is not automatically the right model here.
4. Is the two-card path a layer split or a tensor-parallel split? Start with layers; measure.
5. Does CPU+Arc tandem meet its 1.15x/fit gate, or does it become a documented no-go?
6. Is 33B/Nanbeige in the first Intel claim, or a follow-on after 27B? The first claim is Bonsai
   unless the Nanbeige backend is also ported.
7. Are 8 GB A770/A750 and Windows in scope? Not for the first claim.
8. How strictly is ReBAR enforced? Functional support may exist without it, but a performance
   support claim requires it and must record it.

## Definition of done

The Intel goal is complete only when:

1. one A770 and two A770 configurations are offered by the installer;
2. the advertised Bonsai models load and serve through Kindi's own engine;
3. `kindi verify` is token-exact on every advertised model;
4. prompt cache, streaming, health and metrics work;
5. two-card output is exact and meets the 1.3x gate;
6. CPU+Arc tandem either meets the 1.15x/fit gate or is explicitly closed as a no-go;
7. the documentation records the tested card, driver, oneAPI, context and tokens/s;
8. no manual environment override is needed for a supported setup.

Until then, `docs/CARDS.md`, the README and the installer must not list Arc as supported.

## External Intel work (tracked, not used)

[`red-raine/kindi-intel`](https://github.com/red-raine/kindi-intel) is an AGPL-3.0 fork of
`ahtx/kindi`, created 2026-11-01. Its default `main` branch is stock Kindi 0.3.0 with no Intel path;
all Intel work is on `intel-npu-gpu-acceleration-7749e` (head `ad227a8`, three `qwen.ai[bot]`
"task snapshot" commits, last 2026-11-01), not on `main` and not merged upstream. The branch adds
`kindi/intel.py`, `kindi/ov_build.py`, `kindi/cputune.py` and `kindi/diskp2p.py`, plus small
engine/server/CLI hooks.

- It selects an **OpenVINO** backend (`KINDI_DEVICE=intel` / `openvino`, `KINDI_OV_DEVICE=CPU` /
  `GPU` / `NPU`, `KINDI_INTEL_GPU`/`KINDI_INTEL_NPU`), probes Intel PCI IDs and Level Zero devices,
  and advertises Intel CPUs, integrated Arc and discrete Arc A/B-series through OpenVINO's GPU
  plugin.
- The ternary-pack path is not runnable: `kindi/ov_build.py`'s `build()` is a stub, and its `main()`
  prints "weights are staged, graph emission pending" and returns 0 without writing `model.xml`.
  `kindi export-openvino` therefore cannot produce the IR that `Engine` needs.
- Its GGUF fallback calls `openvino-genai` (`ov_genai.LLMPipeline`) and bypasses Kindi's pack,
  kernels, prompt cache and token-exact contract.
- There is no test, benchmark, reference output or A770 run in the branch; its README is unchanged
  from 0.3.0. Nothing in it is an Arc measurement or a Kindi support claim.

**What to take:** the device naming, `discover()`/Level Zero probe and `KINDI_OV_DEVICE` handling are
useful prior art for `kindi info` and the M1 checklist. **What not to take:** OpenVINO is not the SYCL/Level Zero backend
chosen above, and an unbuilt IR or an `ov_genai` shortcut cannot pass M3/M4. Do not merge it or cite
it as Arc support. Re-check the branch only if it gains a working IR, an A770 run and token-exact
output.

### Verdict on `diskp2p.py` and `cputune.py` (read 2026-11-19)

Neither is worth adapting. Both are read-only prior art for the A770, and nothing here is merged or
copied.

- **`diskp2p.py` — no.** Its `stream_load` re-implements what `kindi/engine.py::file_to_card`
  already does (chunked upload with `posix_fadvise(POSIX_FADV_DONTNEED)` behind each chunk;
  `kindi/tp.py` does the same for the 5-trit cache). `pin_file`'s `MADV_WILLNEED`/`MADV_HUGEPAGE`
  read-ahead is a Linux host-DMA micro-optimization aimed at the iGPU/NPU case and has no Arc
  measurement, and `p2p_enable` is CUDA-only (`cuCtxEnablePeerAccess`), so it does not describe a
  two-A770 exchange. The module's docstring also advertises a `unified_hint(dev)` helper that it
  never implements. If M2/M3 load times turn out disk-bound, the one experiment worth running is a
  pinned staging ring against `file_to_card` on the Arc rig — as a measurement, not an import.
- **`cputune.py` — mostly already in Kindi.** `kindi/cpu.py` already defaults to one thread per
  physical core, sets `OMP_PROC_BIND=close` and binds the workers, so the OpenMP block duplicates
  existing behaviour. Two ideas survive as hypotheses, not code, and are folded into the M1
  arrival checklist above with their decision measurements: disabling THP on Intel too (Kindi's
  `engine._no_thp()` is gated on ROCm today, and host-memory compaction is what a CPU+Arc tandem
  decode would feel), and P-core pinning through `cpu_capacity` if the Arc rig is a hybrid part.
  The cpufreq-governor report is a diagnostic worth copying into the M1 checklist output, not a
  runtime feature.

### Verdict on `intel.py` and `ov_build.py` (read 2026-11-19)

Read at fork head `ad227a8`; nothing here is merged or copied. The useful material is the probe/report
shape, not the backend.

**`intel.py`: probe/report ideas worth reimplementing independently.** The one-machine profile is
the right scope for `kindi intel-info` and M1: CPU vendor/model/family/stepping/cores and feature
flags (`avx2`, `fma`, `avx_vnni`, `amx`), Intel display devices, NPU presence, Level Zero devices,
OpenVINO version/available devices, and oneAPI presence. Keep the current `/sys/class/drm` discovery
as the primary device source and use `lspci` only as corroboration. Preserve the PCI BDF, device id,
driver and render node for each device, and distinguish integrated from discrete entries.

The fork's Level Zero names are useful M1 evidence, but its ctypes probe guesses a
`ze_device_properties_t` layout in a 512-byte buffer and reads the name at a fixed offset; that is
not a stable ABI, so take the names from `sycl-ls --verbose` or a maintained loader binding instead.
Treat `KINDI_OV_DEVICE` and OpenVINO's device list as diagnostic fields only. Kindi's
`KINDI_DEVICE=intel` must remain the SYCL backend selector; do not adopt the fork's OpenVINO
GPU -> NPU -> CPU fallback. A missing probe should be explicit in the report (an error/absence
field), not silently omitted, and the layout must be testable with synthetic sysfs/command output
on a CPU-only host.

The M1 checklist now records `lscpu`, the feature flags, `lspci -nnk` for every Intel display device,
the `/dev/dri/by-path` render-node-to-slot mapping, `sycl-ls --verbose`, and optional OpenVINO
versions/devices as a baseline. The fork does not probe ReBAR or PCIe link state, so the existing
manual `lspci -vv`/sysfs capture stays mandatory.

**Not worth taking from `intel.py`:** `_gen_of`/`_SKU_MODELS` (duplicated and contradictory model
mappings; the profile stores the stepping-derived value in a field named `model`), broad `lspci`
substring matching, `pick_backend`'s OpenVINO precedence, `spec_ready`, and the OpenVINO
`Engine`/`export_ir` path.

**`ov_build.py`: no backend code worth reimplementing.** It is a non-runnable OpenVINO IR builder:
`main()` prints "weights are staged, graph emission pending" and returns 0 without writing
`model.xml`, its `Builder.build()` bodies are `pass`/`raise`, `attn_body()` returns `None`, and
`embed_body()` is a placeholder. `export_ir()` therefore cannot produce the IR `Engine` needs, and
there is no test, benchmark or A770 run. The only transferable test idea is a reduced-dimension
(`--tiny`) synthetic graph for an eventual backend smoke, which belongs in M2/M3, not M1. The graph
and state inventory (residual, GDN convolution/M state, KV cache) is design documentation, not
verified arithmetic. Do not adopt OpenVINO IR or `ov_genai`; M1/M2 remain SYCL/Level Zero and
Kindi's pack format.

## References

- [Intel Arc A770 16 GB specifications](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html)
- [Intel oneAPI system requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-base-toolkit/2025.html)
- [Intel: run LLMs on Intel GPUs with llama.cpp and SYCL](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html)
- [Intel: Resizable BAR on Arc A-series](https://www.intel.com/content/www/us/en/support/articles/000092416/graphics.html)
- [llama.cpp SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [Kindi CPU+GPU tandem design](CPU_GPU_TANDEM_DESIGN.md)
- [Kindi current CPU/GPU support boundaries](CARD-SUPPORT.md)
