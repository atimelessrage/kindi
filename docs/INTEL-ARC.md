# Intel Arc A770 support: goal and implementation plan

Status: **planning and architecture audit, 2026-11-10.** There is no Intel backend in Kindi yet. The
test computer has only an NVIDIA RTX 4060 Ti; no Arc GPU is visible, `sycl-ls` and `clinfo` are not
installed, and no Intel result has been measured. This page is a goal and a plan, **not a support
claim**.

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
| Arc A770 8 GB, A750, integrated Arc | not in the first support claim; measure before adding |
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
| CPU engine | whole-model C engine; verified for 2B/4B/9B, not 27B; no per-layer entry point | CPU+Arc tandem needs a real per-layer CPU API, not just two servers |
| CPU+GPU tandem | design only in [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md); the WIP `tandem.py` is not usable and the roadmap records no runtime | CPU+Arc tandem is the last milestone, not an offload flag |

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

The existing [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md) and
[TANDEM-STATUS.md](TANDEM-STATUS.md) are the starting design, but they are not a runtime. The
current CPU engine supports 2B/4B/9B; it has no 27B per-layer path. Therefore CPU+Arc tandem starts
with 2B/4B/9B; 27B/33B wait for CPU per-layer support, not merely an Arc port.

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
| **M2** Backend smoke test | `libksycl.so`, USM, vector-add/GEMM and device selection | the smoke binary runs on the A770 with no CPU fallback |
| **M3** Single-card 2B exact | plain engine, exact head, `kindi verify` | 4/4 Bonsai prompts token-exact; no fast/spec/TC |
| **M4** Server and cache | `kindi serve`, non-stream/stream, prompt cache, health/metrics | same outputs as M3; cache save/restore works |
| **M5** 4B/9B/27B | model matrix and context fit | exact verify for each advertised model; 8K/24K stability measured |
| **M6** Two Arc cards | layer split, two queues, host exchange | 27B exact; dual is at least 1.3x single-card at the same context; no exchange race |
| **M7** CPU + Arc tandem | per-layer CPU API, scheduler, 1/2 Arc | exact; at least 1.15x Arc-only, or it enables a model/context that Arc-only cannot fit |
| **M8** Installer and support claim | Intel container, detection, docs, CI/manual matrix | installed setup passes the same checks as the NVIDIA rows; no hidden environment override |

M1 is currently **blocked on hardware**: there is no Arc A770 on the test computer. No Arc code has
been executed, so M2 onward are not started.

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

- **No Arc hardware on the test host.** This is the immediate blocker; no amount of code review
  substitutes for `sycl-ls` and a token-exact run.
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
- **CPU tandem.** The CPU engine is whole-model C and has no 27B path; a per-layer CPU API is the
  real prerequisite. The existing design is not a runtime.
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

## References

- [Intel Arc A770 16 GB specifications](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html)
- [Intel oneAPI system requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-base-toolkit/2025.html)
- [Intel: run LLMs on Intel GPUs with llama.cpp and SYCL](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html)
- [Intel: Resizable BAR on Arc A-series](https://www.intel.com/content/www/us/en/support/articles/000092416/graphics.html)
- [llama.cpp SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [Kindi CPU+GPU tandem design](CPU_GPU_TANDEM_DESIGN.md)
- [Kindi current CPU/GPU support boundaries](CARD-SUPPORT.md)
