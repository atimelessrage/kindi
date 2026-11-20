# CPU + Intel integrated-graphics tandem: investigation

Status: **investigation 2026-11-15; G0 slice implemented 2026-11-15, no kernel yet.** No Intel backend
exists in Kindi yet (the A770 route in [INTEL-ARC.md](INTEL-ARC.md) /
[REQ-0020](development/requirements/REQ-0020.md) is the discrete-card plan). This page investigates
the adjacent question: **tandem (CPU + iGPU layer split) using the integrated Intel graphics that
ship in a large installed base of CPUs**, and records the route decision. Tracked as
[REQ-0021](development/requirements/REQ-0021.md) (proposed). **Not a support claim.**

**Implemented (G0 slice, CPU-testable):** `kindi/intel.py` — device discovery from Linux DRM
sysfs/render nodes, eligibility against the oneAPI floor (unknown PCI ids refused, not assumed),
the shared-memory budget model (half-RAM cap; Core Ultra 2+ override), pack fit with reserve, and
the tandem GPU-side contract (`get_hidden`/`set_hidden`/`D`/`vocab`) a future SYCL engine must
expose. `KINDI_DEVICE=intel` refuses loudly from `kindi/engine.py` naming REQ-0020/0021. Covered
by `tests/test_intel.py`. The SYCL kernels themselves (G1+) need a supported iGPU to compile and
verify token-exactness — this host has neither oneAPI nor a 11th-gen+ iGPU, so they are not
started. The code changes below describe that remaining half.

## Why this is interesting for Kindi

Most tandem work so far assumes a discrete GPU with its own VRAM and a PCIe boundary between CPU
and GPU. Integrated Intel graphics remove both: the iGPU uses host system memory (UMA), so there
is no PCIe copy at the layer boundary and no VRAM fit cliff — the Kindi CPU engine already runs
2B/4B/9B on the CPU alone, so a CPU+iGPU tandem is "the same engines, with the GPU side on
shared memory". The installed base is the argument: oneAPI targets Intel UHD Graphics for 11th
generation processors or newer, Intel Iris Xe graphics or newer
([Intel oneAPI Base Toolkit system requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-base-toolkit/2025.html)),
and llama.cpp's SYCL backend verifies iGPUs in 11th–13th generation Core CPUs and built-in Arc
in Meteor Lake/Arrow Lake/Lunar Lake
([llama.cpp SYCL docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)).

## Hardware landscape (what "many CPUs" means)

| iGPU class | examples | architecture | oneAPI/SYCL |
|---|---|---|---|
| UHD 730/770 | 12th/13th gen Core desktop | Xe-LP (Gen12), 32 EU | supported (11th gen+ floor) |
| Iris Xe 80/96 EU | 11th–13th gen laptop Core | Xe-LP | supported |
| Built-in Arc 130V/140V | Core Ultra (Lunar Lake), 12–16 Xe-cores | Xe2-LPG | supported (verified) |
| Built-in Arc 140T | Core Ultra 3 (Panther Lake) | Xe2-LPG | supported via compute runtime |
| UHD 630 and older | 8th–10th gen Core (incl. this host's i7-8086K) | Gen9 | **below the oneAPI floor — excluded** |

The dividing line is Intel's own: oneAPI requires UHD for 11th generation or newer / Iris Xe or
newer. This host's 8086K (UHD 630, Gen9) is therefore not a test target for this route, and
Windows Device Manager here shows only the RTX 4060 Ti anyway.

## Memory: the fit model is different, not absent

- The iGPU has no dedicated VRAM; it uses host RAM. Windows reports up to one-half of system
  memory as the graphics limit for Iris Xe / UHD 730/750/770 / Core Ultra
  ([Intel graphics memory FAQ](https://www.intel.com/content/www/us/en/support/articles/000020962/graphics.html)).
- Core Ultra Series 2 and later gain a "Shared GPU Memory Override"
  (default 57%, example up to 87% of RAM; Panther Lake Arc Pro iGPUs to 93% on a 64 GB system)
  ([Intel support article](https://www.intel.com/content/www/us/en/support/articles/000101789.html),
  [Tom's Hardware](https://www.tomshardware.com/pc-components/gpu-drivers/intels-new-graphics-driver-lets-you-dedicate-up-to-87-percent-of-laptop-memory-capacity-to-the-igpu-for-vram-core-ultra-cpus-get-shared-gpu-memory-override-feature)).
- llama.cpp's SYCL docs give the practical rule: a 7B Q4 model needs ≥8 GB of shared memory on
  an iGPU (vs 4 GB on a discrete card)
  ([llama.cpp SYCL docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)).
  For Kindi's ternary packs the CPU engine already proves 2B/4B/9B fit in host RAM, so the
  relevant budget is: *shared graphics limit ≥ model + K/V + activations*, with the rest of RAM
  left for the OS and the CPU-side layers. `kindi/info` must report the shared limit, not a
  VRAM number.

## Stack options (what was compared)

| stack | verdict for Kindi |
|---|---|
| **SYCL / oneAPI + Level Zero** | **chosen** — same backend as REQ-0020, so iGPU support is a retarget (Xe-LP/Xe2-LPG) of the A770 work, not a second backend. Level Zero USM gives the tandem boundary a zero-copy design (below). Intel's own llama.cpp path; Linux + Windows supported. |
| Vulkan (Mesa ANV on Linux, Intel driver on Windows) | competitive — on Panther Lake it beat SYCL slightly (pp 624 vs 609 tok/s, tg 32.8 vs 30.6 on a small model, [grigio benchmark](https://grigio.org/benchmarking-llama-cpp-backends-on-intel-panther-lake-vulkan-vs-sycl-vs-openvino-vs-cpu)). But no USM/pinned-memory model and a second kernel language; keep as a baseline, not the Kindi backend. |
| OpenVINO | fastest prefill in one Qwen3-4B iGPU test (545 vs 290/229 tok/s) but it is a separate model runtime — same objection as in INTEL-ARC.md: not Kindi's engine, pack format or `kindi verify`. Baseline only. |
| OpenCL (CLBlast) | fallback for pre-11th-gen GPUs; Intel says performance is not optimal. Excluded from the target. |

## Tandem economics: what changes vs the NVIDIA tandem

1. **The boundary transfer nearly disappears.** Today's NVIDIA tandem pays 3.09 ms/token in
   synchronising `get_hidden()`/`set_hidden()` copies (0.44 ms × 7 switches, re-measured
   2026-11-15). With Level Zero USM, the hidden state can live in shared memory: the CPU engine
   writes it, the iGPU reads it, no copy and no PCIe. This is the one structural advantage of
   the iGPU tandem over the discrete tandem — the transfer term in the step-0 decomposition
   becomes ~0 by construction rather than by optimisation.
2. **But CPU and iGPU share one memory bus.** Decode is memory-bandwidth bound; every byte the
   CPU engine reads is a byte the iGPU cannot. The NVIDIA tandem's split rule (measured
   CPU/PCIe ratio, gate 1) does not transfer — the iGPU split rule must be measured as
   CPU-vs-iGPU bandwidth contention, with the display compositor as another consumer (a
   Lunar Lake study found 15–20% lower rates under a desktop session vs TTY,
   [lhl/intel-inference](https://github.com/lhl/intel-inference)).
3. **Decode gains are modest on older iGPUs.** Panther Lake (Xe2-LPG): 1.7× decode over CPU
   (32.8 vs 19.4 tok/s, small model). Iris Xe era: ~5–7 tok/s decode on small models, and one
   4B iGPU test had the iGPU *behind* a 22-core CPU on decode while winning prefill 9×
   ([Qiita iGPU study](https://qiita.com/yanma0102/items/df540c0d8ecbe382c347)). A tandem that
   loses decode to CPU-only is a no-go, same rule as the discrete tandem.
4. **Compute-runtime version matters ~2×.** One Iris Xe report: 2.7 → 5.6 tok/s decode between
   compute-runtime releases on the same binary/model
   ([raimund homelab log](https://raaaimund.github.io/posts/local-llm-homelab-llama-cpp-sycl-hermes)).
   Pin the runtime and record it with every benchmark, exactly as INTEL-ARC.md requires for
   the A770.

## What Kindi needs (smallest honest slice)

- The REQ-0020 SYCL backend, retargeted at Xe-LP/Xe2-LPG, with a **plain engine** first
  (no TC/plan/graphs — same rule as the A770 plan).
- A SYCL implementation of the tandem GPU-side contract
  (`get_hidden()`/`set_hidden()` over USM; the shared-memory design makes these
  pointer exchanges, and the `TandemEngine` layer-handoff code needs no PCIe staging).
- Fit/planning against the **shared-memory budget** (half of RAM, or the override
  percentage on Core Ultra 2+), not VRAM.
- `kindi/info` reporting device, Level Zero identity, driver, shared limit and
  the fitted context; installer detection via `/dev/dri` + `sycl-ls` — all behind
  the A770 gates, no separate installer until then.
- The CPU engine is unchanged; the CPU-side split rule is re-measured for
  bandwidth contention (new gate, below).

## Gates (fixed before implementation)

- **G0 (qualification):** an 11th-gen+ / Iris Xe / built-in-Arc iGPU visible in
  Level Zero with its shared limit recorded; a pinned compute-runtime version.
- **G1 (single device):** 2B plain path, `kindi verify` 4/4 token-exact.
- **G2 (tandem):** CPU+iGPU layer split token-exact 4/4 **and** decode ≥1.15× the
  CPU-only engine on the same host, or the tandem is a documented no-go (the
  NVIDIA tandem's rule). Prefill is expected to win big; decode decides.
- **G3:** 4B/9B within the shared budget, 8K stability measured.
- First hardware target: **built-in Arc (130V/140V/140T)** — that is where "many
  CPUs" are going; UHD/Iris Xe is the compatible second tier; pre-11th-gen is excluded.

## Risks

- **Shared-bandwidth contention** can make the tandem slower than CPU-only at the
  same placement; the split rule must be measured, not assumed.
- **Windows shared-memory cap** (half of RAM) can refuse a model the CPU engine
  would run; the override only exists on Core Ultra 2+.
- **Kernel driver churn** (i915 vs xe on Linux) and compute-runtime version
  swings; record versions with every number.
- **Laptop thermals/power**: iGPU and CPU share one power budget; sustained
  decode rates need a laptop measurement, not a desktop one.
- **Display contention**: a compositor on the same iGPU costs up to ~20%.

## References

- [Intel oneAPI Base Toolkit system requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-base-toolkit/2025.html)
- [Intel: run LLMs on Intel GPUs using llama.cpp](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html)
- [llama.cpp SYCL backend docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [Intel graphics memory FAQ](https://www.intel.com/content/www/us/en/support/articles/000020962/graphics.html)
- [Intel: Shared GPU Memory Override](https://www.intel.com/content/www/us/en/support/articles/000101789.html)
- [Benchmarking llama.cpp backends on Intel Panther Lake](https://grigio.org/benchmarking-llama-cpp-backends-on-intel-panther-lake-vulkan-vs-sycl-vs-openvino-vs-cpu)
- [Intel iGPU + 20 GB RAM Qwen3-4B study](https://qiita.com/yanma0102/items/df540c0d8ecbe382c347)
- [lhl/intel-inference (Lunar Lake)](https://github.com/lhl/intel-inference)
- [Intel Iris Xe architecture (oneAPI optimization guide)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/intel-iris-xe-gpu-architecture.html)
