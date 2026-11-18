# Hardware identification and per-machine tuning

Two commands answer the two questions a user has before choosing any setting: **what can this
machine do** (`kindi caps`), and **which placement does this model actually run best with here**
(`kindi autotune`). Both are evidence-only: the first reports what the driver and the OS say, the
second ranks real `kindi bench` runs and never estimates a throughput.

Requirement: [REQ-0022](development/requirements/REQ-0022.md) (experimental).

## `kindi caps` — what each card and this CPU can do

```bash
kindi caps          # human-readable, one block per card plus the CPU
kindi caps --json   # the same report as data
```

Identification is layered, because the layer that is missing is often the one that matters:

| source | when it is used | what it gives |
|---|---|---|
| `nvidia-smi` | default | index, name, UUID, PCI bus id, **compute capability**, total/free VRAM, driver |
| `pynvml` | nvidia-smi missing, Python bindings installed | name, compute capability, total/free |
| CuPy | no NVIDIA tooling | name, compute capability, total (no free figure) |
| none | no card | the command still answers: "none detected", with the CPU half |

A row with no `compute_cap` (drivers before CUDA 11.6) is reported as **unknown**, not guessed. The
report keeps two lists apart on purpose:

- **hardware** — what the silicon has, from the compute-capability tables: tensor cores (sm_70+),
  BF16/TF32 (sm_80+), FP8 (sm_89+), FP4 and microscaling (sm_100+), async copy and
  FlashAttention-2-class features (sm_80+);
- **Kindi can use** — only the paths that exist in this tree for that capability: the 5-trit
  kernels, the dense CPU+GPU tandem split, the plan engine's tensor-core MMA (sm_70+), and the
  FP8 Gemma 4 path (sm_89+, on by default, `KINDI_G4_FP8=0` disables it).

A card having FP8 tensor cores does not mean Kindi has an FP8 kernel for the model you loaded, which
is why the second list is shorter and why AVX-512 on the CPU is printed as *present but unused*:
there is no AVX-512 kernel, and that is a named roadmap gate rather than a hidden one.

The detection half of the interrupted 2026-11-09 CPUID draft
(`wip/main-interrupted-snapshot-20261109-045352:kindi/kcpu_cpuid.c`) was taken over here — that is
where the AVX-512/AMX reporting started. Its AVX-512 kernels (`fwht_block_avx512`, `rows16_avx512`)
stay archived on purpose: no machine here has AVX-512, so they could not be compiled, verified or
measured, and an unverifiable kernel would be a claim, not a feature. Reopening them needs an
AVX-512 host and the roadmap gate that already names it.

The CPU half reuses the engine's own probe (`kindi/cpu.py`), so the report cannot disagree with the
engine about AVX/AVX2/FMA3/F16C or about which build tier would be selected. On top of that it adds
what the engine does not need but a tuner does: the marketing model name and core counts (from WMI on
Windows, `/proc/cpuinfo` on Linux), L2/L3 size, and whether AVX-512/AMX exist.

Which OS answers which question, so a value is never invented:

| flag | Linux | Windows |
|---|---|---|
| AVX, AVX2 | `/proc/cpuinfo` flags | `kernel32.IsProcessorFeaturePresent(39/40)` |
| AVX-512F | `/proc/cpuinfo` flags | `IsProcessorFeaturePresent(41)` |
| AMX | `/proc/cpuinfo` flags | unknown (no kernel32 constant) |
| FMA3, F16C | `/proc/cpuinfo` flags | unknown — the build compiler probe answers them, which is why the line says so |
| L2/L3 | `lscpu` | WMI `Win32_Processor` |

The engine's own answer always wins where it has one: `kindi caps` only fills values the OS would not
give the engine probe, so it cannot report something the engine disagrees with.

Two honesty rules:

- a feature the OS does not expose is **unknown**, never "unsupported" (Windows hands Python no CPU
  flags; the build compiler probe is the authority there, and the output says so);
- the **CPU engine toolchain** is part of the report. A CPU feature nobody can compile is not an
  available feature: without `gcc`/`clang` (and libomp on Windows) the CPU engine and the tandem
  split cannot be built at all, so `kindi caps` says that instead of implying they work.

Example on this host (RTX 4060 Ti, i7-8086K, Windows 10):

```
GPU: 1 card (identified with nvidia-smi)
  card 0: NVIDIA GeForce RTX 4060 Ti  [sm_89]
           8188 MiB total, 6994 MiB free, driver 617.42, PCI 00000000:01:00.0, UUID GPU-…
           hardware: tensor_cores, bf16, tf32, fp8, async_copy (no fp4)
           Kindi can use: 5-trit kernels, dense CPU+GPU tandem split, plan engine (tensor-core MMA),
                          FP8 Gemma 4 path (on by default, KINDI_G4_FP8=0 disables)
CPU: Intel(R) Core(TM) i7-8086K CPU @ 4.00GHz, 6 physical / 12 logical threads
     AVX=yes, AVX2=yes, FMA3=unknown, F16C=unknown
     L3 12 MiB, L2 1 MiB, AVX-512=no, AMX=unknown
     Kindi CPU path: unknown until the compiler selects the path (build native (-march=native)), 6 threads recommended
     CPU engine toolchain: clang not found on PATH (set KINDI_CC, and KINDI_LLVM on Windows) — the CPU engine and the tandem split cannot be built here
     note: FMA3/F16C not exposed by this OS; the build compiler probe (kindi/cpu.py, KINDI_CC) answers them and picks the tier
```

## `kindi io` — what the memory tiers actually cost

```bash
kindi io --mib 64              # RAM copy, unbuffered disk read, PCIe both ways
kindi io --mib 64 --json       # the same report as data
kindi io --mib 64 --json ~/.kindi/io.json  # save it for plan/autotune
kindi io --skip disk,pcie      # only the host copy rate
```

`--skip` accepts only `ram`, `disk` and `pcie`; a misspelled name exits 2 with that set instead of
running the probe. Every plan that moves experts between tiers has to assume three rates: the host memcpy into a
staging buffer, the disk for an expert that is not in RAM, and PCIe for a host expert copied into a
device slot. `kindi io` measures them on this machine. The disk probe writes a temporary file and
reads it **unbuffered** (`O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` on Windows), because a
buffered read of a file this process just wrote measures the page cache; where the OS refuses, the
report says `buffered` and that rate is a cache/RAM rate, not the device's. The PCIe probe uses
pinned host memory through the active CUDA device, like the engine's staging buffers, and reports
H2D and D2H separately.

`kindi io --json FILE` writes the report as well as printing it. `kindi plan` and `kindi autotune`
accept `--io-report FILE` (or `KINDI_IO_REPORT`) and normalize that saved report exactly like a fresh
probe, so the 64 MiB disk probe runs once instead of on every plan. A missing, unreadable or
rate-less file is reported by name and both commands fall back to one fresh probe; with no report
selected they say how to save one. A report describes this host and this storage, so rerun
`kindi io` after changing either.

Measured on the i7-8086K + RTX 4060 Ti, 2026-11-17, `--mib 64`: RAM copy **13,456.7 MiB/s**, disk
read **2,509.8 MiB/s** unbuffered, PCIe **6.0 GiB/s H2D / 6.1 GiB/s D2H**. These are this host's
numbers, not targets: a different disk or a PCIe x4 riser moves them, which is why they are measured
before a cache or a streaming budget is sized. The probe loads no model, so it runs on a machine with
no pack, in about a second per probe at the default size.

## `kindi autotune` — which placement this model actually likes here

```bash
kindi --pack ~/.kindi/packs/bonsai-2b --ctx 8192 autotune --dry-run   # the plan, loads nothing
kindi --pack ~/.kindi/packs/bonsai-2b --ctx 8192 autotune --max-runs 6 --json tune.json
```

The candidate list is derived from the hardware report and the pack manifest, so the tuner only
proposes what this machine could actually run:

| candidate | offered when | what it changes |
|---|---|---|
| `gpu0` | a card exists | the baseline: one card holds the whole model |
| `tp-2-auto`, `tp-2-equal` | **two or more cards** | tensor-parallel row split (`KINDI_TP=1`, `KINDI_TP_SPLIT=auto` or `equal`) |
| `tandem-auto` | a card, a dense pack, ≥4 physical cores | the smallest CPU slice that makes the requested context fit |
| `tandem-gdn:8`, `tandem-gdn:16` | same | fixed CPU slices: the first N non-attention layers |
| `dcache-<sized>` | a Lightning (MoE) pack | decode LRU slots per expert layer (`KINDI_LT_DCACHE`, default 10) against VRAM; with measured rates the list is the predicted count plus half and double it |
| `expert-cache-<MiB>` | a Lightning pack with measured rates and host-tier experts | the pinned host cache for the cold tier: `KINDI_LT_COLD=1` plus `KINDI_LT_HOST_MB`, sized from the same rate-sized working set and capped by the experts that live on the host tier |
| `cpu` | no card, or `--include-cpu` | the CPU engine (`KINDI_DEVICE=cpu`) |

For a Lightning pack the slot candidates are sized from the `kindi io` rates: one token's `n_used`
experts plus `ceil((RAM copy + disk read) / PCIe H2D)` lookahead of the same working set, capped by
`n_exp` and by the room that still leaves `n_used` experts per layer resident. The plan's `rates:`
line prints the three assumptions. The `expert-cache-<MiB>` candidate keeps that same rate-sized
working set in pinned host RAM, multiplied by the MoE layer count and the expert bytes and capped by
the host tier's `n_exp - n_card`; it sets `KINDI_LT_COLD=1` as well as `KINDI_LT_HOST_MB`, because the
budget is inert on the default pinned-RAM path. Both are predictions; a Lightning-pack comparison is
the gate.

Every candidate is **one real `kindi bench` run in a child process**, with `--sample-vram`, so decode
and read rates, the context actually granted and the device-wide VRAM peak all come from the code
path `serve` uses. Ranking is by measured decode tok/s; a run that had to **reduce the context** is
flagged and ranked below every run that got the requested one, because a faster number at a smaller
context answers a different question. A candidate that fails is printed with the child's own last
line and is never scored — a missing compiler or a refused split stays visible instead of becoming a
zero in a table.

`--json FILE` records the candidates, every run and the winner. The `--dry-run` plan is what to read
before spending measurements.

### Measured on this host (2026-11-17)

Bonsai 2B pack, `--ctx 8192`, RTX 4060 Ti + i7-8086K, one card:

| config | decode | read | peak | outcome |
|---|---|---|---|---|
| `gpu0` | 97.4–97.8 tok/s | 111.7–113.0 tok/s | 3,050 MiB | winner, recommended |
| `tandem-auto` | — | — | — | refused by the planner: "this model already fits at 8192 tokens (~2410 MiB needed, 7075 MiB free); a split would only make it slower" |
| `tandem-gdn:8` | — | — | — | failed: "could not run clang … it needs gcc with OpenMP, or clang and libomp on Windows" |

Both failures are the tuner behaving correctly rather than a tuner bug: the planner refuses a split
that would only cost decode (the 9B/27B measurements in [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md) say
the same), and this shell has no compiler on `PATH`, which `kindi caps` now reports up front.

## What this does not claim

- **No exactness.** `kindi verify` owns token-exactness; a winner that changes placement (tandem,
  two-card split) must pass it before it is served, and the tuner says so in its last line.
- **No two-card numbers yet.** The `tp-2-*` candidates are unit-tested here but unmeasured: this
  host has one card. The same is true for the Lightning decode-slot and host-cache candidates (no
  Lightning pack here).
- **Not a search over everything.** Context sizes, `KINDI_KV` modes, chunk sizes and the sampler are
  out of scope; the tuner covers placement, the split and the MoE decode cache — the settings whose
  best value is machine-specific. Each run is a real model load, so the budget is the user's.
