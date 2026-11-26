# Long-term CPU support

Where this sits: this page is the CPU support contract; [CARDS.md](CARDS.md) and
[SUPPORT-MATRIX.md](SUPPORT-MATRIX.md) cover the cards it runs next to, and [README.md](README.md)
is the index.

Status: **roadmap and acceptance contract, updated 2026-11-12. The i7-8086K engine and native CPU
server chat path on the 2B/4B/9B are verified. The AVX-only fallback and the AVX2/F16C no-FMA
fallback are implemented; a forced AVX-only build passes the 2B `kindi verify` 4/4 token-exact on
the i7-8086K, and a forced AVX2/F16C no-FMA build passes the 2B `kindi verify` 4/4 token-exact (31.7 tok/s). All five forced ISA tiers now pass the 4B and 9B
4/4 token-exact; the installer AVX preflight is published in the refreshed 0.5.0 artifact. Native
Sandy Bridge and native no-FMA hardware are still untested. The other CPUs and a repeat timing run
remain open.** The baseline CPU engine (`kindi/cpu.py`, `kindi/kcpu.c`)
is in `main`; this page records the target, the verified result, and the exact build/runtime
requirements so the work can be implemented and measured without re-discovering them. The tandem
recovery gate is in [TANDEM-STATUS.md](TANDEM-STATUS.md).

## What “supported” means here

A CPU is supported when all of these hold:

1. `libkcpu.so` builds on that CPU with the documented gcc and OpenMP toolchain.
2. `KINDI_DEVICE=cpu` runs the advertised models and the server answers a chat request.
3. `kindi verify` passes for the advertised models (token-exact with llama.cpp for Bonsai).
4. The CPU is listed in `docs/CARDS.md` with its thread count and measured result.
5. It works without an illegal-instruction crash, and without an undocumented manual override.

The current CPU engine covers the **2B, 4B and 9B**, and since 2026-11-24 it also runs the **27B**:
`KINDI_DEVICE=cpu` loads the 27B pack and `kindi verify` is **4/4 token-exact** on the i7-8086K
(3.7 tok/s over the run, 340 s load, 31.9 GiB host RAM). That is an exactness row, not a recommendation: the 27B
is a GPU model and this path is the fallback of a machine without one.

The **Bonsai 2.5 27B** runs on the same path (2026-11-25) and is the one row here without a llama.cpp
reference: the released `PTQ1_0` file converts and packchecks, the CPU engine loads it in 339.9 s
and generates greedily at 4.02 tok/s (best of two prompts, 3.08 tok/s worst) at ctx 2,048. `kindi
verify` refuses it by name instead of scoring it against Bonsai 2's tokens and says how a reference
is supplied (`--reference FILE`, `docs/BONSAI-2-SUPPORT.md` section 1a). So the row is load and
generate, not exactness: nothing here is a token-for-token comparison with another engine. On a host
without a card, `kindi models` lists both 27B releases as `CPU engine` and `kindi plan` prints CPU
follow-up commands rather than the GPU split this engine cannot build (2026-11-26).

Nemotron and Nanbeige are separate backends and are not part of this CPU goal.

## Current backend reality

| gate | current behaviour | consequence |
|---|---|---|
| `kindi/cpu.py` build | Linux: `gcc -O3 <tier flags> -fopenmp -fno-math-errno -shared -fPIC`; Windows: `clang -O3 <tier flags> -fopenmp -fno-math-errno -ffp-contract=off -shared` plus `libomp` (`KINDI_CC`, `KINDI_LLVM`; a bare `clang` that is not on `PATH` is resolved to a standard LLVM install under Program Files, 2026-11-24) | One library per ISA tier, built on demand and cached by OS/architecture/compiler/tier. A shared cache can serve AVX2 and AVX-only hosts without mixing binaries. |
| runtime dispatch | `kindi/cpu.py` selects the best tier from `/proc/cpuinfo` (Linux) or the build compiler's `-march=native` macros (Windows), builds it with explicit flags and loads it. `KINDI_CPU_ISA` pins a tier. `kcpu.c:kc_isa_name()` reports the compiled tier and the loader rejects a mismatch. | The i5-2500K path is selected at runtime, not by a special build; forcing an unsupported tier is an actionable error instead of an illegal instruction. |
| `kindi/kcpu.c` ISA | Requires AVX. The AVX2 path (`_mm256_srlv_epi32`, `_mm256_permutevar8x32_ps`) is used whenever the compiler enables AVX2. F16C (`_cvtsh_ss` / `_cvtss_sh`) is used when available, with a software fp16 fallback. FMA (`_mm256_fmadd_ps`) is optional: without it the AVX2 kernel uses separate multiply/add. An AVX-only build uses the scalar AVX table lookup. A compile-time `#error` names a missing AVX baseline. | AVX2/FMA/F16C hosts get the full fast path; AVX2/F16C hosts without FMA keep the AVX2 table/permute path; an AVX-only CPU such as Sandy Bridge gets the scalar fallback. |
| documented requirement | `docs/CARDS.md` and the README say x86-64 with AVX; AVX2/F16C is the fast path, FMA is optional, and AVX-only is the fallback | This matches the source: AVX is required; `kindi/cpu.py` selects the AVX2/FMA/F16C, AVX2/F16C, AVX2 or AVX-only tier at startup from the host flags, and each tier is built with explicit `-m` flags. |
| `install.sh` preflight | Checks Linux, `x86_64` and AVX; it classifies AVX2/FMA/F16C, AVX2/F16C without FMA, and AVX-only. CPU setups are hidden without AVX, and `--gpu cpu` stops with an actionable message. GPU setups remain available. | A CPU-only machine without AVX is rejected before the build; AVX2/F16C without FMA and AVX-only CPUs are accepted, and the selected path is printed before building. |
| CPU engine thread default | `kindi/cpu.py` counts the physical cores this process may run on (sysfs `thread_siblings_list`, `/proc/cpuinfo` fallback) and sets `OMP_NUM_THREADS`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` before the library loads; `threads=` or an `OMP_NUM_THREADS` already in the environment wins | On a hybrid i9 this counts 24 cores, E-cores included. The best P/E split has not been measured. |
| `install.sh` CPU default | Counts the physical cores of socket 0 (`lscpu -p=CORE,SOCKET`) and exports the same three variables for the CPU container; `--threads N` overrides it | Two sockets are not measured; the engine's own default counts every CPU it may use, not socket 0. |
| build container | `docker/Dockerfile` is Ubuntu 18.04 with the distribution gcc and `libgomp1` | The old gcc may not know Raptor Lake as a `-march=native` target. An explicit AVX baseline with optional AVX2/FMA/F16C flags is the portable build. |
| test coverage | `tests/` has server/config/cache tests plus `test_cpu_build.py`, `test_cpu_threads.py` and the gated `test_cpu_smoke.py`; `test_cpu_build.py` checks the AVX guard, preprocesses every AVX2 variant, compiles and loads the AVX-only, AVX2/F16C no-FMA and AVX2/FMA/F16C builds through `kindi.cpu._lib()` without a pack, and pins the explicit flags and per-tier cache paths; `test_cpu_features.py` pins `select_isa()` including forced and unsupported tiers; the CI `cpu-engine-smoke` job compiles `kcpu.c` with gcc and loads the library | The compile/load/selection path is covered; a full token cannot run without a model pack. |

## Runtime ISA dispatch (2026-11-11)

`kindi/cpu.py` no longer builds one `-march=native` library. It classifies the host, picks one of
five tiers, builds that tier with explicit flags on first use, and loads it. The cached library is
keyed by OS, architecture, compiler and tier, so one cache can serve AVX2 and AVX-only machines
without mixing binaries. `kcpu.c` reports the compiled tier through `kc_isa_name()`, and the loader
refuses a library whose report does not match the selection.

| tier | build flags | kernels |
|---|---|---|
| `avx2-fma-f16c` | `-mavx2 -mfma -mf16c` | AVX2 table/permute, FMA, hardware fp16 |
| `avx2-f16c-no-fma` | `-mavx2 -mno-fma -mf16c` | AVX2 table/permute, separate multiply/add, hardware fp16 |
| `avx2-fma` | `-mavx2 -mfma -mno-f16c` | AVX2 table/permute, FMA, software fp16 |
| `avx2` | `-mavx2 -mno-fma -mno-f16c` | AVX2 table/permute, separate multiply/add, software fp16 |
| `avx-only` | `-mavx -mno-avx2 -mno-fma -mno-f16c` | scalar AVX table lookup, software fp16 |

Selection is `KINDI_CPU_ISA=auto` by default: the best tier whose features `/proc/cpuinfo` (Linux)
or the build compiler's `-march=native` macros (Windows) reports. `KINDI_CPU_ISA=<tier>` pins one
for a benchmark or a scheduler; forcing a tier the CPU explicitly lacks is an error, not a silent
illegal instruction. When the OS exposes no flags, the old `-march=native` path remains the
fallback. `kindi info` prints the detected path and the build that will be loaded.

Measured on the i7-8086K (native Windows, clang 18.1.8, 6 threads, 1,024-token context, 2B pack,
one run per tier):

| tier | 2B verify | tokens/s | evidence |
|---|---|---:|---|
| `avx2-fma-f16c` | 4/4 token-exact | 43.6 | `bench-logs/cpu-dispatch-avx2-fma-f16c.log` |
| `avx2-f16c-no-fma` | 4/4 token-exact | 43.9 | `bench-logs/cpu-dispatch-avx2-f16c-no-fma.log` |
| `avx2-fma` | 4/4 token-exact | 31.9 | `bench-logs/cpu-isa-2b/avx2-fma.log` |
| `avx2` | 4/4 token-exact | 29.1 | `bench-logs/cpu-isa-2b/avx2.log` |
| `avx-only` | 4/4 token-exact | 8.9 | `bench-logs/cpu-dispatch-avx-only.log` |

All five forced tiers now have an end-to-end 2B verify pass. The two software-F16C tiers are slower
than the hardware-F16C paths (31.9 and 29.1 tok/s against 43.6 and 43.9), as expected. These are
forced-tier runs on an AVX2 host, not native Sandy Bridge or no-F16C measurements.
Cross-tier parity is pinned by `test_cpu_isa_tiers_agree_on_tokens_and_hidden_close`
(`tests/test_cpu_layers.py`): the five tier libraries are built and loaded and run the same
fixed decode sequence on the 2B. Greedy token picks are identical across all five tiers; the
hidden states agree to ~3e-4, which is summation-order fp32 rounding (byte-identical hidden
states are not the claim — structurally different accumulation orders cannot be bitwise equal).
The AVX-only
build is about 5x slower on the 2B and 4.9x/5.5x slower on the 4B/9B, so it remains a compatibility
path. No native CPU without F16C hardware was available. The 4B/9B rows below provide the same
end-to-end coverage for those models. Evidence for the two new runs is in
`bench-logs/cpu-isa-2b/RESULTS.md` and their full logs.

### 4B/9B forced-tier runs (2026-11-12)

The same forced-tier protocol was run against the 4B and 9B packs on the i7-8086K: native Windows,
clang 18.1.8, 6 threads, 1,024-token context, one fresh process per tier/model, and a dedicated
`KINDI_CPU_CACHE`. All ten runs exited 0 and reported `ALL PASS` with four prompt passes (the 9B
prompt 3 stops at 13/13 tokens, as in the earlier runs). These are single runs, so the AVX2 rates
are samples, not a tier ranking.

| tier | 4B verify | 4B tok/s | 9B verify | 9B tok/s |
|---|---:|---:|---:|---:|
| `avx2-fma-f16c` | 4/4 | 18.1 | 4/4 | 10.7 |
| `avx2-f16c-no-fma` | 4/4 | 15.8 | 4/4 | 10.4 |
| `avx2-fma` | 4/4 | 17.8 | 4/4 | 10.9 |
| `avx2` | 4/4 | 19.2 | 4/4 | 11.0 |
| `avx-only` | 4/4 | 3.9 | 4/4 | 2.0 |

Evidence: `bench-logs/cpu-isa-4b-9b/RESULTS.md`, `bench-logs/cpu-isa-4b-9b/summary.json`, and the
ten raw logs in `bench-logs/cpu-isa-4b-9b/logs/`. The 4B manifest SHA-256 is
`e7166ca35a0f97fc973acacaff1edb7eeb1f63098e7a343d6f76438c2fedd7dd`; the 9B manifest is
`f5e4a8b1629e30936d5bf0ba32b99a4d460d3e937bbcc1ca76868a1391f72d07`. The AVX-only path is
about 4.9x slower than the fastest 4B AVX2 run and 5.5x slower than the fastest 9B run. Forced-tier
runs on this AVX2 host are compatibility evidence, not native Sandy Bridge or no-FMA support.

### The 27B on the CPU (2026-11-24)

The 27B is a 64-layer gated-attention/GDN model, and `kcpu.c` already implements both block shapes
(`qn[2*HD]` carries the per-head query gate), so the pack loads on the CPU engine unchanged:

```sh
KINDI_DEVICE=cpu python3 -m kindi --pack <packs>/bonsai-27b verify
```

| measurement | result |
|---|---|
| pack integrity before the engine | 10 files, size + fast hash + sha256 (18.4 s) |
| tensor map | 851 tensors, byte ranges tile `bcodes.bin`/`dpm.bin`/`f32.bin` exactly |
| load | 340 s (fresh build) / 417.5 s (cached build), `avx2-fma-f16c`, 6 threads, ctx 1,024 |
| `kindi verify` (4 prompts x 40 greedy tokens) | **4/4 PASS, 40/40 tokens each**, both libraries |
| throughput over the run | **3.7 tok/s** (library built from the current `kcpu.c` in this session), 2.0 tok/s (the cached 2026-11-20 build) |
| exit code | 0 |

That is exactness against llama.cpp's q8_0-K/V reference for the same GGUF, on the CPU engine's own
16-bit K/V. It is not a speed recommendation (3.7 tok/s against 34.8 tok/s for the same 27B on the
RTX 4060 Ti) and it does not make the 27B a CPU target: the row exists so a host without a card can
still prove a 27B pack is intact and its engine decodes faithfully. The 64-layer pack needs about
13 GiB of host memory for the pair codes and scales on top of the 6.8 GiB on disk.

## Target matrix

| CPU | Microarchitecture | Cores / threads | Relevant ISA | Current state | Target |
|---|---|---|---|---|---|
| Intel Core i5-2500K | Sandy Bridge | 4 / 4 | AVX; **no AVX2, FMA or F16C** | **AVX-only tier selected at runtime**: forced `KINDI_CPU_ISA=avx-only` passes 2B 4/4 token-exact at 8.9 tok/s, 4B 4/4 at 3.9 and 9B 4/4 at 2.0 on the i7-8086K; native i5-2500K hardware not available | 2B and 4B; 9B only if measured and useful; 27B no |
| Intel Xeon W-2133 | Skylake-W | 6 / 12 | AVX-512, AVX2, FMA, F16C | Expected to build and run, but **unverified** | 2B, 4B and 9B |
| Intel Core i9-13900K | Raptor Lake (8 P + 16 E) | 24 / 32 | AVX2, FMA, F16C; no AVX-512 | Expected to build and run, but **unverified** | 2B, 4B and 9B; tune P/E thread count |
| Intel Core i7-8086K | Coffee Lake | 6 / 12 | AVX2, FMA, F16C | **Engine and server verified**: native Windows (2B/4B/9B 4/4 at 33.7/16.7/9.1 tok/s, 6 threads auto-detected; 24K generation 8.2/3.1/2.6 tok/s, read 45.1/19.2/9.9), WSL2 history, native CPU server chat requests on the 2B/4B/9B (32.9/19.2/11.5 tok/s, 1,024 context, non-stream and stream, 0 errors), and the **27B 4/4 token-exact at 3.7 tok/s** (2026-11-24) | 2B, 4B, 9B and the 27B (exact, slow); use 6 threads |

Notes:

- The i5-2500K is the only CPU here without AVX2/FMA/F16C. `kindi/cpu.py` now selects the
  `avx-only` tier on such a host: AVX table lookup, separate multiply/add and software fp16. The 2B
  pass is a forced-tier check on an AVX2 host, not a native Sandy Bridge claim; the 4B and 9B
  also pass the forced tier, but a native run is still needed for support.
- An AVX2/F16C host without FMA (for example a VM with FMA masked) uses the AVX2 table/permute
  kernel with separate multiply/add. This is not a native CPU row yet: the forced no-FMA build
  passes 2B 4/4 at 31.7, 4B 4/4 at 15.8 and 9B 4/4 at 10.4 tok/s on the i7-8086K; native no-FMA
  hardware is still needed for a support claim.
- The Xeon W-2133 supports AVX-512, but `kcpu.c` is written with 256-bit AVX2 intrinsics. The compiler
  may auto-vectorize surrounding loops to AVX-512; test whether `-mprefer-vector-width=256` avoids a
  frequency drop without losing throughput.
- The i9-13900K is hybrid. The current installer counts 24 physical cores and OpenMP places threads on
  “cores”, so E-cores will be used. Test `--threads 8`, `16`, `24` and `32`, then choose a default from
  the measurement rather than from the core count.
- The i7-8086K is now measured natively: 2B 45.1 read / 35.9 write at depth 100, 4B 19.2 / 15.2 and
  9B 9.9 / 8.7 tokens/s, with 2B/4B/9B at 8.2/3.1/2.6 tokens/s at 24K. The WSL2 run (35.2/28.2,
  14.2/11.8, 7.6/6.8; 4.6/1.6/1.4 at 24K) is kept as history. The 12-thread configuration is 6-20x
  slower and 10 is slower too; 8 is only a WSL2/libgomp tuning (see the thread sweep below). Use 6
  threads (one per physical core, now the engine's default — see below). See the
  [native benchmark](benchmarks/2026-11-10-i7-8086k-windows-cpu/README.md). The i7-8700 row remains a
  separate 6-core reference point.
- These are single-socket desktop/workstation CPUs. Multi-socket CPU inference is out of scope.

## i7-8086K processor details

The CPU used for the measurements in this page is the Intel Core i7-8086K:

| item | value |
|---|---|
| architecture | Coffee Lake, 14 nm |
| cores / threads | 6 physical cores / 12 logical threads (Intel Hyper-Threading, 2 threads per core) |
| clocks | 4.00 GHz base; up to 5.00 GHz single-core turbo; all-core turbo is typically around 4.3-4.4 GHz, depending on the board, cooling and power limits |
| cache | 12 MB L3 |
| multiplier | unlocked (K suffix) |
| release / positioning | mid-2018 limited 40th-anniversary edition; closely related to the Core i7-8700K, with a higher 5.0 GHz single-core turbo |
| benchmark clocks | no explicit overclock and no thermal or power clamp; per-core clocks were not logged, so turbo residency and the board's power limits are not verified |

The all-core turbo figure is a typical operating range, not a guaranteed clock. The 6/8/10/12
thread results below use the 12 logical CPUs: 6 threads take the six physical cores, 8 and 10 add
SMT pairs, and 12 uses all six pairs.

## Verified i7-8086K result (2026-11-09)

Environment: WSL2 Ubuntu 24.04, kernel 6.18.54.1-microsoft-standard-WSL2, GCC 13.3.0, Intel Core
i7-8086K (6 cores / 12 threads), 15 GiB visible to WSL2 of 32 GB installed, HEAD `0ce4350`, and a
24,576-token benchmark context. The CPU library was built with `gcc -O3 -march=native -fopenmp`.
The native Windows result is recorded in the next section.

Verify used a 1,024-token context:

| Model | Prompts | Tokens/s | Result |
|---|---:|---:|---|
| 2B | 4/4 | 28.1 | token-exact |
| 4B | 4/4 | 13.0 | token-exact |
| 9B | 4/4 | 5.6 | token-exact |

Benchmark at 24,576 tokens with 6 threads (`OMP_NUM_THREADS=6`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`):

| Model | Read | 100 | 2,000 | 8,000 | 16,000 | 24,000 |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 35.2 | 28.2 | 21.3 | 12.2 | 6.7 | 4.6 |
| 4B | 14.2 | 11.8 | 9.1 | 3.9 | 2.3 | 1.6 |
| 9B | 7.6 | 6.8 | 5.8 | 3.2 | 2.0 | 1.4 |

The same benchmark with 12 threads is 6-20x slower (2B read 1.8, 4B 1.2, 9B 1.1 tokens/s); it is
not a recommended configuration. The follow-up probe below identifies libgomp's active spin-wait on
the hyperthread pairs as the cause, so `kindi/cpu.py` now defaults to one thread per physical core.
The native 24K numbers in the next section replace these cells in the cross-mode matrix.

The full CPU test suite passes (`127 passed, 1 warning`), but it does not build `kcpu.c`; the native
runs above are what exercise the C engine. This is one run per model/thread configuration, with no
thermal or power clamp. The server path and native Windows build were not tested in that WSL2 run;
the native results are in the next section.

Evidence: [README](benchmarks/2026-11-09-i7-8086k-cpu/README.md),
[summary.json](benchmarks/2026-11-09-i7-8086k-cpu/summary.json),
[runs.csv](benchmarks/2026-11-09-i7-8086k-cpu/runs.csv), and the raw verify/bench logs indexed there.

## Verified native Windows result (2026-11-10)

Environment: Windows, Intel Core i7-8086K, Python 3.13, clang 18.1.8
(`clang+llvm-18.1.8-x86_64-pc-windows-msvc`) with the MSVC 14.51 toolset and Windows SDK that clang's
driver finds through `vswhere`; no `vcvars` shell is needed. `kindi` builds a keyed `libkcpu-<key>.dll`
on first use into `KINDI_CPU_CACHE`; the key covers the host CPU and compiler.

```bat
set KINDI_CC=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin\clang.exe
set KINDI_LLVM=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc
set KINDI_DEVICE=cpu
python -m kindi verify
```

| Model | Prompts | Tokens/s | Result |
|---|---:|---:|---|
| 2B | 4/4 | 33.7 | token-exact |
| 4B | 4/4 | 16.7 | token-exact |
| 9B | 4/4 (13/13 on prompt 3) | 9.1 | token-exact |

The same verifies on WSL2 read 28.1, 13.0 and 5.6 tokens/s, so the native build is the faster path on
this CPU. Pack load is the slow part on Windows (22.8 s, 50.7 s and 112.3 s for 2B/4B/9B). A second
native run measured 33.5/14.3/7.3, so these single runs vary by up to a quarter.

At 24,576 tokens, six threads, exact head, one run per model:

| Model | Read | 100 | 2,000 | 8,000 | 16,000 | 24,000 |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 45.1 | 35.9 | 29.3 | 19.4 | 11.9 | 8.2 |
| 4B | 19.2 | 15.2 | 13.4 | 6.8 | 4.3 | 3.1 |
| 9B | 9.9 | 8.7 | 7.7 | 4.9 | 3.5 | 2.6 |

The native 24K rates are 8.2/3.1/2.6 tokens/s; the old WSL2 run generated 4.6/1.6/1.4. These rows are the
CPU cells in `CROSS-MODE.md`; raw logs are in the [benchmark](benchmarks/2026-11-10-i7-8086k-windows-cpu/README.md).

The engine resolves 6 threads, not 12, through `GetLogicalProcessorInformationEx`, so the Windows
default avoids the hyperthread collapse below without `OMP_NUM_THREADS`. `libomp.dll` is loaded from
the clang `bin` directory, and `kindi/kcpu.c` marks its four entry points `__declspec(dllexport)` so
the ctypes loader can bind them. The build command and the Linux/Windows split are pinned by
`tests/test_cpu_build.py`, and the topology parser by `tests/test_cpu_threads.py`.

### CPU server chat (2026-11-10)

The server path is exercised on the same native Windows machine for all three CPU models.
`KINDI_DEVICE=cpu`, `KINDI_CTX=1024`, six threads, exact head, one fresh server per pack on
`127.0.0.1:8199`. Each model answered `POST /v1/chat/completions` non-stream and stream with HTTP 200,
0 errors and 0 MiB of GPU memory; every stream ended in `[DONE]`.

| model | engine ready | non-stream | stream |
|---|---:|---|---|
| 2B | 25.5 s | 19 prompt / 9 completion, 32.9 tok/s | 17 / 7, 32.7 tok/s, stop |
| 4B | 55.8 s | 15 / 9, 19.2 tok/s | 17 / 1, stop |
| 9B | 123.7 s | 15 / 9, 11.5 tok/s | 15 / 9, 10.7 tok/s, stop |

The 2B row is from the earlier run; the 4B and 9B rows are the 2026-11-10 extension. The 9B was
also probed with `max_tokens=12`; that stream ended with `finish_reason=length` and `[DONE]`, so a
capped stream terminates cleanly. These are single chat requests, not throughput benchmarks.

### Why 12 threads collapse

A 2B probe at a 512-token context, one run per row, read rate and generation at position 100:

| setting | read | generation |
|---|---:|---:|
| 1 thread | 11.2 | 8.3 |
| 2 threads | 18.0 | 12.5 |
| 4 threads | 29.2 | 22.1 |
| 6 threads, `close`/`cores` | 39.2 | 28.9 |
| 8 threads, `close`/`cores` | 47.9 | 37.3 |
| 12 threads, `close`/`cores` | **1.3** | **1.3** |
| 12 threads, `close`/`cores`, `OMP_WAIT_POLICY=passive` | 12.6 | 11.5 |
| 12 threads, `close`/`cores`, `GOMP_SPINCOUNT=0` | 12.5 | 11.6 |
| 12 threads, default binding | 1.4 | 1.5 |
| 12 threads, `OMP_PROC_BIND=spread` | 1.3 | 1.4 |
| 6 threads, default binding | 38.6 | 28.8 |

The 6- and 8-thread rows in this first probe are one-run values; the follow-up medians below
supersede them and add 10 threads.

Binding is not the variable: 12 threads collapse under `close`, `spread` and no binding at all.
Turning the active spin-wait off is: passive waiting or `GOMP_SPINCOUNT=0` recovers 10x. The cause
is libgomp spinning both hyperthread siblings of a core. `kcpu.c` runs a token as one parallel
region with a barrier between row blocks, so a spinning sibling takes the pipeline from the partner
it is waiting for at every barrier, and the convoy grows with each one. The passive recovery still
only reaches 12.5 tokens/s against 39.2 for 6 threads: waking 12 threads at every barrier is
expensive on its own, and the two siblings of a core share its execution units. For the default,
leaving the siblings out is the stable fix, so `_configure_threads()` in `kindi/cpu.py` resolves the
default to the physical-core count and pins it with `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.

### Thread-count sweep: 6, 8, 10 and 12 (2026-11-10)

The 8-thread row in the table above is not eight physical cores. With `OMP_PLACES=cores`, each place
is a physical core (both SMT siblings), so 6 threads take six places. Eight threads put the two extra
threads on two of those places (two SMT pairs); 10 threads use four pairs; 12 threads double all six.
A thread-affinity probe on both WSL2 and native Windows confirmed this placement.

A follow-up measured six in-process read blocks (20 tokens each) and six generation blocks (10 tokens
each) per process; the medians below use the same 512-token protocol as `kindi bench`. WSL2 (GCC
13.3, libgomp), 2B, read and generation at position 100:

| threads | read | generation |
|---|---:|---:|
| 6 | 38.2 | 28.6 |
| 8 | 44.3 | 31.2 |
| 10 | 25.5 | 24.1 |
| 12 | 1.7 | 1.6 |

At 24,576 tokens on WSL2 (one run each):

| threads | read | 100 | 2,000 | 8,000 | 16,000 | 24,000 |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 38.3 | 28.1 | 22.4 | 11.7 | 7.0 | 4.5 |
| 8 | 40.0 | 31.9 | 26.8 | 15.3 | 7.6 | 6.2 |

At 512 tokens, one run each:

| model | 6 threads (read / gen) | 8 threads (read / gen) |
|---|---:|---:|
| 4B | 14.8 / 12.7 | 10.2 / 13.1 |
| 9B | 7.3 / 6.4 | 7.5 / 7.3 |

Native Windows (clang 18.1.8, libomp), 2B, 512 tokens; median of the available process runs (6/8:
three, 10: two):

| threads | read | generation |
|---|---:|---:|
| 6 | 43.2 | 31.7 |
| 8 | 28.4 | 21.6 |
| 10 | 13.5 | 9.8 |

Stage timers from thread 0 (`kc_times`), medians of six blocks; seconds per 20 read tokens / 10
generation tokens. The native rows use a debug DLL with a read-only accessor; production code is
unchanged.

| runtime | threads | read total | read GEMV | gen total | gen GEMV |
|---|---:|---:|---:|---:|---:|
| WSL2 | 6 | 0.531 | 0.480 | 0.356 | 0.323 |
| WSL2 | 8 | 0.458 | 0.404 | 0.309 | 0.275 |
| WSL2 | 10 | 0.737 | 0.520 | 0.499 | 0.359 |
| native | 6 | 0.465 | 0.406 | 0.312 | 0.278 |
| native | 8 | 0.619 | 0.506 | 0.709 | 0.452 |
| native | 10 | 1.722 | 0.827 | 1.242 | 0.532 |

The stage split answers the "why": GEMV is about 90% of both paths. On WSL2, 8 threads lowers GEMV
by 16-19%; GDN, attention and norm stages barely change or get worse. `rows8` is a latency-heavy
loop (permute, table loads, FMA), and the two SMT workers fill issue and memory gaps. This is not
barrier load balance: the barrier-heavy GDN/attention stages do not improve. On native Windows the
same two SMT pairs make GEMV 25-63% slower and add overhead in the small stages, so 6 wins. The
exact libgomp/libomp and GCC/clang difference is not reduced to a cycle-level model, but the effect
is localized to the GEMV kernel and SMT contention.

At 10 threads four cores are doubled. Both runtimes lose: WSL2 reads 25.5 / generates 24.1 against
44.3 / 31.2 at 8, and native reads 13.5 / generates 9.8 against 43.2 / 31.7 at 6. Twelve doubles all
six and collapses (1.7 tokens/s); passive waiting recovered 12 threads to 12.6 / 11.5 in the earlier
probe, still far below 6. Native passive waiting improved 10 to 20.0 / 14.7, but not enough to beat 6.

**Decision: keep the physical-core default (6 on this CPU).** Do not change it to 8 or 10. Eight
threads is an explicit WSL2/libgomp tuning for the 2B and 9B; 4B read regresses, native Windows 8 is
slower, and the result is compiler/runtime-specific. A startup auto-tune is not warranted now:
read and generation disagree for 4B, the choice is model-dependent, and calibration would add
startup cost and another source of variance; 10/12 must remain bounded. A user with a measured WSL2
workload can set `OMP_NUM_THREADS=8`. Raw probe logs are in `bench-logs/thread-sweep-20261110/`
(untracked).

## Work packages

### P0 — detect the CPU before downloading or compiling

1. Add a CPU feature report (model name, physical/logical cores, AVX, AVX2, FMA, F16C) to `kindi info`
   and to the installer's machine check.
2. In `cpu.py`, check the features before invoking gcc and stop with a clear message:
   “the CPU engine needs AVX; this CPU has …”. A CPU without AVX fails at this check, not with a
   target-mismatch error from inside a container. Implemented in `cpu.feature_report()` /
   `cpu.check_features()` and shown by `kindi info`; `_lib()` runs the check before the first build.
   Feature values are `unknown` on systems that do not expose `/proc/cpuinfo`, so the compile-time
   guard in `kcpu.c` remains the authority there. The installer now checks AVX before it offers a
   CPU setup; `--gpu cpu` stops immediately when the check fails.
3. Use `-march=native` for the host build: it selects the AVX2/FMA/F16C fast path, the AVX2/F16C
   no-FMA path, or the AVX-only fallback. A portable AVX-only artifact is a separate build decision,
   not a runtime requirement.
4. Keep a compile-and-load test for the AVX-only path, the AVX2/F16C no-FMA path and the
   AVX2/FMA/F16C fast path; the feature gate must accept AVX-only and reject a CPU without AVX.

### P1 — the i5-2500K fallback

5. AVX is the baseline. The AVX2 path is selected with `__AVX2__`, F16C with `__F16C__`, and FMA
   with `__FMA__`. An AVX2 build without FMA uses the same table/permute kernel with separate
   multiply/add; an AVX-only build uses an AVX table lookup and software fp16 conversion.
6. Runtime dispatch exists: `cpu.py` selects from explicit `-m` flag tiers (full FMA,
   AVX2/F16C no-FMA, AVX-only) at startup (`ISA_BUILDS`/`select_isa`), so the host compiler
   is no longer free to smuggle in instructions the tier does not allow. A separate
   portable AVX binary is still not produced; native i5-2500K / native no-FMA hardware
   runs are still needed for a support claim.
7. `kindi verify` was run on the 2B with `-mavx -mno-avx2 -mno-fma -mno-f16c`: 4/4 token-exact.
   A forced AVX2/F16C no-FMA build (`-mavx2 -mno-fma -mf16c`) also passes the 2B 4/4
   token-exact at 31.7 tok/s on the i7-8086K. The 4B and 9B also pass the forced AVX-only and
   AVX2/F16C no-FMA builds; native i5-2500K and native no-FMA hardware are still needed for a support claim.

### P2 — tuning the three AVX2 CPUs

8. Measure the i9-13900K with 8, 16, 24 and 32 threads. Decide whether the default should use only
   P-cores or all cores; record the choice in the installer.
9. Measure the Xeon W-2133 with 6 and 12 threads and with AVX2-width versus AVX-512 auto-vectorization.
10. [x] Measure the i7-8086K with 6 and 12 threads. Done 2026-11-09: 6 threads are usable and
    12 threads are not; the verified result and the 24K table are above.
11. [~] Check the 24,576-token context and the 9B memory footprint on each CPU. The i7-8086K
    completed the 6-thread 24K benchmark; 9B used 3.8 GiB during the 1K verify, but the 24K peak was
    not recorded. The other CPUs are still pending.

### P3 — validation and documentation

12. [~] For every cell that is advertised, record the CPU, gcc version, thread count, context, model,
    `kindi verify` result and tokens/s. The i7-8086K row is recorded here, in `docs/BENCHMARKS.md` and
    in `docs/CARDS.md`; its native CPU server chat path is verified for the 2B, 4B and 9B. `README.md` and the other CPUs remain pending.
13. [x] Keep the current CPU-only tests passing and add a CPU build/smoke job that needs no GPU and
    no model pack. Done 2026-11-10: `.github/workflows/ci.yml` adds `cpu-engine-smoke`, which
    compiles `kcpu.c` with gcc for AVX2/FMA/F16C, loads it through `kindi.cpu._lib()` and runs
    `tests/test_cpu_build.py`, `tests/test_cpu_threads.py` and `tests/test_cpu_smoke.py`.

## Acceptance matrix

| CPU | 2B | 4B | 9B | 24K context | Required before a support claim |
|---|---:|---:|---:|---:|---|
| i5-2500K | 2B forced-AVX pass | forced tiers 4/4 | forced tiers 4/4 | 2B/4B if RAM allows | native hardware, verify, bench |
| Xeon W-2133 | target | target | target | yes | verify, bench, thread/AVX-512 result |
| i9-13900K | target | target | target | yes | verify, bench, P/E thread default |
| i7-8086K | engine + server verified | engine + server verified | engine + server verified | yes | repeat run |

A checked cell needs a recorded `kindi verify` pass and a benchmark row. “Builds” alone is not support.
`engine verified` for the i7-8086K means the CPU library built on WSL2 and natively on Windows,
`kindi verify` passed and the 6-thread 24K benchmark completed. Its native CPU server chat path is
now verified for the 2B, 4B and 9B; the remaining open item is a repeat timing run. Forced-tier
results are compatibility evidence, not native support for the i5-2500K.

## Commands for the validation machine

If you build a library manually, use `python -c 'from kindi.cpu import _so_path; print(_so_path())'`
to get the keyed filename the production loader expects.

```sh
# AVX2/FMA/F16C fast path
gcc -O3 -mavx2 -mfma -mf16c -fopenmp -fno-math-errno -shared -fPIC \
    -o /tmp/libkcpu.so kindi/kcpu.c -lm

# AVX2/F16C without FMA for a host or VM with FMA masked
gcc -O3 -mavx2 -mno-fma -mf16c -fopenmp -fno-math-errno -shared -fPIC \
    -o /tmp/libkcpu-avx2-nofma.so kindi/kcpu.c -lm

# AVX-only fallback for Sandy Bridge and other AVX CPUs
gcc -O3 -mavx -mno-avx2 -mno-fma -mno-f16c -fopenmp -fno-math-errno -shared -fPIC \
    -o /tmp/libkcpu-avx.so kindi/kcpu.c -lm

# CPU-only checks; no GPU and no model pack are needed for the first command
KINDI_DEVICE=cpu python3 -m kindi info
KINDI_DEVICE=cpu KINDI_CPU_CACHE=/tmp/kindi-cpu python3 -m kindi verify
KINDI_DEVICE=cpu KINDI_CPU_CACHE=/tmp/kindi-cpu python3 -m kindi bench
# forced AVX-only exactness on a real pack (builds the keyed library itself)
KINDI_CPU_AVX_PACKS=/path/to/pack python3 -m pytest tests/test_cpu_avx_fallback.py -q
# forced AVX2/F16C no-FMA exactness on a real pack
KINDI_CPU_NO_FMA_PACKS=/path/to/pack python3 -m pytest tests/test_cpu_avx_fallback.py -q
```

`kindi verify` needs the converted pack and the matching reference file. The installer's `--threads N`
sets `OMP_NUM_THREADS` for the CPU container; the direct commands above use the OpenMP default unless
`OMP_NUM_THREADS` is set. On Windows set `KINDI_CC` to `clang.exe` and `KINDI_LLVM` to the LLVM
install root (the directory holding `lib/libomp.lib`); `cpu.py` appends `-L <root>/lib -llibomp` and
puts `<root>/bin` on the DLL search path. When `-march=native` does not enable FMA, the Windows
build also passes `/nodefaultlib:libcmt /defaultlib:msvcrt`: the static `libcmt` `expf` contains FMA,
which would fault on Sandy Bridge even when the engine itself is built for AVX only.

## Open decisions

- Can the AVX-only fallback be run on native Sandy Bridge hardware to turn the forced-build result
  into a support claim? The forced build is exact on the 2B, 4B and 9B; native Sandy Bridge hardware is still needed.
- Can the AVX2/F16C no-FMA path be run on native hardware or a VM with FMA masked? The forced
  2B/4B/9B runs are token-exact at 31.7/15.8/10.4 tok/s; native no-FMA hardware is still unmeasured.
- Should the supported tier use explicit `-mavx2 -mfma -mf16c` or `-march=native`? The old container
  gcc and the Raptor Lake detection make this a testable choice, not a preference.
- For the i9-13900K, is the default 8 P-core threads, 16 P-core threads, or all 24 cores? Measure
  prompt reading, generation, and long-context behaviour before deciding.
- For the Xeon W-2133, does AVX-512 auto-vectorization help enough to justify the frequency cost, or
  should the CPU build prefer 256-bit vectors?
