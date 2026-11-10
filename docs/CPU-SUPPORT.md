# Long-term CPU support

Status: **roadmap and acceptance contract, updated 2026-11-10. The i7-8086K engine and native CPU
server chat path are verified; the other CPUs and a repeat timing run remain open.** The baseline CPU engine (`kindi/cpu.py`,
`kindi/kcpu.c`) is in `main`; the WIP ISA/AVX-only changes are uncompiled and not merged. This page
records the target, the verified result, and the exact build/runtime requirements so the work can be
implemented and measured without re-discovering them. The tandem recovery gate is in
[TANDEM-STATUS.md](TANDEM-STATUS.md).

## What “supported” means here

A CPU is supported when all of these hold:

1. `libkcpu.so` builds on that CPU with the documented gcc and OpenMP toolchain.
2. `KINDI_DEVICE=cpu` runs the advertised models and the server answers a chat request.
3. `kindi verify` passes for the advertised models (token-exact with llama.cpp for Bonsai).
4. The CPU is listed in `docs/CARDS.md` with its thread count and measured result.
5. It works without an illegal-instruction crash, and without an undocumented manual override.

The current CPU engine covers the **2B, 4B and 9B**. The 27B, Nemotron and Nanbeige are GPU targets;
they are not part of this CPU goal.

## Current backend reality

| gate | current behaviour | consequence |
|---|---|---|
| `kindi/cpu.py` build | Linux: `gcc -O3 -march=native -fopenmp -fno-math-errno -shared -fPIC`; Windows: `clang -O3 -march=native -fopenmp -fno-math-errno -ffp-contract=off -shared` plus `libomp` (`KINDI_CC`, `KINDI_LLVM`) | The binary is built for the host CPU. There is no prebuilt portable CPU library. |
| `kindi/kcpu.c` ISA | Uses AVX2 (`_mm256_srlv_epi32`, `_mm256_permutevar8x32_ps`), FMA (`_mm256_fmadd_ps`) and F16C (`_cvtsh_ss` / `_cvtss_sh`) unconditionally; no `#ifdef` or runtime dispatch | A CPU without all three cannot build or run the current library. |
| documented requirement | `docs/CARDS.md` and the README say “x86-64 with AVX2” | This is accurate but incomplete: FMA and F16C are also required by the source. |
| `install.sh` preflight | Checks Linux and `x86_64`; does not check AVX2/FMA/F16C | An unsupported CPU fails later inside the container build, with a compiler error rather than an actionable message. |
| CPU engine thread default | `kindi/cpu.py` counts the physical cores this process may run on (sysfs `thread_siblings_list`, `/proc/cpuinfo` fallback) and sets `OMP_NUM_THREADS`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` before the library loads; `threads=` or an `OMP_NUM_THREADS` already in the environment wins | On a hybrid i9 this counts 24 cores, E-cores included. The best P/E split has not been measured. |
| `install.sh` CPU default | Counts the physical cores of socket 0 (`lscpu -p=CORE,SOCKET`) and exports the same three variables for the CPU container; `--threads N` overrides it | Two sockets are not measured; the engine's own default counts every CPU it may use, not socket 0. |
| build container | `docker/Dockerfile` is Ubuntu 18.04 with the distribution gcc and `libgomp1` | The old gcc may not know Raptor Lake as a `-march=native` target. Explicit AVX2/FMA/F16C flags are safer for the supported tier. |
| test coverage | `tests/` has server/config/cache tests plus `test_cpu_build.py`, `test_cpu_threads.py` and the gated `test_cpu_smoke.py`; the CI `cpu-engine-smoke` job compiles `kcpu.c` with gcc and loads the library | The compile/load path is covered; a full token cannot run without a model pack. |

## Target matrix

| CPU | Microarchitecture | Cores / threads | Relevant ISA | Current state | Target |
|---|---|---|---|---|---|
| Intel Core i5-2500K | Sandy Bridge | 4 / 4 | AVX; **no AVX2, FMA or F16C** | **Blocked**: the current source cannot compile for this CPU | 2B and 4B after an AVX/scalar fallback; 9B only if measured and useful; 27B no |
| Intel Xeon W-2133 | Skylake-W | 6 / 12 | AVX-512, AVX2, FMA, F16C | Expected to build and run, but **unverified** | 2B, 4B and 9B |
| Intel Core i9-13900K | Raptor Lake (8 P + 16 E) | 24 / 32 | AVX2, FMA, F16C; no AVX-512 | Expected to build and run, but **unverified** | 2B, 4B and 9B; tune P/E thread count |
| Intel Core i7-8086K | Coffee Lake | 6 / 12 | AVX2, FMA, F16C | **Engine and server verified**: native Windows (2B/4B/9B 4/4 at 33.7/16.7/9.1 tok/s, 6 threads auto-detected; 24K generation 8.2/3.1/2.6 tok/s, read 45.1/19.2/9.9), WSL2 history, and a native CPU server chat request on the 2B (32.9 tok/s, 1,024 context, 0 errors) | 2B, 4B and 9B; use 6 threads |

Notes:

- The i5-2500K is the only CPU here without AVX2/FMA/F16C. It cannot be “supported” by changing a
  flag: the kernel needs a real fallback or a second compiled object. Treat it as P1 work with a
  measured 2B/4B acceptance, not as a documentation change.
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
driver finds through `vswhere`; no `vcvars` shell is needed. `kindi` builds `libkcpu.dll` on first use
into `KINDI_CPU_CACHE`.

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

The last open item of the support contract is exercised on the same native Windows machine.
`KINDI_DEVICE=cpu`, `KINDI_PACK=.../bonsai-2b`, `KINDI_CTX=1024`, six threads: the server built and
loaded the 2B in 25.5 s and served on `127.0.0.1:8199`. `POST /v1/chat/completions` returned HTTP 200
with "Hello! How can I help you today?" (19 prompt, 9 completion tokens, 32.9 tokens/s generation); a
streaming request returned SSE chunks ending in `[DONE]` (17 prompt, 7 completion, 32.7 tokens/s).
`/health` reported 2 requests, 0 errors and 0 MiB of GPU memory, and the log identified the engine as
`kindi.cpu.Engine` with 6 threads. This is the 2B chat path; the 4B and 9B engine verifies are above,
but no separate chat request was sent for them.

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

1. Add a CPU feature report (model name, physical/logical cores, AVX2, FMA, F16C) to `kindi info` and
   to the installer's machine check.
2. In `cpu.py`, check the features before invoking gcc and stop with a clear message:
   “the CPU engine needs AVX2, FMA and F16C; this CPU has …”. The i5-2500K must fail at this check,
   not with a target-mismatch error from inside a container.
3. Prefer explicit build flags for the supported tier (`-mavx2 -mfma -mf16c`) over an unchecked
   `-march=native`, or validate that the image's gcc maps each host correctly. Keep `-march=native`
   only if a test on all four CPUs proves it works.
4. Add a compile-only test to CI for the supported flags, and a fixture that checks the feature gate
   rejects an AVX-only CPU before compilation.

### P1 — the i5-2500K fallback

5. Guard the AVX2/FMA/F16C kernels with `__AVX2__`, `__FMA__`, `__F16C__` and provide a scalar or
   AVX-only implementation of the same arithmetic:
   - `rows8` table lookup without `vpermps` (a scalar gather or an AVX shuffle sequence);
   - FMA as separate multiply and add, with the same rounding policy;
   - software fp16 conversion in place of the F16C intrinsics.
6. Dispatch at runtime or build a baseline object plus an AVX2 object; never execute AVX2 instructions
   on the 2500K.
7. Run `kindi verify` on the 2B and 4B. If the fallback is not token-exact, fix the arithmetic policy
   before advertising it. Measure whether 9B is usable on four threads; it may be a functional but
   impractical target.

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
    in `docs/CARDS.md`; its native CPU server chat path is verified. `README.md` and the other CPUs remain pending.
13. [x] Keep the current CPU-only tests passing and add a CPU build/smoke job that needs no GPU and
    no model pack. Done 2026-11-10: `.github/workflows/ci.yml` adds `cpu-engine-smoke`, which
    compiles `kcpu.c` with gcc for AVX2/FMA/F16C, loads it through `kindi.cpu._lib()` and runs
    `tests/test_cpu_build.py`, `tests/test_cpu_threads.py` and `tests/test_cpu_smoke.py`.

## Acceptance matrix

| CPU | 2B | 4B | 9B | 24K context | Required before a support claim |
|---|---:|---:|---:|---:|---|
| i5-2500K | fallback target | fallback target | measure only | 2B/4B if RAM allows | fallback build, verify, bench |
| Xeon W-2133 | target | target | target | yes | verify, bench, thread/AVX-512 result |
| i9-13900K | target | target | target | yes | verify, bench, P/E thread default |
| i7-8086K | engine + server verified | engine verified | engine verified | yes | repeat run |

A checked cell needs a recorded `kindi verify` pass and a benchmark row. “Builds” alone is not support.
`engine verified` for the i7-8086K means the CPU library built on WSL2 and natively on Windows,
`kindi verify` passed and the 6-thread 24K benchmark completed. Its native CPU server chat path is
now verified for the 2B; the remaining open item is a repeat timing run.

## Commands for the validation machine

```sh
# build the CPU library with the feature set the supported tier requires
gcc -O3 -mavx2 -mfma -mf16c -fopenmp -fno-math-errno -shared -fPIC \
    -o /tmp/libkcpu.so kindi/kcpu.c -lm

# CPU-only checks; no GPU and no model pack are needed for the first command
KINDI_DEVICE=cpu python3 -m kindi info
KINDI_DEVICE=cpu KINDI_CPU_CACHE=/tmp/kindi-cpu python3 -m kindi verify
KINDI_DEVICE=cpu KINDI_CPU_CACHE=/tmp/kindi-cpu python3 -m kindi bench
```

`kindi verify` needs the converted pack and the matching reference file. The installer's `--threads N`
sets `OMP_NUM_THREADS` for the CPU container; the direct commands above use the OpenMP default unless
`OMP_NUM_THREADS` is set. On Windows set `KINDI_CC` to `clang.exe` and `KINDI_LLVM` to the LLVM
install root (the directory holding `lib/libomp.lib`); `cpu.py` appends `-L <root>/lib -llibomp` and
puts `<root>/bin` on the DLL search path.

## Open decisions

- Is the i5-2500K in scope as a 2B/4B CPU, or is a clear “not supported” error the correct long-term
  outcome? A fallback is real engine work and must not be hidden behind an AVX2 flag.
- Should the supported tier use explicit `-mavx2 -mfma -mf16c` or `-march=native`? The old container
  gcc and the Raptor Lake detection make this a testable choice, not a preference.
- For the i9-13900K, is the default 8 P-core threads, 16 P-core threads, or all 24 cores? Measure
  prompt reading, generation, and long-context behaviour before deciding.
- For the Xeon W-2133, does AVX-512 auto-vectorization help enough to justify the frequency cost, or
  should the CPU build prefer 256-bit vectors?
