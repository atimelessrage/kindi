# Long-term CPU support

Status: **roadmap and acceptance contract, 2026-11-09. Not a support claim yet.** The CPU engine files
(`kindi/cpu.py`, `kindi/kcpu.c`) and `install.sh` are dirty in the main worktree. This page records the
target and the exact build/runtime requirements so the work can be implemented and measured without
re-discovering them.

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
| `kindi/cpu.py` build | `gcc -O3 -march=native -fopenmp -fno-math-errno -shared -fPIC` | The binary is built for the host CPU. There is no prebuilt portable CPU library. |
| `kindi/kcpu.c` ISA | Uses AVX2 (`_mm256_srlv_epi32`, `_mm256_permutevar8x32_ps`), FMA (`_mm256_fmadd_ps`) and F16C (`_cvtsh_ss` / `_cvtss_sh`) unconditionally; no `#ifdef` or runtime dispatch | A CPU without all three cannot build or run the current library. |
| documented requirement | `docs/CARDS.md` and the README say “x86-64 with AVX2” | This is accurate but incomplete: FMA and F16C are also required by the source. |
| `install.sh` preflight | Checks Linux and `x86_64`; does not check AVX2/FMA/F16C | An unsupported CPU fails later inside the container build, with a compiler error rather than an actionable message. |
| CPU thread default | Counts the physical cores of socket 0 (`lscpu -p=CORE,SOCKET`), then sets `OMP_NUM_THREADS`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`; `--threads N` overrides it | On a hybrid i9 this counts 24 cores, including E-cores. The best P/E split has not been measured. |
| build container | `docker/Dockerfile` is Ubuntu 18.04 with the distribution gcc and `libgomp1` | The old gcc may not know Raptor Lake as a `-march=native` target. Explicit AVX2/FMA/F16C flags are safer for the supported tier. |
| test coverage | `tests/` has server/config/cache tests but no CPU-engine build or runtime test | A regression in the CPU ISA path is not caught by the current CPU-only suite. |

## Target matrix

| CPU | Microarchitecture | Cores / threads | Relevant ISA | Current state | Target |
|---|---|---|---|---|---|
| Intel Core i5-2500K | Sandy Bridge | 4 / 4 | AVX; **no AVX2, FMA or F16C** | **Blocked**: the current source cannot compile for this CPU | 2B and 4B after an AVX/scalar fallback; 9B only if measured and useful; 27B no |
| Intel Xeon W-2133 | Skylake-W | 6 / 12 | AVX-512, AVX2, FMA, F16C | Expected to build and run, but **unverified** | 2B, 4B and 9B |
| Intel Core i9-13900K | Raptor Lake (8 P + 16 E) | 24 / 32 | AVX2, FMA, F16C; no AVX-512 | Expected to build and run, but **unverified** | 2B, 4B and 9B; tune P/E thread count |
| Intel Core i7-8086K | Coffee Lake | 6 / 12 | AVX2, FMA, F16C | Expected to build and run, but **unverified** | 2B, 4B and 9B |

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
- The i7-8086K is close to the measured 6-core i7-8700 in the benchmark table: 2B 42.50 read /
  28.92 write tokens/s, 4B 17.02 / 12.61. That is a reference point, not a promise for the 8086K.
- These are single-socket desktop/workstation CPUs. Multi-socket CPU inference is out of scope.

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
10. Measure the i7-8086K with 6 and 12 threads. Compare it with the i7-8700 benchmark before setting
    expectations.
11. Check the 24,576-token context and the 9B memory footprint on each CPU; lower the CPU profile's
    default context if the measured free memory requires it.

### P3 — validation and documentation

12. For every cell that is advertised, record the CPU, gcc version, thread count, context, model,
    `kindi verify` result and tokens/s. Update `docs/CARDS.md`, `README.md` and `docs/BENCHMARKS.md`
    when the main worktree hands those files off.
13. Keep the current CPU-only tests passing and add a CPU build/smoke job that needs no GPU and no
    model pack.

## Acceptance matrix

| CPU | 2B | 4B | 9B | 24K context | Required before a support claim |
|---|---:|---:|---:|---:|---|
| i5-2500K | fallback target | fallback target | measure only | 2B/4B if RAM allows | fallback build, verify, bench |
| Xeon W-2133 | target | target | target | yes | verify, bench, thread/AVX-512 result |
| i9-13900K | target | target | target | yes | verify, bench, P/E thread default |
| i7-8086K | target | target | target | yes | verify, bench, thread result |

A checked cell needs a recorded `kindi verify` pass and a benchmark row. “Builds” alone is not support.

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
`OMP_NUM_THREADS` is set.

## Open decisions

- Is the i5-2500K in scope as a 2B/4B CPU, or is a clear “not supported” error the correct long-term
  outcome? A fallback is real engine work and must not be hidden behind an AVX2 flag.
- Should the supported tier use explicit `-mavx2 -mfma -mf16c` or `-march=native`? The old container
  gcc and the Raptor Lake detection make this a testable choice, not a preference.
- For the i9-13900K, is the default 8 P-core threads, 16 P-core threads, or all 24 cores? Measure
  prompt reading, generation, and long-context behaviour before deciding.
- For the Xeon W-2133, does AVX-512 auto-vectorization help enough to justify the frequency cost, or
  should the CPU build prefer 256-bit vectors?
