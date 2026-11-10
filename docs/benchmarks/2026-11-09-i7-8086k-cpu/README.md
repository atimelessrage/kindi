# Kindi CPU engine on the Intel Core i7-8086K

Measured 2026-11-09. This is the first native run of the `kindi/cpu.py` + `kindi/kcpu.c`
engine on this machine. The engine was built and executed inside WSL2 Ubuntu 24.04. The
native Windows build followed on 2026-11-10: see the
[newer benchmark](../2026-11-10-i7-8086k-windows-cpu/README.md).

The short version: **the CPU engine works on the i7-8086K**. All three converted packs
(2B, 4B, 9B) pass `kindi verify` token-exact with llama.cpp, the full CPU-only test suite
passes, and the 6-thread CPU benchmark completes at a 24,576-token context. **Use 6
threads (one per physical core), not 12**: the 12-thread setting is 6-20x slower on this
kernel and is not a usable configuration.

## Headline

Single run per configuration; exact head; one full 24,576-token context allocation.
"Read" is the bench's prompt-reading probe. Generation is at the stated position.
All rates are tokens/s.

### 6 threads — one per physical core (recommended)

| Model | Read | 100 | 2,000 | 8,000 | 16,000 | 24,000 |
|---|---:|---:|---:|---:|---:|---:|
| Ternary-Bonsai-2-2B | 35.2 | 28.2 | 21.3 | 12.2 | 6.7 | 4.6 |
| Ternary-Bonsai-2-4B | 14.2 | 11.8 | 9.1 | 3.9 | 2.3 | 1.6 |
| Ternary-Bonsai-2-9B | 7.6 | 6.8 | 5.8 | 3.2 | 2.0 | 1.4 |

`OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores`.

The 2B and 4B rates are in the same range as the documented i7-8700 reference point
(2B 42.50 read / 28.92 write, 4B 17.02 / 12.61); this run is a new machine measurement,
not a speedup claim. The 24K cost is the current per-token CPU attention path: 2B falls
6.1x from position 100 to 24,000, 4B 7.4x, and 9B 4.9x.

### 12 threads — all logical CPUs, not viable

| Model | Read | 100 | 2,000 | 8,000 | 16,000 | 24,000 |
|---|---:|---:|---:|---:|---:|---:|
| Ternary-Bonsai-2-2B | 1.8 | 1.8 | 1.7 | 1.7 | 1.5 | 1.4 |
| Ternary-Bonsai-2-4B | 1.2 | 1.3 | 1.2 | 1.0 | 0.9 | 0.8 |
| Ternary-Bonsai-2-9B | 1.1 | 1.1 | 1.1 | 1.0 | 0.9 | 0.7 |

`OMP_NUM_THREADS=12` with no explicit `OMP_PROC_BIND`/`OMP_PLACES`. The 6-thread
configuration is 19.6x faster at 2B read, 11.8x at 4B read and 6.9x at 9B read. Binding is
not the cause: `OMP_PROC_BIND=close OMP_PLACES=cores` (1.8 tok/s read) and
`OMP_PROC_BIND=spread` (1.3) collapse the same way. A follow-up probe on the 2B at a
512-token context found the cause in libgomp's active spin-wait — `OMP_WAIT_POLICY=passive`
and `GOMP_SPINCOUNT=0` recover 12 threads to 12.6 and 12.5 tok/s — because `kcpu.c` runs a
token as one parallel region with a barrier between row blocks, and two spinning siblings
on one core starve the partner they are waiting for. The passive recovery is still 3x
below the 6-thread 39.2 tok/s, since the pairs share one core's execution units.

**Recommendation: set `OMP_NUM_THREADS` to the physical-core count (6 here).**
`kindi/cpu.py` now does this by default, with `OMP_PROC_BIND=close` and `OMP_PLACES=cores`;
an explicit `OMP_NUM_THREADS` or `threads=` still wins. The full thread sweep is in
[CPU-SUPPORT.md](../../CPU-SUPPORT.md#why-12-threads-collapse).

## Verification

`kindi verify` checks greedy output token-for-token against llama.cpp's reference. All
four prompts pass for every model:

| Model | Prompts | Tokens/s during verify | Result |
|---|---:|---:|---|
| 2B | 4/4 | 28.1 | **ALL PASS: token-exact** |
| 4B | 4/4 | 13.0 | **ALL PASS: token-exact** |
| 9B | 4/4 | 5.6 | **ALL PASS: token-exact** |

Verify uses a 1,024-token context. The 9B was also the load test for the 15 GiB WSL
memory limit: `kindi-cpu ready` in 102.7 s, no OOM, 3.8 GiB used during the run.

## Full CPU test suite

```sh
KINDI_DEVICE=cpu python -m pytest tests/ -q
```

**127 passed, 1 warning in 5.98 s** (Windows Python 3.13.6). The warning is CuPy's
"CUDA path could not be detected", expected on this CPU-only run. The suite covers the
Python/config/cache/server/plan-engine/installer surfaces; it does not build `kcpu.c`.
The native runs above are what exercise the C engine.

## Rig and toolchain

| Item | Value |
|---|---|
| CPU | Intel Core i7-8086K @ 4.00 GHz |
| Cores / threads | 6 physical / 12 logical |
| Memory | 15 GiB visible to WSL2 (32 GB installed) |
| Host OS | Microsoft Windows 10 Pro 10.0.19045 |
| CPU runtime | WSL2 Ubuntu 24.04, kernel 6.18.54.1-microsoft-standard-WSL2 |
| Compiler | gcc 13.3.0, OpenMP enabled |
| Python (native engine) | 3.12.3 |
| NumPy (native engine) | 2.5.4, installed with `pip --target` in the user home; no system packages changed |
| CPU flags | AVX2, FMA, F16C enabled; `gcc -march=native` confirms all three |
| Kindi HEAD | `0ce4350` on `freebuff/tandem-cpu-recovery` |
| GPU | not used (`KINDI_DEVICE=cpu`) |

The CPU library is compiled at first use with:

```sh
gcc -O3 -march=native -fopenmp -fno-math-errno -shared -fPIC \
    -o ~/.cache/kindi-cpu-i7/libkcpu.so kindi/kcpu.c -lm
```

It is cached outside the repository. Pack sizes on disk: 2B 1.3 GB, 4B 1.1 GB, 9B
5.5 GB. The 9B load path spends about 103 s repacking the 9B weights, so cold-start
latency is model-size dominated.

## Method

Command per benchmark case:

```sh
KINDI_DEVICE=cpu KINDI_CPU_CACHE=<home>/.cache/kindi-cpu-i7 \
  OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
  python3 -m kindi --pack <home>/.kindi/packs/bonsai-2b \
    --ctx 24576 bench
```

`<home>` is a placeholder for the benchmark machine's home directory.

The same command was used with `bonsai-4b` and `bonsai-9b`. For the 12-thread rows,
`OMP_NUM_THREADS=12` and no explicit `OMP_PROC_BIND`/`OMP_PLACES` were used.

`bench` warms each timed operation, then times the prompt-reading probe, then ten
generation steps at positions 100, 2,000, 8,000, 16,000 and 24,000. It uses one fresh
process per configuration and does not compare model outputs. This is a speed
measurement; the separate `verify` runs above are the correctness check.

One run per model/thread configuration, not the three repeats used by the GPU benchmark.
There are no error bars, and the machine was not thermally or power clamped.

## Files

| File | Contents |
|---|---|
| [runs.csv](runs.csv) | six rows: model, threads, binding, read rate and all five generation positions |
| [summary.json](summary.json) | machine-readable rig, method, verification, per-run data and limitations |
| [raw/pytest.log](raw/pytest.log) | full CPU test suite: 127 passed |
| [raw/verify-2b.log](raw/verify-2b.log), [raw/verify-4b.log](raw/verify-4b.log), [raw/verify-9b.log](raw/verify-9b.log) | token-exact verification logs |
| [raw/bench-2b-6t.log](raw/bench-2b-6t.log), [raw/bench-4b-6t.log](raw/bench-4b-6t.log), [raw/bench-9b-6t.log](raw/bench-9b-6t.log) | 6-thread benchmark logs |
| [raw/bench-2b-12t-default.log](raw/bench-2b-12t-default.log), [raw/bench-4b-12t-default.log](raw/bench-4b-12t-default.log), [raw/bench-9b-12t-default.log](raw/bench-9b-12t-default.log) | 12-thread benchmark logs |
| [raw/bench-2b-12t-pinned.log](raw/bench-2b-12t-pinned.log) | 2B 12-thread `close/cores` diagnostic: 1.8 tok/s read |
| [raw/bench-4b-12t-pinned-incomplete.log](raw/bench-4b-12t-pinned-incomplete.log) | interrupted pinned-12-thread diagnostic; not used in the tables |
| [raw/system-info.log](raw/system-info.log) | CPU, memory, gcc, Python, NumPy, target flags, packs, Windows git state |
| [raw/numpy-install.log](raw/numpy-install.log) | WSL NumPy 2.5.4 user-target install |
| [raw/smoke-2b.log](raw/smoke-2b.log) | first 2B native engine smoke test at 1,024 tokens |
| [build_tables.py](build_tables.py) | regenerates `runs.csv` and `summary.json` from the logs |
| [raw/run-remaining.sh](raw/run-remaining.sh) | WSL-side script used for the remaining verification/benchmark runs |
| [raw/bench-remaining-master.log](raw/bench-remaining-master.log) | run timestamps and completion markers for the background batch |

The raw logs are the source of truth; `build_tables.py` parses the standard `kindi bench`
and `kindi verify` lines. `run-remaining.sh` is retained as the exact WSL command record.

## What this proves and what it does not

Proved on this machine:

- `kcpu.c` builds with GCC 13.3 on the i7-8086K's AVX2/FMA/F16C feature set.
- The 2B, 4B and 9B run in the CPU engine and are token-exact with llama.cpp.
- The full CPU-only pytest suite passes (127 tests).
- The 6-thread benchmark completes at 24,576 tokens for all three models.

Not proved or not done:

- **Windows-native CPU engine (since done).** No compiler was available on the Windows side at the
  time, so WSL2 was used. The native build followed on 2026-11-10 with clang 18.1.8 and libomp:
  [2026-11-10-i7-8086k-windows-cpu](../2026-11-10-i7-8086k-windows-cpu/README.md).
- **12-thread usability.** It is measured and is much slower; it is not a recommended
  configuration. The 2026-11-10 thread sweep and `kc_times` stage timers in
  [CPU-SUPPORT.md](../../CPU-SUPPORT.md) explain the 8/10 results and keep 6 as the default.
- **Repeated-run stability and thermals.** One run per configuration; no clock, power or
  temperature logging; no median/variance.
- **Multi-user serving or 27B CPU.** The 2B CPU server chat path was tested later (native Windows,
  see the 2026-11-10 page); this benchmark covers the 2B/4B/9B bench and verify paths only.
- **A general support claim.** The engine result is recorded in
  [../../CPU-SUPPORT.md](../../CPU-SUPPORT.md) and the benchmark index, and the native CPU server
  chat path is verified separately. Repeated-run stability remains open.

## Reproduce

From a WSL2 Ubuntu 24.04 shell with gcc/OpenMP and NumPy available:

```sh
cd <home>/.freebuff/worktrees/f0e51299/kindi

KINDI_DEVICE=cpu KINDI_CPU_CACHE=<home>/.cache/kindi-cpu-i7 \
  OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
  python3 -m kindi --pack <home>/.kindi/packs/bonsai-2b verify

KINDI_DEVICE=cpu KINDI_CPU_CACHE=<home>/.cache/kindi-cpu-i7 \
  OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
  python3 -m kindi --pack <home>/.kindi/packs/bonsai-2b --ctx 24576 bench
```

Regenerate the tables:

```sh
python docs/benchmarks/2026-11-09-i7-8086k-cpu/build_tables.py
```
