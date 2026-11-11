# Kindi CPU engine on native Windows: Intel Core i7-8086K

Measured 2026-11-10 on the Windows host, with no WSL and no Docker. This is the first native
Windows run of `kindi/cpu.py` + `kindi/kcpu.c`; the earlier i7-8086K result
([2026-11-09-i7-8086k-cpu](../2026-11-09-i7-8086k-cpu/README.md)) ran inside WSL2.

The short version: **`libkcpu.dll` builds with clang 18.1.8 and libomp, the engine resolves six
threads on its own, and the 2B, 4B and 9B are token-exact with llama.cpp.** The native verify is
faster than the WSL2 one on every model, and the first native 24,576-token sample supplies the
CPU-only cells in the cross-mode matrix. Two further idle-host samples are recorded below: run 2
is 9-31% above run 1, while run 3 is between 19% below and 6% above it. The CPU numbers should be
read as a range rather than a fixed rate.

## Rig and toolchain

| Item | Value |
|---|---|
| CPU | Intel Core i7-8086K @ 4.00 GHz |
| Cores / threads | 6 physical / 12 logical |
| OS | Windows 10 Pro 10.0.19045 |
| Compiler | clang 18.1.8, `clang+llvm-18.1.8-x86_64-pc-windows-msvc` |
| Headers / linker | MSVC 14.51.36231 (Visual Studio 18 Community), Windows SDK — clang's driver finds both through `vswhere`; no `vcvars` shell is used |
| OpenMP | libomp from the same LLVM install (`lib/libomp.lib`, `bin/libomp.dll`) |
| Python | 3.13.6 |
| Kindi HEAD | `fd3fba5` for the verify/server runs; `4934974` for 24K run 1; `22ac35e` for 24K runs 2 and 3 (`kindi/kcpu.c` unchanged for run 2; run 3 used the current worktree's uncommitted per-layer CPU entry points) |
| GPU | not used (`KINDI_DEVICE=cpu`) |

`kindi/cpu.py` compiles the engine on first use with:

```
clang -O3 -march=native -fopenmp -fno-math-errno -ffp-contract=off -shared
      -o %USERPROFILE%\.cache\kindi\libkcpu.dll kindi/kcpu.c
      -L <llvm>/lib -llibomp
```

`KINDI_CC` points at `clang.exe` and `KINDI_LLVM` at the LLVM install root; without them the
build looks for `clang` on `PATH`. The produced DLL is 142,848 bytes. The engine adds `<llvm>/bin`
to the DLL search path so `libomp.dll` loads, and `kindi/kcpu.c` marks `kc_init`, `kc_step`,
`kc_sizeof_model` and `kc_sizeof_layer` with `__declspec(dllexport)` so ctypes can bind them.

## Threads

The engine prints `6 threads` with no `OMP_NUM_THREADS`: on Windows it counts physical cores with
`GetLogicalProcessorInformationEx(RelationProcessorCore)` instead of the Linux sysfs path. That
matters — twelve threads is the collapse documented in
[CPU-SUPPORT.md](../../CPU-SUPPORT.md#why-12-threads-collapse).

## Verification

`kindi verify` compares greedy output token-for-token with llama.cpp's reference, at a
1,024-token context, one run per model:

| Model | Prompts | Tokens/s | Engine ready | Result |
|---|---:|---:|---:|---|
| Ternary-Bonsai-2-2B | 4/4 | 33.7 | 22.8 s | **ALL PASS: token-exact** |
| Ternary-Bonsai-2-4B | 4/4 | 16.7 | 50.7 s | **ALL PASS: token-exact** |
| Ternary-Bonsai-2-9B | 4/4 (13/13 on prompt 3) | 9.1 | 112.3 s | **ALL PASS: token-exact** |

These are single runs. An earlier native run of the same commands measured 33.5, 14.3 and 7.3
tokens/s with 26.0, 56.0 and 128.7 s of load time: the verify rate and the pack-load time move by
up to a quarter between runs, so read the table as one sample, not a tight measurement. The 9B
prompt 3 stops at 13 tokens, exactly as in the WSL2 reference run.

For comparison, the WSL2 verify read 28.1, 13.0 and 5.6 tokens/s on the same machine
([2026-11-09 benchmark](../2026-11-09-i7-8086k-cpu/README.md)). The native build is faster on all
three models; the 24,576-token benchmark is below.

## 24,576-token benchmark

`kindi bench` with the full 24,576-token context, exact head, six threads, one fresh process per
model and per run. "Read" is the 1,000-token prompt-reading probe; the other columns are
generation at that position. Tokens/s.

### Run 1: 12:50 (cross-mode source)

| Model | Read | 100 | 2,000 | 8,000 | 16,000 | 24,000 |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 45.1 | 35.9 | 29.3 | 19.4 | 11.9 | 8.2 |
| 4B | 19.2 | 15.2 | 13.4 | 6.8 | 4.3 | 3.1 |
| 9B | 9.9 | 8.7 | 7.7 | 4.9 | 3.5 | 2.6 |

Run 1's 24K generation rates are 8.2, 3.1 and 2.6 tokens/s; the WSL2 run measured 4.6, 1.6 and
1.4. The native read is also faster: 45.1/19.2/9.9 against 35.2/14.2/7.6. The engine loaded the
2B, 4B and 9B in 22.5, 50.3 and 111.7 s. The CPU-only cells in
[CROSS-MODE.md](../CROSS-MODE.md) use this run; the GPU cells are the earlier median of three
and the two blocks were not interleaved.

### Run 2: 23:14 (fresh rerun)

The same command and packs were run again on the same host, one fresh process per model, with
6 auto-detected threads and no other benchmark process. Load time is the `kindi-cpu ready` time.

| Model | Read | 100 | 2,000 | 8,000 | 16,000 | 24,000 | Load |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2B | 58.2 | 43.9 | 37.2 | 23.2 | 13.6 | 9.5 | 22.4 s |
| 4B | 24.0 | 19.9 | 16.4 | 8.4 | 4.7 | 3.4 | 46.8 s |
| 9B | 12.4 | 11.1 | 10.0 | 6.3 | 4.1 | 3.0 | 105.5 s |

Every cell is 9-31% above run 1; at 24,000 the increases are 15.9%, 9.7% and 15.4%. The rerun is
not a speedup from a code change: `kindi/kcpu.c` is unchanged, and `kindi/cpu.py` differs from
run 1 only in thread-default docstrings. The two runs were not interleaved and no clock, power or
thermal data was logged, so the difference is a run-to-run range, not a controlled comparison.
The generated cross-mode matrix deliberately stays on run 1 so its CPU/GPU comparison remains
reproducible; run 2 is a stability sample.

### Run 3: 2026-11-11 05:34 (fresh check)

A third fresh-process sample on the same host, with no other benchmark process and the same
6 auto-detected threads. It used the current worktree at HEAD `22ac35e` plus the uncommitted
per-layer CPU entry points; the benchmark path is still `kc_step`. The matrix stays on run 1; this
is a stability check, not a replacement.

| Model | Read | 100 | 2,000 | 8,000 | 16,000 | 24,000 | Load |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2B | 36.6 | 30.4 | 26.3 | 16.7 | 9.7 | 7.4 | 25.1 s |
| 4B | 18.5 | 16.1 | 12.8 | 6.4 | 4.1 | 2.9 | 52.4 s |
| 9B | 9.2 | 8.3 | 7.5 | 4.7 | 3.3 | 2.4 | 114.5 s |

At 24,000 the run-3 rates are 9.8%, 6.5% and 7.7% below run 1 for the 2B, 4B and 9B. The 4B
position-100 cell is the only run-3 cell above run 1 (5.9%); every other cell is 2.6-18.8% below.
The spread is consistent with the host variance already noted for run 2, not a code-speed claim.

## Method

```bat
set KINDI_CC=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin\clang.exe
set KINDI_LLVM=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc
set KINDI_DEVICE=cpu
python -m kindi --pack %USERPROFILE%\.kindi\packs\bonsai-2b --ctx 1024 verify
```

The same command was used with `bonsai-4b` and `bonsai-9b`. For each 24K benchmark, replace
`--ctx 1024 verify` with `--ctx 24576 bench`. Each run writes its own raw log; the engine builds
once and reuses `~/.cache/kindi/libkcpu.dll` afterwards.

## CPU server chat

The native CPU engine served each model on `127.0.0.1:8199` with a 1,024-token context and six
threads; each run used a fresh server and its own state directory. `POST /v1/chat/completions`
returned HTTP 200 for a normal request and a streaming request, and the streaming body ended in
`[DONE]`. `/health` reported 0 errors and 0 MiB GPU memory for every model.

| model | engine ready | non-stream | stream |
|---|---:|---|---|
| 2B | 25.5 s | 19 prompt / 9 completion, 32.9 tokens/s | 17 / 7, 32.7 tokens/s, stop |
| 4B | 55.8 s | 15 / 9, 19.2 tokens/s | 17 / 1, stop |
| 9B | 123.7 s | 15 / 9, 11.5 tokens/s | 15 / 9, 10.7 tokens/s, stop |

The 2B row is from the earlier run; the 4B and 9B rows are the 2026-11-10 extension. The 9B was
also probed with `max_tokens=12`; that stream ended with `finish_reason=length` and `[DONE]`, so a
capped stream terminates cleanly. These are single requests, not a throughput benchmark.

```bat
set KINDI_DEVICE=cpu
set KINDI_CC=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin\clang.exe
set KINDI_LLVM=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc
set KINDI_PACK=%USERPROFILE%\.kindi\packs\bonsai-2b
set KINDI_MODEL_NAME=bonsai2-2b
set KINDI_CTX=1024
python -m kindi serve --host 127.0.0.1 --port 8199
```

## Files

| File | Contents |
|---|---|
| [raw/system-info.log](raw/system-info.log) | run 1 OS, Python, clang version, git HEAD, logical CPU count, CPU model |
| [raw/system-info-rerun.log](raw/system-info-rerun.log) | run 2 OS, Python, clang version, git HEAD, logical CPU count, CPU model |
| [runs.csv](runs.csv) | run 1 native 24K rows in the cross-mode schema |
| [raw/bench-24k-2b.log](raw/bench-24k-2b.log), [raw/bench-24k-4b.log](raw/bench-24k-4b.log), [raw/bench-24k-9b.log](raw/bench-24k-9b.log) | run 1 logs quoted above |
| [raw/bench-24k-2b-rerun.log](raw/bench-24k-2b-rerun.log), [raw/bench-24k-4b-rerun.log](raw/bench-24k-4b-rerun.log), [raw/bench-24k-9b-rerun.log](raw/bench-24k-9b-rerun.log) | run 2 logs quoted above |
| [raw/system-info-run3.log](raw/system-info-run3.log) | run 3 OS, HEAD, Python, clang version and logical CPU count |
| [raw/bench-24k-2b-run3.log](raw/bench-24k-2b-run3.log), [raw/bench-24k-4b-run3.log](raw/bench-24k-4b-run3.log), [raw/bench-24k-9b-run3.log](raw/bench-24k-9b-run3.log) | run 3 logs quoted above |
| [raw/verify-2b.log](raw/verify-2b.log), [raw/verify-4b.log](raw/verify-4b.log), [raw/verify-9b.log](raw/verify-9b.log) | the three token-exact verification logs quoted above |
| [raw/server-chat.log](raw/server-chat.log), [raw/server-chat.json](raw/server-chat.json), [raw/server-chat-stream.txt](raw/server-chat-stream.txt) | the native CPU server start, the OpenAI chat response and the streaming chunks |
| [raw/server-chat-4b-log.txt](raw/server-chat-4b-log.txt), [raw/server-chat-4b.json](raw/server-chat-4b.json), [raw/server-chat-4b-stream.txt](raw/server-chat-4b-stream.txt) | the 4B server start, chat response and stream |
| [raw/server-chat-9b-log.txt](raw/server-chat-9b-log.txt), [raw/server-chat-9b.json](raw/server-chat-9b.json), [raw/server-chat-9b-stream.txt](raw/server-chat-9b-stream.txt) | the 9B server start, chat response and stream |

## What this proves and what it does not

Proved on this machine:

- `kcpu.c` compiles to a DLL with clang 18.1.8 + libomp against the MSVC 14.51 headers, with no
  `vcvars` shell and no POSIX flags.
- The 2B, 4B and 9B run in the native Windows engine and are token-exact with llama.cpp.
- The 24K benchmark completes natively at 24,576 tokens for the 2B, 4B and 9B in three independent
  samples.
- The engine's Windows thread default resolves to 6 physical cores.
- The CPU server answered 2B, 4B and 9B chat requests, non-stream and stream, at 1,024 context:
  HTTP 200, 32.9/19.2/11.5 tokens/s generation, 0 errors and 0 MiB GPU memory.

Not proved or not done:

- **Cross-block comparison.** The CPU run is native Windows and the GPU cells are from 2026-11-09;
  the two blocks were not interleaved.
- **Repeated-run stability and thermals.** Three 24K runs per model span 7.4-9.5, 2.9-3.4 and
  2.4-3.0 tokens/s at 24K; no clock, power or thermal logging, so the source of that range is not
  isolated. The native verify runs also differ by up to a quarter.
- **Other CPUs.** Only the i7-8086K was measured; the i5-2500K, Xeon W-2133 and i9-13900K rows are
  unchanged.
- **A llama.cpp speed comparison.** This is a Kindi-only correctness and speed run.
