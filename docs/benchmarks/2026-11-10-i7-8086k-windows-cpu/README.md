# Kindi CPU engine on native Windows: Intel Core i7-8086K

Measured 2026-11-10 on the Windows host, with no WSL and no Docker. This is the first native
Windows run of `kindi/cpu.py` + `kindi/kcpu.c`; the earlier i7-8086K result
([2026-11-09-i7-8086k-cpu](../2026-11-09-i7-8086k-cpu/README.md)) ran inside WSL2.

The short version: **`libkcpu.dll` builds with clang 18.1.8 and libomp, the engine resolves six
threads on its own, and the 2B, 4B and 9B are token-exact with llama.cpp.** The native verify is
faster than the WSL2 one on every model.

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
| Kindi HEAD | `fd3fba5` on `freebuff/tandem-cpu-recovery` |
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
three models; no 24,576-token native benchmark was run.

## Method

```bat
set KINDI_CC=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin\clang.exe
set KINDI_LLVM=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc
set KINDI_DEVICE=cpu
python -m kindi --pack %USERPROFILE%\.kindi\packs\bonsai-2b --ctx 1024 verify
```

The same command was used with `bonsai-4b` and `bonsai-9b`. Each run writes its own raw log; the
engine builds once and reuses `~/.cache/kindi/libkcpu.dll` afterwards.

## CPU server chat

The native CPU engine served the 2B on `127.0.0.1:8199` with a 1,024-token context and six threads.
The server built and loaded the pack in 25.5 s. `POST /v1/chat/completions` returned HTTP 200 for a
normal request (19 prompt / 9 completion tokens, 32.9 tokens/s) and a streaming request (17 / 7,
32.7 tokens/s), and the streaming body ended in `[DONE]`. `/health` reported 2 requests, 0 errors
and 0 MiB GPU memory.

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
| [raw/system-info.log](raw/system-info.log) | OS, Python, clang version, git HEAD, logical CPU count, CPU model |
| [raw/verify-2b.log](raw/verify-2b.log), [raw/verify-4b.log](raw/verify-4b.log), [raw/verify-9b.log](raw/verify-9b.log) | the three token-exact verification logs quoted above |
| [raw/server-chat.log](raw/server-chat.log), [raw/server-chat.json](raw/server-chat.json), [raw/server-chat-stream.txt](raw/server-chat-stream.txt) | the native CPU server start, the OpenAI chat response and the streaming chunks |

## What this proves and what it does not

Proved on this machine:

- `kcpu.c` compiles to a DLL with clang 18.1.8 + libomp against the MSVC 14.51 headers, with no
  `vcvars` shell and no POSIX flags.
- The 2B, 4B and 9B run in the native Windows engine and are token-exact with llama.cpp.
- The engine's Windows thread default resolves to 6 physical cores.
- The CPU server answered a 2B chat request, non-stream and stream, at 1,024 context: HTTP 200,
  32.9 tokens/s generation, 0 errors and 0 MiB GPU memory.

Not proved or not done:

- **24K context on Windows.** Only the 1,024-token verify was run here; the 24,576-token CPU
  benchmark remains the WSL2 number.
- **Repeated-run stability and thermals.** One run per model, no clock/power logging; the two runs
  differ by up to a quarter.
- **4B/9B server chat.** The server path was exercised with the 2B only; the 4B and 9B
  `kindi verify` runs passed, but no separate server chat request was sent for them.
- **Other CPUs.** Only the i7-8086K was measured; the i5-2500K, Xeon W-2133 and i9-13900K rows are
  unchanged.
- **A llama.cpp speed comparison.** This is a Kindi-only correctness run.
