# 9B CPU-only vs CPU+GPU tandem: controlled same-context pair

`CROSS-MODE.md`'s 9B tandem-to-CPU ratio used to compare two separate runs: the CPU-only row
from 2026-11-10 and the tandem row from 2026-11-13 (median of three). This page records a
same-session pair on the current code, at the same 24,576-token context, so the 9B ratio is a
controlled A/B. The placement matches the recorded 9B tandem row: `cpu-first`, layers 0-5 on
the CPU, layers 6-31 on the GPU.

## Host and settings

| | |
|---|---|
| CPU | Intel Core i7-8086K, 6 physical cores / 12 threads, 32 GB |
| GPU | NVIDIA GeForce RTX 4060 Ti 8 GB |
| OS / Python | Windows 10 22H2, Python 3.13.6, CuPy 14.2.0 |
| Compiler | clang 18.1.8, installed outside `PATH` and prepended to it for these runs |
| Pack | `~/.kindi/packs/bonsai-9b` (`Ternary-Bonsai-2-9B`) |
| Context | 24,576 tokens, not reduced (`ctx_reduced: false`) |
| Mode | exact head, greedy, no speculative drafting; `--split-at none` |
| Tandem | `cpu-first`, 6 CPU layers (`cpu_range` 0-5, `gpu_range` 6-31), q8_0 K/V |

The CPU engine builds `kcpu.c` on first use; the CPU-only run's 123.8 s load includes that
build. The tandem run reused the built library (27.4 s CPU side + 1.6 s plan engine).

## Commands

```bash
export PATH="/path/to/clang-18/bin:$PATH"  # the local clang 18 install

KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 \
  bench --split-at none --json cpu.json

python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 \
  bench --tandem cpu-first --tandem-cpu-layers 6 --sample-vram --split-at none \
  --json tandem.json

KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 verify
python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 verify \
  --tandem cpu-first --tandem-cpu-layers 6
```

## Throughput (tokens/s)

| Placement | Read | 100 | 2K | 8K | 16K | 24K |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | 12.7 | 11.5 | 9.6 | 6.3 | 3.9 | 2.9 |
| GPU+CPU tandem | 43.1 | 34.2 | 36.6 | 28.1 | 20.6 | 16.8 |
| **Tandem / CPU** | **3.4x** | **3.0x** | **3.8x** | **4.5x** | **5.3x** | **5.8x** |

The old cross-run row reported 4.3x/4.5x/5.2x/5.9x/5.8x at 100/2K/8K/16K/24K. The controlled
pair puts the shallow-context gain at 3.0x, not 4.3x: the current CPU engine is faster at
positions 100-2K (11.5/9.6 tok/s against the old 8.7/7.7), while the tandem run at position
100 is slower than the 2026-11-13 median (34.2 against 37.5). At 24K the two agree at 5.8x;
8K-16K is lower than the cross-run row (4.5x/5.3x against 5.2x/5.9x).

## Exactness

| Placement | Result | Tokens | Verify context | Verify rate |
|---|---|---|---:|---:|
| CPU only | 4/4 prompts, `ALL PASS` | 133/133 | 1,024 | 11.0 tok/s |
| GPU+CPU tandem | 4/4 prompts, `ALL PASS` | 133/133 | 1,024 | 41.0 tok/s |

`kindi verify` runs at its own 1,024-token gate context; the benchmark and both loads above
use the requested 24,576-token context.

## Peak VRAM

The tandem run sampled device-wide memory every 0.5 s (59 samples). Idle baseline 925 MiB;
peak **2,864 MiB used**, **1,939 MiB over idle**, **5,086 MiB free**, peak phase `loaded`;
5,245 MiB free after load. The CPU-only placement has no GPU allocation and no sampled peak
(n/a).

## Evidence

- `cpu.json`, `tandem.json` - raw `kindi bench --json` output for the two placements.
- `runs.json` - the two rows plus verify results, commands and host settings, in one file.
- `logs/cpu.txt`, `logs/tandem.txt` - full benchmark logs.
- `logs/verify-cpu.txt`, `logs/verify-tandem.txt` - full exactness logs.

These are single fresh-process runs per placement, with no thermal or run-to-run control;
the position rows time 10 decode steps each.
