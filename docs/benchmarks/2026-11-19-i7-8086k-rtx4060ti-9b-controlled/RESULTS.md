# 9B CPU-only vs CPU+GPU tandem: three controlled same-context pairs

`CROSS-MODE.md`'s 9B tandem-to-CPU ratio used to compare two separate runs: the CPU-only row
from 2026-11-10 and the tandem row from 2026-11-13 (median of three). This page records three
controlled pairs on the current code, at the same 24,576-token context, so the 9B ratio is a
controlled A/B with a measured run-to-run spread. The placement matches the recorded 9B tandem
row: `cpu-first`, layers 0-5 on the CPU, layers 6-31 on the GPU.

The first pair is the original 2026-11-19 controlled run. The second pair was run later in a
fresh process for each placement. The third pair was run on 2026-11-20 after a passive cooldown:
the prior pair finished at 21:13 on 2026-11-19, the machine then sat idle for more than seven
hours, and the GPU was at 32 °C at the start (34 °C at the end). Each process is independent:
each pair ran CPU-only then tandem back to back, and the pairs were sequential rather than
interleaved. This is an observed idle cooldown, not an instrumented thermal experiment.

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
| CPU threads | 6 (one per physical core; `OMP_NUM_THREADS=6`) |

Load times were 123.8 s / 136.4 s / 107.3 s for the three CPU-only runs and 29.0 s / 28.8 s /
28.6 s for the three tandem runs. The CPU engine builds `kcpu.c` on first use; each fresh
CPU-only process pays a load again.

## Commands

```sh
export PATH="/path/to/clang-18/bin:$PATH"  # the local clang 18 install
export OMP_NUM_THREADS=6                   # one thread per physical core

# Pair 1
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 \
  bench --split-at none --json cpu.json
python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 \
  bench --tandem cpu-first --tandem-cpu-layers 6 --sample-vram --split-at none \
  --json tandem.json

# Pair 2
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 \
  bench --split-at none --json cpu-2.json
python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 \
  bench --tandem cpu-first --tandem-cpu-layers 6 --sample-vram --split-at none \
  --json tandem-2.json

# Pair 3 (2026-11-20, after the passive cooldown)
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 \
  bench --split-at none --json cpu-3.json
python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 24576 \
  bench --tandem cpu-first --tandem-cpu-layers 6 --sample-vram --split-at none \
  --json tandem-3.json
```

## Throughput (tokens/s)

CPU and tandem are the three controlled pairs. GPU only is the median of the three 2026-11-09
runs used by `CROSS-MODE.md`; it is shown for comparison and is not part of any pair.

| Placement | Run | Read | 100 | 2K | 8K | 16K | 24K |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU only | 1 | 12.7 | 11.5 | 9.6 | 6.3 | 3.9 | 2.9 |
| CPU only | 2 | 9.3 | 8.4 | 7.2 | 4.6 | 3.0 | 1.8 |
| CPU only | 3 | 8.9 | 7.7 | 7.0 | 4.7 | 3.3 | 2.4 |
| CPU only | **3-run median** | **9.3** | **8.4** | **7.2** | **4.7** | **3.3** | **2.4** |
| GPU only | 3-run median | 90.3 | 77.8 | 76.8 | 71.6 | 67.4 | 62.3 |
| GPU+CPU tandem | 1 | 43.1 | 34.2 | 36.6 | 28.1 | 20.6 | 16.8 |
| GPU+CPU tandem | 2 | 36.5 | 34.6 | 26.6 | 24.3 | 18.4 | 14.8 |
| GPU+CPU tandem | 3 | 35.4 | 32.1 | 28.8 | 23.4 | 18.0 | 14.4 |
| GPU+CPU tandem | **3-run median** | **36.5** | **34.2** | **28.8** | **24.3** | **18.4** | **14.8** |
| **Tandem / CPU (3-run median)** | | **3.9x** | **4.1x** | **4.0x** | **5.2x** | **5.6x** | **6.2x** |
| **Tandem / GPU (3-run median)** | | **0.4x** | **0.4x** | **0.4x** | **0.3x** | **0.3x** | **0.2x** |

The Tandem / GPU row is tandem throughput as a fraction of GPU-only (1.0x is parity).

Run-to-run change:

| Placement | Comparison | Read | 100 | 2K | 8K | 16K | 24K |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU only | run 2 vs run 1 | -26.8% | -27.0% | -25.0% | -27.0% | -23.1% | -37.9% |
| CPU only | run 3 vs run 2 | -4.3% | -8.3% | -2.8% | +2.2% | +10.0% | +33.3% |
| GPU+CPU tandem | run 2 vs run 1 | -15.3% | +1.2% | -27.3% | -13.5% | -10.7% | -11.9% |
| GPU+CPU tandem | run 3 vs run 2 | -3.0% | -7.2% | +8.3% | -3.7% | -2.2% | -2.7% |

**Run 2 was not a run-wide outlier.** Run 3 reproduced its tandem row within 8% at every cell,
and its CPU row within 4-10% at read/100/2K/8K/16K. The one exception is CPU-only 24K, where
run 2's 1.8 tok/s rose to 2.4 tok/s in run 3 (+33%); that is one low CPU point, not evidence
that the whole run should be discarded. Run 1 is the high side of the CPU and tandem spread.
The three-run median is therefore canonical, and the earlier two-run median is superseded:
the shallow ratio is **3.0x-4.2x** (three-run median **4.1x**), not a stable 3.0x or 3.5x; at
24K the per-run range is **5.8x-8.2x** and the median row is **6.2x** (the old two-run median
read 3.5x and 6.7x). The CPU-only pass slowed much more than the tandem pass in run 2, which is
why its 24K ratio was high; run 3 brings both sides back toward run 1 without making run 2 a
different regime.

## Exactness

| Placement | Result | Tokens | Verify context | Verify rate |
|---|---|---|---:|---:|
| CPU only, run 1 | 4/4 prompts, `ALL PASS` | 133/133 | 1,024 | 11.0 tok/s |
| GPU+CPU tandem, run 1 | 4/4 prompts, `ALL PASS` | 133/133 | 1,024 | 41.0 tok/s |

Runs 2 and 3 were throughput repeats and did not rerun `kindi verify`; their rows are not
additional exactness results. `kindi verify` runs at its own 1,024-token gate context; the
benchmarks and loads above use the requested 24,576-token context.

## Peak VRAM

The tandem runs sampled device-wide memory every 0.5 s.

| Run | Samples | Idle baseline | Peak used | Over idle | Free at peak | Free after load | Peak phase |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 59 | 925 MiB | 2,864 MiB | 1,939 MiB | 5,086 MiB | 5,245 MiB | `loaded` |
| 2 | 62 | 1,105 MiB | 2,968 MiB | 1,863 MiB | 4,982 MiB | 5,245 MiB | `loaded` |
| 3 | 61 | 1,020 MiB | 2,959 MiB | 1,939 MiB | 4,991 MiB | 5,245 MiB | `loaded` |

The CPU-only placements have no GPU allocation and no sampled peak (n/a).

## Evidence

- `cpu.json`, `tandem.json` — raw `kindi bench --json` output for pair 1.
- `cpu-2.json`, `tandem-2.json` — raw `kindi bench --json` output for pair 2.
- `cpu-3.json`, `tandem-3.json` — raw `kindi bench --json` output for pair 3.
- `runs.json`, `runs-2.json`, `runs-3.json` — the three pairs plus host settings, cooldown
  record and commands.
- `logs/cpu.txt`, `logs/tandem.txt`, `logs/cpu-2.txt`, `logs/tandem-2.txt`, `logs/cpu-3.txt`,
  `logs/tandem-3.txt` — full benchmark logs.
- `logs/cooldown-3.txt` — start/end timestamps and GPU utilization/temperature for pair 3.
- `logs/verify-cpu.txt`, `logs/verify-tandem.txt` — the run-1 exactness logs.

These are three fresh-process runs per placement, with no instrumented thermal or other
run-to-run control; the third pair followed the passive cooldown described above. The position
rows time 10 decode steps each. The spread above is the observed spread, not a confidence
interval.
