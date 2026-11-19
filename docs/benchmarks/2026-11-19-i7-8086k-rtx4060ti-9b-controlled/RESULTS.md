# 9B CPU-only vs CPU+GPU tandem: two controlled same-context pairs

`CROSS-MODE.md`'s 9B tandem-to-CPU ratio used to compare two separate runs: the CPU-only row
from 2026-11-10 and the tandem row from 2026-11-13 (median of three). This page records two
controlled pairs on the current code, at the same 24,576-token context, so the 9B ratio is a
controlled A/B with a measured run-to-run spread. The placement matches the recorded 9B tandem
row: `cpu-first`, layers 0-5 on the CPU, layers 6-31 on the GPU.

The first pair is the original 2026-11-19 controlled run. The second pair was run later in a
fresh process for each placement. Each process is independent: each pair ran CPU-only then
tandem back to back, and the two pairs were sequential rather than interleaved.

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
| CPU threads | 6 (one per physical core) |

Load times were 123.8 s / 136.4 s for the two CPU-only runs and 29.0 s / 28.8 s for the two
tandem runs. The CPU engine builds `kcpu.c` on first use; each fresh CPU-only process pays that
load again.

## Commands

```bash
export PATH="/path/to/clang-18/bin:$PATH"  # the local clang 18 install

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
```

## Throughput (tokens/s)

CPU and tandem are the two controlled pairs. GPU only is the median of the three 2026-11-09
runs used by `CROSS-MODE.md`; it is shown for comparison and is not part of either pair.

| Placement | Run | Read | 100 | 2K | 8K | 16K | 24K |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU only | 1 | 12.7 | 11.5 | 9.6 | 6.3 | 3.9 | 2.9 |
| CPU only | 2 | 9.3 | 8.4 | 7.2 | 4.6 | 3.0 | 1.8 |
| CPU only | **median** | **11.00** | **9.95** | **8.40** | **5.45** | **3.45** | **2.35** |
| GPU only | 3-run median | 90.3 | 77.8 | 76.8 | 71.6 | 67.4 | 62.3 |
| GPU+CPU tandem | 1 | 43.1 | 34.2 | 36.6 | 28.1 | 20.6 | 16.8 |
| GPU+CPU tandem | 2 | 36.5 | 34.6 | 26.6 | 24.3 | 18.4 | 14.8 |
| GPU+CPU tandem | **median** | **39.80** | **34.40** | **31.60** | **26.20** | **19.50** | **15.80** |
| **Tandem / CPU (median)** | | **3.62x** | **3.46x** | **3.76x** | **4.81x** | **5.65x** | **6.72x** |

Run-to-run change, run 2 against run 1:

| Placement | Read | 100 | 2K | 8K | 16K | 24K |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | -26.8% | -27.0% | -25.0% | -27.0% | -23.1% | -37.9% |
| GPU+CPU tandem | -15.3% | +1.2% | -27.3% | -13.5% | -10.7% | -11.9% |

The shallow-context tandem/CPU ratio was 2.97x in run 1 and 4.12x in run 2, a range of
**3.0x-4.1x**. The two-run median is **3.46x** at position 100, not a stable 3.0x. At 24K the
pair ranges 5.79x-8.22x (median 6.72x). The CPU-only pass slowed much more than the tandem pass,
which is why the ratio moves up in the second run; this is a measured spread, not an inferred
thermal cause.

The old cross-run row reported 4.3x/4.5x/5.2x/5.9x/5.8x at 100/2K/8K/16K/24K. The first
controlled run put the shallow-context gain at 3.0x, not 4.3x. The second run shows that the
single 3.0x reading does not repeat: the two-run median at 100 is 3.5x, and the pair spans
3.0x-4.1x. At 24K the two controlled runs span 5.8x-8.2x against the old 5.8x.

## Exactness

| Placement | Result | Tokens | Verify context | Verify rate |
|---|---|---|---:|---:|
| CPU only, run 1 | 4/4 prompts, `ALL PASS` | 133/133 | 1,024 | 11.0 tok/s |
| GPU+CPU tandem, run 1 | 4/4 prompts, `ALL PASS` | 133/133 | 1,024 | 41.0 tok/s |

Run 2 was a throughput repeat and did not rerun `kindi verify`; its rows are not a second
exactness result. `kindi verify` runs at its own 1,024-token gate context; the benchmarks and
loads above use the requested 24,576-token context.

## Peak VRAM

The tandem runs sampled device-wide memory every 0.5 s.

| Run | Samples | Idle baseline | Peak used | Over idle | Free at peak | Free after load | Peak phase |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 59 | 925 MiB | 2,864 MiB | 1,939 MiB | 5,086 MiB | 5,245 MiB | `loaded` |
| 2 | 62 | 1,105 MiB | 2,968 MiB | 1,863 MiB | 4,982 MiB | 5,245 MiB | `loaded` |

The CPU-only placements have no GPU allocation and no sampled peak (n/a).

## Evidence

- `cpu.json`, `tandem.json` - raw `kindi bench --json` output for pair 1.
- `cpu-2.json`, `tandem-2.json` - raw `kindi bench --json` output for pair 2.
- `runs.json`, `runs-2.json` - the two pairs plus host settings and commands.
- `logs/cpu.txt`, `logs/tandem.txt`, `logs/cpu-2.txt`, `logs/tandem-2.txt` - full benchmark
  logs.
- `logs/verify-cpu.txt`, `logs/verify-tandem.txt` - the run-1 exactness logs.

These are two fresh-process runs per placement, with no thermal or other run-to-run control; the
position rows time 10 decode steps each. The spread above is the observed spread, not a
confidence interval.
