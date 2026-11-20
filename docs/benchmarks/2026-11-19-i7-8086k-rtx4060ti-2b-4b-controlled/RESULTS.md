# 2B/4B CPU-only vs CPU+GPU tandem: controlled same-context pairs

`CROSS-MODE.md` used to compare the 2B and 4B CPU-only rows from a 2026-11-10 run with tandem
rows from a separate 2026-11-12/13 session. This page records same-session pairs on the current
code, at the same 24,576-token context, so both ratios are controlled A/Bs. Each placement is
one fresh process, run back to back for the model.

The placement matches the recorded tandem rows: `cpu-first`, layers 0-5 on the CPU, the rest on
the GPU (6-23 for the 2B, 6-31 for the 4B).

## Host and settings

| | |
|---|---|
| CPU | Intel Core i7-8086K, 6 physical cores / 12 threads, 32 GB |
| GPU | NVIDIA GeForce RTX 4060 Ti 8 GB |
| OS / Python | Windows 10 22H2, Python 3.13.6, CuPy 14.2.0 |
| Compiler | clang 18.1.8, installed outside `PATH` and prepended to it for these runs |
| Packs | `~/.kindi/packs/bonsai-2b` (`Ternary-Bonsai-2-2B`), `~/.kindi/packs/bonsai-4b` (`Ternary-Bonsai-2-4B`) |
| Context | 24,576 tokens for both packs, not reduced (`ctx_reduced: false`) |
| Mode | exact head, greedy, no speculative drafting; `--split-at none` |
| Tandem | `cpu-first`, 6 CPU layers, q8_0 K/V, exact GPU head |
| CPU threads | 6 (one per physical core) |

Load times were 25.0 s for the 2B CPU-only run and 57.7 s for the 4B CPU-only run. The tandem
runs loaded the CPU side in 10.7 s / 16.4 s and the plan engine in 15.5 s / 33.7 s (2B / 4B).

## Commands

```bash
export PATH="/path/to/clang-18/bin:$PATH"  # the local clang 18 install

# 2B
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 \
  bench --split-at none --json cpu-2b.json
python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 \
  bench --tandem cpu-first --tandem-cpu-layers 6 --sample-vram --split-at none \
  --json tandem-2b.json
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 verify
python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 verify \
  --tandem cpu-first --tandem-cpu-layers 6

# 4B
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-4b --ctx 24576 \
  bench --split-at none --json cpu-4b.json
python -m kindi --pack ~/.kindi/packs/bonsai-4b --ctx 24576 \
  bench --tandem cpu-first --tandem-cpu-layers 6 --sample-vram --split-at none \
  --json tandem-4b.json
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-4b --ctx 24576 verify
python -m kindi --pack ~/.kindi/packs/bonsai-4b --ctx 24576 verify \
  --tandem cpu-first --tandem-cpu-layers 6
```

## Throughput (tokens/s)

CPU and tandem are the controlled pair. GPU only is the median of the three 2026-11-09 runs
used by `CROSS-MODE.md`; it is shown for comparison and is not part of that pair.

### 2B

| Placement | Read | 100 | 2K | 8K | 16K | 24K |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | 41.8 | 27.7 | 27.8 | 18.4 | 11.5 | 8.2 |
| GPU only | 190.0 | 169.0 | 163.3 | 175.4 | 171.7 | 160.6 |
| GPU+CPU tandem | 93.5 | 70.8 | 81.8 | 64.1 | 43.8 | 37.4 |
| **Tandem / CPU** | **2.2x** | **2.6x** | **2.9x** | **3.5x** | **3.8x** | **4.6x** |
| **Tandem / GPU** | **0.5x** | **0.4x** | **0.5x** | **0.4x** | **0.3x** | **0.2x** |

The Tandem / GPU row is tandem throughput as a fraction of GPU-only (1.0x is parity).

The old cross-run row reported 2.2x/2.5x/3.0x/3.6x/4.3x at 100/2K/8K/16K/24K. The controlled
pair gives 2.6x/2.9x/3.5x/3.8x/4.6x: the current CPU placement is slower at shallow positions
(27.7 tok/s at 100 against the old run's 35.9), and the tandem is slower there too (70.8 against
79.8), so the ratio moves up even though both absolute rates are lower. At 24K the ratio is 4.6x.

### 4B

| Placement | Read | 100 | 2K | 8K | 16K | 24K |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | 16.4 | 14.1 | 11.9 | 6.7 | 3.9 | 2.9 |
| GPU only | 129.0 | 106.2 | 101.9 | 90.8 | 89.3 | 79.8 |
| GPU+CPU tandem | 53.3 | 39.6 | 37.9 | 26.2 | 18.4 | 14.4 |
| **Tandem / CPU** | **3.2x** | **2.8x** | **3.2x** | **3.9x** | **4.7x** | **5.0x** |
| **Tandem / GPU** | **0.4x** | **0.4x** | **0.4x** | **0.3x** | **0.2x** | **0.2x** |

The Tandem / GPU row is tandem throughput as a fraction of GPU-only (1.0x is parity).

The old cross-run row reported 3.1x/3.3x/4.7x/5.5x/5.4x at 100/2K/8K/16K/24K. The controlled
pair gives 2.8x/3.2x/3.9x/4.7x/5.0x. Position 100 is slower in both placements than the old
standalone runs (CPU 14.1 against 15.2, tandem 39.6 against 47.0); the 24K tandem rate is 14.4
against 16.7, so the 24K ratio falls from 5.4x to 5.0x.

These single runs are not a claim that the current code is intrinsically slower than the old
rows; they are the controlled pair for the ratios, and run-to-run and thermal spread is not
measured here.

## Exactness

| Model | Placement | Result | Tokens | Verify context | Verify rate |
|---|---|---|---|---:|---:|
| 2B | CPU only | 4/4 prompts, `ALL PASS` | 160/160 | 1,024 | 19.4 tok/s |
| 2B | GPU+CPU tandem | 4/4 prompts, `ALL PASS` | 160/160 | 1,024 | 80.3 tok/s |
| 4B | CPU only | 4/4 prompts, `ALL PASS` | 160/160 | 1,024 | 15.2 tok/s |
| 4B | GPU+CPU tandem | 3/4 prompts exact, 1 llama.cpp tie, 0 fail | 159/160 | 1,024 | 56.3 tok/s |

The 4B tandem's prompt 1 (`The sky is blue because`) first differs at token 24, where llama.cpp's
top-2 gap is 0.0157; it matches 39/40 tokens and is recorded as a tie, not a failure. The other
three prompts are 40/40. This is not a 4/4 exactness result for the 4B tandem, so the 4B
throughput ratio is not an exactness claim.

`kindi verify` runs at its own 1,024-token gate context; the benchmarks and loads above use the
requested 24,576-token context.

## Peak VRAM

The tandem runs sampled device-wide memory every 0.5 s.

| Model | Samples | Idle baseline | Peak used | Over idle | Free at peak | Free after load | Peak phase |
|---|---:|---:|---:|---:|---:|---:|---|
| 2B | 51 | 1,022 MiB | 1,627 MiB | 605 MiB | 6,323 MiB | 6,577 MiB | `loaded` |
| 4B | 95 | 1,012 MiB | 2,385 MiB | 1,373 MiB | 5,565 MiB | 5,895 MiB | `stop` |

The CPU-only placements have no GPU allocation and no sampled peak (n/a).

## Evidence

- `cpu-2b.json`, `tandem-2b.json`, `cpu-4b.json`, `tandem-4b.json` - raw `kindi bench --json`
  output for the four placements.
- `runs-2b.json`, `runs-4b.json` - the two controlled pairs plus verify results, commands and
  host settings.
- `logs/cpu-2b.txt`, `logs/tandem-2b.txt`, `logs/cpu-4b.txt`, `logs/tandem-4b.txt` - full
  benchmark logs.
- `logs/verify-cpu-2b.txt`, `logs/verify-tandem-2b.txt`, `logs/verify-cpu-4b.txt`,
  `logs/verify-tandem-4b.txt` - full exactness logs.

Each position row times 10 decode steps. These are single fresh-process runs per placement,
with no thermal or run-to-run control.
