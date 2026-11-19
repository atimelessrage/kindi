# Tandem throughput, i7-8086K + RTX 4060 Ti

Real `kindi bench` runs of the dense tandem runtime on the same machine and
24,576-token context as the cross-mode CPU/GPU cells. The numbers in the table are
the **median of three fresh-process runs per model** (2026-11-13), the same
methodology as the GPU-only cells. The original 2026-11-12 single runs
(2B read 80.9, 4B read 62.5, 9B read 39.9) are kept as `bench-<model>-cpu-first-24k.log`.

| model | placement | CPU layers | read | 100 | 2K | 8K | 16K | 24K |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2B | cpu-first (CPU 0..5, GPU 6..23, GPU head) | 6 | 84.3 | 79.8 | 72.3 | 58.8 | 43.2 | 34.9 |
| 4B | cpu-first (CPU 0..5, GPU 6..31, GPU head) | 6 | 49.3 | 47.0 | 43.8 | 31.7 | 23.7 | 16.7 |
| 9B | cpu-first (CPU 0..5, GPU 6..31, GPU head) | 6 | 38.1 | 37.5 | 34.7 | 25.4 | 20.5 | 15.2 |

> **The single-run rows were off by up to 30%.** One re-measurement (the 2B 6-layer sweep
> run) read 94.6 against the recorded 80.9. Re-running all three models three times moved
> 4B read by **-21%** and 9B at position 100 by **+32%**; the perf gate flagged the 4B row
> (-21% read, -17% at 24K) before the new medians were adopted. Single tandem runs are not
> spec-grade evidence; these medians were [CROSS-MODE.md](../CROSS-MODE.md)'s source until the
> 2026-11-19 controlled pairs replaced them.

Commands:

```sh
PATH='/c/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc/bin:'$PATH \
python -m kindi --pack '~/.kindi/packs/bonsai-2b' bench --tandem cpu-first --tandem-cpu-layers 6
PATH='/c/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc/bin:'$PATH \
python -m kindi --pack '~/.kindi/packs/bonsai-4b' bench --tandem cpu-first --tandem-cpu-layers 6
PATH='/c/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc/bin:'$PATH \
python -m kindi --pack '~/.kindi/packs/bonsai-9b' bench --tandem cpu-first --tandem-cpu-layers 6
```

The raw outputs are the 2026-11-12 single runs `bench-2b-cpu-first-24k.log`,
`bench-4b-cpu-first-24k.log` and `bench-9b-cpu-first-24k.log`, and the 2026-11-13 median
runs `bench-<model>-cpu-first-24k-r1..3.log`. The CPU engine built `avx2-fma-f16c`, 6
threads; loads were 27.4 s (2B), 17.2 s (4B) and 32.4 s (9B, owned-slice CPU load). `bench`
reported free memory after load of 6411 MiB (2B), 5431 MiB (4B) and 4635 MiB (9B)
of 8187 MiB in every run. That is free-after-load, not a sampled peak. `runs.csv` holds
one row per run; `docs/benchmarks/build_cross_mode.py` takes the median of each model’s
rows. The cross-mode tables do not present free memory or claim a peak from it.

All three cross-mode tandem rows are now measured. Tandem still does not beat
GPU-only on this machine (cf. `docs/benchmarks/CROSS-MODE.md`). The 27B gate has
exactness and sampled peaks in `bench-logs/tandem-27b-gate/`; its rate and cache
evidence is in [27B-8K.md](27B-8K.md).

## CPU-layer boundary sweep (2B, 2026-11-13)

`--tandem-cpu-layers n` decides how much of the model leaves the GPU. Same
session, same build, 24,576-token context, `cpu-first`, exact head; rates in
tokens/s.

| CPU layers | read | 100 | 2K | 8K | 16K | 24K |
|---:|---:|---:|---:|---:|---:|---:|
| 2 (CPU 0..1, GPU 2..23) | 122.8 | 118.0 | 109.3 | 98.5 | 98.0 | 118.9 |
| 4 (CPU 0..3, GPU 4..23) | 110.5 | 104.9 | 95.3 | 71.8 | 52.1 | 41.7 |
| 6 (CPU 0..5, GPU 6..23) | 94.6 | 87.7 | 82.0 | 62.6 | 52.1 | 39.5 |
| 8 (CPU 0..7, GPU 8..23) | 85.6 | 87.2 | 77.0 | 54.0 | 31.5 | 22.6 |
| 12 (CPU 0..11, GPU 12..23) | 80.0 | 69.7 | 63.5 | 40.4 | 25.9 | 18.2 |

Each extra CPU layer costs about 8-10% of decode rate near 2K and much more at
long context, where more of the work moves to the CPU: 24K decode falls from
118.9 (2 layers) to 18.2 tok/s (12 layers). The knee is at 4-6 layers; beyond 6
the long-context cost grows faster than the VRAM saving. GPU-only 2B at 24K is
160.6 tok/s, so every split is slower - the point of the knob is memory, not
speed. Commands: `kindi --pack ~/.kindi/packs/bonsai-2b bench --tandem cpu-first
--tandem-cpu-layers N`; raw logs `bench-logs/tandem-cache/sweep-2b-cpu-first-l*.log`.

The row at 6 layers is faster here than the recorded median (94.6 versus 84.3 read);
both are real runs from different sessions, and the sweep's purpose is the same-session
trend, not a new headline number.
