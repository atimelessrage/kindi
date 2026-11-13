# 2026-11-12 tandem throughput, i7-8086K + RTX 4060 Ti

Real `kindi bench` runs of the dense tandem runtime on the same machine and
24,576-token context as the cross-mode CPU/GPU cells, one fresh process per model.

| model | placement | CPU layers | read | 100 | 2K | 8K | 16K | 24K |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2B | cpu-first (CPU 0..5, GPU 6..23, GPU head) | 6 | 80.9 | 65.7 | 71.3 | 55.5 | 43.4 | 35.8 |
| 4B | cpu-first (CPU 0..5, GPU 6..31, GPU head) | 6 | 62.5 | 52.1 | 51.4 | 33.5 | 24.8 | 20.0 |
| 9B | cpu-first (CPU 0..5, GPU 6..31, GPU head) | 6 | 39.9 | 28.5 | 35.8 | 28.2 | 20.1 | 16.0 |

Commands:

```sh
PATH='/c/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc/bin:'$PATH \
python -m kindi --pack '~/.kindi/packs/bonsai-2b' bench --tandem cpu-first --tandem-cpu-layers 6
PATH='/c/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc/bin:'$PATH \
python -m kindi --pack '~/.kindi/packs/bonsai-4b' bench --tandem cpu-first --tandem-cpu-layers 6
PATH='/c/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc/bin:'$PATH \
python -m kindi --pack '~/.kindi/packs/bonsai-9b' bench --tandem cpu-first --tandem-cpu-layers 6
```

The raw outputs are `bench-2b-cpu-first-24k.log`, `bench-4b-cpu-first-24k.log` and
`bench-9b-cpu-first-24k.log`. The CPU engine built `avx2-fma-f16c`, 6 threads;
loads were 27.4 s (2B), 17.2 s (4B) and 32.4 s (9B, owned-slice CPU load). `bench`
reported free memory after load of 6411 MiB (2B), 5431 MiB (4B) and 4635 MiB (9B)
of 8187 MiB. That is free-after-load, not a sampled peak. `runs.csv` is the
machine-readable input consumed by `docs/benchmarks/build_cross_mode.py`; it
records `free_mib` for these runs but the cross-mode tables do not present free
memory or claim a peak from it.

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

The row at 6 layers is faster here than the recorded 2B row above (94.6 versus
80.9 read); both are real single runs from different sessions, and the sweep's
purpose is the same-session trend, not a new headline number.
