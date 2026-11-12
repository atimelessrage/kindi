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
GPU-only on this machine (cf. `docs/benchmarks/CROSS-MODE.md`). The 27B
larger-than-VRAM gate has exactness and sampled peaks in
`bench-logs/tandem-27b-gate/`; it is not used for these rates because it does
not report this 24K-position throughput table.
