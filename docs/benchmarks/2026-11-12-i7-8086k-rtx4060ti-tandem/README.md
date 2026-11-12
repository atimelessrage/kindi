# 2026-11-12 tandem throughput, i7-8086K + RTX 4060 Ti

One real `kindi bench` run of the dense tandem runtime on the same machine and
24,576-token context as the cross-mode CPU/GPU cells.

| model | placement | CPU layers | read | 100 | 2K | 8K | 16K | 24K |
|---|---|---:|---:|---:|---:|---:|---:|
| 2B | cpu-first (CPU 0..5, GPU 6..23, GPU head) | 6 | 80.9 | 65.7 | 71.3 | 55.5 | 43.4 | 35.8 |

Command:

```sh
PATH='/c/Program Files/clang+llvm-18.1.8-x86_64-pc-windows-msvc/bin:'$PATH \
python -m kindi --pack 'C:/Users/ICH/.kindi/packs/bonsai-2b' bench --tandem cpu-first --tandem-cpu-layers 6
```

The raw output is `bench-2b-cpu-first-24k.log`. The CPU engine built
`avx2-fma-f16c`, 6 threads, ready in 27.4 s; the GPU engine was ready in 1.6 s.
`bench` reported 6411 MiB free of 8187 MiB **after load**. That is free-after-load,
not a sampled peak. `runs.csv` is the machine-readable input consumed by
`docs/benchmarks/build_cross_mode.py`; it records `free_mib` for this run but the
cross-mode tables do not present free memory or claim a peak from it.

Only the 2B was measured. 4B and 9B tandem cells stay `n/a` until the same
single-run benchmark is done for them. The 27B larger-than-VRAM gate has exactness
and sampled peaks in `bench-logs/tandem-27b-gate/`; it is not used for these
rates because it does not report this 24K-position throughput table.
