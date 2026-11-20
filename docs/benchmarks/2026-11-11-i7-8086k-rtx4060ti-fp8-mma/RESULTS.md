# Native FP8 MMA microbench on Ada

Measured 2026-11-11 on the i7-8086K + RTX 4060 Ti 8 GB, driver 617.42, Ada sm_89.

`bench-logs/bench_fp8_mma.py` compiled the CUDA module once, ran two warm-up calls and then five
timed calls per row. The time includes per-token activation quantization. E4M3 and E5M2 are within
about 2% of each other; `docs/FP8-BF16.md` quotes the E4M3 table.

```text
device: NVIDIA GeForce RTX 4060 Ti (sm_89), driver 13040
matrix rows x cols | tokens | ms | TFLOPS
E4M3 2560 x 2560 | 128 | 0.11 | 15.1
E4M3 2560 x 2560 | 512 | 0.24 | 28.2
E4M3 10240 x 2560 | 128 | 0.23 | 29.0
E4M3 10240 x 2560 | 512 | 0.76 | 35.5
E5M2 2560 x 2560 | 128 | 0.11 | 15.4
E5M2 2560 x 2560 | 512 | 0.24 | 27.9
E5M2 10240 x 2560 | 128 | 0.23 | 29.0
E5M2 10240 x 2560 | 512 | 0.76 | 35.2
```

The f32 dequantize + cuBLAS fallback was not timed: cuBLASLt is absent on this host. This is a
synthetic kernel measurement, not a real FP8 model or a Blackwell result.
