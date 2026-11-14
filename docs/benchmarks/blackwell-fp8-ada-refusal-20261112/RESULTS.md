# Blackwell FP8 verification

Status: **unsupported**

- requested arch: `auto`
- detected arch: `n/a`
- started: `2026-11-12T18:38:38Z`
- finished: `2026-11-12T18:38:39Z`
- device: NVIDIA GeForce RTX 4060 Ti (device 0), compute capability 89
- memory: 8187.4 MiB
- runtime: Python 3.13.6, NumPy 2.5.1, CuPy 14.2.0, CUDA driver 13040, CUDA runtime 12090

Error: `device does not match the requested Blackwell lane`

## Checks

| check | result | detail | elapsed ms |
|---|---|---|---:|
| `device-arch` | FAIL | compute capability 89 (not Blackwell); this lane requires 100/sm_100 or 120/sm_120 | 0.0 |

A pass requires a Blackwell device with compute capability exactly 100 or 120, loaded native FP8 kernels, and every check above green. Unsupported hardware exits 2.
