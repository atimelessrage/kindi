# Tensor-core (plan engine) throughput on the i7-8086K + RTX 4060 Ti (2026-11-13)

Tensor-core plan-engine (`KINDI_TP=1`, `KINDI_TC=1`) throughput at a 24,576-token context with the
exact head, `KINDI_FD=1`, `KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4`, `KINDI_KV=q8_0`, `KINDI_SPEC=0`. The
9B runs with `KINDI_TC_HP=1` (its 4096 Hadamard blocks are refused otherwise; see `kindi/spec.py`).
Three fresh-process runs per model; `runs.csv` holds one row per run and the matrix uses the
median, like the GPU-only rows.

On this machine the plan engine is ~1.5-1.8x the plain GPU-only engine:

| model | read | 100 | 2K | 8K | 16K | 24K | peak | free after load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B | 471.7 | 349.1 | 327.4 | 325.2 | 319.4 | 299.2 | 1,645 MiB | 6,399 MiB |
| 4B | 214.1 | 174.0 | 173.6 | 156.6 | 141.7 | 127.8 | 3,311 MiB | 4,733 MiB |
| 9B | 133.9 | 114.7 | 111.3 | 108.7 | 103.2 | 97.0 | 3,257 MiB | 4,787 MiB |

The 2B numbers match the earlier plan-vs-plain comparison in
`2026-11-09-i7-8086k-rtx4060ti/2b-plan-vs-plain-24k.csv` (read 470.5, 24K decode 303.5).

Commands (one per run; `--sample-vram` records the sampled device-wide peak):

```sh
KINDI_TP=1 KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1 KINDI_TC_TMAX=4 KINDI_KV=q8_0 KINDI_HEAD=exact \
  python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 bench --sample-vram --json bench-2b-tc-24k.json
# 9B: add KINDI_TC_HP=1
```

These runs feed the `tc` rows of
[`CROSS-MODE.md`](../CROSS-MODE.md) and the `tc/*` cells of
[`perf-baseline.json`](../perf-baseline.json).
