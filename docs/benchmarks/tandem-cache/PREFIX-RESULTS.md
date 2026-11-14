# 2B tandem prefix reuse and checkpoint cost

Status: **ok, 2026-11-12.** At 512 tokens, both dense placements reused the expected joint prefixes
and the joint recurrent-state checkpoint measured **19.27 MiB**. A checkpoint-only follow-up measured
save/restore at 4,096 and 8,192 context in both placements.

This is a real-engine measurement on the 2B pack. It does not decode tokens; it isolates prompt
preparation (`Runner.prepare`) and checkpoint save/restore. Exact cached-vs-cache-off token output
is covered by [RESULTS.md](RESULTS.md).

## Environment

| item | value |
|---|---|
| host | native Windows, i7-8086K |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8,187 MiB |
| model | Ternary-Bonsai-2-2B |
| contexts | 512, 4,096 and 8,192 tokens |
| CPU layers | 6 per placement |
| CPU engine | `avx2-fma-f16c`, 6 threads |
| compiler | clang 18.1.8 with libomp |
| checkpoints | every 256 tokens at 512/4K; 512 at 8K; `prefix_max=0`, `park_max=0` |
| repetitions | 5 save and 5 restore samples per checkpoint path |

Command:

```sh
python docs/benchmarks/tandem-cache/measure_prefix.py \
  --pack "$HOME/.kindi/packs/bonsai-2b" --ctx 512 --cpu-layers 6 --reps 5 \
  --out bench-logs/tandem-cache/prefix-results.json
```

The checkpoint-only follow-up uses the same pack and CPU split, with the prompt and branch
positions aligned to the longer contexts:

```sh
python docs/benchmarks/tandem-cache/measure_prefix.py \
  --ctx 4096 --cpu-layers 6 --reps 5 --t1-tokens 4080 --branch-at 3072 \
  --checkpoint-only --out bench-logs/tandem-cache/checkpoint-cost-4k.json

python docs/benchmarks/tandem-cache/measure_prefix.py \
  --ctx 8192 --cpu-layers 6 --reps 5 --t1-tokens 8176 --branch-at 6144 \
  --checkpoint-only --modes cpu-first \
  --out bench-logs/tandem-cache/checkpoint-cost-8k-cpu.json

python docs/benchmarks/tandem-cache/measure_prefix.py \
  --ctx 8192 --cpu-layers 6 --reps 5 --t1-tokens 8176 --branch-at 6144 \
  --checkpoint-only --modes gpu-first \
  --out bench-logs/tandem-cache/checkpoint-cost-8k-gpu.json
```

`T1` is 320 tokens. The extension prompt is `T1` plus a 5-token suffix (325 tokens); the branch
prompt is `T1[:256]` plus a 5-token suffix (261 tokens). Cache-off runs use
`KINDI_PROMPT_CACHE=0` and reset the engine before reading the full prompt.

## Prefix reuse

| placement | extension | branch | cold T1 |
|---|---|---:|---:|
| `cpu-first` | **320/325 reused (98.46%)**, 5 new | **256/261 reused (98.08%)**, 5 new | 0/320 |
| `gpu-first` | **320/325 reused (98.46%)**, 5 new | **256/261 reused (98.08%)**, 5 new | 0/320 |

Prepare-only timings:

| placement | cold T1 | extension, cache on | extension, cache off | branch, cache on | branch, cache off |
|---|---:|---:|---:|---:|---:|
| `cpu-first` | 4.156 s | **0.059 s** | 3.375 s | **0.071 s** | 2.704 s |
| `gpu-first` | 3.788 s | **0.074 s** | 3.386 s | **0.090 s** | 2.668 s |

The warm extension skipped 320 tokens and the branch restored the 256-token checkpoint. The
cache-off controls read 325 and 261 tokens from zero respectively. The branch's own cache timing
shows `match/restore 0.006s`, `read+step 0.057-0.073s`; the extension has no restore
(`match/restore 0.000s`) because the end state is already in both engines.

These are one-run prepare timings, not generation throughput. The cache-off/cold spread (for
example CPU-first 4.156 s cold vs 3.375 s cache-off) shows the expected run-to-run noise.

## Joint checkpoint cost

The `Runner` checkpoint measured here copies both owners' recurrent `M` and `conv` arrays. It does
**not** copy K/V: each engine keeps its own K/V arrays in place. The K/V arrays in the cache view
are **6.00 MiB** at this context; they are not included in the save/restore numbers below.

| placement | joint state | CPU owner | GPU owner | save median | restore 256 median | restore 320 median |
|---|---:|---:|---:|---:|---:|---:|
| `cpu-first` | **19.266 MiB** | 5.35 MiB | 13.91 MiB | **8.913 ms** | **3.482 ms** | **3.487 ms** |
| `gpu-first` | **19.266 MiB** | 4.28 MiB | 14.98 MiB | **8.687 ms** | **3.257 ms** | **3.262 ms** |

Save samples:

| placement | min | median | max |
|---|---:|---:|---:|
| `cpu-first` | 7.597 ms | 8.913 ms | 11.887 ms |
| `gpu-first` | 8.428 ms | 8.687 ms | 9.005 ms |

Restore samples are tightly grouped: CPU-first 3.475-3.499 ms for the 256 checkpoint and
3.467-3.515 ms for the 320 checkpoint; GPU-first 3.238-3.309 ms and 3.220-3.275 ms respectively.

The end-of-`T1` checkpoint is saved after both owners have processed the prompt. The 256-token
checkpoint was saved while the same prompt was read, so its restore is a genuine branch restore.
The two placements split the 19.27 MiB differently (`cpu-first` gives 5.61 MB to the CPU owner,
`gpu-first` 4.49 MB), but the joint size is identical.

## 4K/8K joint checkpoint cost

The checkpoint-only runs isolate the joint recurrent state; they do not run the continuation or
cache-off controls. Each cell is five samples after one warm-up. `save` is
`Runner._save_ckpt()`, `restore branch` is `_restore()` of the 3,072/6,144-token checkpoint, and
`restore end` restores the end-of-T1 checkpoint.

| context | placement | joint state | save median (min-max) | restore branch median (min-max) | restore end median (min-max) |
|---:|---|---:|---:|---:|---:|
| 4,096 | `cpu-first` | **19.266 MiB** | **14.734 ms** (10.550-16.124) | **3.675 ms** (3.520-5.003) | **5.193 ms** (4.162-11.042) |
| 4,096 | `gpu-first` | **19.266 MiB** | **9.781 ms** (9.516-10.086) | **3.296 ms** (3.230-3.349) | **3.305 ms** (3.247-3.490) |
| 8,192 | `cpu-first` | **19.266 MiB** | **10.873 ms** (8.847-11.224) | **3.322 ms** (3.237-4.497) | **3.520 ms** (3.407-4.305) |
| 8,192 | `gpu-first` | **19.266 MiB** | **10.223 ms** (8.869-11.545) | **3.416 ms** (3.217-4.872) | **3.331 ms** (3.299-5.209) |

The checkpoint size is unchanged across contexts because it contains the two owners' recurrent
`M`/`conv` arrays, not K/V. The K/V cache is **48 MiB at 4K** and **96 MiB at 8K** and is not part
of these timings. The 8K run kept 24 checkpoint keys (dense keys through 3,840 plus every-512 keys
from 4,096 through 8,176); the 6,144-token branch checkpoint was present. The 4K run kept the full
16-key chain. The samples are single-run host measurements, not a scaling claim; the 4K
`cpu-first` restore-end cell includes an 11.042 ms outlier.

## Cache-timing breakdown

With `KINDI_CACHE_TIMING=1`, the runtime reported:

| placement / path | match/restore | read+step | checkpoints inside | end checkpoint |
|---|---:|---:|---:|---:|
| cpu-first cold T1 | 0.000 s | 4.148 s | 0.009 s | 0.008 s |
| cpu-first extension | 0.000 s | 0.050 s | <0.001 s | 0.008 s |
| cpu-first branch | 0.006 s | 0.057 s | <0.001 s | 0.008 s |
| gpu-first cold T1 | 0.000 s | 3.779 s | 0.009 s | 0.009 s |
| gpu-first extension | 0.000 s | 0.065 s | <0.001 s | 0.010 s |
| gpu-first branch | 0.006 s | 0.073 s | <0.001 s | 0.011 s |

The direct checkpoint measurement and the natural request path agree: a save is about 8-9 ms,
and a checkpoint is about 17 ms when a cold prompt creates both the 256- and 320-token copies.

## Limits

- This harness covers **prepare/checkpoint** only. It does not generate tokens or re-check
  token-for-token exactness; the prior [RESULTS.md](RESULTS.md) does that.
- This prepare-only harness disables saved openings (`prefix_max=0`); disk-opening reload cost and
  server-level metrics are measured separately in [RESULTS.md](RESULTS.md). Long-context equality
  (4K/8K) is in [LONG-CONTEXT-RESULTS.md](LONG-CONTEXT-RESULTS.md).
- K/V is not in the measured checkpoint. A saved opening also includes K/V; that restore path is
  covered by the saved-opening and HTTP-server checks in [RESULTS.md](RESULTS.md).
- The 512-token run covers full prefix reuse; the 4K/8K follow-up is checkpoint-only and does not
  re-check token-for-token equality. Equality at those contexts is in
  [LONG-CONTEXT-RESULTS.md](LONG-CONTEXT-RESULTS.md).
- One 2B pack, one RTX 4060 Ti, one i7-8086K, one CPU ISA and one run per cell. The timings are not a
  speed or context-scaling claim.
- No inference source changed. The harness is external and uses the existing `TandemRunner` and
  `cache.Runner` checkpoint methods.

Evidence: `measure_prefix.py`, `prefix-results.json`, `prefix-run.log`,
`checkpoint-cost-4k.json`, `checkpoint-cost-4k.log`, `checkpoint-cost-8k-cpu.json`,
`checkpoint-cost-8k-cpu.log`, `checkpoint-cost-8k-gpu.json` and `checkpoint-cost-8k-gpu.log` in this
directory.
