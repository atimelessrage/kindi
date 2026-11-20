# Real 9B tandem joint-cache check

Status: **PASS, 2026-11-12.** Both dense tandem placements produced token-identical greedy
continuations with the joint cache on and with `KINDI_PROMPT_CACHE=0`. The new runner also loaded
the saved 256-token opening from disk in both placements. The real HTTP server save/restart/reload
path passed separately with a 344-token opening in both placements. A checkpoint-only run at 512
context measured a **50.25 MiB** joint recurrent checkpoint, **15.9/17.4 ms** save and **10.5-11.1 ms**
restore medians, and a device-wide peak of **3,063/3,065 MiB**. At 4,096 and 8,192 tokens, both
placements were also token-identical to cache-off (extension 4,080/8,176 and branch 3,072/6,144).

This is the 9B counterpart to [RESULTS.md](RESULTS.md). The 2B long-context 4K/8K results are in
[LONG-CONTEXT-RESULTS.md](LONG-CONTEXT-RESULTS.md); the 9B run here is at 512 context.

## Environment

| item | value |
|---|---|
| host | native Windows, i7-8086K |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8,187 MiB |
| model | Ternary-Bonsai-2-9B, `Ternary-Bonsai-2-9B-PTQ1_0.gguf` |
| pack source SHA-256 | `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b` |
| layers | 32, width 4,096, 16 query / 4 KV heads |
| contexts | 512, 4,096 and 8,192 tokens |
| CPU layers | 6 per placement |
| CPU engine | `avx2-fma-f16c`, 6 threads |
| compiler | clang 18.1.8 with libomp |
| cache | 512: `prefix_max=1`, `prefix_min=256`, 256-token checkpoints; 4K/8K: `prefix_max=0`, 1,024/2,048-token checkpoints, `park_max=0` |
| tokens | 8 greedy continuation tokens |

The 9B CPU engine loaded in **107.5 s** (`cpu-first`) and **105.4 s** (`gpu-first`); the GPU owner
took 5.8-6.0 s. The total harness load times are in the result JSON. The `cpu-first` ranges were
CPU `0..5` and GPU `6..31`; the `gpu-first` ranges were GPU `0..25` and CPU `26..31`.

## Workload

The harness used the same 320-token `T1` and 8 greedy tokens as the 2B check. The extension prompt is
`T1` plus 5 tokens (325 total); the branch prompt is `T1[:256]` plus a different 5-token suffix (261
total). The cached branch writes a 256-token disk opening; a new `TandemRunner` loads it.

Commands:

```sh
python docs/benchmarks/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-9b" --ctx 512 --cpu-layers 6 --tokens 8 \
  --modes cpu-first --out bench-logs/tandem-cache/results-9b-cpu.json

python docs/benchmarks/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-9b" --ctx 512 --cpu-layers 6 --tokens 8 \
  --modes gpu-first --out bench-logs/tandem-cache/results-9b-gpu.json
```

## Results

| placement | case | cache-on reused | cache-off reused | exact | cache-on seconds | cache-off seconds |
|---|---|---:|---:|---|---:|---:|
| `cpu-first` | extension | 320 | 0 | **yes** | 0.327 | 8.705 |
| `cpu-first` | branch | 256 | 0 | **yes** | 0.334 | 6.997 |
| `cpu-first` | saved opening (new runner) | 256 | 0 | **yes** | 0.463 | 6.882 |
| `gpu-first` | extension | 320 | 0 | **yes** | 0.407 | 8.942 |
| `gpu-first` | branch | 256 | 0 | **yes** | 0.500 | 7.214 |
| `gpu-first` | saved opening (new runner) | 256 | 0 | **yes** | 0.530 | 7.626 |

The runtime logs show the expected transitions in both placements:

```
prompt cache: a checkpoint every 256 tokens (50 MB each)
cache: 320 prompt tokens, 320 shared with the last request, resumed at 256
cache: 261 prompt tokens, 256 shared with the last request, resumed at 256
loaded a saved opening of 256 tokens (0.1s)
```

The cache-off requests show `resumed at 0`, and the saved opening's cache-on count is **256** versus
**0** with the cache off. Save and load each took about 0.1 s.

Generated IDs (identical for both placements):

| case | IDs |
|---|---|
| extension | `[328, 760, 1727, 11316, 11690, 440, 328, 760]` |
| branch / saved opening | `[264, 2086, 3299, 13, 357, 2086, 8602, 11690]` |

## HTTP server save/restart/reload (2026-11-12)

The server harness (`run_server_check.py`) started a real `kindi serve` process, sent a warm system
prompt and a branch request that wrote a 344-token opening, stopped it, started a new server over
the same state directory, and requested the final user turn. A third server with a fresh state
directory and `KINDI_PROMPT_CACHE=0` produced the reference reply. The prompt is 364 tokens; the
opening ends at 344.

| placement | warm branch cached | reload cached | cache-off cached | reload reply | exact | status |
|---|---:|---:|---:|---|---|---|
| `cpu-first` | 344 | **344** | 0 | `Blue` | **yes** | **ok** |
| `gpu-first` | 344 | **344** | 0 | `Blue` | **yes** | **ok** |

Both server logs contain `saved an opening of 344 tokens` and `loaded a saved opening of 344
tokens`; the save took 0.1 s and the load 0.1-0.2 s. The cached and cache-off replies matched on
content, finish reason and completion-token count. The CPU engine started in 115-123 s and the GPU
owner in 6.7-14.7 s during these runs; the server starts dominate wall time.

Command:

```sh
python docs/benchmarks/tandem-cache/run_server_check.py \
  --pack "$HOME/.kindi/packs/bonsai-9b" --ctx 512 --cpu-layers 6 \
  --modes cpu-first,gpu-first \
  --run-dir bench-logs/tandem-cache/server-runs-9b \
  --out bench-logs/tandem-cache/server-results-9b.json
```

## 512-context checkpoint cost and peak VRAM (2026-11-12)

The checkpoint-only run used the same 9B pack and placement split but skipped continuations and
cache-off controls. It measured the joint `Runner` checkpoint (both owners' recurrent `M`/`conv`),
not K/V. Five samples followed one warm-up allocation per path. The VRAM sampler read device-wide
`nvidia-smi memory.used` every 0.5 s; the baseline was taken before the engine constructor.

| placement | joint state | CPU owner | GPU owner | save median | restore 256 median | restore 320 median | device peak | idle baseline | peak delta | peak free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpu-first` | **50.25 MiB** | 10.47 MiB | 39.78 MiB | **15.914 ms** | **11.141 ms** | **10.901 ms** | **3,063 MiB** | 1,184 MiB | 1,879 MiB | 4,887 MiB |
| `gpu-first` | **50.25 MiB** | 8.38 MiB | 41.88 MiB | **17.375 ms** | **10.540 ms** | **11.068 ms** | **3,065 MiB** | 1,293 MiB | 1,772 MiB | 4,885 MiB |

The checkpoint size is unchanged between placements; only its owner split differs. The K/V arrays
are **16 MiB** at 512 context and are not part of the checkpoint. The peak was reached at
`engine-ready` in both runs; the cold-prompt and checkpoint markers stayed at the same 3,063/3,065
MiB. The CPU engine took 107.5-108.0 s and the GPU owner 6.0-6.4 s to load, for 114.1 s total per
placement. The sampler is a 0.5 s device-wide observation, not a CUDA allocator high-water mark;
the baseline includes the desktop and other processes.

Command:

```sh
python docs/benchmarks/tandem-cache/measure_prefix.py \
  --pack "$HOME/.kindi/packs/bonsai-9b" --ctx 512 --cpu-layers 6 \
  --reps 5 --t1-tokens 320 --branch-at 256 \
  --checkpoint-only --vram --vram-sample-s 0.5 \
  --modes cpu-first,gpu-first \
  --out bench-logs/tandem-cache/checkpoint-cost-9b-512.json
```

This companion run does not re-check generated token equality; the 9B exactness result above and
the HTTP path remain the correctness evidence.

## 4K/8K equality (2026-11-12)

This extends the 512-token exactness result to 4,096 and 8,192 context with the same harness and
9B pack. Saved openings are disabled (`--prefix-max 0`) so this run isolates the in-process
checkpoint path. `T1` is 16 tokens below the context, the extension adds 5 tokens, and the branch
is at three-quarters of `T1`.

| context | checkpoint interval | T1 | branch | extension prompt | branch prompt |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,024 | 4,080 | 3,072 | 4,085 | 3,077 |
| 8,192 | 2,048 | 8,176 | 6,144 | 8,181 | 6,149 |

| context | placement | extension cached | branch cached | cache-off | exact | status |
|---:|---|---:|---:|---:|---|---|
| 4,096 | `cpu-first` | **4,080/4,085** | **3,072/3,077** | 0 | **yes** | **ok** |
| 4,096 | `gpu-first` | **4,080/4,085** | **3,072/3,077** | 0 | **yes** | **ok** |
| 8,192 | `cpu-first` | **8,176/8,181** | **6,144/6,149** | 0 | **yes** | **ok** |
| 8,192 | `gpu-first` | **8,176/8,181** | **6,144/6,149** | 0 | **yes** | **ok** |

The runtime logs show the expected `resumed at 4080/3072` and `resumed at 8176/6144` transitions,
and the cache-off requests show `resumed at 0`. The generated extension IDs were identical between
placements at each context; the branch IDs were also identical between placements. These are
one-run prepare/generation timings, not a throughput or long-context scaling claim.

Commands:

```sh
python docs/benchmarks/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-9b" --ctx 4096 --cpu-layers 6 --tokens 8 \
  --t1-tokens 4080 --branch-at 3072 --checkpoint-every 1024 \
  --prefix-max 0 --skip-saved --modes cpu-first,gpu-first \
  --out bench-logs/tandem-cache/context-9b-4k.json

python docs/benchmarks/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-9b" --ctx 8192 --cpu-layers 6 --tokens 8 \
  --t1-tokens 8176 --branch-at 6144 --checkpoint-every 2048 \
  --prefix-max 0 --skip-saved --modes cpu-first,gpu-first \
  --out bench-logs/tandem-cache/context-9b-8k.json
```

## Limits

- One 9B pack, one RTX 4060 Ti, one i7-8086K, one run per placement at each of 512, 4,096 and
  8,192 tokens.
- The CPU engine load dominates the wall time at about 107-108 s; this is not a throughput claim.
  The checkpoint-only companion measured device-wide sampled peak VRAM, not a generation-path or
  CUDA allocator high-water value.
- The run covers the joint checkpoint, end-state reuse, saved-opening restore, the non-stream HTTP
  save/restart/reload path, and 4K/8K exactness. Saved-opening reload at 4K/8K, server-streaming
  metrics, 4K/8K peak VRAM, and a 27B-sized split remain unmeasured.
- No inference source changed for this run. The harness and the prior 2B cache fix are the only
  relevant code under `bench-logs/tandem-cache/` and `kindi/cache.py`.

Evidence: `results-9b-cpu.json`, `results-9b-gpu.json`, `results-9b-cpu.log`,
`results-9b-gpu.log`, `server-results-9b.json`, `server-results-9b.md`, `server-results-9b.log`,
`checkpoint-cost-9b-512.json`, `checkpoint-cost-9b-512.log`, `context-9b-4k.json`,
`context-9b-4k.log`, `context-9b-8k.json`, `context-9b-8k.log`, `server-runs-9b/` and the three
harnesses in this directory.
