# 2B tandem cache at 4K and 8K context

Status: **PASS, 2026-11-12.** Both dense tandem placements produced token-identical greedy
continuations with the joint cache on and with `KINDI_PROMPT_CACHE=0` at 4,096- and 8,192-token
contexts, using 1,024- and 2,048-token checkpoint intervals.

This extends [RESULTS.md](RESULTS.md) beyond the 512-token saved-opening/server checks. The
long-context runs use the same `run_check.py` harness with explicit prompt and checkpoint sizes;
`saved openings` are disabled here (`prefix_max=0`) so the run isolates the longer in-process
checkpoint interval. The 512-token saved-opening and HTTP restart checks remain in
[RESULTS.md](RESULTS.md).

## Environment

| item | value |
|---|---|
| host | native Windows, i7-8086K |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8,187 MiB |
| model | Ternary-Bonsai-2-2B |
| contexts | 4,096 and 8,192 tokens |
| CPU layers | 6 per placement |
| CPU engine | `avx2-fma-f16c`, 6 threads |
| compiler | clang 18.1.8 with libomp |
| cache | `park_max=0`, `prefix_max=0`, `checkpoint_budget_mb=512` |
| tokens | 8 greedy continuation tokens |

## Workload

For context `C`, `T1` is `C - 16` tokens and the extension prompt is `T1` plus a 5-token suffix
(`C - 11` tokens). The branch prompt is `T1[:3C/4]` plus the same 5-token suffix. This leaves room
for the eight generated tokens and puts the branch point exactly on a checkpoint.

| context | checkpoint interval | T1 | branch point | extension prompt | branch prompt |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,024 | 4,080 | 3,072 | 4,085 | 3,077 |
| 8,192 | 2,048 | 8,176 | 6,144 | 8,181 | 6,149 |

The harness first prepared `T1`, ran the cached extension, restored `T1`, and ran the cached
branch. It then ran both prompts with `KINDI_PROMPT_CACHE=0`, which resets the engine and reads the
complete prompt. The cached and cache-off output token IDs were compared.

Commands:

```sh
python bench-logs/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-2b" --ctx 4096 --cpu-layers 6 --tokens 8 \
  --t1-tokens 4080 --branch-at 3072 --checkpoint-every 1024 \
  --prefix-max 0 --skip-saved --modes cpu-first \
  --out bench-logs/tandem-cache/context-4k-cpu.json

python bench-logs/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-2b" --ctx 4096 --cpu-layers 6 --tokens 8 \
  --t1-tokens 4080 --branch-at 3072 --checkpoint-every 1024 \
  --prefix-max 0 --skip-saved --modes gpu-first \
  --out bench-logs/tandem-cache/context-4k-gpu.json

python bench-logs/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-2b" --ctx 8192 --cpu-layers 6 --tokens 8 \
  --t1-tokens 8176 --branch-at 6144 --checkpoint-every 2048 \
  --prefix-max 0 --skip-saved --modes cpu-first \
  --out bench-logs/tandem-cache/context-8k-cpu.json

python bench-logs/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-2b" --ctx 8192 --cpu-layers 6 --tokens 8 \
  --t1-tokens 8176 --branch-at 6144 --checkpoint-every 2048 \
  --prefix-max 0 --skip-saved --modes gpu-first \
  --out bench-logs/tandem-cache/context-8k-gpu.json
```

## Results

| context | placement | case | cache-on reused | cache-off reused | exact | cache-on seconds | cache-off seconds |
|---:|---|---|---:|---:|---|---:|---:|
| 4,096 | `cpu-first` | extension | 4,080 | 0 | **yes** | 0.159 | 46.060 |
| 4,096 | `cpu-first` | branch | 3,072 | 0 | **yes** | 0.162 | 34.743 |
| 4,096 | `gpu-first` | extension | 4,080 | 0 | **yes** | 0.219 | 61.850 |
| 4,096 | `gpu-first` | branch | 3,072 | 0 | **yes** | 0.260 | 36.705 |
| 8,192 | `cpu-first` | extension | 8,176 | 0 | **yes** | 0.186 | 100.952 |
| 8,192 | `cpu-first` | branch | 6,144 | 0 | **yes** | 0.185 | 72.886 |
| 8,192 | `gpu-first` | extension | 8,176 | 0 | **yes** | 0.403 | 123.005 |
| 8,192 | `gpu-first` | branch | 6,144 | 0 | **yes** | 0.284 | 85.043 |

Generated IDs (identical for every row in a case):

| case | IDs |
|---|---|
| extension | `[279, 3299, 328, 760, 3158, 561, 1727, 11316]` |
| branch | `[279, 3841, 13477, 37550, 33075, 888, 279, 15217]` |

The runtime logs confirm the intended spacing:

```
prompt cache: a checkpoint every 1024 tokens ...
cache: 4080 prompt tokens, 4080 shared with the last request, resumed at 3072
cache: 3077 prompt tokens, 3072 shared with the last request, resumed at 3072
```

```
prompt cache: a checkpoint every 2048 tokens ...
cache: 8176 prompt tokens, 8176 shared with the last request, resumed at 6144
cache: 6149 prompt tokens, 6144 shared with the last request, resumed at 6144
```

The cache-off prompts show `resumed at 0` and `cached_tokens: 0`. The extension cache-on times are
one-run harness timings for reading only the suffix and generating eight tokens; they are not a
throughput or speed claim. Joint checkpoint save/restore cost at these contexts is in
[PREFIX-RESULTS.md](PREFIX-RESULTS.md#4k8k-joint-checkpoint-cost).

## Limits

- One 2B pack, one RTX 4060 Ti, one i7-8086K and one run per cell. These are equality checks, not a
  long-context benchmark.
- Saved openings are disabled in these four runs. The 256-token in-process opening and the
  344-token HTTP restart opening are covered at 512 context in [RESULTS.md](RESULTS.md); a
  saved-opening reload at 4K/8K remains open.
- The engine is the dense Bonsai tandem path. No 27B or multi-GPU claim is made.
- The checkpoint interval was intentionally set to 1,024/2,048 tokens. The default auto-spacing
  (256/512 tokens at these contexts) was not separately re-run for this table.

Evidence: `context-4k-cpu.json`, `context-4k-gpu.json`, `context-8k-cpu.json`,
`context-8k-gpu.json`; the 8K runs also have matching `.log` files.
