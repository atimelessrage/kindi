# Real 2B tandem joint-cache check

Status: **PASS, 2026-11-12.** Both dense tandem placements produced token-identical greedy
continuations with the joint cache on and with `KINDI_PROMPT_CACHE=0`, including a saved opening
loaded from disk by a new runner and by a restarted HTTP server. The same harness also passed at
4K and 8K context with 1,024- and 2,048-token checkpoint spacing; see
[LONG-CONTEXT-RESULTS.md](LONG-CONTEXT-RESULTS.md).

This is the real-engine counterpart to the model-free `tests/test_tandem.py` suite. It exercises
`TandemRunner` over the actual 2B CPU and GPU owners, including a checkpoint that contains both
owners' recurrent state. K/V remains in the two engines; saved openings are the path that also
stores K/V. The 9B pack passed the same check separately; see [RESULTS-9B.md](RESULTS-9B.md).

## Environment

| item | value |
|---|---|
| host | native Windows, i7-8086K |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8,188 MiB |
| model | Ternary-Bonsai-2-2B pack |
| context | 512 tokens |
| CPU layers | 6 per tandem placement |
| CPU engine | `avx2-fma-f16c`, 6 threads |
| compiler | clang 18.1.8, libomp from the same LLVM install |
| cache mode | in-process checkpoints and disk openings; `prefix_max=1`, `prefix_min=256`, `park_max=0` |

The harness is `bench-logs/tandem-cache/run_check.py`. The exact command was:

```sh
python bench-logs/tandem-cache/run_check.py \
  --pack "$HOME/.kindi/packs/bonsai-2b" --ctx 512 --cpu-layers 6 --tokens 8 \
  --out bench-logs/tandem-cache/results.json
```

## What was compared

The tokenized prompt `T1` is 320 tokens. Each placement ran these cases:

- **extension**: `T1` plus a short suffix. The cache-on request resumes from the state at the end
  of the preceding 320-token request.
- **branch**: `T1[:256]` plus a different suffix. The cache-on request restores the joint
  checkpoint at token 256, then reads the remaining suffix. The branch request also writes the
  256-token opening to disk.
- **saved opening**: after the branch, a **new** `TandemRunner` opens the same state directory and
  runs the branch prompt. It must load the on-disk opening; the in-memory checkpoint in the old
  runner is not shared with the new one.

For the cache-on pass, the runner first processed `T1`, then ran the continuation. For the
cache-off pass, `KINDI_PROMPT_CACHE=0` reset the engine and read the complete continuation prompt.
Eight greedy tokens were produced from the same continuation prompt in both cases. Token IDs, not
decoded text, are compared.

The 2B runner chose 256-token checkpoints under the 512 MB host budget, so the branch point is 256.
A branch at 192 does not exercise a checkpoint and would resume at 0.

## Results

| placement | case | cache-on reused | cache-off reused | exact | cache-on seconds | cache-off seconds |
|---|---|---:|---:|---|---:|---:|
| cpu-first | extension | 320 | 0 | **yes** | 0.200 | 4.506 |
| cpu-first | branch | 256 | 0 | **yes** | 0.186 | 3.534 |
| cpu-first | saved opening (new runner) | 256 | 0 | **yes** | 0.249 | 3.678 |
| gpu-first | extension | 320 | 0 | **yes** | 0.222 | 4.514 |
| gpu-first | branch | 256 | 0 | **yes** | 0.235 | 3.830 |
| gpu-first | saved opening (new runner) | 256 | 0 | **yes** | 0.313 | 3.618 |

Generated greedy IDs:

| placement | case | IDs |
|---|---|---|
| cpu-first | extension | `[279, 3299, 328, 760, 3158, 271, 760, 1727]` |
| cpu-first | branch | `[279, 3299, 359, 32, 4282, 561, 3841, 13477]` |
| cpu-first | saved opening | `[279, 3299, 359, 32, 4282, 561, 3841, 13477]` |
| gpu-first | extension | `[279, 3299, 328, 760, 3158, 271, 760, 1727]` |
| gpu-first | branch | `[279, 3299, 359, 32, 4282, 561, 3841, 13477]` |
| gpu-first | saved opening | `[279, 3299, 359, 32, 4282, 561, 3841, 13477]` |

Both placements also happened to produce the same IDs in this run. The required result is the
within-placement cache-on/cache-off equality; no cross-card performance claim is made.

The cache-on times include the prompt suffix read and eight generated tokens. The cache-off times
include reading the complete 325-token or 261-token prompt on the CPU/GPU split. These are one-run
harness timings, not a speed claim or a benchmark of the split.

## HTTP server path (2026-11-12)

The server harness starts a real `kindi serve` process for each placement, sends a warm system
prompt and a branch request to write the opening, stops the server, and starts a new server over the
same state directory. The reload request must load the opening from disk. A third server, with a
fresh state directory and `KINDI_PROMPT_CACHE=0`, generates the reference reply.

| placement | opening end | warm branch cached | reload cached | cache-off cached | reply | exact |
|---|---:|---:|---:|---:|---|---|
| `cpu-first` | 344 | 344 | **344** | 0 | `A third color is **purple**.` | **yes** |
| `gpu-first` | 344 | 344 | **344** | 0 | `A third color is **purple**.` | **yes** |

The prompt is 362 tokens; the server-side `usage.prompt_tokens_details.cached_tokens` is 344 after
the restart, while the cache-off server reports 0. Both server logs contain
`saved an opening of 344 tokens` and `loaded a saved opening of 344 tokens`; the on-disk
`meta.json` records `p: 344` for both placements. The replies and all three compared fields
(content, finish reason, completion-token count) are exact. Save and load each took about 0.1 s.

This run found and fixed one real defect: `Openings.load` assigned the NumPy K/V arrays into CuPy
views with `eng.Kc[i][:p] = ...`, which CuPy treated as a scalar fill and rejected
(`non-scalar numpy.ndarray cannot be used for fill`). It now uses the existing `_to_dev` copy
helper, so the CPU and GPU owners' K/V both restore in place. Before the fix the saved opening was
discarded and the request silently fell back to reading from token 0; the cache-on count was 0.

## Longer contexts (2026-11-12)

With the checkpoint interval set explicitly, both placements passed the same equality check at
4,096 and 8,192 context. At 4K, extension reused 4,080 tokens and the branch restored 3,072; at 8K,
extension reused 8,176 and the branch restored 6,144. The checkpoint logs reported 1,024-token and
2,048-token spacing, and every cached token sequence matched cache-off. Saved openings were
intentionally skipped in these four runs; the 512-context saved-opening result above still stands.
Full commands, timings and IDs: [LONG-CONTEXT-RESULTS.md](LONG-CONTEXT-RESULTS.md).

## Limits

- The saved-opening test uses a 256-token in-process opening and a 344-token server opening at a
  512-token context. It is not a long-context or 27B/VRAM gate. The 4K/8K runs exercise in-process
  checkpoints only; a saved-opening reload at those contexts remains open.
- The timings are one-run harness timings. They are not a speed claim or a benchmark of the split.
- No inference source was changed except the CuPy K/V restore path above; the harness is external
  and the CPU library was built on demand from the existing `kindi/kcpu.c`.

Prefix-reuse and checkpoint-cost measurements are in [PREFIX-RESULTS.md](PREFIX-RESULTS.md).

Evidence: `results.json`, `server-results-cpu.json`, `server-results-gpu.json`, the server logs under
`server-runs/`, `run_check.py`, and `run_server_check.py` in this directory.
