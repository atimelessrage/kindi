# 9B default 5-trit vs tensor-core (`KINDI_TC_HP=1`) smoke, RTX 4060 Ti

Single-run smoke on 2026-11-20: token exactness at `kindi verify`'s fixed 1,024-token
context, then decode and memory at 8,192 and 24,576 tokens for the default plan-engine
lane and the tensor-core path with high-precision activations. This is **not a benchmark
claim**: one run per cell, no medians, no repeats, no clock or power pinning.

The 9B pack is folded with Hadamard block 4096. The default path leaves `KINDI_TC` off
and runs the plan engine's 5-trit products; the tensor-core path needs
`KINDI_TC=1` and `KINDI_TC_HP=1`.

## Host

| | |
|---|---|
| Host | Windows 10 Pro 19045 |
| CPU / RAM | Intel Core i7-8086K, 6 physical / 12 logical cores, 32 GB |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8,188 MiB, driver 617.42 |
| Python / CuPy / NumPy | 3.13.6 / 14.2.0 / 2.5.1 |
| Pack | `~/.kindi/packs/bonsai-9b`: Ternary-Bonsai-2-9B, `source_sha256` `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b`, Hadamard block 4096 |
| K/V | q8_0 (host config; the verify log records it) |
| Tree | `ee2ba46`; the bench runs preceded the diagnostic-label-only change and the verify runs followed it. No kernel or model change between the runs |
| Other load | the desktop held roughly 1.0–1.1 GiB of the card; the sampled peaks are device-wide |

## Commands

The host config selected the 9B pack. The exact verify commands were:

```sh
# default path (`KINDI_TP` and `KINDI_TC` unset)
python -m kindi verify

# tensor-core path with high-precision activations
KINDI_TP=1 KINDI_TC=1 KINDI_TC_HP=1 KINDI_FD=1 KINDI_ATT_TC=1 \
KINDI_TC_TMAX=4 KINDI_SPEC=0 KINDI_HEAD=exact \
python -m kindi verify
```

The exact bench commands were:

```sh
python -m kindi --ctx 8192 bench --split-at none --sample-vram \
  --json bench-logs/9b-smoke-2026-11-20/default-8192.json

KINDI_TP=1 KINDI_TC=1 KINDI_TC_HP=1 KINDI_FD=1 KINDI_ATT_TC=1 \
KINDI_TC_TMAX=4 KINDI_SPEC=0 KINDI_HEAD=exact \
python -m kindi --ctx 8192 bench --split-at none --sample-vram \
  --json bench-logs/9b-smoke-2026-11-20/tc-hp-8192.json

python -m kindi --ctx 24576 bench --split-at none --sample-vram \
  --json bench-logs/9b-smoke-2026-11-20/default-24576.json

KINDI_TP=1 KINDI_TC=1 KINDI_TC_HP=1 KINDI_FD=1 KINDI_ATT_TC=1 \
KINDI_TC_TMAX=4 KINDI_SPEC=0 KINDI_HEAD=exact \
python -m kindi --ctx 24576 bench --split-at none --sample-vram \
  --json bench-logs/9b-smoke-2026-11-20/tc-hp-24576.json
```

The JSON and logs were written under the gitignored `bench-logs/9b-smoke-2026-11-20/`.
This page is the tracked record; the tables below are transcribed from those runs.

## Verify

`kindi verify` uses a fixed 1,024-token context and four reference prompts.

| Path | Exactness | tokens/s over the run |
|---|---|---:|
| Default 5-trit plan lane | **4/4 PASS** (40/40, 40/40, 13/13, 40/40) | 95.7 |
| Tensor-core + `KINDI_TC_HP=1` | **4/4 PASS** (40/40, 40/40, 13/13, 40/40) | 119.5 |

Both runs exited 0. In this run the tensor-core verify was **1.25x** the default path.

The default run logged `plan engine ready: 1 card, ctx 1024, K/V q8_0, weights 5-trit
lane` and `speculation: unavailable (this engine has no spec_begin); verifying the plan
engine's 5-trit lane`; it did not log a high-precision-activation line. The tensor-core
run logged one `card 0: 4096 Hadamard blocks run with high-precision activations
(KINDI_TC_HP=1)` line and `tensor-core plans ready: T = 1 .. 4, 493 nodes per decode
token`.

## Bench

All four runs exited 0. `ctx` equals `ctx_requested` and `ctx_reduced` is `false` in every
JSON: **no OOM and no context reduction**. The default path is the plan engine's 5-trit
lane; the tensor-core path is the documented `KINDI_TC=1` / `KINDI_TC_HP=1` configuration.

| Context | Path | Read | Decode @100 | @2,000 | @8,000 | @16,000 | @24,000 | Free after load | Sampled peak used / free |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | default 5-trit | 102.3 | 88.9 | 90.6 | 86.1 | — | — | 5,089 MiB | 3,139 / 4,811 MiB |
| 8,192 | TC + HP | 132.8 | 113.4 | 110.6 | 108.0 | — | — | 5,077 MiB | 3,155 / 4,795 MiB |
| 24,576 | default 5-trit | 102.3 | 91.9 | 89.6 | 86.4 | 81.0 | 76.6 | 4,809 MiB | 3,465 / 4,485 MiB |
| 24,576 | TC + HP | 133.5 | 114.4 | 110.8 | 107.7 | 102.1 | 95.8 | 4,785 MiB | 3,447 / 4,503 MiB |

All rates are tokens/s. In this single run the tensor-core path was 1.22–1.30x the
default path across the measured cells. The sampled peak is device-wide from
`nvidia-smi` and includes the desktop; “free after load” is CuPy's free device memory
after the engine loaded.

## Single-run caveat and earlier observations

- Every row is one fresh process. There are no repeats or medians, and no GPU clock or
  power setting was pinned. The 8K default position-100 value being lower than the 24K
  one is run-to-run variation, not a context trend.
- An earlier 8K short run without `--sample-vram` measured 105.7 / 133.8 tokens/s for
  prompt reading and 92.5 / 114.7 at position 100 (default / TC+HP). It is not averaged
  into the table.
- An earlier verify before the diagnostic-label fix measured 89.6 / 116.5 tokens/s
  (default / TC+HP). The table uses the later post-fix verify; the labels changed, not
  the kernels.
- Exactness was checked only at verify's 1,024-token context. No long-context token
  comparison was run; these 8K/24K runs are decode and memory smokes.
- One card, one pack, one host. No 27B, no other card, no concurrency and no long soak.
