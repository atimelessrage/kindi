# Plain-engine q8_0 K/V and the 27B split, re-measured (2026-11-14)

The plain engine (`kindi/engine.py`) can now keep its K/V cache in q8_0 — the plan engine's kernels
and arithmetic (`k_store_q8`, `k_q8row`, `k_attn_sq8t`, `k_attn_vq8w`, `k_attn_combn`), half the memory —
selected by `KINDI_KV=q8_0` or by `Engine(..., kv="q8_0")`. The tandem's plain GPU owner is built with
`kv="q8_0"`, so an offloaded layer frees its weights *and* its K/V under both owners. Standalone the
default stays fp16 (the 2B/4B/9B verify references are fp16), and with q8 the chunked prompt path is
disabled (it writes an fp16 cache), so prompts feed one token per call at the decode rate.

## Exactness

| setup | result |
|---|---|
| 2B plain fp16 (`KINDI_TP=0`) | 4/4, 182 tok/s |
| 2B plain q8 (`KINDI_TP=0 KINDI_KV=q8_0`) | 4/4, 181 tok/s |
| 2B tandem cpu-first 8 layers, plain owner | 4/4, 74 tok/s |
| 2B tandem cpu-first 8 layers, plan owner | 4/4, 87 tok/s |
| 9B plain q8 (`KINDI_TP=0 KINDI_KV=q8_0`) | 4/4, 78 tok/s |
| 9B tandem `gdn:6`, plain owner | 4/4, 29 tok/s |
| 27B tandem `gdn:16`, plain owner | 4/4, 8.0 tok/s |

Against the same references as the fp16 runs (the 27B against llama.cpp's q8_0 cache): q8_0 is
token-exact at verify length on all three models. Longer generations may diverge from an fp16
reference; that is the documented loss.

## The 27B split at 24,576 tokens

`kindi bench --tandem cpu-first --tandem-cpu-layers gdn:16 --tandem-gpu <owner> --sample-vram`,
RTX 4060 Ti 8 GB, i7-8086K, three fresh processes per owner, pack `~/.kindi/packs/bonsai-27b`
(`bench-logs/tandem-plain-q8/`):

| owner | free after load | sampled peak used | peak over idle | read | decode 100 / 2K / 8K / 16K / 24K |
|---|---:|---:|---:|---:|---|
| plain, q8_0 K/V (new) | 995 MiB | 7,130 / 7,110 / 7,098 MiB | 6,216 / 6,118 / 6,124 MiB | 8.7 / 8.0 / 8.3 | 7.5/7.7/8.1/8.4/7.7, 7.1/6.9/6.7/7.0/7.8, 8.0/7.7/8.0/6.9/7.4 |
| plain, fp16 K/V (2026-11-14 plan-owner round) | 197 MiB | 7,721 MiB | ~6,770 MiB | 7.5 | 4.5 at 24K |
| plan (re-run, same day) | 1,985 MiB | 6,091 MiB ×3 | 5,182 / 5,199 / 5,199 MiB | 9.7 / 9.8 / 8.9 | 7.8/7.4/7.6/7.6/7.4, 8.9/8.1/7.7/7.7/7.4, 8.4/7.7/6.5/6.3/7.1 |
| plan (2026-11-14) | 1,985 MiB | 6,004 MiB (5,862-6,083) | ~5,050 MiB | 7.2 | 5.8 at 24K |

The plain owner's peak drops ~600 MiB (~7,721 → ~7,110 MiB); byte accounting expects ~225 MiB of
that (5 attention layers × 24,576 tokens × 4 K/V heads × (512 − 272) bytes), the rest is the
device-wide baseline varying between measurement days (914-992 MiB of other processes here), so the
delta-over-idle column is the honest comparison. The plan owner stays ~1 GiB below the plain owner
(it reads the f32 file per tensor and keeps the embedding in pinned host memory). Decode is inside
run-to-run noise between the owners; both are far below the unsplit plan engine, because the CPU
layers dominate the token.
