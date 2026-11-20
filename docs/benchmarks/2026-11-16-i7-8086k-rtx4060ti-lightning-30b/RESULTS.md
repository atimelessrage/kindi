# Nemotron Lightning 30B-A3B MoE gate (RTX 4060 Ti, 2026-11-16)

Host: i7-8086K + RTX 4060 Ti 8 GB (8,188 MiB). Model: Nemotron 3.5
Lightning 30B-A3B (Nemotron-H MoE, `nemotron_h_moe`, Q2_0).

The earlier local copy was discarded: its size matched the catalog
(9,472,656,320 bytes) but the catalog/local/manifest SHA-256 values
disagreed and the GGUF reader rejected it. A clean re-download to
`<kindi-model-root>/kindi-lightning-clean/models/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf`
now checks out exactly:

- size: `9472656320` bytes
- SHA-256: `38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3`
- magic: `GGUF`
- `kindi.gguf.GGUF`: arch `nemotron_h_moe`, name `Nemotron-3.5-Lightning-30B-A3B Nota-Ternary (MLX 2-bit) -> Q2_0`, 401 tensors

Fresh pack (manifest only; weights stay in the GGUF):
`<kindi-model-root>/kindi-lightning-clean/packs/nemotron-lightning-30b`.

Commands:

```powershell
python -m kindi convert <kindi-model-root>/kindi-lightning-clean/models/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf --out <kindi-model-root>/kindi-lightning-clean/packs/nemotron-lightning-30b --force
python -m kindi --pack <kindi-model-root>/kindi-lightning-clean/packs/nemotron-lightning-30b plan --ctx 8192
python -m kindi --pack <kindi-model-root>/kindi-lightning-clean/packs/nemotron-lightning-30b verify
python -m kindi --pack <kindi-model-root>/kindi-lightning-clean/packs/nemotron-lightning-30b --ctx 8192 bench --sample-vram
```

## Plan

At ctx 8192 the planner chose the MoE expert-tier strategy: dense core and
shared expert on the card; routed experts tiered card / pinned host / mmap,
with the last-used dcache (default 10 slots/layer) in device slots.
Estimate: ~5.57 GiB on the card at ctx 8192 (5,706 MiB of 7,075 MiB free);
70 to 106 of 128 experts per layer on the card (85.5% of measured picks).
The host keeps the whole expert set + dense core (~8.5 GiB) readable.

## Verify

The tokenizer and chat template matched llama.cpp. Load on device 0
(sm_89) took 13.5 s at ctx 33024 with 66 of 128 experts per layer on the
card.

| prompt | result |
|---|---|
| The capital of France is | PASS, 3 of 3 tokens agree |
| def fibonacci(n): | PASS, 64 of 64 |
| The sky is blue because | TIE at 14 of 64 (top-2 gap 0.0851) |
| Once upon a time, in a small village by the ... | TIE at 16 of 64 (gap 0.0545) |
| 1, 1, 2, 3, 5, 8, | PASS, 64 of 64 |
| chat: What is 17 times 23? | TIE at 0 of 64 (gap 0.1216; `llama.cpp -ub 1` gives Kindi's token) |
| chat: Write a Python function that reverses ... | PASS, 64 of 64 |
| chat: List three uses of the ls command. | TIE at 34 of 64 (gap 0.1008; `llama.cpp -ub 1` gives Kindi's token) |
| long 512 | PASS, 32 of 32 |
| long 2048 | TIE at 6 of 32 (gap 0.0444) |
| long 8192 | TIE at 0 of 32 (gap 0.1699) |
| long 32768 | TIE at 2 of 32 (gap 0.0426) |

Verdict: **PASS: 5 of 12 prompts token-exact, 7 end at a llama.cpp tie
(noise floor 0.217), 0 fail (74 s)**.

## Bench

At ctx 8192 the engine loaded in 12.7–13.1 s with 68–74 of 128 experts per
layer on the card.

| context | read | write |
|---|---|---|
| 512 | 788.3–821.3 tok/s | 165.96–167.41 tok/s at position 528 |

The `--sample-vram` bench run did not print a separate sampled-peak line in
this stdout capture, so no sampled-peak number is claimed here; the planner
estimate above is the VRAM record for this gate.

## 8K / 32K follow-up (2026-11-20): blocked

No `kindi_lightning` pack or Lightning GGUF existed on this host. `kindi models`
listed `lightning-30b` with no presence label, and `~/.kindi/packs` contained
only Bonsai packs. The two intended runs were:

```bash
KINDI_PACK=~/.kindi/packs/nemotron-lightning-30b KINDI_DEVICE=gpu \
  python -m kindi --ctx 8288 bench --sample-vram
KINDI_PACK=~/.kindi/packs/nemotron-lightning-30b KINDI_DEVICE=gpu \
  python -m kindi --ctx 32864 bench --sample-vram
```

Both exited 1 with `FileNotFoundError` for
`.../nemotron-lightning-30b/kindi_pack.json`, before any weights loaded. No 8K
or 32K prefill, decode or peak-VRAM number is claimed.

Two details for the eventual run:

- The 2026-11-16 bench used ctx 8192, so it could not produce the 8K row:
  `lightning_cli.bench` includes a length N only when `N + 96 <= eng.ctx`.
  Use `--ctx 8288` or more for 8K and `--ctx 32864` or more for 32K.
- `kindi bench` dispatches a `kindi_lightning` pack to `lightning_cli.bench`
  before creating its `VramMonitor`; `--sample-vram` does not currently print
  a peak for this backend. A peak-VRAM record needs an external `nvidia-smi`
  sampler or a bench change.

The catalog artifact is `9,472,656,320` bytes, SHA-256
`38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3`;
downloading it is an owner action, not part of this blocked run.

## Scope of the verdict

This closes the Lightning MoE GPU gate on the 4060 Ti for the clean catalog
artifact: the model loads the 8 GB card through expert tiers, the
chat-template ids match llama.cpp, and no prompt fails. It does not claim
parity at other contexts, mixed-card behavior, or Nanbeige/Gemma/DeepSeek
gates, which remain blocked on their own artifacts.
