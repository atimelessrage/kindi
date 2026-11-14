# 27B + 2B drafter at 24K on the 8 GB card (2026-11-14)

Host: i7-8086K + RTX 4060 Ti (8,188 MiB), driver 617.42, tensor-core plans (`KINDI_TC=1`).
Pack: `T:/kindi-packs/bonsai-27b` (source sha256 `53107f53...`, the pinned 27B), drafter pack
`~/.kindi/packs/bonsai-2b`. One run per configuration (no repeats); `kindi bench` times the exact
head only, so the fit comparison below comes from `bench` and the speed comparison from the server
(requests: greedy, `max_tokens: 200`, prompt 5,012 tokens, second request after the prompt-cache
warmup; both completed 136 tokens).

## Fit: the drafter does not hold 24K on 8 GB

| | no speculation | 2B drafter (`KINDI_SPEC=draft2b`) |
|---|---|---|
| requested ctx | 24,576 | 24,576 |
| achieved ctx | 24,576 (unreduced) | **12,032 (reduced)** |
| free after load | 323 MiB | 115 MiB |
| bench gen at 24K | 33.9 tok/s | not measurable (no 24K cell) |

The drafter (2B weights + plans + draft K/V) costs ~1.6 GB: with it, the loader fits only 12,032
tokens. So at a 24K context on 8 GB you choose: 24K without the drafter, or 12K with it.

## Speed where it fits: still 1.5x

| context (prompt) | no speculation | 2B drafter | gain |
|---|---|---|---|
| 5,012 tokens, 136 completed | 5.00 s (27.2 tok/s) | 3.25 s (41.8 tok/s) | **1.54x** |

Consistent with the 8K row in `docs/MTP.md` (1.56x). Ending: the 12K fit makes 24K-with-drafter a
tandem-slice question, not a loader question (see `docs/TANDEM-RUNTIME.md`).

Files: `bench-base.json`, `bench-draft2b.json` (`kindi bench --json`, both runs).
