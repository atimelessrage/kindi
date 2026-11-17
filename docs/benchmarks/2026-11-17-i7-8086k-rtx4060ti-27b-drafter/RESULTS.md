# 27B verify matrix with the 2B drafter (2026-11-17)

Host: i7-8086K + RTX 4060 Ti (8,188 MiB), driver 617.42, Windows, CuPy 14.2.0, CUDA runtime
12090. Target: `bonsai-27b` on `D:/kindi-27b/packs/bonsai-27b`, source sha256
`53107f53...`, Hadamard block 1024. Drafter: `~/.kindi/packs/bonsai-2b`, source sha256
`36061ad4...`, plus the registered `kindi-mtp-2b-v1.npz` head (sha256 `1b5c0c79...`) for `both`.
The engine is the tensor-core plan path (`KINDI_TC=1`, `KINDI_TP=1`, `KINDI_FD=1`,
`KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4`) at the verifier's fixed 1,024-token context.

## Command

```sh
python tools/verify_matrix.py \
  --models 27b --modes tc \
  --spec-models 27b --spec-modes ngram,draft2b,both \
  --pack-dir ~/.kindi/packs \
  --pack-27b D:/kindi-27b/packs/bonsai-27b \
  --spec-pack ~/.kindi/packs/bonsai-2b \
  --mtp-head ~/.kindi/models/kindi-mtp-2b-v1.npz \
  --out docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-27b-drafter \
  --no-serve --no-bench
```

Each cell is a fresh `kindi verify` process. The four-prompt, 40-greedy-token reference matrix is
the same one used for the recorded 27B cell; the drafter modes add `--spec <mode>` and the mode's
own files. One run per cell, no repeats.

## Result

| cell | status | exactness | tokens/s | requested -> effective | drafts accepted |
|---|---|---|---:|---|---:|
| `27b tc` | pass | 3 exact + 1 tie | 41.5 | — | — |
| `27b tc-ngram` | pass | 3 exact + 1 tie | 49.8 | `ngram -> ngram` | 35/148 |
| `27b tc-draft2b` | pass | 3 exact + 1 tie | 75.7 | `draft2b -> draft2b` | 112/162 |
| `27b tc-both` | pass | 3 exact + 1 tie | 67.1 | `both -> both` | 90/211 |

All four cells have the same reference verdict: prompts 2-4 are 40/40 token-exact, and prompt 1
("The sky is blue because") ends at the documented llama.cpp tie after 14 tokens with a top-2 gap
of 0.0064. That tie is present in the no-speculation cell too, so none of the three speculation
modes introduced a new token difference or a failure. This is exact under the same accepted-tie
rule as the existing 27B record, not a claim that the tied prompt is 40/40.

The drafter and its head attached in every relevant cell (`draft2b`/`both`), and each requested
mode was the effective mode; a fallback would have been scored `refused` rather than `pass`. The
throughput column is one run per cell and is therefore indicative, not a median: `draft2b` was
1.82x the no-speculation cell, `both` 1.62x, and `ngram` 1.20x in this short verify matrix. The
fit also matters: with the 2B drafter loaded, the 27B plan engine had 663 MiB free at the 1,024-token
verification context; the earlier 24K measurement still records the drafter reducing the 27B to a
12,032-token context, so this does not claim a 24K drafter configuration.

## Artifacts

- `results.json` — the full matrix record, including requested/effective modes, draft counts,
  per-prompt verdicts, config, card/driver and pack hash.
- `results.csv` — the same rows in the matrix's flat format.
- `logs/27b-tc*.log` — the four raw verifier logs, including the tie location and attached files.

## Not verified

One run per cell, one card and one pack; no repeated throughput measurement. The 27B's accepted
tie remains, so this is not a 4/4-token-exact claim. `dspine` was not run because its weights are
absent, and no 2B/4B/9B drafter pairing or server/cache cycle was measured here.
