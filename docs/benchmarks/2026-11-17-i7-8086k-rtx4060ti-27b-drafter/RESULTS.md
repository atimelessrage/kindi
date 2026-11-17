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
own files. The first pass recorded one run per cell; three later passes repeat the
three speculation cells (and the base cell), and the table reports those three
passes as medians with their range.

## Result

| cell | status | exactness | tokens/s | requested -> effective | drafts accepted |
|---|---|---|---:|---|---:|
| `27b tc` | pass | 3 exact + 1 tie | 38.7 (38.7-38.8) | — | — |
| `27b tc-ngram` | pass | 3 exact + 1 tie | 45.7 (44.5-46.7) | `ngram -> ngram` | 35/148 |
| `27b tc-draft2b` | pass | 3 exact + 1 tie | 71.0 (70.7-71.1) | `draft2b -> draft2b` | 112/162 |
| `27b tc-both` | pass | 3 exact + 1 tie | 62.0 (61.3-62.3) | `both -> both` | 90/211 |

All four cells have the same reference verdict: prompts 2-4 are 40/40 token-exact, and prompt 1
("The sky is blue because") ends at the documented llama.cpp tie after 14 tokens with a top-2 gap
of 0.0064. That tie is present in the no-speculation cell too, so none of the three speculation
modes introduced a new token difference or a failure. This is exact under the same accepted-tie
rule as the existing 27B record, not a claim that the tied prompt is 40/40.

The drafter and its head attached in every relevant cell (`draft2b`/`both`), and each requested
mode was the effective mode; a fallback would have been scored `refused` rather than `pass`. The
throughput column is the median of three passes, with the three-pass min-max in parentheses; the
first pass's 41.5/49.8/75.7/67.1 were one run each and are superseded. On the medians `draft2b` is
1.84x the no-speculation cell, `both` 1.60x and `ngram` 1.18x. The single-pass numbers overstated
the three speculation cells by 6.6-9.0%, so the medians are the record. The
fit also matters: with the 2B drafter loaded, the 27B plan engine had 663 MiB free at the 1,024-token
verification context; the earlier 24K measurement still records the drafter reducing the 27B to a
12,032-token context, so this does not claim a 24K drafter configuration.

## Baseline gate

`verify-baseline.json` holds this run's three speculation cells in the matrix's
`results.json` shape. Unlike the 2B baseline it is tracked, because these are the
only 27B drafter cells and a clean checkout should be able to gate against them.

```sh
python tools/verify_matrix.py \
  --models 27b --modes tc \
  --spec-models 27b --spec-modes ngram,draft2b,both \
  --pack-dir ~/.kindi/packs \
  --pack-27b D:/kindi-27b/packs/bonsai-27b \
  --spec-pack ~/.kindi/packs/bonsai-2b \
  --mtp-head ~/.kindi/models/kindi-mtp-2b-v1.npz \
  --out bench-logs/verify-matrix/27b-spec-gate \
  --no-serve --no-bench \
  --baseline docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-27b-drafter/verify-baseline.json \
  --fail-on-regression --max-tps-drop-pct 10
```

The run also executes `27b tc` because the speculation cells build on a base
mode; the baseline omits that cell, so it prints as `new` and is not
throughput-gated. A failing `27b tc` still fails the run's own exit status. The
gate exits 1 when a saved cell is not `pass`, loses an exact prompt, raises its
fail count, disappears (for example a missing drafter pack or head, or a
narrowed `--spec-modes`), loses its throughput number, or falls more than 10%
below the saved tok/s. New cells and faster runs pass. The per-cell status,
exactness and signed throughput deltas are written to `comparison.csv` and
`comparison.json` under `--out`.

Run against the saved baseline on 2026-11-17, the command passed: all three
cells stayed `3/1/0` and requested their mode, at 48.3/73.3/64.7 tok/s
(-3.0%, -3.2% and -3.6% against 49.8/75.7/67.1), with no missing cells.

The baseline keeps that first pass's one-run values. The three-pass medians are
6.6-9.0% below them, inside the 10% limit, so a typical run passes with less
headroom than the first-pass figures suggest; a regression gate at 10% therefore
still separates a real drop from the spread observed here.

## Artifacts

- `results.json` — the full matrix record, including requested/effective modes, draft counts,
  per-prompt verdicts, config, card/driver and pack hash.
- `results.csv` — the same rows in the matrix's flat format.
- `verify-baseline.json` — the three speculation cells as a tracked verify-matrix
  baseline; the exact gate command is above.
- `median-r{1,2,3}-results.json` — the three repeat matrices behind the table's
  medians (all four cells each, with config, packs and draft counts). Raw logs
  stayed in `bench-logs/verify-matrix/27b-spec-median/`.
- `logs/27b-tc*.log` — the four raw verifier logs, including the tie location and attached files.

## Not verified

Three passes per cell for the reported medians, on one card and one pack. The 27B's accepted
tie remains, so this is not a 4/4-token-exact claim. `dspine` was not run because its weights are
absent, and no 2B/4B/9B drafter pairing or server/cache cycle was measured here.
