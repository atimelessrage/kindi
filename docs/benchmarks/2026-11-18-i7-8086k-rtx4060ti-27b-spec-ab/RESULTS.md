# 27B speculation A/B: plain output versus ngram, draft2b and both (2026-11-18)

Host: i7-8086K + RTX 4060 Ti (8,188 MiB), driver 617.42, Windows, CuPy 14.2.0, CUDA runtime
12090. Target: `bonsai-27b` on `D:/kindi-27b/packs/bonsai-27b`, source sha256
`53107f53...`, Hadamard block 1024. Drafter: `~/.kindi/packs/bonsai-2b`, source sha256
`36061ad4...`, plus the registered `kindi-mtp-2b-v1.npz` head (sha256 `1b5c0c79...`).

This is a direct comparison, not a comparison of two reference verdicts. One 27B engine is built
with the 2B drafter and its head attached; for each reference prompt the harness generates the
plain output and the `ngram`, `draft2b` and `both` outputs from the same engine, prompt and greedy
settings, then compares the generated token ids and the decoded output bytes.

## Command

```sh
python tools/spec_ab.py \
  --pack D:/kindi-27b/packs/bonsai-27b \
  --drafter ~/.kindi/packs/bonsai-2b \
  --head ~/.kindi/models/kindi-mtp-2b-v1.npz \
  --json docs/benchmarks/2026-11-18-i7-8086k-rtx4060ti-27b-spec-ab/results.json
```

The engine is the tensor-core plan path (`KINDI_TC=1`, `KINDI_TP=1`, `KINDI_FD=1`,
`KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4`) at the verifier's 1,024-token context. The harness sets those
switches itself and builds with both drafter modes attachable (`KINDI_SPEC=both`,
`KINDI_SPEC_PACK=<2B pack>`, `KINDI_SPEC_DRAFT_HEAD=<head>`), then switches only `KINDI_SPEC`
between generations.

## Result

All 12 prompt/mode pairs are identical: 4 prompts x 3 modes, 40 greedy tokens per output.

| prompt | plain bytes | ngram | draft2b | both | plain byte sha256 |
|---|---:|---|---|---|---|
| `The sky is blue because` | 216 | identical | identical | identical | `eefabc4d325d...` |
| `The capital of France is` | 164 | identical | identical | identical | `fb566ea206be...` |
| `Water boils at a lower temperature on mountains because` | 192 | identical | identical | identical | `e25f475ef959...` |
| `def add_two_numbers(` | 110 | identical | identical | identical | `aa543290e3ec...` |

The output was also token-identical, not only byte-identical: each mode's token-id list and decoded
byte stream both equal the plain run's. The JSON records both hashes, the effective mode,
drafted/accepted counts and first-mismatch fields for every pair. `ngram` drafted 30-42 tokens per
prompt, `draft2b` 45-54 and `both` 46-60; those statistics are incidental to this comparison and
are not a throughput result.

The direct comparison exited 0 and wrote
[results.json](results.json). Run log: `bench-logs/spec-ab-27b.log` (gitignored).

## Not verified

One pass per prompt and mode, one card, one pack, the 1,024-token verification context, and the
four reference prompts. This does not claim byte equality at 24K, for other sampling settings, or
for the MTP head on the 2B. It closes the direct plain-vs-speculation check for the 27B modes that
have a drafter here.
