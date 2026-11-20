# RTX 4060 Ti verify_matrix record (folded into `card-results.json`)

Source run: `bench-logs/verify-matrix/results.json` (2026-11-10, verify_matrix plain + tc modes,
2B/4B/9B), copied here because the registry link must be tracked (runbook: `docs/OLD-CARD-RIG.md`).
Folded with `python -m kindi.support_matrix --from-verify docs/benchmarks/2026-11-10-i7-8086k-rtx4060ti/results.json --card rtx-4060-ti`.
The registry holds one source run per card; this dual-mode run is the canonical record. Cells it
produced: 2B plain 4/4 @ 193.4 tok/s · tc 4/4 @ 382.7; 4B plain 4/4 @ 123.1 · tc failed (first
diff 6); 9B plain 4/4 @ 79.4 · tc refused; 27B not run **in this run** (later scoped runs: the 27B
TC path is 3 exact + 1 llama.cpp tie at 1,024 tokens — see `SUPPORT-MATRIX.md` Open and STATUS —
and the 9B `tc-hp` run of 2026-11-14 passes 4/4 in `bench-logs/verify-matrix-9b-tc-hp/`, local
evidence). Those supersessions live in the curated STATUS/Open text; this file is the folded record.
