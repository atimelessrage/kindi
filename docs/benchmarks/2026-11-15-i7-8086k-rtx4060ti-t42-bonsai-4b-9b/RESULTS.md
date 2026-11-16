# T4.2 Bonsai adapter: 4B/9B gate extension (RTX 4060 Ti, 2026-11-15)

The T4.2 Bonsai adapter (`kindi/backends/bonsai.py`) was verified only on the
2B (`9f48723`: verify 4/4 on both paths, 103.0/103.2 tok/s). This round
extends the same gate to the other two packs on the host: 4B and 9B.

Host: i7-8086K + RTX 4060 Ti 8 GB (8,188 MiB), ctx 1024 for verify and
24,576 for bench, exact head. Adapter path = default
(`KINDI_LEGACY_ENGINE_PATH` unset, registry resolution);
valve path = `KINDI_LEGACY_ENGINE_PATH=1` (pre-port builder).

## Verify (token-exact vs llama.cpp)

| pack | adapter | valve |
|---|---|---|
| 4B | PASS 3+tie @ 141.0 tok/s | PASS 3+tie @ 143.1 tok/s |
| 9B | ALL PASS 4/4 @ 99.3 tok/s | ALL PASS 4/4 @ 99.6 tok/s |

The 4B tie is the same known one (prompt 3, first difference at token 25,
llama.cpp top-2 gap 0.0012), on both paths. Exactness is identical between
the paths; the tok/s deltas are run-to-run jitter (the adapter builds the
same engine object by construction).

## Bench (exact head, ctx 24576)

Adapter path:

| pack | read tok/s | gen @100 | gen @2K | gen @8K | gen @16K | gen @24K |
|---|---|---|---|---|---|---|
| 4B | 161.0 | 132.9 | 137.4 | 125.6 | 114.1 | 106.8 |
| 9B | 109.2 | 92.6 | 93.8 | 89.5 | 83.5 | 78.7 |

9B valve path (for the jitter band): read 110.7; gen 93.1/94.0/89.8/84.2/78.8.
The adapter and valve rows agree within ~1% everywhere.

## Scope of the verdict

This closes the Bonsai adapter's local gate for the three packs present on
the host. The 27B pack is not on this host, so no 27B adapter row can be
honestly claimed; Nanbeige, Nemotron Lightning, DeepSeek V4.1 and Gemma 4
have no adapters yet (T4.2 remaining families), so they stay on the legacy
path. No arbitrary-GGUF claim is made (T4.4/T4.5 stay decision-gated).

Machine-readable bench output: `4b-adapter.json`, `9b-adapter.json`,
`9b-valve.json` in this directory.
