# T4.2 Bonsai adapter: 27B gate (RTX 4060 Ti, 2026-11-16)

The Bonsai adapter (`kindi/backends/bonsai.py`) was verified on the 2B, 4B and
9B packs; the 27B was left ungated because the project record said the pack
is not on this host. It is in fact present at `T:/kindi-packs/bonsai-27b`
(18 GB), but its `kindi_pack.json` carries `"model": "Hf"` (the 27B GGUF's
`general.name`), so `model_id_for_manifest` returned `None` and the pack
silently took the legacy path — even though the adapters `SIZES` and
`model_ids` already list `27b`. That resolution bug is fixed in this tree
(the size now also comes from `source_file`, the same fallback
`kindi/config.py` uses to name the served model).

Host: i7-8086K + RTX 4060 Ti 8 GB (8,188 MiB), ctx 1024 for verify and
24,576 for bench, exact head. Adapter path = default
(`KINDI_LEGACY_ENGINE_PATH` unset, registry resolution);
valve path = `KINDI_LEGACY_ENGINE_PATH=1` (pre-port builder).
Pack: `T:/kindi-packs/bonsai-27b` (source size 5,946,648,928 bytes).

## Verify (token-exact vs llama.cpp)

| pack | adapter | valve |
|---|---|---|
| 27B | ALL PASS 4/4 @ 34.5 tok/s | ALL PASS 4/4 @ 35.7 tok/s |

Exactness is identical between the paths; the tok/s delta is run-to-run
jitter (the adapter builds the same plan engine by construction — both
runs log `plan engine ready: 1 card, ctx 1024, K/V q8_0`).

## Bench (exact head, ctx 24576, ctx_reduced false)

| path | read tok/s | gen @100 | gen @2K | gen @8K | gen @16K | gen @24K |
|---|---|---|---|---|---|---|
| adapter | 36.7 | 34.8 | 34.4 | 33.1 | 31.6 | 30.2 |
| valve | 37.1 | 34.7 | 34.5 | 33.1 | 31.6 | 30.2 |

The rows agree within ~1% everywhere (identical at 8K/16K/24K).

## Scope of the verdict

This closes the Bonsai adapter's local gate for all four packs present on
this host (2B/4B/9B/27B). It says nothing about the adapter for other
families: Nanbeige's GPU gate stays blocked on a working GGUF
(`docs/MANUAL-HELP-NEEDED.md`), and Nemotron Lightning / DeepSeek V4.1 /
Gemma 4 have no adapters yet. No arbitrary-GGUF claim is made (T4.4/T4.5
stay decision-gated).

Machine-readable bench output: `27b-adapter.json`, `27b-valve.json` in
this directory.
