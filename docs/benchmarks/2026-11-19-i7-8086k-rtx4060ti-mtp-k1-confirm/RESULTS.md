# Fresh mode A k=1 A/B at 512 / 24K (2B)

Run: 2026-11-19. Host: i7-8086K (6 cores / 12 threads), 32 GB RAM, Windows 10
22H2, RTX 4060 Ti 8 GB. Pack: `bonsai-2b` (`Ternary-Bonsai-2-2B`). Head:
`kindi-mtp-2b-v1.npz` (sha256 `1b5c0c79…`, the registered file).

This run rechecks the 2026-11-17 k=1 depth row with **`KINDI_SPEC_K` unset**, so
the engine selects explicit mode A's shipped default of 1. `KINDI_TC=1`; the
head attached and the startup log reported `speculation: requested mtp,
effective mtp`.

## Method

`tools/mtp_context_bench.py` builds the engine once per invocation and, for each
position, alternates plain and mode A on the same engine. Only `KINDI_SPEC` and
`spec_begin` change between the pair, so weights, prompt and kernel caches are
shared. The prompt is the repository's current `docs/*.md` joined in name order
(306,836 tokens available); prefill is excluded from the clock and 128 greedy
tokens are timed. Two runs per mode at 512 (ctx 1024), four per mode at 24,576
(ctx 26,624), with medians reported.

```sh
KINDI_TC=1 python tools/mtp_context_bench.py \
  --pack ~/.kindi/packs/bonsai-2b \
  --head ~/.kindi/models/kindi-mtp-2b-v1.npz \
  --ctx 1024 --positions 512 --tokens 128 --runs 2 \
  --json docs/benchmarks/2026-11-19-i7-8086k-rtx4060ti-mtp-k1-confirm/mtp-context-k1-512-unset.json

KINDI_TC=1 python tools/mtp_context_bench.py \
  --pack ~/.kindi/packs/bonsai-2b \
  --head ~/.kindi/models/kindi-mtp-2b-v1.npz \
  --ctx 26624 --positions 24576 --tokens 128 --runs 4 \
  --json docs/benchmarks/2026-11-19-i7-8086k-rtx4060ti-mtp-k1-confirm/mtp-context-k1-24k-unset.json
```

## Result

| context | runs | plain tok/s | mode A tok/s | ratio | drafts accepted |
|---:|---:|---:|---:|---:|---:|
| 512 | 2 | 341.8 | **378.2** | **1.11x** | 52/75 (69.3%) |
| 24,576 | 4 | 301.7 | **343.2** | **1.14x** | 54/74 (73.0%) |

Per-run rows: 512 plain 337.7 / 345.9, mode A 362.8 / 393.6; 24K plain
303.0 / 301.6 / 301.2 / 301.8, mode A 338.7 / 344.3 / 342.2 / 344.2. The JSON
files preserve every row, including the draft and acceptance counts.

## Confirmation

The 2026-11-17 k=1 row measured 1.12x at 512 and 1.13x at 24K. This fresh run
measures 1.11x and 1.14x with the same direction and the same acceptance shape:
69.3% at 512 against 69%, and 73.0% at 24K against 72%. It **confirms** the
measured k=1 row and therefore the shipped default; no default change is
needed.

The prompt is not byte-identical to the 2026-11-17 prompt: the current
`docs/*.md` set has grown from 281,805 to 306,836 available tokens, so this is a
confirmation on the current tree rather than an identical replay. The 512 and
24K positions are the same, and the generation clock excludes prefill as before.

## Artifacts

- `run.log` - the raw run output, with the home-directory and Python-install
  prefixes normalized to `~` and `<python>`; the exact raw copy is kept in the
  ignored `bench-logs/mtp-k1-confirm-20261119/run-raw.log`.
- `mtp-context-k1-512-unset.json` and `mtp-context-k1-24k-unset.json` - the
  per-run rows.

No `docs/claims/` entry is added: `docs/MTP.md` is not one of the claimed docs,
and this microbenchmark makes no `kindi verify` or model-row claim.
