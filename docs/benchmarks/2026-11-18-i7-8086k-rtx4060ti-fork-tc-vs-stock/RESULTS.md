# Fork tensor-core plan path against the stock v0.6.1 rows (2026-11-18)

**Different configuration, not a same-settings A/B.** This page adds the fork's
default tensor-core plan path at `--ctx 8192` to the 2B and 9B rows in §4.6 of
`CURRENT-VS-KINDI-0.7.md`. The stock reference rows are the 2026-11-17
same-settings medians already quoted there; no stock process was run in this
capture. Stock v0.6.1 has no plan engine or tensor-core path, so its rows cannot
be used as a controlled estimate of the effect of those settings.

Host: i7-8086K + NVIDIA GeForce RTX 4060 Ti, 8,188 MiB, driver 617.42, native
Windows, Python 3.13.6, CuPy 14.2.0, CUDA runtime 12.9 (12090). Packs:
`~/.kindi/packs/bonsai-2b` and `~/.kindi/packs/bonsai-9b`, the same source
packs as the stock and same-settings pages.

## Fork settings

The documented default tensor-core plan configuration is used:

```sh
KINDI_TC=1 KINDI_TP=1 KINDI_FD=1 KINDI_ATT_TC=1 KINDI_TC_TMAX=4 \
KINDI_HEAD=exact KINDI_SPEC=0 \
python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 8192 bench --split-at off
```

and the same command for `bonsai-9b`. The plan engine uses its q8_0 K/V default;
the 9B pack's Hadamard block 4096 automatically selects `KINDI_TC_HP=1`, the
exact tensor-core activations (the log reports that selection). No speculation,
tandem, cache or split-at request is enabled. Stock has no equivalent path.

## Results

Five interleaved passes per model, medians in tok/s; the range is the five-pass
min-max. The stock rows are the reference already recorded in §4.6, repeated
here to keep the comparison visible.

| model | engine | prompt reading | gen @100 | @2,000 | @8,000 |
|---|---|---:|---:|---:|---:|
| 2B | stock v0.6.1 (reference, 2026-11-17) | 215.5 | 180.5 | 182.9 | 176.4 |
| 2B | **this fork, tc plan — different configuration** | **484.9** (484.2-485.9) | **354.9** (354.2-362.5) | **333.8** (333.0-339.8) | **331.0** (328.0-346.2) |
| 9B | stock v0.6.1 (reference, 2026-11-17) | 83.9 | 83.8 | 81.3 | 77.4 |
| 9B | **this fork, tc plan — different configuration** | **137.2** (137.1-137.3) | **118.3** (117.9-118.6) | **115.2** (115.1-115.4) | **111.8** (111.6-112.0) |

Free memory after the fit was 6,515 MiB on the 2B and 5,077 MiB on the 9B. The
runs used `ctx_reduced: false`; the requested and actual context are both 8,192.

These are absolute TC-path rates against the same bench cells, not an estimate
of the TC settings' effect. The fork's own plain-engine row in §4.6 is the
controlled comparison; the stock rows are shown only because they are the
published reference rows this table extends.

## Commands

```sh
# for each model and pass 1..5, from the worktree:
KINDI_TC=1 KINDI_TP=1 KINDI_FD=1 KINDI_ATT_TC=1 KINDI_TC_TMAX=4 \
KINDI_HEAD=exact KINDI_SPEC=0 \
python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 8192 bench --split-at off \
  --json tc-2b-r1.json
```

`summary.json` holds the per-run values, medians and ranges. The raw logs and
JSONs for every pass are in this directory.

## Limits

One card, one pack per size, five passes, exact head, no token-exactness or
server/cache measurement. The stock numbers were not re-measured in this
capture and come from the earlier same-settings page. GPU clocks and Windows
WDDM scheduling are uncontrolled; ranges are reported rather than a best pass.
This is not a same-settings comparison and must not be substituted for the plain
fork-vs-stock rows in §4.6.
