# Rig session: P106-100, Quadro P4000, Quadro M4000, Quadro M5000, Quadro RTX 4000 (Turing)

Status: **prepared, not measured.** Nothing on this page is a support claim. It is the script for the
session on the other rig, written so that each command produces evidence that can be entered into
[CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here) without guessing afterwards.

The five cards are the last ones with routing in `install.sh` and no GPU run behind them:

| card | silicon | compute | VRAM | installer class | engine path |
|---|---|---|---|---|---|
| P106-100 | GP106 | sm_61 | 6,077 MiB | `maxwell` (6 GB sits under the 7,500 MiB `pascal8` start) | plain plan engine, dp4a available but `KINDI_TC` not enabled by the profile |
| Quadro P4000 | GP104 | sm_61 | 8,192 MiB | `pascal8` | plain plan engine; the Pascal tensor-core kernels (`tcp.cu`, dp4a) exist but are unmeasured |
| Quadro M4000 | GM204 Maxwell 2.0 | sm_52 | 8,192 MiB | `maxwell` | plain plan engine only: `tc.supported()` requires sm_60+, so `KINDI_TC=1` refuses instead of crashing |
| Quadro M5000 | GM204 Maxwell 2.0 | sm_52 | 8,192 MiB | `maxwell` | plain plan engine only, like the M4000; a user report (below) already covers 2B/4B/9B/27B verify, but there is no `verify_matrix` row yet |
| Quadro RTX 4000 | TU104 Turing | sm_75 | 8,192 MiB | `turing` | Turing tensor-core backend with `KINDI_TC=1` (`mma.m8n8k16` decode/prompt), fallback to the plain 5-trit path with `KINDI_TC=0` |

## What is already in place (verified without the hardware)

- Routing and profiles: `tests/test_installer_cards.py` covers all three as single cards, the
  P106+P4000 and M4000+P4000 pairs (mixed generations force the plain split and 2-bit weights) and
  the Kepler rejections. `python -m pytest -q tests/test_installer_cards.py` runs anywhere.
- Host toolchain: `docker/Dockerfile` is the CUDA 10.2.89 + CuPy 9.6 image (the last one that targets
  sm_52 and sm_61). The host driver has to be at least 440.33 for that toolkit; a 535 branch driver
  covers Maxwell and Pascal. There is also an upper bound: **R580 is the last NVIDIA driver branch
  that supports Maxwell, Pascal and Volta** (support is dropped in R590), so a 617.x host driver
  defeats this path regardless of what the card names itself — pin a <=580 branch before the session.
  CUDA 13 dropped offline compilation below sm_75, so pre-Turing cards need a CUDA <= 12.x toolkit
  (CUDA 12.4 builds sm_50+); the 10.2 image above is one such.
- Evidence capture: `tools/verify_matrix.py` now writes an `environment` block into `results.json` -
  card name, compute capability, driver, VRAM, CuPy and CUDA runtime version, and for each pack its
  `source_sha256`, fold and size. On a host without `nvidia-smi` it records `"no card detected"`
  instead of failing, so the same command works in a CPU container.
- The 9B Hadamard-4096 default (`KINDI_TC_HP=1`) does not apply here: none of these cards reaches
  the tensor-core path unless `KINDI_TC=1` is forced, and on sm_52 it then refuses.
- M5000 and RTX 4000 share this page's toolchain (CUDA 10.2.89 + CuPy 9.6, driver >= 440.33):
  sm_52/sm_75 are both in its target range, and the Turing backend's PTX is JIT-built for sm_75 by the
  driver. The M5000 user report above ran on the 470/580 drivers; record the driver in the session too.

## Before the session

1. Driver installed, `nvidia-smi` works, and the card is not in "display off" mode in `nvidia-smi -q`.
2. Image built or pulled: `docker build -t kindi:old docker/` (or `bash install.sh`, which picks it
   for pre-Ampere cards). Inside the container `python -c "import cupy; print(cupy.__version__)"`
   must print `9.6`.
3. Packs present, built on this or another host: `bonsai-2b`, `bonsai-4b`, `bonsai-9b`, and
   `bonsai-27b` for the P4000 / M4000 / M5000 / RTX 4000 (8 GB; the M5000 user report and the RTX 4000
   section both name the 27B as the fit check). Record the pack `source_sha256` - `verify_matrix`
   does that for you.
4. If the card sits on a mining riser, note the PCIe link width: `nvidia-smi --query-gpu=pcie.link.width.current
   --format=csv`. An x1 link explains slow load times and a slow CPU/GPU split, and is not a card defect.

## The session, per card

Run this once per card, in a fresh container, with the other cards removed or hidden
(`CUDA_VISIBLE_DEVICES=0`). On the RTX 4000, run the matrix twice: once in the installer `turing`
profile (`KINDI_TC=1`, see below) and once with the TC fallback (`KINDI_TC=0`).

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap,pcie.link.width.current \
    --format=csv
python -m kindi info                      # engine choice, VRAM, pack

# one command covers the whole support contract (docs/CARD-SUPPORT.md): the
# verify cells, a `kindi bench --json` measured row per model, and a server/cache
# cycle per model (boot, two requests with the repeat a cache hit, an invalid
# 400 that must not clear the cache, restart, cache hit again, bit-identical
# replies). The catalog offer (criterion 1) is recorded from the card's VRAM.
python tools/verify_matrix.py --pack-dir ~/.kindi/packs --models 2b,4b,9b \
    --modes plain,tc --cards 1 --out bench-logs/verify-matrix-<card>

# memory and speed at the contexts that matter, one file per card (the measured
# row above covers the contract; this adds the sampled-peak capture)
python -m kindi --pack ~/.kindi/packs/bonsai-2b bench --sample-vram --json bench-logs/<card>-2b.json
python -m kindi --pack ~/.kindi/packs/bonsai-9b bench --sample-vram --json bench-logs/<card>-9b.json
```

For the 8 GB cards also try the 27B. This is the fit check that decides the support row: the 27B pack
is about 6.7 GB, so a few hundred MiB is left for K/V, and `kindi bench --sample-vram` records both the
sampled peak and any context reduction:

```bash
python tools/verify_matrix.py --models 27b --modes plain --cards 1 \
    --pack-27b ~/.kindi/packs/bonsai-27b --out bench-logs/verify-matrix-<card>-27b
python -m kindi --pack ~/.kindi/packs/bonsai-27b bench --sample-vram --ctx 65536 \\
    --json bench-logs/<card>-27b.json
```

Record the fitted context (`/health` reports it) and the sampled peak. A reduced context is a result,
not a fault: the M5000 user report already found the 27B fitting at 24K-28K with the 580 driver
fitting 28,672 down to 27,904 tokens, and the RTX 4000 profile targets about 65,536 tokens. The
supported row names whichever context actually fit, with its peak.

For a pair (P106+P4000, M4000/M5000+P4000), the same command with `--cards 2` and both cards visible; the
installer forces the plain split and 2-bit weights, so expect lower throughput than one card and
record it as the mixed-pair number.

## M5000 specifics (sm_52, `maxwell`, 8 GB)

Same profile as the M4000; nothing extra to tune: plain plan engine, no dp4a, no tensor cores, and
`KINDI_TC=1` refuses by design (`tc.supported()` requires sm_60+) — record the refusal, don't debug it.
Mixed pairs (M5000+P4000) set `KINDI_G5=0` so both cards use the 2-bit path. The user report in
[CARDS.md](CARDS.md#quadro-m5000-8-gb-maxwell-sm52-user-report) (Kindi 0.4.0, packs reused from a K4000
without conversion: verify 4/4 on 2B/4B/9B/27B; 27B at 17.8 tokens/s write / 39.0 tokens/s read at 512;
Nemotron served at its full 131,072 tokens) is the baseline the rig run confirms or corrects — and
repeats with the current installer (which now reports sm_52, fixing the pre-0.6 sm_50 misread). On the
580 driver set `KINDI_BLOCKING_SYNC=0` (blocking sync cost 5-15% there); the 470 driver does not show it.

## RTX 4000 (Turing, sm_75, `turing`, 8 GB)

The installer `turing` profile sets `KINDI_TC=1`, `KINDI_FD=1`, `KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4` and
`KINDI_BLOCKING_SYNC=0`, and offers 27B/9B/4B/2B. Turing's int8 tensor cores come from a separate
backend (`tc_turing`, [TURING-TENSOR-CORES.md](TURING-TENSOR-CORES.md)): `mma.m8n8k16` for the decode
products and the prompt chunks, exact integer sums, so greedy output should be token-identical to the
5-trit path — but that is the claim the verify run tests, not a given. TU116/TU117 parts report sm_75
without tensor cores: `tc_turing.supported()` checks the device name and leaves them on the plain
engine; record the device name in the run.

Rig commands for the Turing card:

```bash
# the TC path (the interesting experiment)
KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1 KINDI_TC_TMAX=4 KINDI_BLOCKING_SYNC=0 \\
  python tools/verify_matrix.py --models 2b,4b,9b,27b --modes tc --cards 1 \
  --pack-27b ~/.kindi/packs/bonsai-27b --out bench-logs/verify-matrix-rtx4000

# the TC fallback (plain 5-trit path)
KINDI_TC=0 KINDI_BLOCKING_SYNC=0 \\
  python tools/verify_matrix.py --models 2b,4b,9b,27b --modes plain --cards 1 \
  --pack-27b ~/.kindi/packs/bonsai-27b --out bench-logs/verify-matrix-rtx4000-plain
```

If the TC run fails or diverges, fall back in this order: `KINDI_TC=0` (the 5-trit path), then
`KINDI_FD=0`, then drop `KINDI_SPEC=draft2b` — and record which flag changed the outcome. The 27B fit
check (sampled peak + fitted context at 65,536) runs under both paths.

## What to send back

- `bench-logs/verify-matrix-<card>/results.json` - it carries the card, driver, CuPy/CUDA and the
  pack hash, so the run is reproducible. Copy it into the tracked tree and fold it into the matrix;
  this is the step that stops the rig numbers becoming a hand-typed table:

```bash
mkdir -p docs/benchmarks/<date>-<card>
cp bench-logs/verify-matrix-<card>/results.json docs/benchmarks/<date>-<card>/results.json
python -m kindi.support_matrix --from-verify docs/benchmarks/<date>-<card>/results.json
```

`--from-verify` writes the card's cells into the tracked registry
`docs/benchmarks/card-results.json` (deriving exactness, tie and speed from the run's rows - a
refusal is recorded as a result, never as an error) and regenerates `docs/SUPPORT-MATRIX.md`. It
refuses a path under `bench-logs/` because a link to gitignored evidence would break; it also
refuses an unrecorded card name (register it in `kindi/support_matrix.py` first) and a multi-card
run without `--card <slug>`.
- `bench-logs/<card>-2b.json`, `-9b.json` (and `-27b.json` when it ran).
- `nvidia-smi -q -d MEMORY,CLOCK` if a token rate or a peak looks wrong.
- The full log of the first failing cell, if any, not a summary line.

Anything quoted in a tracked page has to be copied into `docs/benchmarks/<date>-<card>/` first
(`docs/README.md`: `bench-logs/` is gitignored, `benchmarks/` is the record).

## What makes it a support row

The contract in [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here), unchanged: an installer
offer (the catalog fit against the card's VRAM), a loading model and a serving/cache check (the
server cycle: boot, repeat, invalid, restart), a passing `kindi verify` (the cells) and a measured
row (`kindi bench --json`). The one command above covers all five; its statuses land in `results.json`
under `offered`, `serve`, `bench` and `results` respectively.

## Expected outcomes (to confirm, not to assume)

| card | plain 2B/4B/9B | tc | 27B |
|---|---|---|---|
| P106-100 6 GB | expected to load and pass; 9B context profile is 24,576 tokens | `KINDI_TC=1` on sm_61 is unmeasured - record pass or refusal as found | not offered (6 GB) |
| Quadro P4000 8 GB | expected pass | the dp4a path exists; `pascal8` does not enable it, so forcing it is the interesting experiment | 8 GB fit unverified - the most likely first failure is a context reduction, which is a result, not a fault |
| Quadro M4000 8 GB | expected pass (Maxwell, no dp4a, `KINDI_G5=0` for the 5-trit lane) | refused by `tc.supported()`, and that refusal is the correct recorded outcome | same as P4000, and Maxwell has no int8 path at all in pairs (2-bit weights on both cards) |
| Quadro M5000 8 GB | user report says pass (4/4 on 2B/4B/9B/27B); rig confirms with the current installer | refused by `tc.supported()` (sm_52), the expected record | user report fits 24K-28K; the sampled-peak + fitted-context number from the fit check is the support row |
| Quadro RTX 4000 8 GB | expected pass (the `KINDI_TC=0` fallback) | `turing` profile is the experiment: `mma.m8n8k16` Turing backend, expected token-identical to the 5-trit path; TU116/TU117 stay plain by device name | profile targets about 65,536 tokens; the TC/ plain sampled-peak pair at that context decides the row |

## When something fails

Record which of these changes the behaviour, in this order, and put the answers in the report rather
than only the final error:

1. `KINDI_TC=0` - takes the plain engine (already the default on these cards; on the RTX 4000 this is
   the Turing TC fallback and the other half of the card's two-path record).
2. `KINDI_TP=0` - disables the two-card split for a pair.
3. `KINDI_G5=0` - disables the 5-trit lane layout, the fallback for Maxwell.
4. `KINDI_CHUNK_Q8=0` - disables int8 activations (already off below sm_61).
5. `KINDI_KV=fp16` - the plan engine's q8_0 K/V needs the plan engine; the plain engine's fp16 K/V is
   the conservative choice.
6. On the RTX 4000 only, after `KINDI_TC=0`: `KINDI_FD=0` (no split flash-decode attention), then drop
   `KINDI_SPEC=draft2b` — record which one restored exactness.

A refusal with a measured reason is a result. A traceback is a bug: send the log.
