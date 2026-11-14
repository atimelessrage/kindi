# Rig session: P106-100, Quadro P4000, Quadro M4000

Status: **prepared, not measured.** Nothing on this page is a support claim. It is the script for the
session on the other rig, written so that each command produces evidence that can be entered into
[CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here) without guessing afterwards.

The three cards are the last ones with routing in `install.sh` and no GPU run behind it:

| card | silicon | compute | VRAM | installer class | engine path |
|---|---|---|---|---|---|
| P106-100 | GP106 | sm_61 | 6,077 MiB | `maxwell` (6 GB sits under the 7,500 MiB `pascal8` start) | plain plan engine, dp4a available but `KINDI_TC` not enabled by the profile |
| Quadro P4000 | GP104 | sm_61 | 8,192 MiB | `pascal8` | plain plan engine; the Pascal tensor-core kernels (`tcp.cu`, dp4a) exist but are unmeasured |
| Quadro M4000 | GM204 Maxwell 2.0 | sm_52 | 8,192 MiB | `maxwell` | plain plan engine only: `tc.supported()` requires sm_60+, so `KINDI_TC=1` refuses instead of crashing |

## What is already in place (verified without the hardware)

- Routing and profiles: `tests/test_installer_cards.py` covers all three as single cards, the
  P106+P4000 and M4000+P4000 pairs (mixed generations force the plain split and 2-bit weights) and
  the Kepler rejections. `python -m pytest -q tests/test_installer_cards.py` runs anywhere.
- Host toolchain: `docker/Dockerfile` is the CUDA 10.2.89 + CuPy 9.6 image (the last one that targets
  sm_52 and sm_61). The host driver has to be at least 440.33 for that toolkit; a 535 branch driver
  covers Maxwell and Pascal.
- Evidence capture: `tools/verify_matrix.py` now writes an `environment` block into `results.json` -
  card name, compute capability, driver, VRAM, CuPy and CUDA runtime version, and for each pack its
  `source_sha256`, fold and size. On a host without `nvidia-smi` it records `"no card detected"`
  instead of failing, so the same command works in a CPU container.
- The 9B Hadamard-4096 default (`KINDI_TC_HP=1`) does not apply here: none of these cards reaches
  the tensor-core path unless `KINDI_TC=1` is forced, and on sm_52 it then refuses.

## Before the session

1. Driver installed, `nvidia-smi` works, and the card is not in "display off" mode in `nvidia-smi -q`.
2. Image built or pulled: `docker build -t kindi:old docker/` (or `bash install.sh`, which picks it
   for pre-Ampere cards). Inside the container `python -c "import cupy; print(cupy.__version__)"`
   must print `9.6`.
3. Packs present, built on this or another host: `bonsai-2b`, `bonsai-4b`, `bonsai-9b`, and
   `bonsai-27b` for the P4000 / M4000 (8 GB). Record the pack `source_sha256` - `verify_matrix`
   does that for you.
4. If the card sits on a mining riser, note the PCIe link width: `nvidia-smi --query-gpu=pcie.link.width.current
   --format=csv`. An x1 link explains slow load times and a slow CPU/GPU split, and is not a card defect.

## The session, per card

Run this once per card, in a fresh container, with the other cards removed or hidden
(`CUDA_VISIBLE_DEVICES=0`):

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap,pcie.link.width.current \
    --format=csv
python -m kindi info                      # engine choice, VRAM, pack

# the recorded matrix: plain and tc, 2B/4B/9B (add 27b when the pack is on disk)
python tools/verify_matrix.py --pack-dir ~/.kindi/packs --models 2b,4b,9b \
    --modes plain,tc --cards 1 --out bench-logs/verify-matrix-<card>

# memory and speed at the contexts that matter, one file per card
python -m kindi --pack ~/.kindi/packs/bonsai-2b bench --sample-vram --json bench-logs/<card>-2b.json
python -m kindi --pack ~/.kindi/packs/bonsai-9b bench --sample-vram --json bench-logs/<card>-9b.json
```

Then the serving check from [CARDS.md](CARDS.md) (start, two requests, restart, second request must
be a cache hit). For the 8 GB cards also try the 27B:

```bash
python tools/verify_matrix.py --models 27b --modes plain --cards 1 \
    --pack-27b ~/.kindi/packs/bonsai-27b --out bench-logs/verify-matrix-<card>-27b
```

For a pair (P106+P4000, M4000+P4000), the same command with `--cards 2` and both cards visible; the
installer forces the plain split and 2-bit weights, so expect lower throughput than one card and
record it as the mixed-pair number.

## What to send back

- `bench-logs/verify-matrix-<card>/results.json` - it carries the card, driver, CuPy/CUDA and the
  pack hash, so the run is reproducible.
- `bench-logs/<card>-2b.json`, `-9b.json` (and `-27b.json` when it ran).
- `nvidia-smi -q -d MEMORY,CLOCK` if a token rate or a peak looks wrong.
- The full log of the first failing cell, if any, not a summary line.

Anything quoted in a tracked page has to be copied into `docs/benchmarks/<date>-<card>/` first
(`docs/README.md`: `bench-logs/` is gitignored, `benchmarks/` is the record).

## What makes it a support row

The contract in [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here), unchanged: an installer
offer, a loading model, a passing `kindi verify`, a serving/cache check, and a measured row. The
matrix alone gives the third; the bench and serving runs give the fifth.

## Expected outcomes (to confirm, not to assume)

| card | plain 2B/4B/9B | tc | 27B |
|---|---|---|---|
| P106-100 6 GB | expected to load and pass; 9B context profile is 24,576 tokens | `KINDI_TC=1` on sm_61 is unmeasured - record pass or refusal as found | not offered (6 GB) |
| Quadro P4000 8 GB | expected pass | the dp4a path exists; `pascal8` does not enable it, so forcing it is the interesting experiment | 8 GB fit unverified - the most likely first failure is a context reduction, which is a result, not a fault |
| Quadro M4000 8 GB | expected pass (Maxwell, no dp4a, `KINDI_G5=0` for the 5-trit lane) | refused by `tc.supported()`, and that refusal is the correct recorded outcome | same as P4000, and Maxwell has no int8 path at all in pairs (2-bit weights on both cards) |

## When something fails

Record which of these changes the behaviour, in this order, and put the answers in the report rather
than only the final error:

1. `KINDI_TC=0` - takes the plain engine (already the default on these cards).
2. `KINDI_TP=0` - disables the two-card split for a pair.
3. `KINDI_G5=0` - disables the 5-trit lane layout, the fallback for Maxwell.
4. `KINDI_CHUNK_Q8=0` - disables int8 activations (already off below sm_61).
5. `KINDI_KV=fp16` - the plan engine's q8_0 K/V needs the plan engine; the plain engine's fp16 K/V is
   the conservative choice.

A refusal with a measured reason is a result. A traceback is a bug: send the log.
