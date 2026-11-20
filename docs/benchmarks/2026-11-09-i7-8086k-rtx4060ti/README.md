# Kindi 2B / 4B / 9B on the i7-8086K + RTX 4060 Ti

Measured 2026-11-09. The main 2B/4B/9B comparison is the plain exact-head
path; a separate 2B plan comparison is included below. It compares the current
branch with tag `v0.4.0`.

The short version: the plain benchmark path did not change. `kindi/bench.py` and
`kindi/engine.py` are byte-identical between `v0.4.0` and current main. The only
changes on this path are the native-Windows driver selection in `kindi/gpudrv.py`,
`kindi/sched.py` and `kindi/chunk1.py`. The current branch adds a Turing
tensor-core backend, plan-engine guards, server features, installer safety and
tests; the plain rows below do not exercise them.

The separate 2B plan engine reaches 302.4 tok/s at 24K versus 170.6 plain
(1.77x); that comparison is below.

## Headline

Median of three fresh processes per model and source; `±` is the sample standard
deviation. All rates are tokens/s. "24K" is generation at position 24,000 after a
24,576-token context allocation.

| Model | Current read | v0.4.0 + loader read | Δ | Current 24K | v0.4.0 + loader 24K | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 190.0 ± 23.6 | 161.5 ± 9.3 | +17.6% | 160.6 ± 1.9 | 165.5 ± 7.1 | -3.0% |
| 4B | 129.0 ± 1.6 | 132.9 ± 6.7 | -2.9% | 79.8 ± 1.1 | 82.1 ± 1.2 | -2.8% |
| 9B | 90.3 ± 9.3 | 90.8 ± 2.3 | -0.6% | 62.3 ± 0.3 | 62.7 ± 1.3 | -0.6% |

The 2B read figure is the noisiest cell (SD 23.6); do not read the +17.6% as a
speedup. The 24K generation deltas are -3.0%, -2.8% and -0.6%. Since the engine
source is identical, these are run-to-run effects, not a code-level regression.

## Rig

| Item | Value |
|---|---|
| CPU | Intel Core i7-8086K @ 4.00 GHz |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8 GB, sm_89 |
| GPU memory | 8,188 MiB |
| GPU driver | 617.42 |
| GPU power limit | 160 W |
| GPU maximum clock | 3,105 MHz |
| GPU observed clock | 2,805–2,820 MHz during the current runs |
| GPU observed temperature | 35–47 °C during the current runs |
| OS | Microsoft Windows 10 Pro 10.0.19045 |
| Python | 3.13.6 |
| CuPy | 14.2.0 |
| NumPy | 2.5.1 |

The clock and temperature range comes from the current-run logs. The baseline
logs do not contain the clock/temperature samples; the same machine and power
limit were used.

## Method

The command was, for each model and repetition:

```sh
python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 bench
python -m kindi --pack ~/.kindi/packs/bonsai-4b --ctx 24576 bench
python -m kindi --pack ~/.kindi/packs/packs/packs/bonsai-9b --ctx 24576 bench
```

`bench` first warms each timed operation, then reads a synthetic 1,000-token
prompt and generates 10 tokens at positions 100, 2,000, 8,000, 16,000 and
24,000. The head is exact and speculation is off. Each repetition is a new
process. The table uses the median of the three runs; `stats.csv` also has the
variance, sample standard deviation, minimum and maximum.

The current runs were all taken first, then the baseline runs. They were not
interleaved, so a slow drift in clocks or host state can appear as a few percent
of difference between the two blocks.

### The v0.4.0 baseline

Unmodified `v0.4.0` hardcodes `libcuda.so.1` in `kindi/sched.py` and
`kindi/chunk1.py`, so it cannot load CUDA on native Windows. The baseline is the
exact `v0.4.0` tree (`cee44886c3b8b9436320b787caf638c9d027f46f`) with only the
current native-Windows driver loader:

1. copy `kindi/gpudrv.py` from current main (`2abf9ad`);
2. in `kindi/sched.py`, use `cu = gpudrv.driver()` instead of
   `ctypes.CDLL("libcuda.so.1")`;
3. in `kindi/chunk1.py`, import `gpudrv` and use `drv = gpudrv.driver()`.

No engine, attention, quantization, chunking, scheduling or benchmark logic was
edited. The exact recipe is in [baseline-loader.md](baseline-loader.md). The
current benchmark was run at `2abf9ad`; the benchmark path is unchanged at
`46b63f8`. The baseline tag's `bench.py` and `engine.py` have the same Git blob
hashes as current main:

```
bench.py   b79c6838d2af462e49bca600dbb57a171dc08cae
engine.py  b7afba3d500569433ef8e8f458c73040ba2bb934
```

## Full generation results

### Ternary-Bonsai-2-2B

| Position | Current | v0.4.0 + loader | Δ |
|---:|---:|---:|---:|
| 100 | 169.0 ± 6.9 | 158.6 ± 6.0 | +6.6% |
| 2,000 | 163.3 ± 7.8 | 172.7 ± 7.4 | -5.4% |
| 8,000 | 175.4 ± 5.7 | 178.5 ± 1.8 | -1.7% |
| 16,000 | 171.7 ± 3.1 | 164.3 ± 10.3 | +4.5% |
| 24,000 | 160.6 ± 1.9 | 165.5 ± 7.1 | -3.0% |

Read: current 190.0 ± 23.6, baseline 161.5 ± 9.3 (+17.6%).

### Ternary-Bonsai-2-4B

| Position | Current | v0.4.0 + loader | Δ |
|---:|---:|---:|---:|
| 100 | 106.2 ± 5.8 | 103.0 ± 7.3 | +3.1% |
| 2,000 | 101.9 ± 0.5 | 105.5 ± 8.7 | -3.4% |
| 8,000 | 90.8 ± 5.8 | 99.6 ± 2.0 | -8.8% |
| 16,000 | 89.3 ± 1.0 | 91.2 ± 1.3 | -2.1% |
| 24,000 | 79.8 ± 1.1 | 82.1 ± 1.2 | -2.8% |

Read: current 129.0 ± 1.6, baseline 132.9 ± 6.7 (-2.9%).

The 4B 8,000-token cell is the largest difference. Current runs were 84.9, 89.2
and 98.3; baseline runs were 102.9, 99.4 and 99.6. It is a single block of three
runs, not an interleaved A/B, so it is not evidence of a code regression.

### Ternary-Bonsai-2-9B

| Position | Current | v0.4.0 + loader | Δ |
|---:|---:|---:|---:|
| 100 | 77.8 ± 5.5 | 79.0 ± 0.6 | -1.5% |
| 2,000 | 76.8 ± 0.3 | 77.1 ± 1.2 | -0.4% |
| 8,000 | 71.6 ± 2.3 | 73.3 ± 1.3 | -2.3% |
| 16,000 | 67.4 ± 2.2 | 67.8 ± 1.4 | -0.6% |
| 24,000 | 62.3 ± 0.3 | 62.7 ± 1.3 | -0.6% |

Read: current 90.3 ± 9.3, baseline 90.8 ± 2.3 (-0.6%).

## 2B plan engine vs plain at 24K

Same rig, same 24,576-token context, exact head. Three fresh processes per
engine, alternating order: r1 plain then plan, r2 plan then plain, r3 plain then
plan.

| Engine | Read | 100 | 2K | 8K | 16K | 24K |
|---|---:|---:|---:|---:|---:|---:|
| Plain | 225.3 ± 12.2 | 184.2 ± 8.6 | 156.1 ± 23.9 | 185.7 ± 0.5 | 183.5 ± 0.8 | 170.6 ± 17.3 |
| Plan | 470.5 ± 0.6 | 350.2 ± 1.7 | 335.6 ± 2.2 | 329.2 ± 7.7 | 322.2 ± 5.9 | 302.4 ± 1.7 |
| Speedup | 2.09x | 1.90x | 2.15x | 1.77x | 1.76x | 1.77x |

At 24K the plan engine is **302.4 tok/s against 170.6 tok/s plain: 1.77x, or
+77.3% (+131.8 tok/s)**. The plain 24K runs were 172.9, 170.6 and 141.8; the plan
runs were 302.4, 303.5 and 300.1. The plan side is stable; the plain side has one
low run, which is why the median speedup is lower than the ratio of the best
runs.

Configurations (no speculative drafting in either):

```sh
# plain single-card engine
KINDI_TP=0 KINDI_SPEC=0 python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 bench

# 2B tensor-core plan engine; T=4 is the exact plan
KINDI_TP=1 KINDI_CARDS=1 KINDI_TC=1 KINDI_TC_TMAX=4 KINDI_ATT_TC=1 KINDI_FD=1 KINDI_SPEC=0 \
  python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 bench
```

The plan engine uses TMAX=4 because the T=4 2B plan is the token-exact
configuration; the T=8 default is not exact for this model. The benchmark does
not compare token outputs. The T=4 plan was previously verified 4/4 token-exact;
this run measures speed only. Per-run data is in
[2b-plan-vs-plain-24k.csv](2b-plan-vs-plain-24k.csv), the machine-readable form
in [2b-plan-vs-plain-24k.json](2b-plan-vs-plain-24k.json), and the six logs in
[raw/plan-vs-plain/](raw/plan-vs-plain/).

## Files

| File | Contents |
|---|---|
| [runs.csv](runs.csv) | 18 rows: source, model, repetition, read rate and every generation position |
| [stats.csv](stats.csv) | n, median, variance, sample standard deviation, minimum, maximum and the three raw values per cell |
| [summary.json](summary.json) | machine-readable rig, method, source commits, per-run data, statistics and deltas |
| [2b-plan-vs-plain-24k.csv](2b-plan-vs-plain-24k.csv) | six rows: plan and plain, three repetitions, read and all five generation positions |
| [2b-plan-vs-plain-24k.json](2b-plan-vs-plain-24k.json) | machine-readable plan-vs-plain configs, runs, statistics and speedups |
| [raw/plan-vs-plain/](raw/plan-vs-plain/) | the six plan-vs-plain logs |
| [raw/current/](raw/current/) | the nine current-run logs |
| [raw/v0.4.0+native-loader/](raw/v0.4.0+native-loader/) | the nine baseline logs |
| [baseline-loader.md](baseline-loader.md) | the exact three edits that made v0.4.0 run natively on Windows |

All 24 runs (18 plain/current-vs-v0.4.0 plus six plan-vs-plain) exited 0 and
contain the expected read and five generation rows.

## What changed with our modifications and additions

The important distinction is between the plain inference path and the rest of
the branch. `kindi/bench.py` and `kindi/engine.py` are unchanged, so the
current-vs-v0.4.0 table cannot show a speedup from the other work. The separate
2B plan section does exercise the tensor-core plan path.

| Area | Added or changed since v0.4.0 | Exercised by this benchmark? | What that means |
|---|---|---|---|
| Native Windows loader | `kindi/gpudrv.py` selects `nvcuda.dll` on Windows and `libcuda.so.1` elsewhere; `sched.py` and `chunk1.py` use `gpudrv.driver()` | Yes, needed to run | Loader only; there is no kernel or arithmetic change. This is why v0.4.0 needed the small baseline patch. |
| Turing tensor cores | `kindi/tc_turing.py`, `kindi/tc_turing.cu`, `kindi/tca_turing.cu`, `docs/TURING-TENSOR-CORES.md` | No | Targets sm_75. The 4060 Ti is sm_89, so it uses the plain path. No Turing gain is measured here. |
| Plan / speculative engine | `kindi/spec.py`: draft clamp, 2B one-split rule, 9B guard | Yes, the 2B plan at 24K; no speculative drafting | The plan section measures 302.4 vs 170.6 tok/s (1.77x). Drafting remains unmeasured. |
| Agent serving layer | `kindi/server.py`: bounded queue with 429, `/metrics`, Ollama-compatible `/api/tags`, `/api/chat`, `/api/generate` | No | Offline `bench` does not start the server. No HTTP throughput or queue result is claimed. |
| Installer safety | `install.sh` GRUB edit guard, card/Turing profiles | No | No runtime effect on this benchmark. |
| Configuration, CI and tests | generated config reference, CPU-only tests, plan-engine tests, installer fixture tests | No | They protect behavior; they do not change the measured kernels. |
| 4B model | `bonsai-4b` was already in `v0.4.0`'s `kindi/catalog.py` | Yes, as a model | This adds the measured 4B row on this rig; it is not a new model-support change. |
| CPU and tandem | CPU support roadmap, recovered tandem design and status; no tandem runtime | No | No CPU or tandem number appears here. |

In one sentence: the branch adds portability, a Turing path for sm_75, plan
correctness guards, serving features and safety/tests; the plain 2B/4B/9B
exact-head GPU path is still the 0.4.0 engine, which is why its measured
performance is essentially unchanged.

## Variance and limits

- This was not a controlled interleaved A/B. The per-position deltas move in
  both directions, and some cells have a standard deviation of several percent.
  The source identity is the stronger evidence: the benchmark and engine files
  are the same.
- No llama.cpp comparison was run here. The earlier P100 and K1200 tables in
  [../../BENCHMARKS.md](../../BENCHMARKS.md) are different cards and are not
  comparable rows for this rig.
- The 2B plan comparison is included; no speculative drafting, Turing, server,
  CPU or tandem measurement is included.
- The 24,576-token context is the tested limit. The packs advertise a 262,144
  token native context, but this run did not test beyond 24K.
- The benchmark is not a correctness test. The current 4B pack passed
  `kindi verify` 4/4 token-exact; no separate `verify` was run for the baseline.
- The GPU clocks were observed, not locked. The current logs show 2,805–2,820
  MHz and 35–47 °C; the baseline did not log those samples.

## Reproduce

Current branch:

```sh
python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 24576 bench
python -m kindi --pack ~/.kindi/packs/bonsai-4b --ctx 24576 bench
python -m kindi --pack ~/.kindi/packs/packs/packs/bonsai-9b --ctx 24576 bench
```

For the `v0.4.0` side, export the tag, apply the three loader edits in
[baseline-loader.md](baseline-loader.md), and run the same commands from that
tree. The model files and SHA-256 values are recorded in
[summary.json](summary.json); the 4B GGUF is
`Ternary-Bonsai-2-4B-PTQ1_0.gguf`, SHA-256
`56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60`.
