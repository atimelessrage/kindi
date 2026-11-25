# Ternary Bonsai 2.5 27B on the CPU engine: load, greedy runs, determinism

Host: i7-8086K, Windows, Python 3.13, NumPy (no CUDA device). Date: 2026-11-25. Model:
`Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` from `prism-ml/Ternary-Bonsai-2.5-27B-GGUF`
(5,993,834,848 bytes; SHA-256 `929560ec…`, re-hashed here and matching `kindi/catalog.py`).

## What was done

```sh
python -m kindi convert Ternary-Bonsai-2.5-27B-PTQ1_0.gguf --out pack     # 7,242 MB in 370 s
python -c "from kindi.packcheck import check; check('pack', 'full', ...)"   # 10 files, 15.7 s
KINDI_DEVICE=cpu python -m kindi --pack pack verify                        # exit 2: no reference
KINDI_DEVICE=cpu python run_cpu.py --pack pack --out run1.json --tokens 16
KINDI_DEVICE=cpu python run_cpu.py --pack pack --out run2.json --tokens 16  # second process
```

`run_cpu.py` loads the pack on the CPU engine at ctx 2048, builds the prompt text itself, generates
`16 tokens` greedily for each of `2 prompts`, repeats each generation once in the same process, and
records the token ids, the text, the timings and a fingerprint of the ids.

## Results

| measurement | run 1 | run 2 |
|---|---:|---:|
| load | 339.9 s | 323.8 s |
| decode, prompt 1 (`In one sentence: what is a ternary weight?`) | 3.75 tok/s | 3.08 tok/s |
| decode, prompt 2 (`2 + 2 = ? Answer with the number only.`) | 4.02 tok/s | 3.63 tok/s |
| replies identical to a repeat in the same process | yes | yes |
| fingerprints identical across the two processes | yes | yes |

Engine line: `kindi-cpu ready: Ternary-Bonsai-2.5-27B, avx2-fma-f16c, 6 threads, ctx 2048`.
The manifest resolves to the `bonsai` backend as `bonsai2.5-27b`, and the pack reports `851 tensors`,
vocabulary `248320`, context `262144`, hadamard block 1024 and a 8952-character chat template - the
same facts the header-only run recorded, now from a built pack whose `bcodes`/`dpm`/`f32` sizes equal
the manifest's declared byte counts.

Prompt 1's reply is on-topic (`A ternary weight is a weight in a set of weights where each weight is`),
which is what shows the pack is loaded coherently rather than merely loaded. Prompt 2 replies `4` and
then repeats end-of-turn markers for the remaining tokens.

## What this shows

- The converter's 27B profile covers the 2.5 weights: the pack builds, packchecks (10 files, sha256)
  and loads on the CPU engine without a size, name or layout complaint.
- The registry identifies the release: the same pack would take Bonsai 2's id without the
  `Ternary-Bonsai-2.5-` prefix rule.
- Greedy CPU decoding is reproducible: identical token ids within a process and across two processes.
- `kindi verify` says `no llama.cpp reference for this model in this version of Kindi` and exits 2,
  which is the release-aware behaviour this round added - a 2.5 pack must not be scored against
  Bonsai 2's reference tokens.

## What this does not show

- **No exactness result.** llama.cpp has no published reference row for this release here, so nothing
  in this directory is a token-for-token comparison with another engine.
- **No GPU result.** The Arc A770 path is still a probe-only toolchain (`kindi intel-sycl`), and this
  host has no CUDA device, so the GPU verify cells stay `not run`.
- No sampling beyond greedy, no tool calling, no long context (2048 tokens), and no comparison of
  quality between Bonsai 2 and 2.5 on the same prompt.
- Decode rates are single-host CPU figures with 6 threads; the two runs differ by up to 20% between
  prompts and runs, so they are not a tuning comparison.
