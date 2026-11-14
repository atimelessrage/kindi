# BF16 and FP8 support

Status: **native FP8 prompt GEMM implemented for sm_89+ (Ada, Hopper, Blackwell); synthetic tests
only; no Blackwell hardware result.** No Gemma 4 BF16/FP8 GGUF is available on the test host, so
end-to-end token agreement, real-model throughput and the Blackwell execution path are unmeasured.
A manual Blackwell sm_100/sm_120 lane exists; its local hardware gate on 2026-11-12 returned
**unsupported (exit 2)** because the host has only an Ada sm_89 device; see
[Blackwell sm_100/sm_120 verification lane](#blackwell-sm_100sm_120-verification-lane). The CPU
codecs, the decode GEMV and the new tensor-core kernels are covered by synthetic tests. The local
stores were checked on 2026-11-11; see [Real-model check](#real-model-check-2026-11-11).

## Supported formats

| GGUF type | Name | Layout | Kindi path |
|---|---|---|---|
| 30 | BF16 | one 16-bit value per weight | Gemma 4 decode GEMV, prompt dequant to f32 + cuBLAS, embeddings; Bonsai auxiliary tensors widened to f32 at convert time |
| 43 | F8_E4M3 | one byte per weight, OCP E4M3FN (bias 7, no infinities) | Gemma 4 decode GEMV; prompt batch uses native FP8 tensor cores on sm_89+; dequant to f32 + cuBLAS elsewhere; embeddings; Bonsai auxiliary tensors widened to f32 at convert time |
| 44 | F8_E5M2 | one byte per weight, IEEE E5M2 (bias 15, inf/NaN) | same as E4M3, with E5M2 tensor-core operands |

The FP8 ids follow GGML's appended `GGML_TYPE_F8_E4M3` (43) and `GGML_TYPE_F8_E5M2` (44). They do
not collide with the ternary ids Kindi uses (143/142/42) or with `TQ1_0`/`TQ2_0` (34/35).

## Gemma 4

`kindi/gemma4.py` accepts the three types in `inspect_model` and keeps the packed bytes resident.
Decode uses `k_g4_bf16_mv` / `k_g4_fp8_mv`, which read the packed weight directly. Token and
per-layer embeddings read BF16/FP8 rows directly. Q/K/V, FFN and projection matrices use the same
dispatch on one card.

### Native FP8 prompt GEMM

`kindi/gemma4_fp8.cu` adds the prompt GEMM on the FP8 tensor cores:

- `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` for GGUF type 43 and the `.e5m2.e5m2`
  form for type 44, with fp32 accumulators. The sm_89 instruction is the portable (non-`a`) form,
  so a Blackwell driver JITs the same PTX for sm_100/sm_120 tensor cores.
- A block computes 64 weight rows x 64 tokens with eight warps; each warp owns 16 rows x 32 tokens
  (four n8 tiles). K advances in 32-byte steps through shared-memory weight and activation tiles.
- Activations are quantized per token before the product: `scale = max|x| / 448` for E4M3 or
  `max|x| / 57344` for E5M2, with hardware `cvt.rn.satfinite` and rounding to nearest even. The
  epilogue multiplies each token's result by its scale. Weights are used in their stored FP8 form.
- This is a W8A8 product. It is not bit-exact with the old f32 dequantize-and-cuBLAS path; decode
  is unchanged and still dequantizes FP8 weights element by element.

`Ops.linear_batch` selects the native kernels when all of these hold: CUDA (not HIP), compute
capability >= 8.9, `cols % 32 == 0`, the module compiles and loads, and `KINDI_G4_FP8` is not `0`.
`KINDI_G4_FP8=0` forces the old path. Pre-Ada cards, non-multiple-of-32 widths and a failed module
load also fall back to dequantize-to-f32 + cuBLAS. `qkv_batch` and the FFN calls route through the
same decision. The two-card path calls each card's `forward_batch`, but its batch path has not been
re-run with the native kernels.

### Measured kernel timings

Synthetic random FP8 weights and f32 activations on the RTX 4060 Ti 8 GB (Ada sm_89, driver 617.42),
module compiled once, two warm-up calls then five timed calls. The time includes the per-token
quantization of the activations. E4M3 and E5M2 are within about 2% of each other; the table is
E4M3. Evidence: `bench-logs/fp8-mma-ada.log`.

| weight matrix (rows x cols) | tokens | time | TFLOPS |
|---|---:|---:|---:|
| 2560 x 2560 | 128 | 0.11 ms | 15.1 |
| 2560 x 2560 | 512 | 0.24 ms | 28.2 |
| 10240 x 2560 | 128 | 0.23 ms | 29.0 |
| 10240 x 2560 | 512 | 0.76 ms | 35.5 |

These are kernel microbenchmarks, not model throughput. The f32 dequantize + cuBLAS fallback could
not be timed on this host because cuBLASLt is not installed, so no speedup over it is claimed.

### Blackwell sm_100/sm_120 verification lane

`tools/verify_blackwell_fp8.py` is the hardware gate for the path that the Ada numbers above do not
cover. It reads the active device's compute capability and accepts exactly `100` (`sm_100`) or
`120` (`sm_120`). A non-Blackwell device, an architecture mismatch for `--arch`, a missing CUDA
device or a missing CuPy is recorded as **unsupported** and exits 2; it is never a green FP8 lane.
A kernel compile/load failure or a numerical mismatch is a **fail** and exits 1.

On supported hardware the tool checks both E4M3 and E5M2 quantization, the native GEMM at
full-tile and tail shapes (64x64x64, 37x96x5 and 130x128x70), and the `Ops.linear_batch` dispatch
against the same dequantized f32 reference and `2e-3 * max|ref|` tolerance used by the GPU tests.
It records the device, compute capability, driver/CuPy/CUDA versions, per-check errors and timings
in `results.json`, `RESULTS.md` and `run.log`.

```sh
python tools/verify_blackwell_fp8.py --dry-run
python tools/verify_blackwell_fp8.py --arch sm_100 --out bench-logs/blackwell-fp8-sm100
python tools/verify_blackwell_fp8.py --arch sm_120 --out bench-logs/blackwell-fp8-sm120
```

The manual [.github/workflows/blackwell-fp8.yml](../.github/workflows/blackwell-fp8.yml) lane runs
the same tool on a self-hosted runner labelled `self-hosted`, `gpu` and the selected architecture
(`sm_100` or `sm_120`); dispatch it once with each architecture to cover both variants. It needs
no model pack. This host has only an Ada RTX 4060 Ti. A local gate run on 2026-11-12 (`--arch auto`) detected
compute capability 89 and returned **unsupported** with exit 2; the result is recorded in
[bench-logs/blackwell-fp8-ada-refusal-20261112/RESULTS.md](benchmarks/blackwell-fp8-ada-refusal-20261112/RESULTS.md).
No Blackwell line has been produced; the lane's existence is not a hardware result.

A dispatch attempt on 2026-11-12 was blocked before any runner job: the local GPU is sm_89, the
workflow was untracked and absent from both remote branches, and no authenticated Actions client
was available. The exact checks are recorded in
[bench-logs/blackwell-fp8-dispatch-20261112/DISPATCH-BLOCKED.md](benchmarks/blackwell-fp8-dispatch-20261112/DISPATCH-BLOCKED.md).
Neither Blackwell architecture has a hardware result.

## Bonsai packs

`kindi convert` accepts BF16 and E4M3/E5M2 auxiliary tensors and widens them exactly to float32 in
`f32.bin`. The ternary language matrices must still be PTQ1_0, PQ2_0 or Q2_0: FP8 is not a ternary
replacement and there is no FP8 branch in the Bonsai tensor-core or plan engines.

## Real-model check (2026-11-11)

The available local stores were checked before running the gate:

- `D:\ollamaModels` contains `gemma4:26b`: Q4_K_M, 30 layers, width 2816, 128 routed experts
  (top-8) plus a dense shared MLP, expert width 704; 658 tensors, 17,074,419,072 B. The routed expert
  weights are 14,479,917,056 B and the dense core is 2,578,661,496 B. Language tensor types are
  Q4_K/Q5_K/Q5_1/Q8_0/F32. It has no BF16 (type 30), E4M3 (43) or E5M2 (44) language tensors, and
  `kindi.gemma4.inspect_model` rejects it on the unsupported type 7 before reaching the E4B
  dimension check. The cache adaptation, slot sizes and gates are in
  [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md).
- `T:\strata\models\models` contains Qwen3.6-35B-A3B UD-Q4_K_XL (`qwen35moe`), not a Gemma 4 model.
- There is no llama.cpp CLI in the workspace or on `PATH`, and no `reference-*.json` for a Gemma 4
  E4B model. `kindi/reference-nemotron-bolt-e4b.json` belongs to the Nemotron Bolt E4B and cannot
  stand in for another checkpoint.

The Gemma 4 engine is E4B-only (42 layers, 2560 width), while the 26B has 15.9 GiB of weights and
would not fit the 8 GB card even if its tensor types were supported. The real-model token-exactness
gate therefore remains open. A suitable input is a Gemma 4 E4B GGUF whose language matrices are
BF16 or FP8, plus a pinned llama.cpp greedy reference for the same file.

## What is not claimed

- No real FP8 or BF16 Gemma 4 GGUF has been validated; there is no end-to-end token agreement or
  model throughput result. The local stores hold only a Q4_K_M 26B Gemma 4 and a Qwen3.6 model (see
  [Real-model check](#real-model-check-2026-11-11)).
- The native prompt product is W8A8, not the f32 path. The per-token activation scale keeps the
  range, but the result differs from the dequantized f32 reference by FP8 rounding.
- Blackwell (sm_100/sm_120) execution is enabled by capability and uses the same portable PTX. A
  dedicated hardware lane exists; the local hardware gate returned `unsupported` on the only
  available Ada device and no Blackwell runner was available, so there is no Blackwell pass or
  throughput result.
- No FP8 KV cache or FP8 attention path; no FP8 Bonsai language weights. BF16 has no native MMA
  path and still dequantizes to f32 for prompt batches.
- The NVFP4/FP8/AWQ/GPTQ/EXL2 boxes in a generic GPU table are hardware capabilities, not claims
  that Kindi uses them. The 5060 Ti's native FP4 remains unused.

## Verification commands

```sh
python -m pytest tests/test_dtypes.py tests/test_gemma4_fp8.py tests/test_gemma4_fp8_mma.py -q
python -m pytest tests/test_verify_blackwell_fp8.py -q
python tools/verify_blackwell_fp8.py --dry-run
python bench-logs/bench_fp8_mma.py
python -m compileall -q kindi
```

The CUDA test files need CuPy and an Ada/Blackwell device; they skip otherwise. The Blackwell
tool needs a real sm_100 or sm_120 device for anything beyond `--dry-run`; its CPU-side test file
runs without CUDA.
