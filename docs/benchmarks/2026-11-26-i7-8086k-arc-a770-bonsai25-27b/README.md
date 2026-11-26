# Run Bonsai 2.5 27B on an Arc A770 (2026-11-26)

This is the first iteration where **Bonsai 2.5 itself runs on the A770**, not just a 2B probe.
The 27B is the only size the 2.5 release has (`docs/BONSAI-2-SUPPORT.md` §1a), it is 6.72 GB of
ternary codes, and it did not run before for three separate reasons — all three are fixed here.

| | |
|---|---|
| host | i7-8086K, 32 GiB RAM, Windows |
| device | Intel(R) Arc(TM) A770 Graphics, 512 CU, 15932 MiB, driver 32.0.101.9175, OpenCL C 1.2, no `cl_khr_fp64` |
| pack | `bonsai-2.5-27b`, built by `kindi use bonsai-2.5-27b` from `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` (SHA-256 checked against `kindi/catalog.py`), 7242 MB, 851 tensors, 64 layers / D 5120 / FFN 17408 |
| oracle | `kindi.cpu.Engine` on the same pack (it fits: 17 GiB free with both loaded) |

## What blocked the 27B, and what fixed it

1. **The driver refuses a 6.7 GB allocation.** `CL_DEVICE_MAX_MEM_ALLOC_SIZE` on this A770 is
   **4095 MiB**, so `clCreateBuffer` on the whole code blob returned `0xffffffc3`
   (`CL_INVALID_BUFFER_SIZE`). The codes are now grouped into buffers of at most 2 GiB **on tensor
   boundaries**, so a tensor never straddles a buffer and each GEMV's byte offset is chunk-relative.
2. **The head is not the embedding.** The 2B ties its head to `token_embd.weight`; the 27B ships a
   separate `output.weight` and its manifest names it (`head = output.weight`). The step hardcoded
   the tied name, so the first 27B run decoded to non-Latin tokens — a wrong-head bug, fixed by
   reading `head` from the manifest.
3. **7 GB of codes does not belong in RAM.** The loader read `bcodes.bin` whole; it now memory-maps
   it, and the chunk uploads are views of that mapping, so residency is the page cache's business.

Plus the run path itself: `kindi intel-run` decodes a prompt on the device without constructing a
CPU engine, which is what makes a model this size runnable at all on a 32 GiB host.

## Results

| run | command tail | result |
|---|---|---|
| teacher-forced step vs the CPU engine | `intel-step --steps 4` | **4/4 tokens match**, max \|Δlogit\| 2.18e-04 |
| step cost | same | **779.14 ms/step**, 1366 enqueues/step |
| completion | `intel-run --prompt "The capital of France is"` | `" Paris."` then `<|im_end|>` (3 tokens) |
| chat completion, 24 tokens | `intel-run --chat --prompt "Name the three primary colors."` | `The three primary colors are **red**, **blue**, and **yellow**.` — 816.41 ms/step |

The tokens the Arc produces are the tokens the CPU engine produces, at the same positions, on the
same pack — so this is a correctness result, not just "it printed something".

## What this is not

- **Not fast.** 779 ms/step is 1.3 tokens/s. The cost is the same one the 2B step has, 32× over:
  one blocking read per GEMV group, 1366 enqueues a token, and the host running the delta rule,
  conv, attention and softmax for 64 layers in numpy. The next order of magnitude is Phase B in
  `docs/INTEL-ARC-KERNELS.md` (K5/K6/K7 plus the per-head state update on the device), not wiring.
- **Not a backend.** There is no `kindi serve` path for this; it is a probe that runs the model.
- **Not the MTP file.** This page is the plain `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf`; the `-mtp`
  pack is measured separately.

## Reproduce

```
python -m kindi use bonsai-2.5-27b
python -m kindi intel-step --pack ~/.kindi/packs/bonsai-2.5-27b --steps 4 \
    --device-name "Intel(R) Arc(TM) A770 Graphics" --json step.json
python -m kindi intel-run --pack ~/.kindi/packs/bonsai-2.5-27b \
    --prompt "Name the three primary colors." --tokens 24 --ctx 1024 --chat \
    --device-name "Intel(R) Arc(TM) A770 Graphics" --json run-chat.json
```

`step.json`, `run-plain.json` and `run-chat.json` in this directory are the runs above, unedited.
