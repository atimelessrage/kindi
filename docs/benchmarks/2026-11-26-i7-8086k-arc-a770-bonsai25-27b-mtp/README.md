# Bonsai 2.5 27B, the `-mtp` file, on an Arc A770 (2026-11-26)

The plain 2.5 27B runs on the A770 (see `../2026-11-26-i7-8086k-arc-a770-bonsai25-27b/`). This page
is the **`-mtp` release file**, `Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` (6,232,818,944 B, SHA-256
checked against `kindi/catalog.py`), converted and decoded on the same card.

| | |
|---|---|
| host | i7-8086K, 32 GiB RAM, Windows |
| device | Intel(R) Arc(TM) A770 Graphics, 512 CU, 15932 MiB, driver 32.0.101.9175, OpenCL C 1.2 |
| pack | `bonsai-2.5-27b-mtp`, 64 layers / 851 tensors, head `output.weight`, 1366 enqueues/step |
| oracle | `kindi.cpu.Engine` on the same pack |

## What the `-mtp` file is, and what ran

`blk.64` is the draft head, not a model layer (15 tensors, `qwen35.nextn_predict_layers = 1`), and
`kindi convert` keeps it **out** of the pack: the pack is the 64-layer language model and the
manifest records the head under `mtp_head` with the note that Kindi does not draft with the 27B head
(`docs/MTP.md`). So "the MTP version runs on the Arc" means exactly this: **the `-mtp` release file's
base model loads and decodes on the device**, with the head recorded and unused — the same thing it
does on any other backend today. No speculative decoding happens on the Arc, because it does not
happen for the 27B head at all yet.

## Results

| run | result |
|---|---|
| `intel-step --steps 4` (teacher-forced vs the CPU engine) | **4/4 tokens match**, max \|Δlogit\| 2.18e-04, **869.65 ms/step** |
| `intel-run --prompt "The capital of France is"` | `[11751, 13, 248046]` = `" Paris.<|im_end|>"` — 911.14 ms/step |
| `intel-run --chat --tokens 24` | `The three primary colors are **red**, **blue**, and **yellow**...` — 890.26 ms/step |

The strongest check here is a cross-file one: the token ids are **the same as the plain pack's**, and
so are the CPU engine's reference tokens `[9624, 271, 2, 561]` at the four compared positions. The
two release files carry the same language model, and the Arc path reads both identically.

## Reproduce

```
python -m kindi use bonsai-2.5-27b-mtp
python -m kindi intel-step --pack ~/.kindi/packs/bonsai-2.5-27b-mtp --steps 4 \
    --device-name "Intel(R) Arc(TM) A770 Graphics" --json step.json
python -m kindi intel-run --pack ~/.kindi/packs/bonsai-2.5-27b-mtp \
    --prompt "The capital of France is" --tokens 4 --ctx 1024 \
    --device-name "Intel(R) Arc(TM) A770 Graphics" --json run-plain.json
```

Correctness, not speed: ~1.1 tokens/s, with conv, the delta rule, attention, softmax and the FFN
elementwise still on the host (`docs/INTEL-ARC.md`, acceptance matrix P5).
