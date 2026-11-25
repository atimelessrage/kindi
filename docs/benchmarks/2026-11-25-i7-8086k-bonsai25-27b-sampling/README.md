# Bonsai 2.5 27B publisher sampling on the CPU engine

Host: i7-8086K, Windows-10-10.0.19045-SP0, 12 cores, Python 3.13.6, no CUDA device. Date: 2026-11-25.

Both Bonsai 2.5 27B files in `T:/kindi-models` were measured: the plain
`Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` and the `-mtp` file (the same model plus its draft head
as a 65th block). Everything below is the **CPU** engine: the Arc A770 has no inference
backend yet, so no GPU figure exists.

## What was measured

Prism's card gives one sampling set per mode and three reasoning efforts (`xhigh` default,
`medium`, and `low`, which the model does not support and treats close to `xhigh`).
`tools/bonsai_sampling_bench.py` asks `kindi.profiles` for the set (so it cannot drift from
the server), renders the 27B template and samples with `kindi.sampling.Sampler`.

Settings the runs used (read back from the JSON, not retyped):

| mode | temperature | top_p | top_k | min_p | presence_penalty | repeat_penalty |
|---|---:|---:|---:|---:|---:|---:|
| thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| instruct | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |

```sh
python -m kindi convert Ternary-Bonsai-2.5-27B-PTQ1_0.gguf --out pack
KINDI_DEVICE=cpu python tools/bonsai_sampling_bench.py --pack pack \
    --out run-base.json --tokens 32
python -m kindi convert Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf --out pack
KINDI_DEVICE=cpu python tools/bonsai_sampling_bench.py --pack pack \
    --out run-mtp.json --tokens 32
```

Each pack loads once; `instruct` requests run with the instruct set, and thinking requests
run at `xhigh`, `medium` and `low`. Each row is 32 sampled tokens on one of two prompts, with
one seed per prompt, so the efforts differ only by the instruction the template renders.

## Plain Bonsai 2.5 27B pack

Load: 341.1 s. Engine: `kindi.cpu.Engine`.

| mode | reasoning effort | prompt | decode | reply |
|---|---|---|---:|---|
| instruct | - | In one sentence: what is a ternary weight? | 2.54 tok/s | `A ternary weight refers to the use of the number 3 (or ba...` |
| instruct | - | 2 + 2 = ? Answer with the number only. | 2.01 tok/s | `4<\|im_end\|> <\|im_start\|><\|im_end\|> <\|im_start\|><\...` |
| thinking | xhigh | In one sentence: what is a ternary weight? | 1.88 tok/s | `The user asks: "In one sentence: what is a ternary weight...` |
| thinking | xhigh | 2 + 2 = ? Answer with the number only. | 2.39 tok/s | `The user is asking a simple math question: 2 + 2 = ? They...` |
| thinking | medium | In one sentence: what is a ternary weight? | 2.09 tok/s | `The user asks for a one-sentence definition of a "ternary...` |
| thinking | medium | 2 + 2 = ? Answer with the number only. | 1.95 tok/s | `The user asks a simple math question: 2 + 2 = ? and wants...` |
| thinking | low | In one sentence: what is a ternary weight? | 2.25 tok/s | `The user asks: "In one sentence: what is a ternary weight...` |
| thinking | low | 2 + 2 = ? Answer with the number only. | 2.35 tok/s | `The user is asking a simple math question: 2 + 2 = ? They...` |

`low` and `xhigh` produced identical token ids on both prompts: **yes**.

## `-mtp` pack

Load: 333.5 s. Engine: `kindi.cpu.Engine`. The manifest records the head's block and tensor names, but
nothing drafted: this pack is the same 64-layer model (`docs/MTP.md`; section 1a of
`docs/BONSAI-2-SUPPORT.md` covers the intake).

| mode | reasoning effort | prompt | decode | reply |
|---|---|---|---:|---|
| instruct | - | In one sentence: what is a ternary weight? | 3.46 tok/s | `A ternary weight refers to the use of the number 3 (or ba...` |
| instruct | - | 2 + 2 = ? Answer with the number only. | 3.29 tok/s | `4<\|im_end\|> <\|im_start\|><\|im_end\|> <\|im_start\|><\...` |
| thinking | xhigh | In one sentence: what is a ternary weight? | 3.46 tok/s | `The user asks: "In one sentence: what is a ternary weight...` |
| thinking | xhigh | 2 + 2 = ? Answer with the number only. | 3.31 tok/s | `The user is asking a simple math question: 2 + 2 = ? They...` |
| thinking | medium | In one sentence: what is a ternary weight? | 3.29 tok/s | `The user asks for a one-sentence definition of a "ternary...` |
| thinking | medium | 2 + 2 = ? Answer with the number only. | 3.47 tok/s | `The user asks a simple math question: 2 + 2 = ? and wants...` |
| thinking | low | In one sentence: what is a ternary weight? | 2.58 tok/s | `The user asks: "In one sentence: what is a ternary weight...` |
| thinking | low | 2 + 2 = ? Answer with the number only. | 3.50 tok/s | `The user is asking a simple math question: 2 + 2 = ? They...` |

`low` and `xhigh` produced identical token ids on both prompts: **yes**.

Every row's token fingerprint equals the plain pack's: **yes** - the head is recorded in the
manifest and not part of what runs, so the two files are the same model numerically.

## What this shows

- Both publisher sets run end to end on both 2.5 27B files: the profile supplies exactly the
  card's values (table above), and the two modes produce different text from the same model.
- `reasoning_effort=low` is accepted and renders the `xhigh` instruction (DEC-0013), so on a
  shared seed its ids equal `xhigh`'s - the check the table reports per pack.
- The `-mtp` file needs no separate sampling path: with its head recorded but not drafting, it
  samples through the same profile and sampler as the plain file.

## What this does not show

- **No GPU number.** The Arc A770 toolchain is probe-only, so these rates are CPU-only.
- **No quality ranking.** Two prompts and 32 tokens each say nothing about which effort or
  mode answers better; they show that the settings run and how fast.
- **No exactness claim.** There is no llama.cpp reference for this release; `kindi verify`
  reports that and exits 2.
- Decode rates are single-host figures and vary run to run; do not read a mode difference
  into them.
