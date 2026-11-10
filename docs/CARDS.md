# How to run Kindi on each card

The installer ([INSTALL.md](INSTALL.md)) uses these settings. It selects them from the cards that it finds.

Blocking sync (`KINDI_BLOCKING_SYNC`, on by default) lets the host thread sleep while it waits for the card. It
costs 2-3% of the decode speed on the GTX 1080. With one server on its own card, set `KINDI_BLOCKING_SYNC=0`
(the installer does this). The RTX 3080 profile keeps it on: there, many servers share few CPU cores.

This page gives the settings for each tested card. It also gives the speeds that we measured with these settings.
All the cards use the same Docker image (CUDA 10.2, CuPy 9.6). The `kindi` folder of this repository is the code.

To make a pack, use `kindi convert` (see the README). Put the pack in `$HOME/.kindi/<pack>`. In the container, it
is at `/data/<pack>`. The examples serve on `127.0.0.1:8093`.

```
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
  -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data -e HOME=/tmp \
  -e KINDI_PACK=/data/pack -e KINDI_STATE_DIR=/data/state ... kindi:0.4.0
```

Replace `...` with the variables in the section for your card. To use one card of two, set `--gpus device=N`.

## Quadro K420, 1 GB (Kepler, sm_30)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-2B |
| variables | `KINDI_CTX=24576` (the default) |
| generation | 22.9 tokens/s (fast head, 2,000 tokens of context); 20.8 tokens/s (exact head, 512 tokens) |
| prompt reading | 31.6 tokens/s at 3K tokens, 28.1 at 12K (one token at a time: 25.3 and 23.4; dev build of 10-21) |

- The prompt path uses chunks of 128 tokens on this card. Then the context stays at 24,576 tokens.
- To read prompts one token at a time, set `KINDI_CHUNK=0`.

## Quadro K420, 2 GB (Kepler, sm_30)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-4B (the installer's choice), or the 2B |
| variables | `KINDI_CTX=23808` for the 4B (the context that fits), `KINDI_CTX=65536` for the 2B |
| generation | 2B: 21.2 tokens/s (exact head), 24.8 (fast head); 4B: 9.8 tokens/s |

## Quadro K1200, 4 GB (Maxwell, sm_50)

| item | value |
|---|---|
| models | 2B (`/data/pack`), 4B (`/data/pack4b`) or 9B (`/data/pack9b`) |
| variables | `KINDI_CTX=24576` |
| generation (exact / fast head) | 2B 71.6 / 88.3 tokens/s, 4B 35.0 / 42.2, 9B 20.7 / 24.9 |
| prompt reading at 3K tokens | 2B 186.6 tokens/s, 4B 74.5, 9B 40.0 |

- The card in an x16 slot is faster than the card in an x4 slot.

## Two Quadro K1200 cards (tensor parallel)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B (`/data/pack27b`) |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CTX=24576`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=4096`, `KINDI_REPEAT_STOP=1` |
| docker | `--gpus all` (the engine uses both cards), `--memory 12g` |
| generation | 12.6 tokens/s |
| prompt reading | 20.9 tokens/s at 512 tokens, 18.4 at 22K (dev code of 10-21, the chunk-path copy) |

- The K/V cache is q8_0. The output matches llama.cpp with `-ctk q8_0 -ctv q8_0`.
- Set the host to text mode. A desktop session turns on the CUDA watchdog, and the watchdog stops the exchange
  kernels.

## Quadro K6000, 12 GB (Kepler, sm_35)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` |
| generation | 20.4 tokens/s at 512 tokens, 16.4 at 8K, 13.1 at 24K |
| prompt reading | 36.2 tokens/s at 512 tokens, 30.2 at 8K, 26.2 at 24K |

- The card needs the NVIDIA 470 driver. The 470 driver does not report the compute capability, so the installer
  reads it from the card name.
- With all of the 12 GB, the engine fits a context of 156,672 tokens. The installer sets 65,536.
- Nemotron also runs on this card (experimental): 20.1 tokens/s of generation at 512 tokens. The installer does not
  offer it on Kepler cards.

## GeForce GTX 1080, 8 GB (Pascal, sm_61)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` |
| generation | 28.4 tokens/s at a short context, 16.0 at 60K |
| prompt reading | 249 tokens/s at 512 tokens, 237 at 3K, 213 at 12K, 188 at 22K |

- The first start writes the 5-trit weights into the pack (`g5n_*.bin`, 5.6 GB). This takes about 12 minutes.
  Later starts take about 30 seconds.
- The prompt products use int8 activations on this card. For fp32, set `KINDI_CHUNK_Q8=0` (110 tokens/s at 512
  tokens).
- On a host with more than one GPU, set `CUDA_VISIBLE_DEVICES` to the GTX 1080.
- On Windows (WSL2 or native), the WDDM driver keeps part of the card memory. Kindi fits the context to the free
  memory, so an 8 GB card gets a smaller context. A GTX 1070 with no monitor connected got 43,008 tokens for the 27B.
  `KINDI_MEM_RESERVE_MB` sets the memory that the fit keeps free. A smaller value gives more context, but the
  risk of an out-of-memory error is higher.

## GeForce RTX 3080, 10 GB (Ampere, sm_86)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=131072`, `KINDI_MEM_RESERVE_MB=128`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=1024`, `KINDI_PARK_MAX=1`, `KINDI_REPEAT_STOP=1` |
| docker | `--memory 14g` (the first load needs it) |
| context | 110,592 tokens (the fit to the free memory) |
| generation | 47-49 tokens/s at a short context, 29 at 64K |
| prompt reading | 491 tokens/s at 512 tokens, 381 at 22K |

These figures are from kindi-rel, 10-21, not the 0.2.0 build.

Tensor cores and speculative decoding (sm_80 and later). This is the profile of the RTX 3080 cluster servers, and
the installer uses it:

| variable | value | |
|---|---|---|
| `KINDI_TC` | `1` | int8 tensor-core products and CUDA graphs for the decode |
| `KINDI_FD` | `1` | split flash-decode attention: the K/V is read once per query, in chunks of 512 positions |
| `KINDI_ATT_TC` | `1` | the attention of the prompt chunks on the tensor cores |
| `KINDI_SPEC` | `draft2b` | speculative decoding with the 2B as the draft model (also `ngram` or `both`). Off by default |
| `KINDI_SPEC_K` | `3` | the draft tokens per verify pass (default 4) |
| `KINDI_SPEC_PACK` | the 2B pack | for `draft2b` (for example `/data/pack2b`) |
| `KINDI_SPEC_WINDOW` | `8192` | the context of the draft model |
| `KINDI_TC_TMAX` | `4` | the most tokens in one verify pass |
| `KINDI_CKPT_VRAM` | `1` | a checkpoint is a copy in card memory; its copy to the host waits until the server is idle |
| `KINDI_MEM_RESERVE_MB` | `160` | the memory that the CUDA graphs need after the context fit |
| `KINDI_DENSE_SPACING` | `2048` | the spacing of the checkpoints in the opening of a conversation |
| `KINDI_CHECKPOINT_BUDGET_MB` | `1280` | the host memory for checkpoints and parked conversations (`1024` on the agent servers) |
| `KINDI_PARK_MAX` | `1` | parked conversations (`0` on the agent servers: one conversation each) |
| `KINDI_PREFIX_MAX` | `2` | saved openings on disk |

With this profile: context 84,992 tokens; at 190 W on agent prompts 99.9 tokens/s greedy and 91.3 sampled; prompt
reading 1,093 tokens/s at 512 to 855 at 22K.

- With `KINDI_TC=1`, the server reads prompts in passes of up to `KINDI_TC_TMAX` tokens.
- Speculative greedy output is the same as the output without speculation.
- The cluster runs seven servers on one 4-core host with `--memory 6g`. One server on a computer of its own can use
  the default budgets.

## Tesla P100, 16 GB (Pascal, sm_60)

The Tesla P100 is a supported card in Kindi 0.4.0. A used card costs $78 to $160. It has 16 GB of HBM2 memory with a
bandwidth of 732 GB/s. Photo: GP100 chip of a Tesla P100, by Fritzchens Fritz, Wikimedia Commons (CC0).

| item | value |
|---|---|
| models | 2B, 4B, 9B, 27B; the 33B on two cards (`kindi use nanbeige-33b`) |
| variables | `KINDI_CARDS=1`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` (the installer sets them) |
| docker | `--memory 14g` |
| generation, 27B, one card (SXM2, 300 W) | 36.0 tokens/s at 512 tokens, 34.6 at 8K (llama.cpp: 18.7 and 17.7) |
| generation, 27B, two cards | 39.7 at 512 tokens, 39.0 at 8K |
| prompt reading, 27B, one card | 167.6 tokens/s at 512 tokens, 159.6 at 8K (llama.cpp: 146.4 and 142.7) |
| generation, 33B, two cards (PCIe, 200 W) | 21.0 tokens/s at 512 tokens, 19.6 at 8K, 15.7 at 32K, 12.3 at 64K (llama.cpp: 9.6 at 512, 8.9 at 8K) |

- On the same P100, Kindi writes the 27B 1.9 to 2.7 times as fast as llama.cpp (one card, SXM2 box: 1.9; PCIe box: 2.0
  on one card and 2.5 to 2.7 on two cards).
- The fp16 prompt path is on by default for sm_60 (`KINDI_P100_FP16_PROMPT=0` turns it off). With it, Kindi reads a
  prompt 14% (512 tokens) and 12% (8K) faster than llama.cpp on the same card. Without it, the read speed is below
  llama.cpp.
- Two P100 cards (PCIe, 200 W) hold the full 262,144-token context of the 27B, with 8.6 GB free on each card. Kindi reads a prompt of 262,016 tokens at 67.6 tokens/s and writes 15.8 tokens/s at that depth. Two cards also run the 33B.
- Speculative decoding is off on the P100. On this card, a verify pass over several tokens costs about as much as the
  same number of single steps, so a draft cannot save time.
- `kindi verify` passes on one P100 for the 2B, 4B and 27B (token-exact), and on two P100 cards for the 33B.
- The numbers come from two rented machines (SXM2 with NVLink, and PCIe). The per-row setup notes are internal
  and are not part of this release.

## NVIDIA CMP 170HX, 40 GB (GA100, sm_80)

The CMP 170HX is a GA100 mining card. The boards that we measured have 40 GB of HBM2e (device id 0x2082, VBIOS
92.00.6D.00.0A), PCIe gen4 x16 and no limit on the FP32 FMA rate. Other CMP 170HX boards have 8 GB and a slower PCIe
link. We did not test the 8 GB board. Check `nvidia-smi` before you buy one. Measured bandwidth: 1,821 GB/s (read).

| item | value |
|---|---|
| models | 2B, 4B, 9B, 27B (one or two cards), AH-Nanbeige4.3-33B IQ3_XXS (one or two cards) |
| variables | `KINDI_TC=1`, `KINDI_FD=1`, `KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4` (plus `KINDI_TP=1` for the 2B, 4B and 9B; `KINDI_CARDS=2` for two cards) |
| generation, 27B, one card | 113.4 tokens/s at 512 tokens, 110.1 at 8K, 92.9 at 32K, 77.2 at 64K (llama.cpp: 106.1, 96.3, 74.1, 56.1) |
| generation, 27B, two cards | 121.2 at 512 tokens, 118.8 at 8K, 107.8 at 32K, 93.4 at 64K (llama.cpp: 107.3, 97.8, 76.1, 58.2) |
| generation, 33B, one card | 59.7 at 512 tokens, 55.5 at 8K, 42.0 at 32K, 32.6 at 64K (llama.cpp: 48.5, 45.9, 40.0, 34.3) |
| generation, 33B, two cards | 64.0 at 512 tokens, 59.1 at 8K, 45.0 at 32K, 35.3 at 64K (llama.cpp: 53.1, 50.2, 43.2, 37.0) |
| prompt reading, 27B, one card | 1,785 tokens/s at 512 tokens, 1,607 at 8K (llama.cpp: 1,398 and 1,381) |

- On sm_80 cards with 20 GB or more, the decode products use the b2 weight copy. On this card the five-trit decode,
  not the memory, limited the products.
- `kindi verify` passes on one and two cards for the 27B and the 33B. On two cards, the 27B ends one prompt at a
  llama.cpp tie (gap 0.0064 nats).

## GeForce RTX 20, 30, 40 and 50 series

Kindi 0.4.0 supports all four series. The Ampere tensor-core path and speculative decoding use cards of compute
capability 8.0 and later (RTX 30, 40 and 50). The RTX 20 series (Turing, sm_75) has its own tensor-core backend
([TURING-TENSOR-CORES.md](TURING-TENSOR-CORES.md)), selected with `KINDI_TC=1`; see the RTX 4000 section below. Its
runtime verification on sm_75 hardware is still pending.

The installer classifies a card by compute capability, not by memory: every sm_80 card gets the `tc` profile,
including the 8 GB RTX 4060 Ti, whose default context is 24,576 tokens. Two cards of the same class with at least
4 GB each can run one model together; two sm_80 cards get the `tc2` profile (tensor cores on, speculative decoding
off until the two-card verify is recorded).

| card | what we measured |
|---|---|
| RTX 3080 (190 W) | 27B: 62.3 tokens/s at 512 tokens. With the 2B draft model: 91.1 (1.46 times) at 512 tokens, 103.2 (1.68 times) at 8K with k=2, and 111.5 (1.82 times) at 8K with k=3. The same output as without the draft model. Against llama.cpp: see [BENCHMARKS.md](BENCHMARKS.md) |
| RTX 3060 12 GB | 27B: 41.1 tokens/s at 512 tokens; with the 2B draft model 60.6 (1.48 times), 73.5 (1.82 times) at 8K, 66.7 (1.81 times) at 24K |
| RTX 4060 Ti 8 GB | tc plan: 2B 4/4 token-exact; 4B 2/4; 9B refused (Hadamard 4096). Plain 4B/9B 4/4. Details below |
| other RTX 20, 40 and 50 cards | supported on the plain engine; `kindi verify` passes. We give no speed numbers |

### RTX 4060 Ti 8 GB (Ada, sm_89)

| item | value |
|---|---|
| models | 2B, 4B, 9B |
| tc plan | `KINDI_TC=1`, `KINDI_TP=1`, `KINDI_FD=1`, `KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4`, `KINDI_SPEC=0`, `KINDI_HEAD=exact`; verify uses its fixed 1,024-token context |
| verify, tc plan (2026-11-10) | 2B **4/4 token-exact**, 354.5 tokens/s; 4B **2/4**: prompt 2 6/40 (first difference at token 6, llama.cpp top-2 gap 0.0447), prompt 3 37/40 (first difference at token 37); 9B **refused before graph build**: the tensor-core plan supports Hadamard blocks 512 and 1024, and this pack has 4096 |
| verify, plain engine | 4B **4/4 token-exact**, 114.9 tokens/s; 9B **4/4 token-exact** (13/13 on prompt 3), 72.5 tokens/s |
| result | The tc plan is verified for the 2B only. The 4B divergence and the 9B Hadamard block are open; [CARD-SUPPORT.md](CARD-SUPPORT.md) tracks them |

- The 8 GB `tc` installer profile only adds tensor-core settings to the 27B (and the 2B when its MTP head is installed); the 2B/4B/9B settings above are the documented tc plan. The plain 4B/9B paths remain the safe default.

## AH-Nanbeige4.3-33B IQ3_XXS

| item | value |
|---|---|
| model | AH-Nanbeige4.3-33B IQ3_XXS (our IQ3_XXS quantization of Nanbeige4.3-33B, made with bartowski's imatrix): AH-Nanbeige4.3-33B-IQ3_XXS.gguf (`kindi use nanbeige-33b`); model card: https://huggingface.co/ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF |
| cards | one card with 16 GB or more (for example an RTX 3090), or several cards with about 17 GB in total (measured: 2x RTX 3080, 2x Tesla P100); a Quadro K6000 with layers in host memory |
| variables | `KINDI_CTX=16384`, `KINDI_REPEAT_STOP=1`, `CUDA_VISIBLE_DEVICES=<the two cards>` |
| docker | `--memory 10g` |
| generation | 2x RTX 3080 (190 W): 45.5 tokens/s at a short context, 44.3 at 2K, 40.7 at 8K, 37.0 at 16K |
| generation | 2x Tesla P100: 21.0 tokens/s at 512 tokens, 19.6 at 8K. Quadro K6000: 3.56 tokens/s at 8K |
| prompt reading | 1,047 tokens/s at 512 tokens, 1,209 at 16K (2x RTX 3080) |

It is a strong model for coding work, and it can do even better on a slightly faster GPU.

The backend puts the layers in order on the cards. Only the hidden state goes between the cards, so PCIe x1 is
sufficient. The GGUF file has no MTP head. Prompt lookup is on by default: when the model copies or edits text, it
writes about 2 times as fast (91.6 against 45.1 tokens/s on two RTX 3080 cards at 8K).

On one Quadro K6000 (12 GB), 13 layers stay in host memory (at a context of 8,448 tokens). The card copies them in
while it computes, and the CPU computes 55% of the FFN rows of each host layer. Kindi writes 3.9 tokens/s at 480
tokens and 3.56 at 8,160 tokens, and llama.cpp (the same card, 10 layers on the CPU) writes 2.19 and 1.52.

An agent tool such as opencode can use the 27B and the 33B together, because both run on Kindi's OpenAI-compatible
server. Use the 27B for fast steps and the 33B for harder coding steps.

## AMD Radeon 780M (RDNA3, gfx1103)

The Radeon 780M is the graphics unit of the Ryzen 7 7840HS and Ryzen 9 7940HS. It has no memory of its own and uses the
memory of the computer (32 GB or more for the 27B). Kindi runs it through ROCm, in the `kindi-rocm` image, with the kernel
option `amdgpu.cwsr_enable=0`. [AMD.md](AMD.md) gives the settings.

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B, Nemotron or the 2B |
| generation | 10.6 tokens/s at 512 tokens (27B) |
| prompt reading | 103.7 tokens/s at 512 tokens (27B) |

## GeForce RTX 4000, 8 GB (Turing, sm_75)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B (the 9B, 4B and 2B also run) |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact` or `fast`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536`, `KINDI_CHUNK_Q8=1`, `KINDI_BLOCKING_SYNC=0` |
| tensor cores | `KINDI_TC=1`, `KINDI_FD=1`, `KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4` |
| docker | `--memory 14g` (the first load needs it) |
| generation | not yet measured; expect Pascal-class speed or better (the tensor-core path is new) |

- Turing has int8 tensor cores. Kindi has a separate Turing backend
  ([TURING-TENSOR-CORES.md](TURING-TENSOR-CORES.md)): with `KINDI_TC=1` on an
  `sm_75` card it uses `mma.m8n8k16` for the decode products and the prompt
  chunks. The Ampere backend (`sm_80` and later) is unchanged.
- The installer's `turing` profile sets `KINDI_TC=1`, split flash-decode
  attention (`KINDI_FD=1`), tensor-core prompt attention (`KINDI_ATT_TC=1`) and,
  for the 27B, the 2B draft model (`KINDI_SPEC=draft2b`). It also adds the 2B
  pack.
- The weights keep the existing `tc` layout, and every Turing MMA is an exact
  integer sum, so greedy output should be token-identical to the 5-trit path.
  Verification on this card is still pending; `KINDI_TC=0` returns to the
  5-trit path.
- The 8 GB card fits about 65,536 tokens of context with the 27B. Larger Turing
  cards (11 GB and more) get `KINDI_CTX=131072` and `KINDI_MEM_RESERVE_MB=160`
  (the smaller reserve that context needs, as on the RTX 3080).

## Quadro P4000, 8 GB (Pascal, sm_61)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact` or `fast`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536`, `KINDI_CHUNK_Q8=1`, `KINDI_BLOCKING_SYNC=0` |
| generation | not yet measured; expect GTX 1080-class speed |

- Single-slot Pascal card. Use the same profile as the GeForce GTX 1080.

## NVIDIA P106-100, 6 GB (Pascal, sm_61)

| item | value |
|---|---|
| models | 2B, 4B or 9B; 27B on two cards (12 GB total) |
| variables | `KINDI_CTX=24576`, `KINDI_CHUNK_Q8=1`, `KINDI_BLOCKING_SYNC=0` |
| generation | not yet measured |

- A mining card with no display output. It works as a headless GPU, but check its PCIe link: many P106 cards run at x1 or x4 and are slower on prompt reading.
- The 27B does not fit in 6 GB. Two P106 cards can run it together: the installer pairs two cards of the same class with at least 3,800 MiB each.

## Quadro M4000, 8 GB (Maxwell 2.0, sm_52)

| item | value |
|---|---|
| models | 2B, 4B or 9B (`/data/pack`, `/data/pack4b`, `/data/pack9b`) |
| variables | `KINDI_CTX=24576`, `KINDI_BLOCKING_SYNC=0` |
| generation | not yet measured; expect Maxwell-class speed |

- 8 GB of memory. The 27B can be tried manually with `KINDI_PACK=/data/pack27b`; the engine fits the context to the free memory. It is not offered by default.

## GeForce 940M, 2 GB (Maxwell 1.0, sm_50)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-2B |
| variables | `KINDI_CTX=24576`, `KINDI_BLOCKING_SYNC=0` |
| generation | not yet measured |

- Laptop card, often behind Optimus. On most laptops the CPU engine is faster and simpler to use.

## GeForce GT 1030, 2 GB (Pascal, sm_61)

| item | value |
|---|---|
| models | 2B or 4B |
| variables | `KINDI_CTX=24576`, `KINDI_CHUNK_Q8=1`, `KINDI_BLOCKING_SYNC=0` |
| generation | not yet measured |

- Low-memory Pascal card. Keep the context at 24,576 tokens or less.

## Pascal and Turing card families

These two generations contain the cards that the installer profiles above cover. The class column is what
`card_class` in `install.sh` returns; the profile is the variables it sets. All the cards below run with `KINDI_TC=0`
(the 5-trit path). `KINDI_TC=1` adds tensor cores only where the card has them.

### Pascal (GP10x, `sm_61`; GP100 is `sm_60`)

| line | cards | compute | tensor cores | class / profile |
|---|---|---|---|---|
| consumer | Titan X (Pascal), Titan Xp, GTX 1080 Ti, 1080, 1070 Ti, 1070, 1060 6 GB and 3 GB, 1050 Ti, 1050, GT 1030 | sm_61 | no | `pascal8` at 8 GB and more; `maxwell` below (the GT 1030 section above) |
| workstation | Quadro P6000, P5000, P4000, P2200, P2000, P1000, P620, P600, P400 | sm_61 | no | `pascal8` at 8 GB and more, `maxwell` below (the P4000 section above) |
| mining | P106-100, P106-090, P104-100, P102-100 | sm_61 | no | `pascal8` at 8 GB and more, `maxwell` below (the P106-100 section above) |
| datacenter | Tesla P40, Tesla P4 | sm_61 | no | `pascal8` / `maxwell` |
| datacenter / workstation | Tesla P100 (12/16 GB HBM2), Quadro GP100 | **sm_60** | no | `p100`: the `tcp.cu`/`tcl.cu` lane kernels (GP100 has no dp4a) |

Pascal has no int8 tensor cores. With `KINDI_TC=1` the plans run the dp4a kernels of `tcp.cu` (`sm_61`) or the
`tcl.cu` lane-table kernels (`sm_60`); the products are the same integers as on the tensor-core path, so the output
is unchanged. A Pascal card needs the NVIDIA 470 driver branch (or newer for later cards).

### Turing (TU10x / TU11x, `sm_75`)

| line | cards | tensor cores | class / profile |
|---|---|---|---|
| consumer, RTX | Titan RTX, RTX 2080 Ti, 2080 Super, 2080, 2070 Super, 2070, 2060 Super, 2060 (6 GB and 12 GB) | yes (TU102/TU104/TU106) | `turing` (8 GB and more) |
| workstation, RTX | Quadro RTX 8000, 6000, 5000, 4000 (and RTX 3000 mobile) | yes | `turing` (8 GB and more) |
| datacenter | Tesla T4 (16 GB) | yes (TU104) | `turing` |
| mining | CMP 40HX (8 GB) | yes (TU106) | `turing` |
| consumer, GTX | GTX 1660 Ti, 1660 Super, 1660, 1650 Super, 1650, 1630 | **no** (TU116/TU117) | `maxwell` (plain engine, under 7.5 GB) |
| workstation, entry | Quadro T400, T600, T1000, T2000 | **no** (TU117) | `maxwell` (plain engine, under 7.5 GB) |
| mobile | GeForce MX450, MX550 (TU117) | **no** | `maxwell` |
| mining | CMP 30HX (6 GB) | **no** (TU116) | `maxwell` (under 7.5 GB) |

Only the TU102/TU104/TU106 parts have tensor cores, and only those of 7.5 GB and more take the `turing` profile and
the `tc_turing.cu` backend (a 6 GB RTX 2060 keeps the plain profile in this release; its tensor cores can be enabled
by hand with the variables from the RTX 4000 section). TU116/TU117 has the same compute capability 7.5 but no tensor
cores; an `mma` instruction there traps, so `tc_turing.supported()` also checks the device name and leaves those
cards on the plain engine (see [TURING-TENSOR-CORES.md](TURING-TENSOR-CORES.md)). Turning on `KINDI_TC=1` by hand on
a GTX 16-series card is a no-op: the backend is not selected.

## Unsupported NVIDIA cards

| card | reason |
|---|---|
| GeForce GT 430 and other Fermi / pre-Kepler cards (below sm_30) | Kindi needs Kepler (sm_30) or later; the CUDA 10.2 image does not support these cards |

- Use the CPU engine (`KINDI_DEVICE=cpu`) on these machines, or remove the card.

## AMD Radeon 840M (RDNA 3.5, gfx1152)

The Radeon 840M is the graphics unit of the Ryzen AI 5 340. It has 4 compute units and uses the memory of the computer.
ROCm runs it natively (no `HSA_OVERRIDE_GFX_VERSION`). We measured it in a container with 2 GB of host memory: Kindi
loads the 27B there, because it copies the weights to the card one tensor at a time.

| item | value |
|---|---|
| model | Ternary-Bonsai-2 2B, 4B, 9B or 27B |
| generation | 104.7 (2B), 49.1 (4B), 25.5 (9B), 7.90 (27B) tokens/s at 512 tokens |
| prompt reading | 265 (2B), 105 (4B), 56 (9B), 30.2 (27B) tokens/s at 512 tokens |

The installer builds the `kindi-rocm7` image (ROCm 7, HIP 7.16) for the 840M.

## AMD Radeon 8060S (Strix Halo, RDNA 3.5, gfx1151)

The Radeon 8060S is the graphics unit of the Ryzen AI MAX+ 395. It has 40 compute units and uses the memory of the
computer (256-bit LPDDR5X, measured read 240 GB/s). ROCm runs it natively. The installer builds the `kindi-rocm7`
image for it.

| item | value |
|---|---|
| model | Ternary-Bonsai-2 2B, 4B, 9B or 27B |
| generation | 260.0 (2B), 143.2 (4B), 88.8 (9B), 30.59 (27B) tokens/s at 512 tokens (llama.cpp: 236.1, 128.6, 86.0, 28.31) |
| prompt reading | 3,985 (2B), 1,952 (4B), 1,174 (9B), 375 (27B) tokens/s at 512 tokens (llama.cpp: 3,153, 1,642, 746, 301) |

- Kindi writes and reads faster than llama.cpp for all four models, at 512 and 8K tokens.
- All four models use the plan engine. Kindi waits for the unit in a loop (`KINDI_BLOCKING_SYNC=1` keeps the blocking
  wait).
- The AH-Nanbeige4.3-33B backend of Kindi needs CUDA. It does not run on AMD graphics.

## CPU (x86-64 with AVX2, FMA and F16C)

| item | value |
|---|---|
| models | 2B, 4B or 9B |
| variables | `KINDI_DEVICE=cpu`; optional `OMP_NUM_THREADS=<cores of one socket>`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` (the default is one thread per physical core, bound to those cores) |

- The CPU engine builds at the first start into `KINDI_CPU_CACHE`: `libkcpu.so` with gcc and OpenMP on
  Linux, `libkcpu.dll` with clang and libomp on Windows (`KINDI_CC`, `KINDI_LLVM`).
- The output is token-exact with llama.cpp. Verified on the i7-8086K: 2B/4B/9B 4/4 at 33.7/16.7/9.1
  tokens/s natively on Windows and 28.1/13.0/5.6 under WSL2
  ([benchmark](benchmarks/2026-11-10-i7-8086k-windows-cpu/README.md), [CPU-SUPPORT.md](CPU-SUPPORT.md)).
- The engine defaults to one thread per physical core of the CPUs it may run on. On the i7-8086K the 2B
  runs 39.2 tokens/s with 6 threads and 1.3 with 12: a token is one OpenMP parallel region with a
  barrier between row blocks, and the two hyperthread siblings of a core spin into each other at every
  one. See [CPU-SUPPORT.md](CPU-SUPPORT.md#why-12-threads-collapse).

### Intel Core i7-8086K (6 cores / 12 threads)

| item | value |
|---|---|
| models | 2B, 4B or 9B |
| variables | `KINDI_DEVICE=cpu`; six threads (the engine's physical-core default); on Windows set `KINDI_CC` and `KINDI_LLVM` to clang 18 and the LLVM install |
| build | `libkcpu.dll` with clang 18.1.8 and libomp on Windows, `libkcpu.so` with gcc and OpenMP on Linux |
| verify | 2B 4/4 at 33.7 tokens/s, 4B 4/4 at 16.7, 9B 4/4 at 9.1 (native Windows, 1,024-token context) |
| generation (24,576-token context, six threads, native Windows) | 2B 35.9 at 100 tokens, 29.3 at 2K, 19.4 at 8K, 11.9 at 16K, 8.2 at 24K; 4B 15.2 / 13.4 / 6.8 / 4.3 / 3.1; 9B 8.7 / 7.7 / 4.9 / 3.5 / 2.6 |
| prompt reading | 45.1 (2B), 19.2 (4B), 9.9 (9B) tokens/s (native Windows, 1,000-token prompt) |
| server | `kindi serve` with `KINDI_DEVICE=cpu` answered `POST /v1/chat/completions` non-stream and stream on the 2B: 19 prompt / 9 completion tokens at 32.9 tokens/s generation, 0 errors, 0 MiB GPU memory (native Windows, 1,024-token context, 2026-11-10) |

- The 12-thread setting is 6-20x slower; use 6 threads. Eight threads were faster on this chip in the
  probe (47.9 read, 37.3 generation), but they are not the general default.
- The 24K and prompt-reading rows are the native Windows 2026-11-10 run; the WSL2 numbers are archived
  in the benchmark page.
