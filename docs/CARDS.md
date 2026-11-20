# How to run Kindi on each card

The installer ([INSTALL.md](INSTALL.md)) uses these settings. It selects them from the cards that it finds.

Blocking sync (`KINDI_BLOCKING_SYNC`, on by default) lets the host thread sleep while it waits for the card. It
costs 2-3% of the decode speed on the GTX 1080. With one server on its own card, set `KINDI_BLOCKING_SYNC=0`
(the installer does this). The RTX 3080 profile keeps it on: there, many servers share few CPU cores.

This page gives the settings for each tested card. It also gives the speeds that we measured with these settings.
The pre-Ampere NVIDIA cards and the CPU use the CUDA 10.2 / CuPy 9.6 image; AMD cards use the ROCm images ([AMD.md](AMD.md)). Ada/Blackwell were measured on a modern CuPy host, and the CUDA 10.2 image path for sm_89/sm_120 is not yet verified; see [CARD-SUPPORT.md](CARD-SUPPORT.md). The `kindi` folder of this repository is the code.

To make a pack, use `kindi convert` (see the README). Put the pack in `$HOME/.kindi/<pack>`. In the container, it
is at `/data/<pack>`. The examples serve on `127.0.0.1:8093`.

```
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
  -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data -e HOME=/tmp \
  -e KINDI_PACK=/data/pack -e KINDI_STATE_DIR=/data/state ... kindi:0.7.0-fork
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
- The 43,008-token limit comes from the WDDM driver, not from Kindi or the card. On the same GTX 1070 in TCC mode
  (a driver change that Kindi does not need or support), the 27B got 66,560 tokens. `kindi verify` passed in both
  modes, and the write speed was the same (22.0 to 22.9 tokens/s). Report: epscheel (GitHub), issue #33.

## GeForce RTX 3080, 10 GB (Ampere, sm_86)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=131072`, `KINDI_MEM_RESERVE_MB=128`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=1024`, `KINDI_PARK_MAX=1`, `KINDI_REPEAT_STOP=1` |
| docker | `--memory 14g` (the first load needs it) |
| context | 110,592 tokens (the fit to the free memory) |
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

The Tesla P100 is a supported card in Kindi 0.4.0. A Tesla P100 costs $78 to $160. It has 16 GB of HBM2 memory with a
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
  prompt 14% (512 tokens) and 12% (8K) faster than llama.cpp on the same card.
- Two P100 cards (PCIe, 200 W) hold the full 262,144-token context of the 27B, with 8.6 GB free on each card. Kindi reads a prompt of 262,016 tokens at 67.6 tokens/s and writes 15.8 tokens/s at that depth. Two cards also run the 33B.
- Speculative decoding is off on the P100. On this card, a verify pass over several tokens costs about as much as the
  same number of single steps, so a draft cannot save time.
- `kindi verify` passes on one P100 for the 2B, 4B and 27B (token-exact), and on two P100 cards for the 33B.
- The numbers come from two rented machines (SXM2 with NVLink, and PCIe). The per-row setup notes are internal
  and are not part of this release.
- `KINDI_CHUNK_Q8` stays off on the P100, and setting it to 1 does not turn it on. The int8 prompt path uses the
  `dp4a` instruction, which starts at sm_61. The fp16 prompt path above replaces it on sm_60.
- A user report (Kindi 0.3.0, one P100-PCIE-16GB at 125 W): the 9B fitted 426,752 tokens of context and wrote
  41.3 tokens/s at that context; two 9B servers on one card gave 65.8 tokens/s in total (1.64 times one server).
  Report: scuderiapk (GitHub), issue #22.

## Tesla V100S, 32 GB (Volta, sm_70): user report

| item | value |
|---|---|
| models | 2B, 27B (Kindi 0.3.0, the stock Docker image, `KINDI_BLOCKING_SYNC=0`, exact head, context 24,576) |
| generation, 27B | 58.5 tokens/s at a short context, 52.0 at 16K, 48.2 at 24K; 56.6 over 512-token replies |
| prompt reading, 27B | 282 tokens/s on a 1,732-token prompt |
| generation, 2B | about 160 to 170 tokens/s at all depths to 16K |

- `kindi verify` passed. The card was capped at 185 W (its default is 250 W).
- Report: vntgeai (GitHub), issue #25. We did not measure this card ourselves.

## Quadro M5000, 8 GB (Maxwell, sm_52): user report

| item | value |
|---|---|
| models | 2B, 4B, 9B, 27B, Nemotron (Kindi 0.4.0, the packs of a K4000 reused without conversion) |
| generation | 2B 139, 4B 72, 9B 50, 27B 17.8 tokens/s at 512 tokens |
| prompt reading | 2B 695, 4B 264, 9B 142, 27B 39.0 tokens/s at 512 tokens |
| context | Nemotron at its full 131,072 tokens; the 27B at 24K to 28K |

- `kindi verify` passed on the 2B, 4B, 9B and 27B.
- With the NVIDIA 580 driver, set `KINDI_BLOCKING_SYNC=0`: the default (blocking sync) cost 5 to 15% of the write
  speed there (2B: 117 instead of 137 tokens/s). The 470 driver did not show this.
- The 580 driver also uses 29 MiB more card memory, so the 27B at 28,672 tokens was fitted down to 27,904.
- Report: rojecas (GitHub), issue #30. Installers before 0.6.0 showed this card as sm_50 on the 470 driver; it is
  sm_52.

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

Kindi 0.5.0 supports the RTX 20, 30 and 40 series. The RTX 20 tensor-core specialization still needs its sm_75
runtime verify. The Ampere tensor-core path and speculative decoding use cards of compute capability 8.0 and later
(RTX 30 and 40; the RTX 50/Blackwell path is detected but not yet verified).
The RTX 20 series (Turing, sm_75) has its own tensor-core backend
([TURING-TENSOR-CORES.md](TURING-TENSOR-CORES.md)), selected with `KINDI_TC=1`; see the RTX 4000 section below. Its
runtime verification on sm_75 hardware is still pending.

The installer classifies a card by compute capability, not by memory: every sm_80 card gets the `tc` profile,
including the 8 GB RTX 4060 Ti, whose default context is 24,576 tokens. Any two cards in the pair-supported
classes (Maxwell, Pascal, Turing, tensor-core or P100) with at least 4 GB each can be paired. Two sm_80 cards
get the `tc2` profile (tensor cores on, speculative decoding off until the two-card verify is recorded); mixed
generations use the plain `k1200x2` split, and a pre-6.0 + 6.0-or-newer pair adds `KINDI_G5=0` so both cards
use the 2-bit path. A pair of unequal-capacity cards (for example a 6 GB card with an 8 GB card) selects
`KINDI_TP_SPLIT=auto`, so the larger card receives the larger aligned share; equal-capacity pairs retain the
historical equal split. Kepler stays single-card. No mixed-card execution has been recorded yet.

| card | what we measured |
|---|---|
| RTX 3080 (190 W) | 27B: 62.3 tokens/s at 512 tokens. With the 2B draft model: 91.1 (1.46 times) at 512 tokens, 103.2 (1.68 times) at 8K with k=2, and 111.5 (1.82 times) at 8K with k=3. The same output as without the draft model. With DSpine: 115.5 tokens/s at 32K ([DSPINE.md](DSPINE.md)). Against llama.cpp: see [BENCHMARKS.md](BENCHMARKS.md) |
| RTX 3060 12 GB | 27B: 41.1 tokens/s at 512 tokens; with the 2B draft model 60.6 (1.48 times), 73.5 (1.82 times) at 8K, 66.7 (1.81 times) at 24K |
| RTX 4060 Ti 8 GB | tc plan (2026-11-11 one-pass matrix): 2B 4/4 token-exact, 370.2 tokens/s; 4B 3 exact + 1 tie (FD fallback); 9B 4/4 with `KINDI_TC_HP=1` (122.7 tokens/s; HB-4096 packs now get that exact setting automatically, 3/4 without it); 27B passes (3 exact + 1 tie; 38.9 one-pass, **38.1 median of 3**); 27B 24K benchmark 33.9 tokens/s at position 24,000 and server/cache pass. Plain 4B/9B 4/4. Details below |
| RTX 5060 Ti 16 GB | installer classifies it as `tc` (sm_80+), but there is no Kindi exactness or speed run on Blackwell yet; this is not a support claim. See [CARD-SUPPORT.md](CARD-SUPPORT.md) |
| other RTX 20, 40 and 50 cards | supported on the plain engine; `kindi verify` passes. We give no speed numbers. RTX 50/Blackwell is unverified (see the 5060 Ti row) |

### RTX 4060 Ti 8 GB (Ada, sm_89)

| item | value |
|---|---|
| models | 2B, 4B, 9B, 27B |
| tc plan | `KINDI_TC=1`, `KINDI_TP=1`, `KINDI_FD=1`, `KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4`, `KINDI_SPEC=0`, `KINDI_HEAD=exact`; verify uses its fixed 1,024-token context; the 24K run below uses the same profile with `KINDI_KV=q8_0` and `--ctx 24576` |
| benchmark, tc plan (2026-11-11) | 27B at ctx 24,576: prompt read 40.5 tokens/s; generation **37.6 / 37.1 / 36.6 / 34.7 / 33.9** tokens/s at positions 100 / 2,000 / 8,000 / 16,000 / 24,000; 724 nodes/decode token; **351 MiB free** |
| server/cache, tc plan (2026-11-11) | three real servers at ctx 24,576: `/health` 200, non-stream and stream chat, 888-token prompt; a **867-token opening** saved and loaded after restart; **cached 867/888** in process and after restart; replies bit-identical; 350-398 MiB free |
| verify, tc plan (2026-11-11 one-pass matrix) | 2B **4/4 token-exact**, 370.2 tokens/s; 4B **3/4 token-exact + 1 llama.cpp tie** with the automatic FD fallback (the forced-FD run was 2/4: prompt 2 6/40 at token 6, prompt 3 37/40 at token 37); 9B **4/4 token-exact with `KINDI_TC_HP=1`** (122.7 tokens/s; the pack's Hadamard 4096 kernels exist, but with the default int8 activations one prompt flips at token 36, so `make_engine` selects the high-precision activations for HB-4096 packs (`KINDI_TC_HP=0` still refuses, with that measurement in the message)); 27B **PASS: 3/4 token-exact + 1 llama.cpp tie** (tie at token 14, gap 0.0064), **38.9 tokens/s** in the one-pass matrix; the tc plan built T=1..4 with 724 nodes/decode token and left 1,128 MiB free |
| verify, plain engine | 4B **4/4 token-exact**, 114.9 tokens/s; 9B **4/4 token-exact** (13/13 on prompt 3), 72.5 tokens/s |
| result | The tc plan passes on the 2B (4/4), the 4B (3 exact + 1 accepted tie with FD fallback), the 9B (4/4 with `KINDI_TC_HP=1`) and the 27B (3 exact + 1 accepted tie). The 27B single-card path now has a 24K benchmark and a real server/cache restart cycle, but no long soak or concurrency; [CARD-SUPPORT.md](CARD-SUPPORT.md) tracks the remaining gaps |

- A single-run 2026-11-20 smoke on this host records the default 5-trit lane and the TC+HP path at 8K and 24K, with the exact commands and the single-run caveat: [9B default vs TC+HP smoke](benchmarks/2026-11-20-i7-8086k-rtx4060ti-9b-default-vs-tc/RESULTS.md).
- The 8 GB `tc` installer profile only adds tensor-core settings to the 27B (and the 2B when its MTP head is installed); the 2B/4B/9B/27B settings above are the documented tc plan. The plain 4B/9B paths remain the safe default.

#### Stock Kindi v0.6.1 user report (plain engine, 2026-11-17)

These rows are **stock upstream Kindi v0.6.1** (`7fececb`), not this fork and
not its `tc` or speculation tables. They use the stock plain engine and exact
head. The commands, logs and loader qualification are in the
[benchmark record](benchmarks/2026-11-17-i7-8086k-rtx4060ti-stock-0.6.1/RESULTS.md).

| model | context | prompt reading | generation @100 / 2,000 / 8,000 tokens | free |
|---|---:|---:|---|---:|
| 2B | 512 | 225.9 (216.0-226.8) tokens/s | 177.9 (162.8-188.4) / — / — | 6,505 MiB |
| 2B | 8,192 | 228.4 (223.3-231.0) tokens/s | 187.8 (187.8-191.4) / 187.7 (183.4-191.1) / 188.4 (185.4-191.7) | 6,411 MiB |
| 9B | 512 | 63.6 (52.3-93.9) tokens/s | 83.9 (83.2-85.6) / — / — | 4,613 MiB |
| 9B | 8,192 | 69.0 (55.3-91.0) tokens/s | 84.1 (82.4-86.3) / 82.1 (81.7-84.3) / 77.4 (77.4-79.7) | 4,371 MiB |

- Three passes per cell; medians, with the three-pass min-max in parentheses.
  Decode is stable across passes; prompt reading is not (the 9B at ctx 512 spans
  52.3-93.9), so no reading shape is claimed. Stock's fixed position sweep prints
  only positions that fit, so the 512-context runs have only position 100.
- A separate five-pass harness ([`pos512_stock.py`](benchmarks/2026-11-17-i7-8086k-rtx4060ti-stock-0.6.1/pos512_stock.py))
  fills the literal position-512 gap for both models at ctx 8192: **190.3 tok/s**
  (2B) and **84.5 tok/s** (9B), medians from five paired bench-then-harness passes.
  Its position-100 anchor matched the immediately preceding bench (191.3 vs 189.1
  on the 2B, 86.3 vs 86.3 on the 9B), so no downward harness offset is applied. The
  earlier standalone medians (171.2/80.7) were cold-clock runs and are superseded;
  ranges and the full anchor table are in the [benchmark record](benchmarks/2026-11-17-i7-8086k-rtx4060ti-stock-0.6.1/RESULTS.md).
- Unmodified stock v0.6.1 stops on native Windows at `libcuda.so.1`; the measured
  runs use a loader shim that maps that one name to `nvcuda.dll`. No Kindi module,
  kernel or engine decision is changed. The failed unmodified run is in the benchmark
  directory.

#### Verify matrix (RTX 4060 Ti)

`tools/verify_matrix.py` runs each selected model/mode cell as a fresh
`python -m kindi --pack ... verify` process with a clean `KINDI_*` environment. It writes
`results.csv`, `results.json` and raw logs. `--save-baseline PATH` copies a run's
`results.json` as a reference, keeping each cell's `logs` provenance from a baseline it
replaces only when the list names that cell's `log`, and refusing to drop one; to replace a
baseline with a fresh measurement, save to a new path or add an explicit `logs` list naming the
run's `log`. `--baseline PATH` loads that file on a later run and writes
`comparison.csv`/`comparison.json` with status and exactness transitions plus signed
throughput deltas. The process exit status still reflects the current run's pass/fail;
the comparison itself is diagnostic.

```sh
python tools/verify_matrix.py --models 2b,4b,9b,27b --modes tc \
  --pack-27b /path/to/bonsai-27b \
  --out bench-logs/verify-matrix \
  --save-baseline bench-logs/verify-matrix/baseline.json

python tools/verify_matrix.py --models 2b,4b,9b,27b --modes tc \
  --pack-27b /path/to/bonsai-27b \
  --out bench-logs/verify-matrix \
  --baseline bench-logs/verify-matrix/baseline.json
```

Current single-card 4060 Ti results:

| cell | status | exactness | throughput |
|---|---|---|---|
| 2B tc | pass | 4/4 | 370.2 tokens/s |
| 4B tc | pass | 3 exact + 1 tie (FD fallback; forced FD was 2/4) | 164.6 tokens/s |
| 9B tc | refused | — | — |
| 27B tc | pass | 3 exact + 1 tie (token 14, gap 0.0064) | 38.9 one-pass; **38.1 median** over three runs |
| 27B tc-ngram | pass | 3 exact + 1 tie (token 14, gap 0.0064) | **45.7 median** over three runs |
| 27B tc-draft2b | pass | 3 exact + 1 tie (token 14, gap 0.0064) | **71.0 median** over three runs |
| 27B tc-both | pass | 3 exact + 1 tie (token 14, gap 0.0064) | **62.0 median** over three runs |
| 4B plain | pass | 4/4 | 114.9 tokens/s |
| 9B plain | pass | 4/4 (13/13 on prompt 3) | 72.5 tokens/s |

The 27B tc three-run repeat was **38.2 / 38.1 / 36.7 tokens/s** (median 38.1, range 1.5,
4.1%), with the same exactness in every run. The older 2026-11-10 **40.5** rerun was not
reproduced; 40.5 is also the separate 24K benchmark's prompt-read rate, not verify throughput.
The three speculation rows use the 2B drafter and head and are the tracked verify-matrix
baseline; the command, per-run ranges and `verify-baseline.json` are on the
[27B drafter baseline page](benchmarks/2026-11-17-i7-8086k-rtx4060ti-27b-drafter/RESULTS.md).
The full workflow and qualifications are in [CARD-SUPPORT.md](CARD-SUPPORT.md#verify-matrix).

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

## Nemotron 3.5 Lightning 30B-A3B (ternary)

| item | value |
|---|---|
| model | Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf (`kindi use lightning-30b`): Nota AI's ternary version of NVIDIA Nemotron 3.5 Lightning 30B-A3B, as GGUF by boro64; model card: https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF |
| licence | the weights are not part of Kindi; their licence is OpenMDW 1.1 (https://openmdw.ai/license/1-1/) |
| cards | one card with 10 GB or more holds the full model. On cards with 6 GB or 8 GB, the least used experts stay in host memory |
| variables | `KINDI_CTX` (0: the largest context that fits; 60,416 positions on an RTX 3080 10 GB) |
| RTX 3080 (190 W) | write: 310 tokens/s at 512 tokens, 299 at 8K, 277 at 32K. Read: 4,260 tokens/s at 512, 3,780 at 32K |
| RTX 3090 (220 W) | write: 341 tokens/s at 512 tokens, 326 at 8K, 295 at 32K. Read: 4,790 tokens/s at 512, 4,010 at 32K |
| RTX 2060 12 GB | write: 164 tokens/s at 512 tokens, 155 at 8K, 135 at 32K. Read: 1,950 tokens/s at 512, 1,730 at 32K |
| RTX 2060, 6 GB held back | 60 of 128 experts per layer on the card. Write after a text prompt: 53 tokens/s at 512 tokens, 57 at 32K. Read: 1,415 tokens/s |
| GTX 1080 8 GB | 95 of 128 experts per layer on the card. Write after a text prompt: 75 tokens/s at 512 tokens, 89 at 8K, 74 at 32K. Read: 745 tokens/s at 512, 536 at 32K |
| Tesla P100 16 GB | write: 155 tokens/s at 512 tokens, 145 at 8K, 118 at 32K. Read: 654 tokens/s at 512, 605 at 32K |
| Quadro K6000 | write: 61.9 tokens/s at 512 tokens, 48.3 at 8K, 32.8 at 32K. Read: 206 tokens/s at 512, 192 at 8K |

The model has 52 layers: Mamba-2 layers, mixture-of-experts layers (128 experts, 6 for each token) and attention
layers. The engine reads the weights from the GGUF file. `kindi convert` writes only a small manifest. The
products use int8 activations in runs of 32, as llama.cpp does. `kindi verify` compares the output with llama.cpp.
Where the output is different, verify shows that llama.cpp's top two tokens were almost equal there.

Prompt reuse: the Mamba state cannot go back to an earlier position. So the server keeps a copy of the state at
the start of the last assistant turn of each prompt. The next turn of the same chat starts from that copy.

The Nemotron 3.5 Lightning backend was started by the work of slavalevski-tech
([github.com/slavalevski-tech/kindi](https://github.com/slavalevski-tech/kindi)). slavalevski-tech first ran this model on Kindi
and shared that backend under the Kindi CLA 1.1. The Kindi code is a new implementation. Thank you.

Tool calls use the XML form of the chat template. After a tool result, the model often calls the same tool again.
llama.cpp does the same with the same file. This is the behaviour of the model.

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
| models | 2B, 4B or 9B on one card; 27B on two cards (12 GB total) |
| single-card variables | `KINDI_CTX=24576`, `KINDI_BLOCKING_SYNC=0` (the installer sets both; `KINDI_CHUNK_Q8` defaults to 1 on sm_61) |
| two-card variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=2`, `KINDI_CTX=24576`, `KINDI_CHECKPOINT_BUDGET_MB=4096`, `KINDI_REPEAT_STOP=1`, `KINDI_BLOCKING_SYNC=0` |
| generation | not measured on a P106; the installer offer and pairing are unit-tested only |

- Official specifications: 1,280 CUDA cores, 80 TMUs, 48 ROPs, 1,506 MHz base / 1,708 MHz boost, 4.37 TFLOPS FP32, 6 GB GDDR5 on a 192-bit bus (192.2 GB/s), PCIe 3.0 x16, 120 W, one 6-pin connector and no display outputs.
- It is a mining variant of the GTX 1060. Run it headless: use the integrated graphics or another card for the console, and leave the P106 for CUDA. `nvidia-smi` still reports it.
- Many P106 boards link at x1 or x4 although the slot is x16. Prompt reading and the two-card exchange are sensitive to that link; check `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`.
- The installer classifies the 6 GB card as `maxwell`, its 24,576-token memory profile, rather than `pascal8` (which starts at 7,500 MiB). That is the context that fits 6 GB; it does not mean the card is Maxwell.
- The 27B does not fit in 6 GB. Two P106 cards run it together; the installer also offers a P106 + P4000 pair, which keeps the 5-trit path on both cards. Any pair from the supported pair classes with at least 3,800 MiB each uses the legacy two-card profile. A pre-6.0 + 6.0-or-newer pair sets `KINDI_G5=0` and runs both cards on the 2-bit path. These mixed offers are not measured yet.
- The installer's shared 6 GB list also offers Nemotron; it is experimental and has not been measured on a P106.
- CPU+GPU tandem is implemented for dense Bonsai models on the measured host, but it is not supported on the P106: no P106 tandem result is recorded and P106 acceptance is missing. (The Nanbeige backend has a separate host-layer path; it does not apply here.) The gate is in [P106.md](P106.md#cpu-tandem-status); do not advertise it yet.
- Verification runner: `python3 tools/verify_p106.py` records the single-card 2B/4B/9B verify, the server/cache cycle and the two-card 27B verify. It has not run on P106 hardware; see [P106.md](P106.md#one-command-verification-record).

## Quadro M4000, 8 GB (Maxwell 2.0, sm_52)

| item | value |
|---|---|
| silicon | GM204, 28 nm, 1,664 CUDA cores, 1,500 MHz boost, about 4.99 TFLOPS FP32 |
| memory | 8 GB GDDR5, 256-bit, 256 GB/s, on-die ECC; PCIe 3.0 x16, 120 W |
| installer class | `maxwell` (sm_52 has no `pascal8` branch; the class is Maxwell by compute capability, not by size) |
| models | 2B, 4B or 9B (`/data/pack`, `/data/pack4b`, `/data/pack9b`); 27B is not offered |
| variables | `KINDI_CTX=24576`, `KINDI_BLOCKING_SYNC=0` |
| tensor cores | none. `KINDI_TC=1` refuses on sm_52 by design (`tc.supported()` requires sm_60+); that refusal is the expected record, not a failure |
| pairs | M4000 + P4000 is offered with `KINDI_G5=0` (both cards drop to the 2-bit path) |
| generation | not yet measured; expect Maxwell-class speed |
| page | [M4000.md](M4000.md); session script [OLD-CARD-RIG.md](OLD-CARD-RIG.md) |

- 8 GB of memory. The 27B can be tried manually with `KINDI_PACK=/data/pack27b`; the engine fits the context to the free memory. It is not offered by default.
- The rig session script (commands, evidence, fallbacks) is [OLD-CARD-RIG.md](OLD-CARD-RIG.md); `KINDI_TC=1` on sm_52 refuses by design and that refusal is the expected record.

## Quadro P4000, 8 GB (Pascal, sm_61)

| item | value |
|---|---|
| silicon | GP104, 16 nm, 1,792 CUDA cores, 1,505/1,709 MHz, about 5.48 TFLOPS FP32 |
| memory | 8 GB GDDR5, 256-bit, 256 GB/s, on-die ECC; PCIe 3.0 x16 half-height, 105 W, four DisplayPort 1.3 |
| installer class | `pascal8` (sm_61 with 8,192 MiB clears the 7,500 MiB gate) |
| models | 27B, 9B, 4B, 2B in that order - one of only three classes that offers the 27B on one card; the 8 GB fit is unverified |
| variables | `KINDI_CTX=65536`, `KINDI_BLOCKING_SYNC=0`, `KINDI_CHUNK_Q8` on by default on sm_61 |
| tensor cores | none; the dp4a Pascal kernels exist (`tc.pascal()` accepts major 6) but `pascal8` does not enable `KINDI_TC=1`, so forcing it is the interesting experiment |
| pairs | P4000 + P106 is offered without `KINDI_G5=0` (both cards 5-trit); M4000 + P4000 with `KINDI_G5=0` |
| generation | not yet measured; the GTX 1080 (same sm_61 family) is 4/4 token-exact on the 27B, but the P4000 still needs its own run |
| page | [P4000.md](P4000.md); session script [OLD-CARD-RIG.md](OLD-CARD-RIG.md) |

- The 27B weights are about 6.7 GB, so a few hundred MiB is left for the K/V; `kindi bench --sample-vram` records the peak and the context reduction, which is the number that decides the support row.

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
| GeForce GT 430 and other Fermi / pre-Kepler cards (below sm_30) | Kindi needs Kepler (sm_30) or later; no code path compiles a kernel for sm_21 and the card is ignored as a compute device |

- **Legacy Fermi path:** on a machine whose only NVIDIA card is Fermi, the installer leaves the card unused and offers the CPU engine (`KINDI_DEVICE=cpu`) when the CPU supports AVX. Removing the card is not required; CUDA only rejects it when a CUDA engine is selected, so the legacy path is a graceful-degradation route, not GPU support.

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

## Intel Arc A770 16 GB (Alchemist, DG2): purchased, in transit, not supported

One A770 16 GB was purchased 2026-11-15 and is on its way to us; it is not installed yet. Kindi has
no Intel backend, so nothing is measured and the installer does not offer this card. The plan is a
SYCL/oneAPI backend (Linux-first; milestones, gates and the M1 arrival checklist are in
[INTEL-ARC.md](INTEL-ARC.md), tracked as [REQ-0020](development/requirements/REQ-0020.md)). The card
is registered in the support matrix as `not measured` under the `arc-a770` slug, ready for a rig run
to fold into when it lands.

| item | value |
|---|---|
| memory | 16 GB GDDR6, 256-bit |
| models | not yet measured |
| variables | none (no backend) |

## CPU (x86-64 with AVX; AVX2/FMA/F16C fast path)

| item | value |
|---|---|
| models | 2B, 4B or 9B |
| variables | `KINDI_DEVICE=cpu`; optional `OMP_NUM_THREADS=<cores of one socket>`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` (the default is one thread per physical core, bound to those cores) |

- The CPU engine builds at the first start into `KINDI_CPU_CACHE`: a keyed `libkcpu-<key>.so` with gcc and
  OpenMP on Linux, or `libkcpu-<key>.dll` with clang and libomp on Windows (`KINDI_CC`, `KINDI_LLVM`).
- AVX is the baseline. An AVX-only build (`-mavx -mno-avx2 -mno-fma -mno-f16c`) passed 2B/4B
  `kindi verify` 4/4 at 6.4/3.0 tokens/s; native Sandy Bridge hardware was not available
  ([CPU-SUPPORT.md](CPU-SUPPORT.md)).
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
| processor | Intel Core i7-8086K, Coffee Lake (14 nm), 6 physical cores / 12 logical threads (Hyper-Threading), 12 MB L3 |
| clocks | 4.00 GHz base; up to 5.00 GHz single-core turbo; all-core turbo is typically 4.3-4.4 GHz, depending on the board, cooling and power limits; unlocked multiplier |
| variables | `KINDI_DEVICE=cpu`; six threads (the engine's physical-core default); on Windows set `KINDI_CC` and `KINDI_LLVM` to clang 18 and the LLVM install |
| build | keyed `libkcpu-<key>.dll` with clang 18.1.8 and libomp on Windows, keyed `.so` with gcc and OpenMP on Linux |
| verify | 2B 4/4 at 33.7 tokens/s, 4B 4/4 at 16.7, 9B 4/4 at 9.1 (native Windows, 1,024-token context) |
| generation (24,576-token context, six threads, native Windows) | 2B 35.9 at 100 tokens, 29.3 at 2K, 19.4 at 8K, 11.9 at 16K, 8.2 at 24K; 4B 15.2 / 13.4 / 6.8 / 4.3 / 3.1; 9B 8.7 / 7.7 / 4.9 / 3.5 / 2.6 |
| prompt reading | 45.1 (2B), 19.2 (4B), 9.9 (9B) tokens/s (native Windows, 1,000-token prompt) |
| server | `kindi serve` with `KINDI_DEVICE=cpu` answered `POST /v1/chat/completions` non-stream and stream on the 2B, 4B and 9B: 32.9/19.2/11.5 tokens/s generation, 0 errors, 0 MiB GPU memory (native Windows, 1,024-token context, 2026-11-10; [details](benchmarks/2026-11-10-i7-8086k-windows-cpu/README.md#cpu-server-chat)) |

- The 8086K is the 2018 40th-anniversary limited edition, closely related to the i7-8700K with a
  higher single-core turbo. These runs used no explicit overclock and no thermal or power clamp;
  per-core clocks were not logged.
- The default six threads is the portable choice. The 12-thread setting is 6-20x slower and 10 is
  slower too; eight threads are a WSL2/libgomp tuning only (2B/9B gain, 4B read regresses, native
  Windows 8 loses). See [CPU-SUPPORT.md](CPU-SUPPORT.md#thread-count-sweep-6-8-10-and-12-2026-11-10).
- The 24K and prompt-reading rows are the native Windows 2026-11-10 run; the WSL2 numbers are archived
  in the benchmark page.
