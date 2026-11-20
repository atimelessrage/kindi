# Dory1.0-4B D0 reference evidence

Status: **D0 reference audit complete; numeric performance gate still provisional** (11 November
2026). This is reference evidence for the Dory design, not a Kindi support claim. There is no Dory converter, engine, catalog entry or installer offer yet; D1
is the next phase.

## Pinned inputs

### Q5_K_M GGUF

- repository: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF`
- repository revision: `df53755cb8741843af8e019e088819fbef17e29b`
- file: `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`
- size: `3,673,288,864` bytes
- SHA-256: `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`
- local storage used for this audit: `~/dory-d0\model\NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`
  (not copied into the repository)

The Hugging Face download metadata records revision `df53755...` and the same SHA-256. The
checksum was recomputed locally with `sha256sum`; it matches.

### BF16 metadata

Only the config/tokenizer metadata was downloaded. The 7.95 GB BF16 weights were deliberately not
downloaded.

- repository: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
- repository revision: `f6c832e6963ddb74f62c75813a3e8c74b1af67b1`
- files and checksums: [`bf16/SHA256SUMS`](bf16/SHA256SUMS)

| file | bytes | SHA-256 |
|---|---:|---|
| `config.json` | 2,267 | `d2d086f5949598a043a00d70e2c28c45305bca7703682de3e4f990a7fd3279a4` |
| `generation_config.json` | 225 | `55535fcf0e43be7b7968d51a4223182316000b2799c46d75037c23dcca2e467d` |
| `tokenizer_config.json` | 177,209 | `10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc` |
| `special_tokens_map.json` | 563 | `e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b` |
| `chat_template.jinja` | 10,152 | `82753bef5cedc4932c1ed509b5c9a12be680fd86d1adb65bc3f7398d11c8eebc` |

The released `config.json` declares 72 layers, 16 input + 40 recurrent + 16 output, GQA 16 query /
4 KV heads, `head_dim` 256, hidden 2,560, 131,072 vocabulary, 262,144 context and
`n_recurrent_loops: 2`. The GGUF below declares `dory.num_loops=4`; see the discrepancy note.

### llama.cpp reference

- PR: [ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306)
- tested revision: `1c3cdd23787750814aa21a070dc4d6dc451af871` (PR tip)
- tested source archive: `dory-1c3c.zip`, SHA-256
  `16880d0fb4491769a12cdb8720c613991a6472aa7490d417c91f0f0628217332`
- earlier documented pin: `98a74c97ea09ac72f6ceda1559808fc0a2f447ce` (archive SHA-256
  `31385d783b346468b98eb8ed4f90b9aa659a6157ca6e2afe0aa27123b14d723c`), **not used** for the CPU
  run because later PR commits fix Dory CPU tensor placement (`9e2f3be…` and `3834b4…`)

The CPU build used MSVC 14.51 and CMake 4.4.2, Release, Ninja:

```text
cmake -S <dory-1c3c>/llama.cpp-1c3cdd23787750814aa21a070dc4d6dc451af871 \
  -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build <build> --config Release -j 12
```

`--override-kv` is present in this tree; the harness uses the equivalent
`llama_model_kv_override` with key `dory.num_loops`.

The token-ID harness is [`harness/dory_golden.cpp`](harness/dory_golden.cpp). It is built by copying
that file to `examples/simple/dory_golden.cpp` and appending the `dory-golden` target to
`examples/simple/CMakeLists.txt`; the harness prints the prompt token IDs, generated token IDs and
decoded text as JSON. It is not part of Kindi.

## GGUF inventory

Full metadata and the 507-tensor manifest: `tensor-inventory.json` in the local raw run directory
(`bench-logs/dory-d0/`, gitignored, 1.3 MB) - it is a dump of the GGUF, not a tracked deliverable; the
pinned hashes and the goldens that the rows above cite are tracked next to this page.
(1,280,464 bytes). Generated with:

```text
python tools/dory_gguf_inventory.py <model.gguf> -o bench-logs/dory-d0/tensor-inventory.json
```

Summary:

- 507 tensors: **252 Q5_K, 36 Q6_K, 219 BF16**
- 44 metadata keys; `data_start` 7,901,344; `general.file_type` 17 (Q5_K_M)
- 72 physical layers, 16 input + 40 recurrent + 16 output
- 27 sliding-attention + 9 full-attention entries in the 72-character layer pattern
- 36 each of the attention (`attn_gate`, `attn_k`, `attn_output`, `attn_q`, `attn_v`) and FFN
  (`ffn_down`, `ffn_gate`, `ffn_up`) matrices, plus 36 each of their BF16 scale tensors;
  72 `residual_alpha` tensors
- 262,144 context, 1,024 sliding window, 16/4 GQA, 256 key/value length, 7,168 FFN width;
  RoPE bases 10,000,000 and 10,000
- `dory.recurrent_kv_cache_mode = per_loop`
- **`dory.num_loops = 4`** in this GGUF, while the BF16 `config.json` says
  `n_recurrent_loops: 2`. The golden suite therefore overrides `dory.num_loops` explicitly for
  every k; it never relies on the file default.

Representative tensor families:

| family | count | type |
|---|---:|---|
| `blk.*.attn_{gate,k,output,q}` | 144 | Q5_K |
| `blk.*.attn_v` | 36 | Q6_K |
| `blk.*.ffn_{gate,up}` | 72 | Q5_K |
| `blk.*.ffn_down` | 36 | Q6_K |
| `blk.*.residual_alpha` and attention/FFN scales | 216 | BF16 |
| `output.weight` | 1 | BF16 |
| `token_embd.weight` | 1 | BF16 |
| `output_scale.weight` | 1 | BF16 |

## Golden protocol

The suite is four fixed raw-completion prompts, 32 greedy tokens each, at explicit k=1, 2 and 4.
It uses six CPU threads, no GPU layers, a 4,096-token context, and 512-token batch/ubatch.

Prompts, in order:

1. `The capital of France is`
2. `Once upon a time, there was a`
3. `def fibonacci(n):`
4. `The sky appears blue because`

Exact single-run command:

```text
PATH=<build>/bin:$PATH dory-golden.exe \
  -m <model.gguf> -p "The capital of France is" \
  -n 32 -k 2 -t 6 -ngl 0
```

The complete 12-run command is:

```text
python bench-logs/dory-d0/run_goldens.py \
  --model <model.gguf> --harness <build>/bin/dory-golden.exe \
  --out bench-logs/dory-d0/goldens --n-predict 32 --threads 6 --ngl 0
```

The harness uses `llama_sampler_init_greedy()` (temperature 0), adds BOS/special tokens during
tokenization, evaluates the prompt once and prints the full prompt and generated token IDs. There
is no random seed because no stochastic sampler is used. Each run is a fresh process and a fresh
context; no KV or prompt cache is reused.

## Pinned results

Canonical output: [`goldens/goldens.json`](goldens/goldens.json) (17,755 bytes). All 12 runs exited
0. The stderr log for every run is retained beside it; the repeatability run is
`goldens/k2-p1-repeat.json`.

| run | prompt IDs | generated tokens | first 8 generated IDs |
|---|---|---:|---|
| k1-p1 | 1784,8961,1307,5498,1395 | 32 | 6993,1046,1531,8961,1307,9591,1395,56452 |
| k1-p2 | 23322,6610,1261,2142,1044,2156,1486,1261 | 32 | 3709,1044,1457,93651,8256,4418,93186,80083 |
| k1-p3 | 3149,9111,87539,4990,4244 | 32 | 1693,1308,9120,1032,1049,1058,1850,1308 |
| k1-p4 | 1784,21283,10249,10991,3147 | 32 | 1307,13488,70972,35805,1046,1531,24233,8688 |
| k2-p1 | 1784,8961,1307,5498,1395 | 32 | 6993,1046,1531,8961,1307,10787,1395,10874 |
| k2-p2 | 23322,6610,1261,2142,1044,2156,1486,1261 | 32 | 3709,11525,27128,1970,1294,1278,7008,1307 |
| k2-p3 | 3149,9111,87539,4990,4244 | 32 | 1693,1308,9120,1032,1049,1058,1850,1308 |
| k2-p4 | 1784,21283,10249,10991,3147 | 32 | 1307,13488,70972,35805,1046,16178,1562,1278 |
| k4-p1 | 1784,8961,1307,5498,1395 | 32 | 6993,1046,1531,8961,1307,9591,1395,56452 |
| k4-p2 | 23322,6610,1261,2142,1044,2156,1486,1261 | 32 | 1894,8876,3986,2274,1486,1261,3435,10973 |
| k4-p3 | 3149,9111,87539,4990,4244 | 32 | 1693,1308,9120,1032,1049,1058,1850,1308 |
| k4-p4 | 1784,21283,10249,10991,3147 | 32 | 1307,13488,70972,35805,1046,2813,4391,22405 |

Observed behavior:

- k=2 prompt 1 was rerun after the suite; prompt IDs, all 32 generated IDs and text were identical.
- k=1 and k=4 prompt 1 produced the same 32 tokens; k=1, 2 and 4 prompt 3 also produced the same
  32 tokens. This is an observation, not evidence that loop count is generally inert: prompts 2
  and 4 differ between all three counts.
- Wall times ranged from 5.14 s to 11.18 s per fresh process (model load plus 32 generated tokens).
  These are not throughput claims.

## Checksums

[`SHA256SUMS`](SHA256SUMS) covers the GGUF, BF16 metadata, inventory, golden JSON and harness/runner
sources. Source archive hashes are in the reference section above.

## Limits and next step

- CPU-only reference. No CUDA build, GPU comparison, batching or throughput claim was tested.
- Only `per_loop` KV was run. `last_loop`, k=3, long contexts and a second rebuilt binary remain
  untested.
- The BF16 weights (7.95 GB) were not downloaded; only their config/tokenizer metadata is pinned.
- The earlier documented `98a74…` revision was not built. The tested `1c3cdd…` tip includes the
  later Dory CPU-placement fixes.
- The model is not in Kindi's catalog, converter, server, installer or support matrix. D1 has not
  started.
