# Looped-model GGUF headers (2026-11-22)

**Scope: metadata inventory only.** No weights were downloaded, no model was loaded and no
output, speed or memory number is claimed. This page is the Q0 evidence for
[REQ-0031](../../development/requirements/REQ-0031.md).

## Method

Each file's first 32 MiB was fetched with an HTTP range request and parsed with the
repository's own reader. GGUF metadata and the tensor index live before the tensor data, so the
header is enough to record the architecture and the stored-block map.

```sh
curl -sL -r 0-33554431 -o header.gguf.part "<resolve-url>"
python -c "from kindi.gguf import GGUF; g = GGUF('header.gguf.part'); print(g.kv['general.architecture'])"
```

The range request is public for both files: SenseNova's GGUF reports `gated: false`, and the
Unsloth mirror of Inkling 2 Mini is public. The host was the i7-8086K Windows machine; the
download was 32 MiB per file, not the multi-GB weights.

## SenseNova-Loop-9B

| field | value |
|---|---|
| repository | `sensenova/SenseNova-Loop-9B-GGUF` |
| repo revision | `b489b15c1e359e6ad429566ab02067bc4b1b12b5` |
| file | `SenseNova-Loop-9B-Q4_K_M.gguf`, 5,818,631,072 B |
| other files | BF16 18,419,909,536 B; Q5_K_M 6,668,501,920 B; Q8_0 9,798,648,736 B; mmproj 918,165,984 B |
| `general.architecture` | `sensenova_loop` |
| `block_count` | 57 (logical) |
| stored `blk.*` indices | 0-27 and 52-56 (33) |
| loop | `loop.start=4`, `loop.end=28`, `loop.count=2` |
| mHC | `start=4`, `end=28`, `streams=2`, `iterations=20`, `fixed_identity=false`, `pre_scale=1.0` |
| attention | 16 query / 4 KV heads, key/value length 256, `full_attention_interval=4` |
| linear attention | `ssm.conv_kernel=4`, `group_count=16`, `state_size=128`, `time_step_rank=32`, `inner_size=4096` |
| geometry | embedding 4096, FFN 12288, context 524,288 |
| rope | dimension 64, freq base 1e7, YaRN factor 2.0 over 262,144 |
| nextn | 1 MTP layer |
| tensors | 490; types F32 / Q4_K / Q6_K |
| tokenizer | 248k vocabulary, EOS 248046, `qwen35` pre-tokenizer |

The post-loop decoder layers are stored under their **logical** indices 52-55, and the MTP layer
is 56. Logical layers 28-51 are the second pass over 4-27; there are no tensors for them.

## Inkling 2 Mini

| field | value |
|---|---|
| repository | `unsloth/Inkling-2-Mini-GGUF` (base `thinkingmachines/Inkling-2-Mini`, revision `b85c3cfa615fc413cc9af3379f491b963f3b586a`) |
| file | `Inkling-2-Mini-UD-Q4_K_M.gguf`, 16,610,543,328 B |
| other files | UD-IQ1_S 7,764,291,776 B; UD-Q8_K_XL 22,867,451,616 B; DSpark drafter 402,278,560 B; mmproj ~640 MB |
| `general.architecture` | `inkling2` |
| `block_count` | 48 (logical) |
| stored `blk.*` indices | 0-23 (24) |
| loop | `loop.core_start=6`, `loop.core_end=18`, `num_loops=3` |
| loop routing | `loop_res_query.weight`, `loop_kv_res_query.weight`, `loop.gate_rank=64` |
| MoE | `expert_count=128`, `expert_used_count=8`, expert FFN 1152, `expert_weights_scale=8.0` |
| attention | key/value length 128, sliding window 1024, relative length 16 / extent 1024, `attn_res.block_size=6` |
| geometry | embedding 2304, FFN 1152, context 1,048,576 |
| activations | SITU (`situ_beta=4.0`, `situ_linear_beta=25.0`), `shortconv.l_cache=4` |
| logits | `final_logit_softcapping=30.0`, `qk_clip=22.627`, `log_scale_alpha=0.1` |
| tensors | 734; types F32 / Q8_0 / Q5_K / IQ4_XS / MXFP4-class |
| tokenizer | 201,024 padded / 200,058 unpadded, EOS 200006, `inkling2` pre-tokenizer |

The post-loop blocks 42-47 are stored at the **physical** indices 18-23. Logical layers 18-41 are
the second and third passes over 6-17; there are no tensors for them. The bundled DSpark drafter
is a separate file and is not part of this inventory.

## `kindi loop-info` on the same 32 MiB ranges (2026-11-23)

Both partial files were checked with the repository's own command, once as text and once with
`--json`. GGUF metadata and the tensor index come before the tensor data, so the range is enough
for this map check; no tensor byte was read and no weight was downloaded.

| check | SenseNova-Loop-9B | Inkling 2 Mini |
|---|---|---|
| revision the range came from | `b489b15c1e35...` | `800c946d7b9c...` (the mirror's `main` on 2026-11-23) |
| 32 MiB range sha256 | `e9c95bf1a787...` | `229bbecf356c...` |
| stored blocks present / expected | **33 / 33** | **24 / 24** |
| missing / unexpected stored blocks | none / none | none / none |
| exit code | 0 | 0 |

```text
$ python -m kindi loop-info sensenova.gguf.part
SenseNova-Loop-9B (sensenova_loop)
  stored blocks: 33; logical layers: 57; loop: 4-27 x2
  post-loop stored at 52; tensor coverage: 33/33 stored blocks present
  plan: pre 0-3; loop pass 1 4-27; loop pass 2 28-51; post 52-56
  no looped loader or engine is implemented (REQ-0031); this is a map check.

$ python -m kindi loop-info inkling.gguf.part
Inkling 2 Mini (inkling2)
  stored blocks: 24; logical layers: 48; loop: 6-17 x3
  post-loop stored at 18; tensor coverage: 24/24 stored blocks present
  plan: pre 0-5; loop pass 1 6-17; loop pass 2 18-29; loop pass 3 30-41; post 42-47
  no looped loader or engine is implemented (REQ-0031); this is a map check.
```

The plan's `loop pass` ranges are logical layers, not stored blocks: Inkling's second pass over
logical 18-29 reads stored 6-17, and its post-loop logical 42-47 reads stored 18-23.

**No coverage mismatch on either file.** The map the command prints is the one the header table
above recorded, and each tensor index carries every stored block the map expects. One provenance
difference from the 2026-11-22 inventory: the Unsloth mirror's `main` resolved to
`800c946d7b9c830c86290df49afbbe91db8455f5` for this run (last modified 2026-11-23), while the
sections above record only the base model's revision. The Q4_K_M file's size is unchanged at
16,610,543,328 B, and the evidence names the mirror revision actually read; SenseNova's revision
is unchanged.

The exact text is [sensenova-loop-info-32m.txt](sensenova-loop-info-32m.txt) and
[inkling-loop-info-32m.txt](inkling-loop-info-32m.txt);
[loop-info-32m.json](loop-info-32m.json) carries both range hashes, both full JSON reports and the
coverage fields. This remains a map check only: no loader, engine, pack, catalog entry or support
claim, and no full-file SHA-256 (only the 32 MiB ranges were read).

## What this does and does not establish

- It establishes the architecture key, the logical/stored block map, the loop and mHC/MoE fields,
  and the quantization types of the public Q4_K_M files.
- It does **not** establish the full-file SHA-256 (the weights were not downloaded), a tokenizer
  round-trip, any output token, any speed or any memory measurement.
- It is not a support claim. `kindi convert` recognizes both architectures and refuses them with
  this map; `kindi loop-info FILE.gguf [--json]` prints the same logical plan and checks the tensor
  index for missing stored blocks. No pack, catalog entry or engine path exists yet.

The machine-readable form is [headers.json](headers.json).
