# Ternary Bonsai 2.5 27B: the released GGUF header, read without downloading the weights

Host: i7-8086K, Windows, Python 3.11. Date: 2026-11-24. Source: the published
`prism-ml/Ternary-Bonsai-2.5-27B-GGUF` repository, file `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf`
(5,993,834,848 bytes, SHA-256 `929560ec…`).

## What was done

A single HTTP range request (`Range: bytes=0-12582911`, HTTP 206) fetched the first 12 MB of the
6 GB file, which contains the GGUF metadata block and the tensor table. `parse_header.py` parsed them
into `header.json`. No tensor data was downloaded, so nothing here is a conversion, a load, a verify
or a speed measurement.

Commands:

```sh
curl -r 0-12582911 -o head.gguf \
  https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/resolve/main/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf
python parse_header.py head.gguf > header.json
```

## What it says

| question | answer |
|---|---|
| architecture | `qwen35` |
| GGUF version / metadata entries / tensors | 3 / 48 / 851 |
| geometry (blocks, width, ffn, heads, kv heads, groups) | 64 / 5120 / 17408 / 24 / 4 / 16 |
| `kindi/convert.py`'s 27B profile | the same tuple (`convert_profile_match: true`) |
| tensor types | 402 of type 143 (`PTQ1_0`) + 449 of type 0 (`f32`) |
| tensor names | `blk.0…63` plus `output`, `output_norm`, `token_embd`; no extra or unknown names |
| `prism.hadamard.block_size` | 1024 |
| vocabulary / context | 248320 tokens / 262144 |
| chat template | 8952 characters, with `reasoning_effort` and both `xhigh` and `medium` |

That is the same architecture, geometry, tokenizer size and template family as Bonsai 2 27B, which is
what the engine and converter already support; the 2.5 file differs in its weights and its template
text. The file also ships `-mtp`, `PQ2_0`, `Q2_0` and `mmproj` variants that this header run did not
touch.

## What this does not show

No pack was built from the 2.5 weights, no engine loaded it, no prompt was generated, and no number
in `docs/BONSAI-2-SUPPORT.md` section 1a's support row is a measurement of this file. The support
matrix row for `bonsai-2.5-27b` reads `not run` / `not converted` for exactly that reason.
