# PQ2_0 / Q2_0 import: identity gate, verify and benchmark (2026-11-10)

Host: i7-8086K + RTX 4060 Ti 8 GB (sm_89), Windows, Python 3.13 + CuPy.
Branch `freebuff/tandem-cpu-recovery`, three PTQ1_0-reference packs under `~/.kindi/packs`:

| pack | source GGUF | source type | pack bytes |
|---|---|---|---:|
| `bonsai-2b` | `Ternary-Bonsai-2-2B-PTQ1_0.gguf` | 143 | 506 MB |
| `bonsai-2b-pq2` | `Ternary-Bonsai-2-2B-PQ2_0.gguf` | 142 | 506 MB |
| `bonsai-2b-q2` | `Ternary-Bonsai-2-2B-Q2_0.gguf` | 42 | 506 MB |

Both new GGUFs were downloaded from the publisher and checked against `SHA256SUMS`
(`sha256sum -c` -> both `OK`): PQ2_0 517,312,992 B, Q2_0 546,689,504 B.

## 1. Conversion identity gate

`python bench-logs/convert_pq2_q2.py --ref ~/.kindi/packs/bonsai-2b --gguf <file> --out <pack>`
converts the file and compares every pack stream with the PTQ1_0 pack:

| stream | PQ2_0 | Q2_0 |
|---|---|---|
| `bcodes.bin` | identical | identical |
| `dpm.bin` | identical | identical |
| `f32.bin` | identical | identical |
| `signs_2048.f32` | identical | identical |
| `signs_6144.f32` | identical | identical |
| `vocab.json` | identical | identical |
| `merges.txt` | identical | identical |
| `bcodes_index.json` | identical | identical |
| `index.json` | same tensor table, source types `[0, 142]` | same tensor table, source types `[0, 42]` |
| `kindi_pack.json` | same fields | same fields |

`ALL PASS` for both. The imported packs are byte-identical to the PTQ1_0 pack; only the recorded
source type and `source_format` differ. Import cost is the normal convert time (20-21 s).

Two conversion bugs were found and fixed on the way, both visible only in the new formats:

- `q2_groups` assumed one 128-weight group per row; a row can hold several (`spr = ne0/128`).
- The pack's `bc_off`/`dpm_off` accumulators used the *source* groups-per-row (Q2_0 has twice as
  many 64-groups), so `bcodes_index.json` and the manifest sizes were wrong even though the stream
  contents were right. They now advance by `ne0/4` bytes and `ne0/128` scales.

Unit coverage: `tests/test_convert.py` (10 tests) pins the slot decode, the element order, the
code-3 refusal, Q2_0 half folding (matching halves accepted, mismatched refused) and the
multi-group row.

## 2. Verify

`python -m kindi --pack <pack> verify` against the llama.cpp reference:

| pack | prompts | throughput |
|---|---|---|
| `bonsai-2b` | 4/4 token-exact, 40/40 each | 198.8 tok/s |
| `bonsai-2b-pq2` | 4/4 token-exact, 40/40 each | 182.0 tok/s |
| `bonsai-2b-q2` | 4/4 token-exact, 40/40 each | 185.5 tok/s |

## 3. Benchmark at the same fitted context

`python -m kindi --pack <pack> --ctx 24576 bench`, three runs per pack, greedy, exact head.
All three packs fit **24,576 tokens** with the same free memory (`6213 MiB free of 8187`), so the
comparison uses one common context with no truncation. Numbers are tokens/s; medians in bold.

| pack | run | read | 100 | 2,000 | 8,000 | 16,000 | 24,000 |
|---|---|---:|---:|---:|---:|---:|---:|
| PTQ1_0 | 1 | 184.0 | 177.1 | 183.2 | 174.1 | 185.2 | 170.8 |
| PTQ1_0 | 2 | 233.4 | 162.6 | 191.0 | 178.3 | 148.8 | 162.1 |
| PTQ1_0 | 3 | 207.7 | 192.1 | 176.6 | 181.1 | 167.6 | 172.7 |
| **PTQ1_0** | **median** | **207.7** | **177.1** | **183.2** | **178.3** | **167.6** | **170.8** |
| PQ2_0 | 1 | 230.3 | 151.3 | 190.3 | 181.7 | 183.9 | 169.3 |
| PQ2_0 | 2 | 214.7 | 182.6 | 170.4 | 166.1 | 179.2 | 162.1 |
| PQ2_0 | 3 | 205.9 | 184.8 | 192.3 | 182.7 | 185.1 | 160.7 |
| **PQ2_0** | **median** | **214.7** | **182.6** | **190.3** | **181.7** | **183.9** | **162.1** |
| Q2_0 | 1 | 214.5 | 193.4 | 187.0 | 178.9 | 175.5 | 160.8 |
| Q2_0 | 2 | 150.2 | 184.9 | 187.5 | 186.6 | 181.6 | 157.5 |
| Q2_0 | 3 | 160.1 | 153.1 | 164.1 | 171.8 | 186.0 | 172.8 |
| **Q2_0** | **median** | **160.1** | **184.9** | **187.0** | **178.9** | **181.6** | **160.8** |

Median 24K generation: PTQ1_0 170.8, PQ2_0 162.1 (-5.1%), Q2_0 160.8 (-5.9%). Read: 207.7,
214.7 (+3.4%), 160.1 (-22.9%).

**Read the deltas as noise, not format cost.** The three packs are byte-identical, so the engine
executes the same weights, kernels and buffers; only the pack directory differs. The run-to-run
spread within one pack already spans 148.8-192.1 tok/s (PTQ1_0, position 16,000) and 150.2-214.5
read (Q2_0), i.e. wider than every between-pack difference. A format cannot make a bit-identical
pack slower; `PQ2_0`/`Q2_0` support is a download-compatibility feature.

## Caveats

- Only the 2B was downloaded and measured. The 4B/9B/27B `PQ2_0`/`Q2_0` files use the same type
  ids but were not converted or verified here (disk: ~3.9 GB free on `C:`).
- The Q2_0 pack needs both 64-weight halves of a 128-weight group to carry the same fp16 scale; a
  file where they differ is refused rather than silently mis-scaled. Both publisher files fold
  cleanly.
- No GPU-side difference is expected or claimed; no 27B, CPU or server run was made for this task.
