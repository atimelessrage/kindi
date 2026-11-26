# Pack load: 5.5x faster tri-code decoding (2026-11-26)

The Bonsai 2.5 27B's only working path here is the CPU engine, and its first cost was the load:
**339.9 s** before the first token on the i7-8086K, measured on 2026-11-25
([benchmarks/2026-11-25-i7-8086k-bonsai25-27b-cpu](../2026-11-25-i7-8086k-bonsai25-27b-cpu/README.md)).
96% of that was `kindi.cpu.paircodes`, the decoder that turns each pack's base-3 weight byte into the
uint32 words the C engine reads.

`paircodes` used to derive four base-3 trits per byte with `% 3` / `// 3`, stack them, and gather a
uint32 code per column pair with a per-trit multiply-add index - about twelve passes over the matrix.
It now maps the whole byte through one 256-entry `uint16` table (`cpu._PAIR_LUT`) and assembles each
word from four bytes, so a matrix takes a gather, one shift-or and one byte transpose:

```
pairs = _PAIR_LUT[bytes].view(uint8).reshape(rows, C // 2)     # 2 pair codes per byte
by    = rows8[:, 0::2] | (rows8[:, 1::2] << 4)                 # rows 2l, 2l+1 per byte
word  = by.transpose(0, 2, 1).copy().view(uint32)              # 8 nibbles, 8 rows
```

## Measured

Same host, same pack files, same process: each pack is loaded once with the frozen per-trit decoder
(kept in `tests/test_cpu_paircodes.py` and `tools/cpu_pack_load_check.py`) and once with the shipping
one. ctx 512, 4 threads, `avx2-fma-f16c`, i7-8086K, Windows 11. Best of 1 run per pack (a second run
of the 2B measured 3.84-4.04 s).

| pack | pack size | before | after | speedup | after throughput |
|---|---|---|---|---|---|
| Bonsai 2 (2B) | 1,340.5 MB | 21.13 s | **3.84 s** | 5.50x | 349 MB/s |
| Bonsai 4 (4B) | 2,985.0 MB | 47.15 s | **8.16 s** | 5.78x | 366 MB/s |
| Bonsai 9 (9B) | 5,886.6 MB | 109.97 s | **19.91 s** | 5.52x | 296 MB/s |

Raw rows: [`loads.json`](loads.json), written by
`python tools/cpu_pack_load_check.py --runs 1 --pack ... --out loads.json`.

**The 27B is a projection, not a measurement.** Its pack is not on this host, so nothing here was
loaded at 27B size. The scaling is linear in pack bytes across the three packs measured (296-366
MB/s), and the 27B pack is 7,242 MB, so the same decoder puts its load in the 20-25 s range for the
decode stage against the 326 s it spent there - the 339.9 s figure should fall to roughly a minute,
of which the rest is the scale expansion and `f32.bin`. That is an estimate off the three measured
packs; measure it on the host that has the pack before quoting it.

## What did not change

The output bytes. `tests/test_cpu_paircodes.py` compares the new decoder against a frozen copy of the
old one over the real code range and over all 256 byte values, across six matrix shapes, both chunk
paths and a memmap slice; `tests/test_cpu_batch.py` and the CPU `kindi verify` matrix re-check the
whole model end to end:

```
KINDI_DEVICE=cpu python -m kindi verify            # 9B pack: 4 prompts, 40/40, 40/40, 13/13, 40/40 PASS
KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-2b verify   # 2B pack: ALL PASS
```

Both are token-for-token against llama.cpp, so a single wrong trit anywhere in the load would fail
them.

## Re-running

```sh
python tools/cpu_pack_load_check.py --runs 2                       # every pack in ~/.kindi/packs
python tools/cpu_pack_load_check.py --pack ~/.kindi/packs/bonsai-9b --out loads.json
```
