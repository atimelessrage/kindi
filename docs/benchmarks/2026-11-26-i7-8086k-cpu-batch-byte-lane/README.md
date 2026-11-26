# CP3: byte-lane verify GEMV prototype — built, bit-identical, and the gate is missed (2026-11-26)

Roadmap gate **CP3** (`docs/ROADMAP.md`) is the CP1 reopen: CP1 measured the batched pass exact but
a 5-row pass costing 4.4 plain rows because the pair-code GEMV is permute-bound (T rows need 2T
`vpermps` per weight word). CP3 asked for a byte-lane formulation that shares the nibble expansion
instead of repeating it per row, gated on a 5-row pass costing **<=1.5 plain rows** with 4/4
token-exact prompts and a synthetic-pack test pinning rows/logits/rollback.

- Host: Intel i7-8086K, 6 threads, AVX2/FMA/F16C, no AVX-512 (so no VBMI 64-lane permute);
  Windows, clang 18.1.8.
- The 27B pack is **not on this host** (and needs ~32 GiB RAM against this box's 16), so the gate
  number here is on the **bonsai-9b** pack — the largest pack present — with the GEMV probe swept on
  the 27B's matrix shapes. CP1's 4.4 plain rows was measured on the 27B; the 9B engine cost below
  (5.02) is the same wall.
- Harness: an untracked standalone probe (`bench-logs/byte_lane_probe.c`) that replays
  `rows8_multi`'s arithmetic exactly and checks every output float against it, plus
  `tools/cpu_batch_check.py --pack <9b> --rows 15 --batch 5` for the engine pass.

## Variants prototyped

All four keep the current kernel's accumulation order and rounding (pair sum formed as
`w0*x0 + w1*x1` with plain mul/add — multiplication by ±1 is exact — and the same even/odd
accumulators), so every arm is **bit-identical** to the shipped kernel on every shape tried
(0 differing floats):

| arm | idea | 12288×4096 | 4096×12288 | 4096×4096 |
|---|---|---|---|---|
| current (`rows8_multi`) | 2T `vpermps` per weight word | 12.97 ms | 11.70 ms | 3.40 ms |
| `lane` | sign vectors extracted once (2 permutes per word), broadcast + mul + add per row | 14.53 ms | 14.84 ms | 4.78 ms |
| `lane_g` | lanes become the T rows instead of the weight rows | 20.09 ms | 20.55 ms | 6.71 ms |
| `lane_h` | group outer: one group's tables serve every row block (accumulators in memory) | 13.22 ms | **10.45 ms** | **3.34 ms** |
| `lane_hl` | group outer + shared sign extraction | 17.18 ms | 15.80 ms | 5.10 ms |

Single thread, 20 reps, weights L3-resident; the head shape (248320×5120) has `lane_h` 10% *slower*
(the 5 MB accumulator array is evicted). The shared-expansion arms lose because their broadcasts
cost what the permutes cost; the T-lane arm loses to scalar nibble extraction and accumulator spills.
`lane_h`'s 1.01–1.11× on the smaller shapes is table cache locality, and it conflicts with the
engine's split-by-row-block threading (each thread would re-read the tables anyway). None of it
approaches the ~3× the gate needs.

## The engine gate, on the 9B

`tools/cpu_batch_check.py` on bonsai-9b (ctx 2048, 6 threads): plain 16 rows in 1.94 s (8.26 rows/s);
three 5-row passes 15 rows in 1.82 s (8.22 rows/s, 1.00× plain); tokens equal, max logit delta
0.000e+00, rollback to 2 rows then 13 plain rows equal. One 5-row pass is 0.608 s against a plain
row's 0.121 s: **5.02 plain rows per pass**, against the gate's <=1.5. The pass is still exact; the
cost is still ~T on the arithmetic.

## Verdict

**CP3 gate missed; no kernel lands.** Sharing the nibble expansion does not remove the per-row work
on AVX2: the two candidate formulations either trade permutes for equally expensive broadcasts or
spend the saving on scalar extraction and spills. What remains named in the roadmap row is an
AVX-512 VBMI host (64-lane permute) or a different weights layout (a transposed code store the
converter would have to write). The prototype stays untracked in `bench-logs/`; the synthetic-pack
test (`tests/test_cpu_batch_synth.py`) already pins rows/logits/rollback for the shipped kernel, and
the exactness invariant is unchanged.

## Files

`results.json` holds every figure above (probe arms per shape, engine plain/batch/rollback times,
and the 5.02 ratio as `5.022`).
