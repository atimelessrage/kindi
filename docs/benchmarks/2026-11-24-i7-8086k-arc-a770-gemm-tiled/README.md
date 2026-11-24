# Intel Arc A770: the q8 GEMM, pinned work-group and locally staged tiles (2026-11-24)

The first `kindi intel-gemm` page measured the un-tiled kernel and named local-memory staging as
the next step. Doing it turned up two things worth recording: the launch, not the tiles, was the
large win, and the device index is not a safe way to select the Arc.

## 1. The work-group size was the 1.7x

The first page launched the GEMM with the global size `(256, tiles)` and **no** local size, so the
runtime chose the work-group shape. Pinning it to `(256, 1)` - one 256-wide group per
128-token x 128-row tile - keeps a tile's 128 activations and 128 rows inside one group of
consecutive work-items, which is what the kernel's index arithmetic assumes.

Same session, same kernel, 512 x 512 x 4096, 10 launches per run, 3 runs per arm
([workgroup-ab.json](workgroup-ab.json)):

| work-group | 10 launches (3 runs) | GFLOP/s | mismatches |
|---|---|---|---|
| `(256, 1)` pinned | 322.958 / 324.201 / 323.917 ms | 66.49 / 66.24 / 66.30 | 0 |
| runtime-chosen | 555.061 / 555.002 / 555.114 ms | 38.69 / 38.69 / 38.69 | 0 |

**1.71x** from the launch alone, bit-identical output. The first page's 581.6 ms at this shape was
this launch shape, not a different machine state.

Pinning is also a correctness fix, not only a speed one: with the local-staged tile the same
unpinned launch let a work-group span two tiles, so its staging and barriers mixed them and the
result was wrong (0 mismatches at one tile, mismatches at 2+ tiles). The probe now always launches
`(256, 1)`; that is why every shape below, including 512 rows x 512 tokens, is exact.

## 2. Local staging: a small, consistent win

`gemm_tile_local` stages the group's activation words and decoded weight words in local memory (as
`k_mm_g5q` stages its A/B tiles), so each word is read from global memory once per block. Arithmetic
is unchanged - same int32 dot per group in the same `k4` order, one scale multiply - so both
variants are held to the same bit-exact reference.

`python -m kindi intel-gemm --rows R --tokens T --cols K --staging {local,global} --repeats 10
--device-name Arc --json`, three runs per cell, median quoted ([summary.json](summary.json)):

| rows x tokens x cols | staging | 10 launches | GFLOP/s | speedup |
|---|---|---|---|---|
| 128 x 128 x 512 | global | 39.3 ms | 4.27 | - |
| 128 x 128 x 512 | local | 37.6 ms | 4.46 | 1.05x |
| 256 x 128 x 1024 | global | 77.3 ms | 8.68 | - |
| 256 x 128 x 1024 | local | 74.1 ms | 9.05 | 1.04x |
| 256 x 256 x 2048 | global | 153.9 ms | 17.44 | - |
| 256 x 256 x 2048 | local | 147.5 ms | 18.20 | 1.04x |
| 512 x 512 x 4096 | global | 324.6 ms | 66.17 | - |
| 512 x 512 x 4096 | local | 316.9 ms | 67.77 | 1.02x |

0 mismatches and 0 XQ/XS mismatches in every one of the 24 runs. The gain is small because the
bottleneck is the byte-wise dot and the per-word 4-trit decode, not global traffic: the staged
variant removes the re-reads (8 consumers per staged word) but not the arithmetic. That is the
honest reason `dp4a`-style packing, not more staging, is the next lever.

## 3. Device selection is not stable, so the probe selects by name

On this host the driver has returned the **integrated UHD 630 first** and the Arc A770 second, and
on other runs the reverse. A benchmark that selects `--device 0` can therefore measure the iGPU and
be published as an Arc number. All three OpenCL probes (`intel-cl`, `intel-ternary`, `intel-gemm`)
now take `--device-name SUBSTR` and record a warning in the report and in the printed output when
the selected device's name does not contain `Arc`. Every run on this page was
`--device-name Arc`; each report's `device.name` reads `Intel(R) Arc(TM) A770 Graphics` with 512
compute units and 15,932 MiB.

## Scope and limits

- Still a probe: no model, no engine path, no SYCL, no support claim. M2 stays partial and Arc stays
  unsupported.
- Windows and OpenCL are precursor work; the plan's target is SYCL/Level Zero on Linux.
- The weight bytes are the pack's on-disk 4-trit layout, not the engine's load-time 5-trit lane
  repack, and the grid is one tile per work-group (no split-K, no subgroup reductions).
- Timings are within-session comparisons of one kernel shape on one card; the GFLOP/s figures are
  observations from a probe, not a throughput claim about the engine.
