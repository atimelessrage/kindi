# Lightning Q2_0 expert lanes: CPU parity, host-mapped GPU reads and the copy lane

Run: 2026-11-19. Host: i7-8086K (6 cores / 12 threads), 32 GB RAM, Windows 10
22H2, RTX 4060 Ti 8 GB, Python 3.13.6, CuPy 14.2.0, Numba 0.67.0. No model
pack was loaded; this is an expert-level gate, not a model row.

## What was tested

`kindi/moe_expert_lane.py` computes one routed expert of the Lightning big MoE
(Nemotron 3.5 Lightning 30B-A3B: hidden 2688, expert width 1856, 23 MoE layers,
128 routed experts top-6 plus one shared) on the host CPUs. The arithmetic is
the kernel's, not an equivalent one:

- a Q2_0 run is `sum(c * q) - sum(q)` over 32 ternary weights, an exact int32;
- each run's value enters lane `h % 32` with one fused multiply-add, in run order;
- the 32 lane sums fall through the same five-step XOR tree.

`tools/q2_expert_lane.py` compiles the real `lightning.cu` with the engine's
options (`--use_fast_math`, `-std=c++14`) and compares, on synthetic Q2_0
weights of the real geometry:

| matrix | kernel | compared elements | device-resident weights, differing | host-mapped weights, differing |
|---|---|---:|---:|---:|
| up (1856 x 2688) | `lt_mv_q2` | 1856 | **0** | **0** |
| down (2688 x 1856) | `lt_mv_q2_down` (one expert, weight 1, shared branch zero) | 2688 | **0** | **0** |

The intermediate activation is quantized by the card's own `lt_quant` and its
`xq/dx/sx` are fed to the host, because the engine's fast-math division is not
reproducible on the host. Both matrices are bit-identical with device-resident
weights and with host-mapped weights, so the lane is exact for the Q2_0 expert
output at this geometry.

The host-mapped lane is the engine's actual host-expert path, not a copy lane:
`kindi/lightning.py:_q2` allocates one `cupy.cuda.alloc_pinned_memory` block per
matrix stack, places every expert's codes first and its fp16 scales after them,
and hands `lt_mv_q2`/`lt_mv_q2_down` a device `uint64` table of those pinned host
addresses. The tool reproduces that layout with `_host_tables()` and passes the
pointer tables to the real kernels. It does not call `cp.asarray` on the weights
and does not perform an explicit H2D copy; the kernels read the pinned weights
over PCIe. The 0.662 ms host-mapped row is a full up + `lt_quant` + down pass,
so it is directly comparable to the CPU expert and to the copy lane.

## Cost

One expert is 2,806,272 bytes (1,769,472 up + 1,036,800 down).

| step | ms |
|---|---:|
| CPU up | 1.070 |
| CPU down | 1.325 |
| **CPU expert** | **2.395** |
| H2D copy of the expert's bytes (4.065 GiB/s) | 0.643 |
| GPU up + `lt_quant` + down (device-resident weights) | 0.140 |
| **copy lane (copy + GPU)** | **0.783** |
| **GPU up + `lt_quant` + down (host-mapped pinned weights)** | **0.662** |
| lane's small transfers (up out, `hq/hd/hs` back, output in) | 0.406 |
| **CPU lane total (CPU + transfers)** | **2.801** |

Medians of seven passes in each of three fresh processes. Top-level component
fields are medians of the three run medians; `cpu_expert`, `copy_lane` and
`lane_total` are sums of the reported component medians, so the table adds up.
The per-process CPU totals were 2.392 / 2.436 / 2.387 ms; the 2.395 ms row is
1.070 + 1.325, not a median of those totals. The previous page's 2.475 ms was a
median of per-process totals and did not equal its 1.086 + 1.420 components;
this artifact makes the additive definition explicit in `runs_note`.

**Verdict: the direct host-mapped GPU read wins on this host.** The 0.662 ms
lane is 1.18x faster than the 0.783 ms copy lane and 3.62x faster than the
2.395 ms CPU expert; the CPU lane's total with its small transfers is 2.801 ms.
The CPU lane is also 3.06x slower than the copy lane. The CPU lane is exact, but
it is not adopted: the engine's pinned-host pointer path already reads the
weights directly and is the fastest of the three here. A host with a narrower
PCIe link or more CPU cores could change the ratios; that is not measured.

## Blocker for the model row

No `kindi_lightning` pack or Lightning GGUF exists on this host, so the
capacity-bound model row (`kindi verify` 4/4, decode tok/s with the lane
enabled) is **not run**. The CPU lane is also not wired into
`kindi.lightning.Engine`. The mixed-layer combine contract is now implemented
and gated in [the combine page](../2026-11-19-i7-8086k-rtx4060ti-q2-combine/RESULTS.md):
`Q2Matrix.partials`, `lt_mv_q2_down_parts`, `lt_q2_shared_parts` and
`lt_q2_combine` recreate `lt_mv_q2_down`'s reduction without moving a final
host vector into the tree. This tool still exercises the allocation and
pointer-table layout from `lightning.py:_q2`; it is not a new Engine code path.

## Reproduce

```sh
for i in 1 2 3; do
  python tools/q2_expert_lane.py --passes 7 --json "bench-logs/q2-host-mapped-20261119-run$i.json"
done
```

Merge the three per-process medians into the tracked `results.json` as the
`runs_note` describes. `results.json` carries the parity booleans (including
the host-mapped booleans), the host-mapped metadata, the three per-process rows
and the medians used above. `python -m pytest tests/test_moe_expert_lane.py -q`
covers the CPU arithmetic against a pure-Python `math.fma` reference (no GPU).
