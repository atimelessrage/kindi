# CPU expert lane for the Lightning Q2_0 format: parity gate and copy-lane comparison

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

| matrix | kernel | compared elements | differing |
|---|---|---|---:|
| up (1856 x 2688) | `lt_mv_q2` | 1856 | **0** |
| down (2688 x 1856) | `lt_mv_q2_down` (one expert, weight 1, shared branch zero) | 2688 | **0** |

The intermediate activation is quantized by the card's own `lt_quant` and its
`xq/dx/sx` are fed to the host, because the engine's fast-math division is not
reproducible on the host. Both matrices are bit-identical, so the lane is
exact for the Q2_0 expert output at this geometry.

## Cost

One expert is 2,806,272 bytes (1,769,472 up + 1,036,800 down).

| step | ms |
|---|---:|
| CPU up | 1.086 |
| CPU down | 1.420 |
| **CPU expert** | **2.475** |
| H2D copy of the expert's bytes (4.05 GiB/s) | 0.646 |
| GPU up + `lt_quant` + down (device-resident weights) | 0.142 |
| **copy lane (copy + GPU)** | **0.788** |
| lane's small transfers (up out, `hq/hd/hs` back, output in) | 0.412 |
| **lane total (CPU + transfers)** | **2.887** |

Medians of seven passes in each of three fresh processes (CPU expert:
2.418 / 2.475 / 4.557 ms; one process was an outlier on a busy host, the other two
agree; copy: 0.64-0.65 ms; GPU: 0.14 ms).

**Verdict: the copy lane wins on this host.** 0.788 ms against 2.475 ms for
the CPU expert (3.14x), and 2.887 ms including the lane's transfers. The
expert-level gate passes, so the lane is exact, but it does not satisfy the
roadmap's "beats the copy lane" clause here. A host with a narrower PCIe link
or more CPU cores could change the ratio; that is not measured.

## Blocker for the model row

No `kindi_lightning` pack or Lightning GGUF exists on this host, so the
capacity-bound model row (`kindi verify` 4/4, decode tok/s with the lane
enabled) is **not run**. The lane is also not wired into
`kindi.lightning.Engine`: a layer that mixes card and CPU experts still needs
the combine contract, because `lt_mv_q2_down` combines per-lane partials
before its warp tree and a host-computed final vector cannot enter that
reduction unchanged.

## Reproduce

```sh
python tools/q2_expert_lane.py --json docs/benchmarks/2026-11-19-i7-8086k-rtx4060ti-q2-expert-lane/results.json
```

`results.json` carries the parity booleans, the three per-process rows and the
medians used above. `python -m pytest tests/test_moe_expert_lane.py -q` covers
the CPU arithmetic against a pure-Python `math.fma` reference (no GPU).
