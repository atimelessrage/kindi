# Lightning Q2_0 mixed card/CPU combine: bit-exact against `lt_mv_q2_down`

Run: 2026-11-19. Host: i7-8086K (6 cores / 12 threads), 32 GB RAM, Windows 10
22H2, RTX 4060 Ti 8 GB, Python 3.13.6, CuPy 14.2.0, Numba 0.67.0. No model pack
was loaded; this is a synthetic expert gate, not a model row.

## Why a combine contract

`lt_mv_q2_down` does not reduce each expert's output before adding them. Every
lane accumulates its own runs, the lane combines the routed experts with the
gateway weights and the shared expert's partial, and only then does the warp
XOR tree produce the row. A host-computed final expert vector therefore cannot
enter that reduction unchanged: the host must provide the same pre-tree lane
partials.

The implemented contract is:

| piece | output | producer |
|---|---|---|
| host routed expert | float32 `[rows, 32]` | `kindi.moe_expert_lane.Q2Matrix.partials` |
| card routed experts | float32 `[nslot, rows, 32]` | `lt_mv_q2_down_parts` |
| shared expert | float32 `[rows, 32]` | `lt_q2_shared_parts` |
| final reduction | `x[r] += tot` | `lt_q2_combine` |

`parts[k]` and `gw[k]` are in local slot order. `lt_q2_combine` starts from
`shparts[r][lane]`, adds `gw[k] * parts[k][r][lane]` in slot order, runs the
same five-step XOR tree as `lt_mv_q2_down`, and accumulates into `x[r]`. The
host partials use the lane's `h = lane, lane + 32, ...` run order and one
`fmaf` per run, exactly as the down kernel does.

## What was tested

`tools/q2_combine_lane.py` compiles the real `lightning.cu` with the engine's
options (`--use_fast_math`, `-std=c++14`) and builds synthetic Q2_0 weights at
the Lightning down geometry: hidden 2688, expert width 1856, four routed
experts, shared width 3712. Experts 0 and 2 run on the host; experts 1 and 3
and the shared expert run on the card.

Each routed slot has its own down input, quantized by the card's own
`lt_quant` and stacked the way `lt_mv_q2_down` reads it (`hq` holds one block of
`runs * 8` words per slot). The reference is one `lt_mv_q2_down` launch with
`ids = [0, 1, 2, 3]`, four gateway weights and the shared weights. The mixed
path is one `lt_mv_q2_down_parts` call over all four slots (which exercises the
`[nslot][rows][32]` output), the CPU slots compared with the host partials and
then replaced by them, `lt_q2_shared_parts` for the shared expert, and one
`lt_q2_combine`.

| check | differing elements |
|---|---:|
| mixed vs `lt_mv_q2_down`, `x = 0` | **0 / 2688** |
| mixed vs `lt_mv_q2_down`, non-zero residual `x` | **0 / 2688** |
| shared-only combine vs `lt_mv_q2_down` with no routed slots | **0 / 2688** |
| host partials vs the multi-slot `lt_mv_q2_down_parts` output, slots 0 and 2 | **0 / 86,016 per slot** |

The zero-residual case compares the combined sums; the residual case also
checks the kernel's `x[r] += tot` path. The host-partial check is independent
of the combine: it compares every lane of the host result with the same slot
from the card's multi-slot parts call. The shared-only branch was also exercised through the same
combine and matches the reference's shared contribution.

## Blocker and scope

No `kindi_lightning` pack or Lightning GGUF exists on this host. This page
claims only the synthetic Q2_0 combine gate: it is **not** an Engine row, not a
model `kindi verify`, and not a tok/s result. The mixed path is not wired into
`kindi.lightning.Engine`; the engine still uses `lt_mv_q2_down` for its routed
experts. The CPU lane's cost is measured separately in
[the Q2 expert-lane page](../2026-11-19-i7-8086k-rtx4060ti-q2-expert-lane/RESULTS.md).

## Reproduce

```sh
python tools/q2_combine_lane.py \
  --json docs/benchmarks/2026-11-19-i7-8086k-rtx4060ti-q2-combine/results.json
python -m pytest tests/test_moe_expert_lane.py -q
```

The tool exits 0 only if both combine comparisons, the shared-only comparison
and the host-partial comparison have zero differing elements.
