# Bisecting the 4B split flash-decode divergence (2026-11-16)

`KINDI_FD`'s split flash-decode attention fails the 4B gate at prompt 2 token 6 and prompt 3
token 37 — the gate numbers are this directory's `RESULTS.md`. This bisect asked whether that
is a defect in the kernel (masking, head grouping, chunk merge) or something else.

**Answer: no kernel defect.** FD computes the same quantities as the standard decode
attention and only orders the additions differently; the 32-layer residual stream amplifies
that ~6e-8 median difference to ~1e-1 in the logits, which crosses llama.cpp's top-2 gap at
both tokens. Nothing in `tcf.cu` was changed.

Host: i7-8086K, RTX 4060 Ti 8 GB, CUDA 12.9, CuPy 14.2. Pack `~/.kindi/packs/bonsai-4b`
(width 2560, 16 q heads, 4 KV heads, head_dim 256, QL = 4096). FD_CH = 512 (the card has 34
SMs, so `fd_ch()` keeps the default).

## 1. Kernel against kernel, identical inputs

Both paths are fed the same q8_0 K/V/Q bytes and the same gate vector, on the engine's own
buffers: `k_fd_q8_4` + `k_fd_comb` against `k_attn_sq8t` + `k_softmax_rows` + `k_attn_vq8w` +
`k_attn_combn`, one full-length decode per row (n = positions):

```
n=12    max|Δ| 1.8e-07   median_rel 8.1e-08
n=512   max|Δ| 7.5e-08   median_rel 2.3e-07
n=513   max|Δ| 5.6e-08   median_rel 2.5e-07      (first two-chunk merge)
n=600   max|Δ| 6.0e-08   median_rel 2.4e-07
n=1024  max|Δ| 5.2e-08   median_rel 2.7e-07
n=1500  max|Δ| 3.7e-08   median_rel 2.6e-07
```

With spiky inputs (one position × 6, one query head × 6) both paths stay within 1e-6 of a
float64 reference, so neither is "the correct one" — they differ only in association order.

## 2. Masking (the part a single-query test does not cover)

Four queries in one launch, query q seeing exactly `n0 + q` positions, against the standard
verify kernels (`k_attn_sq8m` / `k_softmax_m` / `k_attn_vq8w_m` / `k_attn_combn_m`):

```
n0=1    2.4e-07      n0=255  8.9e-08
n0=7    2.4e-07      n0=511  5.9e-08
n0=33   2.4e-07      n0=512  8.9e-08
                     n0=600  1.0e-07
```

Per-chunk `(max, exp-sum)` matches the standard scores' max and `exp(s - max)` sum to ~1e-7,
and per-chunk value partials match a float64 reference to 4.6e-06 (median rel 2.4e-07) —
including a last chunk holding a single position. Masks and chunk boundaries are correct.

## 3. Real model state at the failing step

Prompt 2 (`The capital of France is`), marched to the diverging position with the standard
plans, then both attention paths run on the captured layer buffers (all 8 attention layers,
n = 11, scores up to 19.9 in magnitude):

```
kvi=0..7: max|fd-std| 2.4e-07 … 9.5e-07,  rms 1.6e-08 … 4.0e-08,  median_rel ~6e-08
```

The harness is faithful: recomputing the standard attention for the last attention layer
reproduces the standard plan's `OUT` bit-exactly (max Δ = 0).

## 4. Where the divergence comes from

Plan-level comparison over the same weights, K/V and tokens — T=1 plans whose only difference
is the attention nodes (380 nodes standard vs 364 with FD, i.e. 2 vs 4 per attention layer).
`XP` is the pre-head normalised activation, `YH` the logits:

| feed | decode | tokens | XP max vs std | logits max vs std |
|------|--------|--------|---------------|-------------------|
| std  | std    | 40/40  | 0             | 0                 |
| FD   | std    | 40/40  | 0.162         | 0.108             |
| std  | FD     | 40/40  | 0.117         | 0.115             |
| FD   | FD     | **6/40** (token 6) | 0.130 | 0.119        |

Repeating the same plan is bit-identical (control `XP` max = 0), so this is not run-to-run
noise. Each FD phase *on its own* already moves the pre-head activation by ~0.1 (rms 0.03)
while staying token-exact: against the ~2e-8 rms difference at the attention output that is
~1.5e6× over the 28 layers from the first attention layer to the head (≈1.7× per layer), an
ordinary growth rate for a residual stream with RMSNorm. Failing needs FD in *both* phases —
the two independent shifts compound, and the total crosses llama.cpp's 0.0447 gap at token 6
(and the token-37 point in prompt 3).

## 5. Why "2560-wide" is the wrong attribution

The FD kernels read `QL = NH · HD`, `NHKV` and `HD` — never the residual width. The 4B and the
9B have *identical* attention geometry (NH = 16, NHKV = 4, HD = 256, QL = 4096); the 9B passes
its FD gate because its reference prompts do not sit on a 0.04-logit tie, not because 4096 is
a multiple of 1024. `fd_supported()`'s `width % 1024` test therefore keys on a quantity the
attention kernels do not use.

## 6. Consequences

- Nothing in `tcf.cu` needs fixing: the kernel is equivalent to the standard decode attention
  up to float association.
- The DEC-0003 guard stays (the gate still fails); only its rationale changes — it protects a
  near-tie-sensitive gate, not a broken geometry.
- Making FD shippable on the 4B would need either an accepted-tie rule in the gate (like the
  Nanbeige backend's) or a merge order that matches the standard path bit for bit; the second
  keeps "verify 4/4" meaningful but costs the split's single-pass K/V read.

## Reproducing

The comparison needs the DEC-0003 guard bypassed (`kindi.tp.fd_supported` stubbed to `True`
before the engine is built), because the guard also disables FD. Build the engine with
`KINDI_TC=1 KINDI_TP=1 KINDI_FD=1 KINDI_ATT_TC=1`, then:

1. Kernel comparison — call `k_fd_q8_4` + `k_fd_comb` and the four standard decode kernels
   directly on `s.Kq[kvi]` / `s.Vq[kvi]` / `tcb["QQ"]` / `tcb["QD"]` / `tcb["QN"]` with the
   same `n`, and diff the two `OUT` buffers.
2. Phase split — build T=1 plans with `FDP`/`FDM`/`fps`/`fms` temporarily removed from
   `s.tcb` (that alone selects the standard attention nodes), then swap the two plan pairs
   into `s.tc_plans` independently for the feed and the decode pass and compare `XP`/`YH`.
