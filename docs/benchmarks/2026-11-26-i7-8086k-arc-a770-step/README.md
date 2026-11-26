# Arc A770: the ternary GEMV actually computes a GEMV, and a full decode step on the device

Host: i7-8086K, Windows 10, Arc A770 16 GB (driver 32.0.101.9175, OpenCL C 1.2, no
`cl_khr_fp64`). Date: 2026-11-26. Pack: the 2B (`bonsai-2b`, 320 tensors, Hadamard block 512).
Probes: `python -m kindi intel-ternary`, `python -m kindi intel-pack`, `python -m kindi intel-step`
(`--pack ~/.kindi/packs/bonsai-2b --device-name Arc --steps 8`).

This is K3 of [INTEL-ARC-KERNELS.md](../../INTEL-ARC-KERNELS.md): a layer loop over a resident
pack. In this wiring the device does the norm/fold and every ternary GEMV; the host does the conv,
the delta rule, attention, the FFN elementwise math and the argmax. The reference is the CPU engine
(`kcpu.c` through `cpu.Engine`), token for token, teacher-forced.

## The bug this iteration found: `pack_gemv4` truncated its input

The shipped `pack_gemv4`/`pack_gemv_persist` kernels multiplied each trit by `(int)x[c]`: the
folded input was truncated to an integer before the dot product. On a real activation vector that
is garbage — 99.5% of the outputs of `blk.0.attn_qkv` moved by more than 1e-3 against the CPU's
`gemv_rows` on the identical input — yet the probe reported "bit for bit" because its validation
input was integer-valued (`((i*5+3)%7-3).astype(f32)`), where int and float accumulation agree
exactly. The kernels and the host model in `intel_ternary` now accumulate in float32, per slot
grouped (`t0*x0 + t1*x1 + t2*x2 + t3*x3` as one expression), scaled once per 128-weight phase —
the same order `kcpu.c`'s GEMV uses — and the validation inputs in `intel-ternary` and `intel-pack`
are fractional so an int-truncating kernel cannot pass silently again.

## Results

| probe | what | result |
|---|---|---|
| `intel-ternary` (scalar + rows4, 1024x512, fractional x) | device vs float host model, bit for bit | **0 mismatches** |
| `intel-pack` (real pack, `blk.3.attn_k.weight`, fractional x) | device vs float host model, bit for bit | **0 mismatches** |
| `intel-step` (full 2B decode step vs `cpu.Engine`) | 8 greedy tokens after a 4-token prefix | **8/8 matched** |

## The step's launch budget

174.89 ms/step was mostly reads: 115 blocking reads at ~0.77 ms each (every one drains the
queue its phase queued behind). Three changes, none of them a new kernel family:

* **host writes are non-blocking** — the in-order queue orders a write against the kernels
  launched after it, so a blocking write was buying a sync the reads already pay;
* **one read per GEMV group** — a 7-line `concat3` kernel copies the group's GEMV outputs into
  one buffer and the phase reads it once, including the GDN's folded input `[xn | qkv | gate]`;
* **the FFN's silu(gate)×up runs on the device** (`silu_mul` + the FWHT fold in place, K4's
  first half), so gate and up never return to the host and the `m` vector is never written.

73 reads, 516 enqueues, **141.72 ms/step** (max |Δlogit| 2.63e-04 — the device's `exp` inside
silu differs from numpy's by an ulp). The remaining ~73 reads are the architectural floor of
this wiring: the host needs each phase's GEMV outputs for conv, the delta rule and sampling.
One read per *step* needs those on the device too (Phase B).

Logit agreement over the 8 steps: max |Δ| **2.63e-04** (steps 3–10, absolute; the logit range is
about ±30). The GEMV itself agrees with the CPU engine's row GEMV to **1.19e-06** on
`blk.0.attn_qkv`, i.e. one float32 ulp at that magnitude — AVX2 lane order vs the kernel's slot
order, both accumulating per phase.

Timing: **141.72 ms/step** at **516 enqueues/step** on the 2B (24 layers × ~23 launches, plus the
head). The GEMV stream itself is only a few ms of that; the rest is 563 launches with a blocking
read each (~0.3 ms of sync). The next order of magnitude needs the conv/delta-rule/attention moved
on-device, which is new kernels, not wiring.

## Numbers in the claims file

`docs/claims/intel-step-20261126.json` gates on: 8/8 tokens, 2.63e-04 max |Δlogit|, 141.72 ms/step,
516 enqueues/step, the GEMV's 1.19e-06, and 0 validation mismatches in both `ternary.json` and
`pack2b.json`. The older pack pages (`2026-11-25-...-pack`) measured the int-truncating kernels;
their GB/s numbers are weight-streaming rates and stand, but nothing there ran a GEMV on real
activations.

Not a support claim: 2B only (no 27B pack on this host), one prefix, and the host still does the
GDN delta rule, attention and the FFN elementwise work.
