# Tandem overlap scheduler — step 0 re-measurement (2026-11-15)

Host: i7-8086K + RTX 4060 Ti (8,188 MiB), 27B pack at `T:/kindi-packs/bonsai-27b`,
`KINDI_TC=0`, ctx 24,576, Windows native (`clang+llvm-18.1.8` added to PATH for the
CPU library build; CPU tier `avx2-fma-f16c`, 6 threads). Same command and placement
as [the 2026-11-14 step 0](../2026-11-14-i7-8086k-rtx4060ti-tandem-overlap/RESULTS.md):

```sh
python -m kindi --pack T:/kindi-packs/bonsai-27b bench \
    --tandem cpu-first --tandem-cpu-layers gdn:12 --sample-vram --json <out>.json
```

## Fresh rows (2026-11-15)

| run | read | gen @100/2K/8K/16K/24K | sampled peak |
|---|---|---|---|
| `baseline-no-probe.json` | 11.3 | 9.3 / 8.4 / 8.1 / 8.9 / **8.2** | 6,723 MiB used |
| `baseline-probe.json` (`KINDI_TANDEM_DEBUG_TIMING=1`) | 11.3 | 10.0 / 9.4 / 8.5 / 9.3 / **8.4** | 6,723 MiB used |

Slower than the 2026-11-14 row (9.9 tok/s at 24K): the CPU library is the Windows
native build here, and background load on this host is unknown. The placement,
context and GPU side are identical, so the *decomposition* below is the decision
input, not the absolute rate.

## Decomposition (probed run, 76 tokens, `tandem_timing` in `baseline-probe.json`)

Per token: **cpu 74.7 ms** (12 layers/token → **6.2 ms/layer**) + **gpu 8.1 ms** +
**transfer 3.09 ms** (7.0 switches/token → 0.44 ms each) + **head 17.6 ms** =
**103.4 ms**.

Against the design's premise (docs/TANDEM-OVERLAP-DESIGN.md §2): the CPU run is
**not** paying a ~6 ms/layer dispatch gap — 6.2 ms/layer sits at the raw-kernel
band (4–6 ms), exactly as the 2026-11-14 run found (5.18 ms/layer). There is no
2× CPU dispatch term for a worker thread to remove. The two serial costs after
CPU compute are the **GPU head (17.6 ms, 17% of the token)** and the **7 boundary
transfers (3.09 ms, 0.44 ms each)** — the same ordering as 2026-11-14.

## Decision

**The worker thread (§3 of the design) is not justified and should not be built.**
Its premise — half the token is dispatch gap — is absent at the 27B/24K gate row
in both the 2026-11-14 and 2026-11-15 measurements. If the >=25% gate
(≥12.4 tok/s at 24K against the 9.9 baseline) is pursued, the next engineering
in order is (1) head-side work (17.6 ms/token is the largest non-compute slice)
and (2) the double-buffered pinned staging for the 7 transfers (3.09 ms/token,
and it removes the per-switch synchronisation that serialises the GPU stream).
The design doc's §2 table and §3 worker-thread build stay as the documented
no-go rationale; the `rate` split rule and capacity mode are unaffected.
