# Tandem overlap scheduler — design draft

Status: **draft, not implemented; step 0 landed 2026-11-14 and was re-measured 2026-11-15** (the
decomposition probe and the re-recorded gate baseline — see
[RESULTS](benchmarks/2026-11-14-i7-8086k-rtx4060ti-tandem-overlap/RESULTS.md) and the fresh
[2026-11-15 re-measurement](benchmarks/2026-11-15-i7-8086k-rtx4060ti-tandem-overlap/RESULTS.md)).
**Decision (2026-11-15): the §3 worker thread is not justified and should not be built** — the
CPU run is at the raw-kernel rate in both measurements (5.18 and 6.2 ms/layer), so there is no
~6 ms/layer dispatch gap for it to remove. Head work (17.6 ms/token) and double-buffered
staging (3.09 ms/token) come first if the >=25% gate is pursued.
Targets the ROADMAP goal "Kindi tandem scheduler (tandem.cpp-grade overlap)": beat the 27B at 24K
tandem row by **>=25%** with token-exact **4/4** (`kindi verify`: `KINDI_TC=0` vs tandem vs full GPU,
for every tuning change). The gate row is the **re-recorded 9.9 tok/s** at the 24K position (the old
7.9 row predates the plan-engine owner), so the target is ≥12.4 tok/s. The memory half of that goal
is done (plan-engine owner, gate 1's measured split rule); this doc designs the speed half. Evidence
conventions follow [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md); the original system sketch is
[CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md) (its worker-thread sketch is where this starts).

**Step 0's result changes the premise below.** At 24K with the plan owner the CPU run is already at
the raw kernel rate (5.18 ms/layer, vs the 8K effective 11.7 ms), so the "~6 ms dispatch gap" this
design targets is not there at the gate row; the head (14.2 ms) and 7 boundary transfers (2.85 ms,
0.41 ms each) are the next serial costs. §3's worker thread must be re-justified against the stepped
decomposition before it is built; double-buffered staging and head-side work come first on those
numbers.

## 1. What the measurement says (and what must be re-measured first)

Per-token time in the current `TandemEngine._run` (`kindi/tandem.py:616`) is a strict layer-by-layer
ping-pong: layer 0 → … → layer NL−1, and every CPU/GPU run switch does a full `get_hidden()`
(device→host copy of the residual) + `set_hidden()` (host→device), each synchronising. With a
`gdn:N` placement the CPU set is scattered (attention layers stay on the GPU), so a 27B token at
`gdn:12` crosses the boundary roughly 2×(12/6) + 2 ≈ 6 times.

Measured inputs (split-ratio calibration, 27B): the **effective CPU GDN layer costs 11.7 ms** while
the **raw GDN kernel is 4–6 ms** — a ~6 ms dispatch/sync gap per layer, paid on the critical path.
The PCIe handover itself is **0.009 ms** (measured staged handover, never the constraint). CPU-side
time therefore dominates the 27B-24K token (7.9 tok/s ≈ 126 ms/token), and most of the overlap
opportunity is *removing per-layer dispatch cost*, not hiding PCIe latency.

**Step 0 of the build is a decomposition re-measurement**, because the 11.7 ms figure comes from a
sweep at 8K while the gate row is at 24K. Required before any tuning change is believed:

1. `kindi bench --tandem cpu-first --tandem-cpu-layers gdn:12 --sample-vram --json` at 24K (the gate
   row: now 9.9 tok/s at the 24K position, 5,949 MiB sampled peak, plan owner, `rate` pick —
   re-recorded 2026-11-14, in the RESULTS link above).
2. A per-run timing probe (env-gated, off by default) that records, per token, the wall time in each
   CPU run, each GPU run, and each hidden transfer — so the serial chain is split into named parts.
   Landed 2026-11-14 as `KINDI_TANDEM_DEBUG_TIMING`: `TandemEngine.timing_summary()` plus the
   `tandem_timing` bench field. The gate math uses these numbers: after the work below, the CPU-run
   figure must match the raw kernel figure (±15%) instead of 2× it. (It already does at 24K;
   the 2× figure was an 8K measurement, which is why the premise is now in question.)

## 2. The bottleneck, in order

| cost | size (27B 24K, gdn:12) | overlappable? |
|---|---|---|
| CPU layer compute (raw kernel) | 4–6 ms/layer | no — on the dependency chain |
| CPU dispatch/sync gap (11.7 − raw) | ~6 ms/layer | **yes — the target** |
| hidden transfer per boundary | 0.009 ms | no need — already negligible |
| GPU idle during CPU runs (and vice versa) | most of the token | only across tokens — see §4 |

The dependency chain is the honest limit: within one token, layer *i*+1 needs layer *i*'s residual,
so CPU and GPU **cannot** compute two layers of the same token concurrently. Anything that claims
otherwise is wrong. The remaining wins are therefore:

1. **Remove the dispatch gap** (worker thread + one dispatch per run, §3) — converts ~6 ms/layer of
   serial overhead into pipelined work. This alone is where ≥25% lives: if CPU time halves, the
   token time drops by ~40% at the same placement.
2. **Double-buffered pinned staging** (§3) — takes the (already small) boundary copy off the
   critical path; matters once the dispatch gap is gone, and removes the per-switch synchronisation
   that serialises the GPU stream.
3. **Cross-token overlap** (§4) — the only place the two devices truly compute in parallel. Prefill
   and spec-verification only; decode stays serial by dependency.

## 3. Design: CPU worker thread + double-buffered staging

Ownership split (mirrors the existing doorbell sketch in `CPU_GPU_TANDEM_DESIGN.md:168-210`, now
instantiated against `TandemEngine._run`):

- **Main thread** (as today): owns the GPU plan engine, the token loop, and the output head. It
  enqueues one *run descriptor* (backend, layer range, token, pos) and waits on a per-run event.
- **CPU worker thread** (new): owns the CPU engine. It blocks on a doorbell queue; per run it
  executes the CPU engine's per-layer loop for its whole range in **one dispatch** (the CPU engine
  gets a `run_layers(tok, pos, range)` entry point instead of per-layer Python calls from `_run`),
  then posts the hidden into the staging slot and signals completion.
- **Staging (two slots, pinned host memory)**: each slot holds one residual vector plus a sequence
  number. The main thread writes its output residual into slot *i* with an async H2D copy and does
  not block on it; the worker reads from slot *i*, computes its run, and writes to slot *i+1*.
  Sequence numbers make a late or corrupt slot detectable instead of silently wrong (a mismatch is
  a loud error, never a wrong token).

The boundary count does not change (one transfer per run switch); what changes is that the switch
no longer costs a synchronising `get_hidden()` + per-layer Python hops. Steady state: the main
thread is either waiting on the worker's event or submitting GPU work — never in per-layer Python
between the two.

New env knobs (config_reference-visible, defaults keep today's behaviour until the gate passes):

| knob | default | meaning |
|---|---|---|
| `KINDI_TANDEM_WORKER` | `0` | `1`: the worker-thread path (the overlap scheduler) |
| `KINDI_TANDEM_STAGING` | `2` | staging slots (double-buffered at 2) |
| `KINDI_TANDEM_DEBUG_TIMING` | `0` | per-run timing probe (§1 step 2) |

`KINDI_TANDEM_WORKER=0` is today's `_run`, byte for byte. The exactness gate compares worker-on
against worker-off and full GPU (`kindi verify`, `KINDI_TC=0`, 4/4 all three).

## 4. What does *not* overlap (stated so it is not re-proposed)

- **Decode within a token**: impossible (dependency chain, §2). `gdn:` placements scatter CPU
  layers between GPU attention layers; each run still depends on the previous one.
- **Prefill chunk pipelining**: chunk *k*+1's CPU layers need chunk *k*'s GPU output through the
  same chain. The only pipelinable prefill work is the *staging copy itself* (async, §3).
- **The LRU hot-weight cache (ROADMAP wording)**: the CPU engine already holds the whole model in
  host RAM; there is nothing to cache for dense models. It stays a MoE-only future item
  (`MOE-TANDEM.md`), not part of the 25% path.

The genuine cross-token overlap is limited to two narrow places, recorded here as follow-ups, not
part of the gate: (a) **spec verification** — while the GPU verify step for a drafted block runs,
the CPU engine can start the *next* block's CPU-owned early layers for the token the draft
predicts (discarded if the draft is rejected; exactness is unaffected because verify still
recomputes); (b) **chunked prefill as decode batches** — the plan engine's T-token batch plans
already move multiple tokens per launch; giving the CPU engine the same batch shape lets one
dispatch cover a run across tokens. Both need a speculative-correctness argument before they are
gate-relevant.

## 5. The gate, and how it is measured

1. Decomposition probe (§1) lands first; the 27B-24K `gdn:12` plan-owner row (7.9 tok/s) is
   re-recorded as the baseline, with sampled peak.
2. Worker thread + one-dispatch-per-run + double buffering behind `KINDI_TANDEM_WORKER=1`.
3. Gate run (4060 Ti + 8086K, 27B pack, `KINDI_TC=0`, ctx 24576, plan owner, `rate` pick):
   - gen at 24K ≥ 7.9 × 1.25 ≈ **9.9 tok/s** (median of three; sampled peak recorded),
   - `kindi verify` **4/4** on all three (worker-on, worker-off, full GPU),
   - if the miss is within noise of the CPU-dispatch term, the design is wrong — not the threshold;
     re-measure §1 before touching the threshold.
4. Recorded in `docs/benchmarks/<date>-<host>-tandem-overlap/RESULTS.md`, ROADMAP goal row updated
   from "not started" to the measured verdict.

Scope guardrails: no placement change (the split rule is gate 1's, already landed), no new
quantisation, no second sequence execution (single-sequence server semantics unchanged), and the
capacity-mode path (`KINDI_TANDEM_WORKER=0`) remains the fallback until the gate passes.
