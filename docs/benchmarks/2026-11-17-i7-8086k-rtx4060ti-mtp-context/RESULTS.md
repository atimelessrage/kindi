# Mode A (MTP head) at depth: 512 / 8K / 24K on the 2B

Measured 2026-11-17 on the i7-8086K + RTX 4060 Ti, `KINDI_TC=1`, pack `bonsai-2b`, head
`kindi-mtp-2b-v1.npz` (sha256 `1b5c0c79…`, the registered file). Prompt: the repository's `docs/*.md`
joined in name order and tokenized (281,805 tokens available), truncated to the position. 128 new
greedy tokens per run, 2 runs per mode unless a row says otherwise, medians reported.

Method: `tools/mtp_context_bench.py` builds the engine once, then for each position runs plain and
mode A alternately **on the same engine** - only `KINDI_SPEC` and `spec_begin` change - so the A/B
shares weights, prompt and kernel caches. Prompt feeding is excluded from the clock; the timer
covers the 128 steps. This is why the numbers differ from the 400-token request row in
`docs/MTP.md` even at the same default `KINDI_SPEC_K=4`.

| context | `KINDI_SPEC_K` | plain tok/s | mode A tok/s | ratio | drafts accepted |
|---:|---:|---:|---:|---:|---|
| 512 | 1 | 335.4 | **375.6** | **1.12x** | 52/75 (69%) |
| 512 | 4 (default) | 347.6 | 307.9 | 0.89x | 77/200 (39%) |
| 8192 | 1 | 327.4 | 329.1 | 1.01x | 43/84 (51%) |
| 8192 | 4 (default) | 330.1 | 218.3 | 0.66x | 60/272 (22%) |
| 24576 | 1 | 303.6 | **344.4** | **1.13x** | 53/74 (72%) |
| 24576 | 2 | 292.2 | 302.2 | 1.03x | 64/126 (51%) |
| 24576 | 3 | 293.1 | 279.2 | 0.95x | 71/168 (42%) |
| 24576 | 4 (default) | 292.1 | 254.4 | 0.87x | 75/208 (36%) |
| 24576 | 6 | 292.5 | 194.1 | 0.66x | 75/312 (24%) |

The 24576 / k=1 row is the median of **4** runs: plain 304.8, 303.6, 303.5, 301.8; mode A 338.0,
344.6, 345.4, 344.2 - the run-to-run spread is under 3%, so the 13% gap is not noise.

Exactness at the winning setting: `KINDI_SPEC_K=1 kindi verify --spec mtp` is **4/4 token-exact**
with llama.cpp (160/160 tokens, 65/94 drafts accepted, 405.4 tok/s).

## What it decides

- Acceptance does rise with depth: 69-72% at k=1 (512 and 24K), 51% at 8K, against 22% at 8K and
  66% at 512 with k=4. The 8K dip is real in this prompt set: 51% at k=1 against 69% at 512.
- Mode A is a win **only at k=1**, and only by about 12-13% (512 and 24K); at 8K it is a wash
  (1.01x). The shipped default `KINDI_SPEC_K=4` loses everywhere measured, and k>=3 at 24K loses.
- `KINDI_SPEC=auto` cannot reach the win: it caps mode A at k=2 below 8K and turns it off above
  (`AUTO_MTP_K2_MAX`), so at 24K it runs the plain engine.

Artifacts in this directory: `mtp-context.json` (8K and 24K, k=4), `mtp-context-k{1,2,3,6}.json`
(24K), `mtp-context-k1-8k.json`, `mtp-context-k1-512.json`, `mtp-context-k4-512.json`,
`mtp-context-k1-24k-r4.json`. Harness: `tools/mtp_context_bench.py`. No number here has been
registered in `docs/claims/` because `docs/MTP.md` is not one of the claimed docs; it quotes these
figures directly.
