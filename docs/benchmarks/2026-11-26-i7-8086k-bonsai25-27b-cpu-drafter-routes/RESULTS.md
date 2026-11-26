# 27B CPU drafters: the 2B pack with and without the MTP head in the file (2026-11-26)

The routes a Bonsai 2.5 27B CPU deployment can draft with, on the same weights and the same
prompts:

1. **the 2B pack drafter on the model as shipped** - the base `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf`
   converted to a pack (no `blk.64`), `KINDI_SPEC=draft2b`'s route;
2. **the `-mtp` file's own head, as the library loads it** - no 2B pack, no extra file;
3. **the 2B pack drafter on the `-mtp` file** - the same model with `blk.64` present and unused.

## Method

The two packs are the same 64-layer model converted from the two GGUF files (7,242 MB each,
`kindi convert`), so route 1 and routes 2-3 differ only in whether the file carries the head. One
process per target pack (`bench-logs/compare_three.py`), plain greedy decode of the two prompts
once, then each drafter scored against those same ids:

```
KINDI_DEVICE=cpu python bench-logs/compare_three.py --pack PACK --drafter-pack PACK_2B \
    --tokens 16 --k 4 --out run.json
KINDI_DEVICE=cpu python bench-logs/compare_three.py --pack PACK --gguf <the -mtp GGUF> \
    --drafter-pack PACK_2B --tokens 16 --k 4 --out run.json
```

Same prompts as the [head vs drafter page](../2026-11-25-i7-8086k-27b-cpu-mtp/RESULTS.md)
(`In one sentence: what is a ternary weight?`, `2 + 2 = ? Answer with the number only.`), 16 greedy
tokens, `k=4`, ctx 2048, 6 threads, i7-8086K (`avx2-fma-f16c`). The head runs with its library
defaults - `Head()`: raw residual, no verified-row seed, window 64, raw embedding row - which is
what "the MTP model as is" means for this tree. Drafts are scored, never substituted, so every run
is token-identical to plain greedy decode (`exact: true` on all six route/prompt rows).

## Result

| route | accepted | rate | depth [1..4] | drafted tok/s (per prompt) | plain tok/s |
|---|---|---|---|---|---|
| 2B drafter, base pack (no `blk.64`) | 33/108 | 30.6% | [15, 8, 7, 3] | 1.19 / 1.14 | 1.98 / 1.98 |
| `blk.64` head as loaded, `-mtp` pack | 9/108 | 8.3% | [1, 1, 4, 3] | 0.10 / 0.10 | 2.14 / 2.06 |
| 2B drafter, `-mtp` pack | 33/108 | 30.6% | [15, 8, 7, 3] | 1.21 / 1.12 | 2.14 / 2.06 |

Per prompt: the 2B drafter accepted 23/54 and 10/54 on **both** packs; the head accepted 2/54 and
7/54. Its 120 head rows cost 159 s and 156 s, against 13-14 s for a 2B-drafted prompt.

## What it decides

**The 2B drafter's acceptance does not depend on the file.** Drafts are scored against the model's
own tokens, and the 2B's drafts come from the 2B engine; adding `blk.64` to the file changes
neither. So the `-mtp` download buys nothing for a `draft2b` deployment and costs 239 MB, and route 1
is the one to ship until CP1's batched verify exists.

**The head's library default is the raw-hidden one.** Route 2 measures 9/108 (8.3%), the `eh` row of
the [variant ladder](../2026-11-25-i7-8086k-27b-cpu-mtp-variants/RESULTS.md), not the 11/108 the
ladder's `eh-normed` row reaches: `Head()` still defaults `hidden="raw"`, so the output-normed
hidden, the verified-row seed and the prompt's last row are probe switches, not shipped behaviour.
Whatever a deployment gets from `blk.64` today is the 8.3% row.

**Neither route is a speedup on this host.** The CPU engine verifies one row per pass, so a drafted
run is strictly slower than plain decode; the head is by far the most expensive (0.10 vs 1.2 tok/s)
because each of its rows costs a 5120-wide attention layer plus the 248,320-token output head, and
it is right 8.3% of the time. CP1 (batched T-row verify) is what makes either route pay.

## What was not verified

No GPU run. One host, two prompts, 16 tokens, `k=4`: a two-prompt sample, not a distribution. The
`-mtp`-pack 2B numbers repeat the earlier tracked run
([head vs drafter](../2026-11-25-i7-8086k-27b-cpu-mtp/RESULTS.md)) exactly, which is the expected
reproducibility check rather than an independent one. `results.json` is the sanitized record of the
two runs (machine paths stripped); the harness and sanitizer stay untracked in `bench-logs/`.
