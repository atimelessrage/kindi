# Other drafter/target pairs (2026-11-14)

Same host and method as RESULTS.md (i7-8086K + RTX 4060 Ti 8 GB, tensor-core plans, greedy,
prompt 5,012 tokens, second request after prompt-cache warmup; one run per configuration).

## 4B drafter -> 27B target: does not fit at all

The 4B drafter loads (0.92 GB of weights, plans ready, 4,979 MiB free), then the 27B cannot even
stream its 5.60 GB of weights into what is left: the loader refuses with "not enough GPU memory
for a 1024-token context" and the server exits. 4B->27B is a tandem-slice question on 8 GB, exactly
like 24K-with-2B-drafter, but worse: there is no usable context at all.

## 2B drafter -> 4B target: fits, good acceptance, still loses

| | no speculation | 2B drafter, k=4* | result |
|---|---|---|---|
| ctx achieved | 24,576 (unreduced) | 24,576 (unreduced) | fits fully |
| 111 tokens, warm | 1.30 s (85.6 tok/s) | 1.47 s (75.5 tok/s) | **0.88x** |
| acceptance | - | 73/160 = 45.6% (40 passes) | good - and still loses |

*`drafting.last`: 40 passes, 160 drafted, 73 accepted.

The 4B target pass is too cheap next to a drafter step + T=4 verify for 45% acceptance to pay -
the same law as 9B (0.98x at 41%), one size smaller. Fits does not imply helps.
