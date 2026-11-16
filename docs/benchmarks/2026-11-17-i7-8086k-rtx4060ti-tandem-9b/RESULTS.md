# 9B tandem vs full GPU: the split buys headroom, never context (2026-11-17)

The 27B answered this question on this card (the loader reduces a 65,536/131,072-token
request to 37,120 on the full-GPU engine and to 30,464 on the tandem, so the split bought
headroom, not context). The 9B had only been spot-checked at 8K and 24K, where the split
saved no peak. This page measures it properly across the whole context range, so the gap
between a split and a single-card run is a number rather than an impression.

Host: i7-8086K, 6 threads, `avx2-fma-f16c`, native Windows, clang 18.1.8 + libomp;
RTX 4060 Ti 8,188 MiB, driver 617.42. Pack: `bonsai-9b` (2.3 GiB, 5-trit lane, plan engine
with q8_0 K/V on the GPU side). Method: `python -m kindi bench --sample-vram` once per cell
(`KINDI_CTX` sets the request; the loader may reduce it), `--tandem cpu-first
--tandem-cpu-layers gdn:N` for the split. Device-wide sampled peak; "over idle" subtracts the
card's baseline use. One run per cell, so treat rates as ±5% (the same grid's GPU-only cells
repeat within that) and peaks as ±32 MiB sampling granularity. Raw JSON: gitignored
`bench-logs/tandem-9b/*.json`.

`gdn:8` puts the CPU on layers 0,1,2,4,5,6,8,9; the GPU keeps attention layers 3,7 and
10-31 plus the head. `gdn:16` adds 10,12,13,14,16,17,18,20 to the CPU side.

## Capacity: nothing reduces

Every request is kept in full by both placements, up to the 131,072-token ceiling tested.
The full-GPU engine still has 3,003 MiB free after loading 131,072 tokens.

| requested | full GPU: kept | full GPU: peak (over idle) | full GPU: free after load | tandem `gdn:8`: kept | tandem peak (over idle) | tandem free after load |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 3,063 (2,094) | 5,089 | 8,192 | 2,577 (1,608) | 5,577 |
| 32,768 | 32,768 | 3,481 (2,512) | 4,671 | 32,768 | 2,993 (2,024) | 5,161 |
| 49,152 | 49,152 | 3,761 (2,792) | 4,391 | 49,152 | 3,273 (2,304) | 4,881 |
| 65,536 | 65,536 | 4,037 (3,068) | 4,115 | 65,536 | 3,549 (2,580) | 4,605 |
| 131,072 | 131,072 | 5,149 (4,180) | 3,003 | 131,072 | 4,661 (3,692) | 3,493 |

The split's peak saving is flat at **486-488 MiB** across the whole ladder — the eight CPU
layers' weights move off the card and nothing else changes size enough to show. It is the same
saving at 8K, where 5,089 MiB was already free, as at 131K, where 3,003 MiB was.

## Rate at comparable contexts

`kindi bench` reports generation at fixed positions inside the allocated context, so these
cells are the same work in every column. Read is prompt ingestion.

| context | full GPU read | full GPU gen @100 / 8,000 / 24,000 | tandem `gdn:8` read | tandem gen @100 / 8,000 / 24,000 | gen ratio (24,000) |
|---:|---:|---|---:|---|---:|
| 8,192 | 112.0 | 98.3 / 92.4 / — | 34.9 | 33.5 / 30.4 / — | 3.0x |
| 32,768 | 112.2 | 97.0 / 92.5 / 82.0 | 31.4 | 30.3 / 31.8 / 29.4 | 2.8x |
| 49,152 | 111.8 | 95.6 / 92.4 / 81.9 | 37.0 | 30.1 / 33.3 / 29.6 | 2.8x |
| 65,536 | 112.3 | 98.8 / 91.6 / 81.8 | 35.2 | 29.7 / 28.2 / 29.5 | 2.8x |
| 131,072 | 111.8 | 97.3 / 91.4 / 82.1 | 35.0 | 29.1 / 31.9 / 31.8 | 2.6x |

The split does not decay with context the way the full-GPU path does (82-98 tok/s down to
82 tok/s) because the CPU layers it owns are independent of the GPU's KV size; it simply
starts 3x lower. Prompt reading is a flat 3.2x behind (112 vs 31-37), since every prompt
token crosses the split boundary too.

## The exchange rate, and what more layers buy

| placement | peak at 8,192 (over idle) | peak at 131,072 (over idle) | gen @24,000 | saving vs full GPU | rate cost |
|---|---:|---:|---:|---:|---:|
| full GPU | 3,063 (2,094) | 5,149 (4,180) | 82.0 | — | — |
| tandem `gdn:8` | 2,577 (1,608) | 4,661 (3,692) | 29.4-29.6 | 486-488 MiB | 2.8x |
| tandem `gdn:16` | 2,191 (1,222) | 4,275 (3,306) | 18.4-18.6 | 872-874 MiB | 4.5-4.6x |

Doubling the CPU-owned layers buys another ~386 MiB and costs another ~1.6x. That is the
same shape the 27B sweep showed on this card: a steady exchange, no knee.

## Verdict

**For the 9B on this card the split buys headroom, not context, and the headroom has no use.**
The full-GPU engine keeps every request up to the 131,072 tested with 3.0 GiB free after
loading; the split lowers the peak by under 0.5 GiB (gdn:8) or 0.87 GiB (gdn:16) at 2.8-4.6x
the token cost. So the remaining gap against a single-card run is **throughput, not capacity**:
2.8-3.0x on `gdn:8` and 4.5-4.6x on `gdn:16`, at every context tested, with the whole pack
already resident.

The 27B is the opposite case and the only one on this card that needs the split: at 65,536 the
full-GPU loader reduces the request to 37,120 tokens, while the plan-engine owner with 19 GDN
layers keeps a 49,152-token request that the unsplit plan engine reduces to 30,976. Read the
two together: the split is the tool for a pack that does not fit, and a net loss for one that
does. See [27B-8K.md](../2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md) and
[plan-owner RESULTS](../2026-11-14-i7-8086k-rtx4060ti-plan-owner/RESULTS.md).

Exactness was not re-measured here; the 9B tandem is token-exact against the full-GPU engine
in both head placements at 512 context (2026-11-12) and with the plan owner at `gdn:6`
(4/4, 2026-11-14). `kindi bench` is a throughput probe, not an exactness gate.

## Reproduce

```bash
export KINDI_PACK=~/.kindi/packs/bonsai-9b KINDI_THREADS=6
export KINDI_CC="$LLVM/bin/clang.exe" KINDI_LLVM="$LLVM"   # Windows CPU library only

for ctx in 8192 32768 49152 65536 131072; do
  KINDI_CTX=$ctx python -m kindi bench --sample-vram --json bench-logs/tandem-9b/gpu-$ctx.json
  KINDI_CTX=$ctx python -m kindi bench --tandem cpu-first --tandem-cpu-layers gdn:8 \
    --sample-vram --json bench-logs/tandem-9b/tandem-gdn8-$ctx.json
done

KINDI_CTX=131072 python -m kindi bench --tandem cpu-first --tandem-cpu-layers gdn:16 \
  --sample-vram --json bench-logs/tandem-9b/tandem-gdn16-131072.json
```
