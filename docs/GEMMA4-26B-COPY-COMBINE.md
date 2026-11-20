# Gemma 4 26B-A4B: expert copy lane and ordered combine

Status: **design only, 2026-11-12. This fixes the eight-slot copy path, the deterministic
expert-id combine and the failure/forced-eviction contract for a future Gemma 4 26B-A4B backend.
There is no 26B parser, backend, CUDA copy path or measured result. The local artifact is Q4_K_M,
which is not an exactness reference.**

This is the execution-level companion to [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md) and the generic
state machine in [MOE-TANDEM.md](MOE-TANDEM.md). The policy-preserving job/completion seam between
the cache and the copy worker is [EXPERT-CACHE-COPY-INTERFACE.md](EXPERT-CACHE-COPY-INTERFACE.md).
The model constants, expert sizes and acceptance gates stay in the profile. This document adds the
concrete slot image, copy sequence, combine order and hard-error behavior that the backend must
implement without changing the arithmetic.

The cache remains a **weight cache** keyed by `(layer, expert_id)`. It is never part of a prompt
checkpoint, saved opening or KV state.

## 1. Fixed geometry

| item | value |
|---|---:|
| routed experts per layer | 128 |
| selected per token per layer | 8 |
| text layers | 30 |
| maximum slot size | 4,336,640 B (4.14 MiB) |
| cache slots | **8** |
| transient slots | **1** |
| cache slot bytes | 34,693,120 B |
| cache + transient bytes | 39,029,760 B |
| host pinned staging | 2 slots = 8,673,280 B |
| per-expert scale storage | 30 x 128 x 4 = 15,360 B, resident |

The slot is the largest complete expert image, not an average:

```text
slot base + 0                 gate_up[gate_up_bytes]   # up to 2,725,888 B
slot base + gate_up_bytes     down[down_bytes]         # up to 2,106,368 B
slot base + used_bytes        unused tail              # never read
```

`gate_up_bytes` and `down_bytes` are layer-dependent. The normal layer has 2,230,272 + 1,486,848
bytes; layers 9 and 15 have 2,725,888 + 1,486,848; layer 29 has 2,230,272 + 2,106,368. All segment
sizes and the 4,336,640-byte slot are 256-byte aligned, so the arena can be one contiguous
allocation with no per-expert alignment gap.

The per-expert scale is **not** copied into a slot. It remains resident and is applied to the router
weights at the reference point. If a future implementation puts the scale in the slot, the slot
becomes 4,336,656 bytes and the ready event must cover that third copy as well. That is a different
layout and needs its own exactness test.

The GPU arena is nine fixed-sized slots: eight cache slots with indices `0..7` and one transient
slot with index `-1`. A normal top-8 layer uses eight cache slots. The transient slot is an overflow
reserve, not a ninth routed expert and not a second cache.

## 2. Host source and staging

The host pack is memory-mapped and remains resident. Each `(layer, expert)` index entry contains:

```text
gate_up_offset, gate_up_bytes
down_offset, down_bytes
scale_offset, scale_bytes
```

At pack load, every entry is validated against the mapped file:

- `gate_up_bytes + down_bytes <= 4_336_640`;
- each segment is within its tensor and 16-byte aligned;
- the layer's gate/up and down quant formats match the manifest;
- the scale entry covers 128 fp32 values;
- the expert index covers all 30 x 128 keys, with no duplicate or missing key.

The copy lane does **not** hash every miss. The pack checksum is verified once at load; per-expert
hashing would add host CPU work to every cold layer and is not part of the arithmetic guarantee.

Two pinned host staging slots form a small ring. A miss stages the two source segments into one
slot image:

```text
stage[0 .. gate_up_bytes)             <- host gate_up
stage[gate_up_bytes .. used_bytes)    <- host down
stage[used_bytes .. 4_336_640)        <- untouched
```

For the normal Q4 layer the staged copy is 3,717,120 B; layer 29 is 4,336,640 B. The tail is never
part of the copy and never read by the expert kernel. Staging into one contiguous image gives the
copy lane one H2D operation and one completion event per expert instead of two independently
completing segments.

`cudaMemcpyAsync` is only truly asynchronous from page-locked memory. The pack mapping is pageable,
so the host-to-pinned memcpy is done first, then the pinned-to-device copy. The staging ring has two
slots; a third miss waits for a free stage. A stage image remains owned until its copy event has
completed, and is not overwritten while the device may still read it. This bounds host staging
memory and prevents an unbounded copy queue.

## 3. Eight-slot copy path

The per-layer scheduler is the owner of the copy stream and the cache API. CUDA streams and events
are illustrative; the contract is what matters.

```text
copy_stream    one non-blocking stream for all expert H2D copies
compute_stream one non-blocking stream for expert and combine kernels
stage_ring     two pinned 4,336,640-byte images
slot_events    one completion event per in-flight expert copy
```

The normal sequence for one layer is:

1. The router runs on the GPU and produces the selected ids and router weights. Keep the top-k
   order for scheduling; derive the combine order separately.
2. Build eight `ExpertKey(layer, id)` values in router order and call
   `cache.plan(keys, token=(request_id, layer))`. The plan pins all eight selections and reserves a
   slot for each miss. It is atomic: if eight slots cannot be reserved, it raises
   `ExpertCacheFull` before changing a slot.
3. For each plan entry, in router order:
   - **hit**: no copy; the slot is already `READY` and pinned.
   - **miss**: enqueue one complete slot fill: stage both host segments into a pinned image, then
     `cudaMemcpyAsync(stage, slot_ptr, used_bytes, H2D, copy_stream)`, and record the event.
   - **coalesced**: if the key was already `LOADING`, wait for the existing event; do not start a
     second copy and do not allocate a second slot.
4. When a copy event completes, query its status. On success call `cache.mark_ready(key)`; on
   failure take the failure path in section 6. A slot is never `READY` with only one segment
   present.
5. On the compute stream, wait for that expert's event, call `cache.begin_use(key)`, launch the
   fused gate/up kernel followed by the down kernel, and record an expert-completion event. Hits
   and completed copies can run while later copies are still in flight.
6. Wait for all eight expert-completion events. Do not read an expert output before its event.
7. Call the ordered combine in section 4. The shared/dense MLP is a separate branch and is not a
   cache entry.
8. After the combine, call `cache.end_use(key)` for each expert, then `cache.release(token)`.
   `READY` cache slots stay resident; the transient slot is freed when its last pin is released.

The pilot is batch size 1. Eight selections per layer are handled one layer at a time; the backend
must never pin all 240 selected experts across 30 layers. The top-8 plan is released before the next
layer starts.

### Copy ordering and slot independence

Copies are scheduled in router order, because that is the natural order in which the selected
weights are available. This is only a scheduling choice. The slot index is also only a storage
choice. The combine maps `expert_id -> output` explicitly; it never derives arithmetic order from
the slot number, the copy completion order or the router's descending-probability order.

A warm hit and a cold miss therefore feed the same expert kernel with the same bytes and produce the
same output vector. A change in copy order, stage-ring occupancy or slot assignment must not change
a single output bit.

## 4. Deterministic expert-id combine

### 4.1 Selected set and combine order

The router contract is:

```text
p = softmax(router_logits, dim=-1)       # 128 experts
top_p, top_i = topk(p, 8)                # reference selection
top_p = top_p / top_p.sum()              # renormalise the selected eight
weights = top_p * per_expert_scale[top_i]
```

`top_i` is in the reference's top-k order, normally descending probability. The official
`Gemma4TextExperts` path collects `expert_hit = nonzero(selected_mask)` and adds each expert in that
order. `nonzero` yields ascending expert ids, so the arithmetic order is:

```text
combine_ids = sorted(top_i)              # ascending expert id, always
combine_weight[id] = weights[index_of(id)]
```

The backend stores both arrays:

```text
selected_ids_router_order = (97, 3, 41, 8, 55, 12, 120, 76)
selected_ids_combine_order = (3, 8, 12, 41, 55, 76, 97, 120)
```

The expert kernel itself must have a fixed reduction order. The copy path removes cache-state and
completion-order variability; it cannot make an otherwise nondeterministic expert kernel exact.

The router order may drive copies. The combine order is ascending id. A backend must not use
`atomicAdd`, completion order, slot order, a parallel reduction tree or `topk` order for the
combine.

`torch.topk` has no specified tie rule. If two experts have equal router probability, the selected
set can depend on the reference implementation. This design does **not** invent a lexicographic
tie-break; G0 must pin the reference and a router tie test. Sorting the already selected ids does not
resolve a tie in the selection itself.

The ids are expected to be unique. `ExpertCache.plan()` deduplicates defensively, but a duplicate
selected id in the backend is a router bug and must fail the layer rather than silently changing the
weight sum.

### 4.2 Combine kernel

The combine is a deterministic ordered accumulation over the eight complete expert outputs:

```text
out[hidden] = 0
for id in combine_ids:                    # ascending expert id
    for h in range(hidden):
        out[h] += expert_out[id][h] * weight[id]

# then the shared/dense branch is applied at the reference point
layer_out = shared_combine(routed_post_norm(out), dense_post_norm(shared_out))
```

One work item owns one output element and walks the eight experts in ascending id order. There is no
cross-work-item reduction. The expert output buffers are keyed by expert id, not by slot index.
The kernel must use the reference's accumulator dtype and operation order; do not introduce TF32,
tensor-core reduction or a fused scale in the down-projection epilogue unless the pinned reference
does the same and the result is bitwise-equal.

The per-expert scale is applied to the router weight at the reference point, before the combine. Do
not fold it into the expert down epilogue in the first path. The local tensor is named
`ffn_down_exps.scale`; G0 must pin exactly where the official implementation applies it.

The dense/shared MLP is computed in parallel with the routed branch. It is not appended as a ninth
expert, is not evicted, and is not included in expert hit-rate counters. For Gemma 4, the routed and
shared branches have separate pre/post norms; both post-norm outputs are added in the reference
order. The expert cache is responsible only for the routed part.

### 4.3 Why this is deterministic

Two runs can differ in:

- which eight slots hold the selected experts;
- the order in which copies complete;
- which expert kernel launches first;
- whether a slot was cold, warm or reloaded.

None of those enter the combine. All eight outputs exist before the combine starts, and the combine
walks `sorted(selected_ids)`. Given the same selected ids, router weights and expert bytes, the
result is the same bit sequence. The exactness check must randomise slot assignment and copy
completion order and still compare bit-for-bit.

## 5. Forced eviction

### 5.1 Capacity preflight

`ExpertCache.plan()` counts:

```text
available = free_cache_slots
          + unpinned READY cache slots eligible for LRU eviction
          + free transient slots
```

The current plan's hit slots are pinned and protected. `LOADING` and `IN_USE` slots are never
victims. If `misses > available`, the plan raises `ExpertCacheFull` without changing any slot, and
the request fails. There is no partial plan and no "evict anyway" mode.

The backend requires at least **eight cache slots plus one transient slot**. It must reject the
configuration at load time if the configured cache cannot provide that minimum; it must not discover
the shortfall after the router has selected an expert set.

### 5.2 Victim selection

An eviction is allowed only for an unpinned `READY` cache slot. The victim order is deterministic:

```text
(oldest last_used, lowest slot index)
```

`last_used` is advanced on a hit, on admission, on `mark_ready()` and on use. Ties therefore resolve
to the lowest slot index, not to Python set order, dictionary order or copy completion order.
Randomised victim selection is forbidden in the exact lane.

Eviction is cache-only and has **no rollback**. Once a slot is evicted, its previous expert is gone
from VRAM. If a later copy in the same plan fails, the old experts are not restored; the request
fails and a retry recopies whatever it needs. This is correct for a cache but must be visible in the
failure metrics.

### 5.3 The forced-eviction trace

A test must exercise all eight selected experts missing while the cache is under pressure:

1. Warm eight unrelated experts into slots `0..7`; release the warm plan so all eight are `READY`
   and unpinned.
2. Hold a synthetic test pin on slot `0` (or pin one expert with a second test token). Seven slots
   are now evictable.
3. Route a token whose top-8 is disjoint from the eight resident experts. The plan needs eight
   slots, so it evicts the seven unpinned slots and takes the one transient slot.
4. Inject a slow copy on the transient expert. Assert:
   - seven evictions, one transient admission;
   - the pinned slot is unchanged;
   - all eight selected experts become `READY` only after their complete copy;
   - peak used slots is nine, never more;
   - the combine still sees eight outputs in ascending expert-id order;
   - the released plan leaves the seven new cache slots resident and frees the transient slot.
5. Run the same prompt with a warm cache, with an all-miss cache, and after a clean reload. The
   generated ids and logits must be identical.

If all eight cache slots are unpinned and a disjoint top-8 arrives, the plan evicts all eight and no
transient slot is used. That is a valid forced-eviction run, but it does not test the overflow path.
The seven-eviction-plus-transient trace above is the required G2 case.

If all eight cache slots are pinned, the transient slot cannot absorb eight misses. The plan must
fail with `ExpertCacheFull`; "forced eviction" never means evicting a pinned or in-use expert.

## 6. Failure behavior

All failures are **hard request failures**. The first exact copy lane has no CPU-compute fallback,
no approximate expert, no skipped expert, no reduced top-k and no partial token. A cache miss may
change latency, never the arithmetic.

| failure | detection | slot/cache action | request action |
|---|---|---|---|
| insufficient capacity | plan preflight | none; plan is atomic | fail before copy; `ExpertCacheFull` |
| cache below 8+1 at load | startup fit check | do not start backend | refuse the model/configuration |
| pack index out of bounds or misaligned | pack validation or stage bounds check | no copy; release plan; fail any loading slots | fail; no expert output |
| staging ring timeout | bounded wait for a free stage | release plan; drain stage jobs | fail; no partial combine |
| copy launch error | CUDA return code | no `mark_ready`; discard the slot after its event is accounted for | fail layer; no combine |
| async copy error | event query/synchronize | discard the whole slot, never expose a partial image | fail layer; no combine |
| only one segment arrived | event/ready invariant | discard the whole slot | fail layer; no combine |
| copy timeout or device stall | bounded event wait | abort copy; synchronize; do not reuse the slot until the event is accounted for | fail; clear sequence state |
| expert kernel error | CUDA return code/event | abort kernel; reset the whole cache for the first implementation | fail layer; no combine |
| combine kernel error | CUDA return code | abort combine; no output is committed | fail request; no token streamed |
| generated token mismatch at a gate | exactness harness | cache state is suspect | fail gate; do not ship the backend |
| cancellation | request cancellation | release pins; synchronize copy/compute; fail loading slots | discard the request; no prompt-cache commit |

### 6.1 Copy failure sequence

The cache prototype has `LOADING`, `READY` and `IN_USE` states and no separate `FAILED` state. The
backend therefore uses this order:

1. On a copy error, stop scheduling new copies for the layer.
2. Record the error and the slot/key. Do not call `mark_ready()`.
3. Wait for the copy event to be complete (or synchronize the copy stream) before touching the slot
   again. This is required even when the event reports failure: the hardware may still be writing
   the slot buffer.
4. Release the plan's pins. A `LOADING` slot remains unavailable until it is accounted for.
5. Call `cache.fail(key)` after the release. It increments `copy_failures`, drops the key and frees
   the slot with `aborted_loads`.
6. Do not launch the expert kernel, do not combine and do not emit a partial token.
7. If the failure indicates a sticky device error, reset the CUDA context and discard the entire
   expert cache; a per-slot cleanup is not sufficient.

No automatic retry occurs inside the token. A client or request-level retry starts a fresh plan and
recopies the missing experts; evicted experts are not restored. A failed request must not commit a
prompt-cache checkpoint or a saved opening.

### 6.2 Kernel and combine failure

An expert kernel failure happens after the slot is `READY`/`IN_USE`. The slot's weights are not
necessarily corrupt, but the output buffer may be. The backend must:

- synchronize the compute stream and read the CUDA error;
- abort the whole layer; do not use the partial expert output;
- release the plan and reset the whole expert cache for the first implementation. This avoids
  inventing a `FAILED` slot state that the current model-free prototype does not have; a later
  per-slot quarantine must have its own invalidation tests;
- return a request error and stream no partial answer.

A combine error is a request-level failure. There is no partially committed residual. For a sticky
error, reset the context and rebuild the resident core and the cache before serving another request.
The shared/dense branch is recomputed from the layer's input on a retry; it is not cached as an
expert.

### 6.3 Cancellation

Cancellation follows the same fence: release the plan, drain the copy and compute streams, account
for every `LOADING` slot with `mark_ready()` or `fail()`, and only then allow reuse. A slot whose copy
was cancelled must not be marked `READY`. A cancelled request does not write a prompt checkpoint,
does not emit a partial stream and does not leave a transient slot pinned.

### 6.4 Forced-eviction failure

Forced eviction and copy failure can combine. If seven slots were evicted and the transient copy
fails, the request fails; the seven successfully copied cache slots remain resident (they are
complete), the transient is discarded and the evicted experts are not restored. This is a cache
state change, not an arithmetic change: the next request recopies the missing experts and must
produce the same output as the pre-eviction run.

## 7. Metrics and diagnostics

The backend must expose at least:

```text
slots=8, transient=1, slot_bytes=4,336,640
expert_hits, expert_misses, coalesced_loads
evictions, admissions, transient_admissions, transient_drops
copy_enqueued, copy_completed, copy_failed, aborted_loads
copy_bytes, copy_wait_ms, stage_wait_ms
expert_kernel_ms, combine_ms
peak_used_slots (<= 9), peak_pinned
combine_order_hash, selected_ids_count
```

`combine_order_hash` is a hash of `sorted(selected_ids)`, not of the router order. It lets a test
prove that two schedules used the same arithmetic order without logging prompt content. Hit rate
must be reported next to bytes copied and slot count; a hit rate alone is not a memory or
performance claim.

A cache error, failed copy, partial expert, bad slot state or parity failure is a hard error. It is
never reported as a cache miss with a silent CPU fallback.

## 8. Required tests before G2

1. **All-hit layer:** eight warm experts; no copies; combine equals the reference.
2. **All-miss layer:** eight cold experts; eight complete copies; exact output.
3. **Forced eviction:** the seven-eviction-plus-transient trace in section 5.3; pinned slot stays
   resident; all eight selected experts are exact.
4. **All-eight-eviction:** eight unpinned residents replaced by eight disjoint missing experts; no
   transient; exact output.
5. **Copy failure:** inject a failure on one of eight copies after eviction; no combine, no partial
   token, failed slot freed, process can serve a fresh request.
6. **Partial-segment failure:** gate/up succeeds and down fails; slot is discarded, never `READY`.
7. **Copy-order determinism:** reverse copy completion order and randomise slot assignment; output
   bits are unchanged.
8. **Combine-order determinism:** use distinct synthetic expert outputs; verify the result equals
   ascending-expert-id accumulation, not router order, slot order or atomic order.
9. **Router tie:** pin the reference's `topk` behavior; do not impose a new tie-break. A tie that
   changes the selected set is a G0 router result, not a cache result.
10. **Capacity rejection:** all eight slots pinned; a new top-8 plan raises `ExpertCacheFull` without
    changing a slot.
11. **Cancellation:** abort during a copy; no slot is reused before its event is accounted for; no
    prompt checkpoint is committed.
12. **Reload:** cold, warm, forced-eviction and failed-copy requests all match cache-off/cold token
    ids at the same context.

The G2 gate in [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md#7-gates) must require the forced-eviction trace
and the copy-failure trace, not only a warm run that happens to fit.

## 9. Non-claims and open items

This design does **not**:

- add a Gemma 4 26B parser, backend, CUDA kernel or cache integration;
- claim a copy path exists in `kindi/`;
- claim Q4_K_M is exact against the official BF16 model;
- claim a throughput, hit-rate or peak-VRAM result;
- support batch size greater than one, speculative prefetch or CPU expert compute;
- permit eviction of pinned, `LOADING` or `IN_USE` experts.

Before implementation, G0 must pin:

- the exact reference dtype and accumulation order;
- the official router's `topk` tie behavior;
- the exact application point of `ffn_down_exps.scale`;
- the global-layer V derivation and the full 30-layer schedule.

Until those are pinned and the tests above pass on real hardware, this is a design contract, not a
support claim.

Related records: [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md), [MOE-TANDEM.md](MOE-TANDEM.md),
[EXPERT-CACHE-COPY-INTERFACE.md](EXPERT-CACHE-COPY-INTERFACE.md), [RAON-1-MINI.md](RAON-1-MINI.md),
[ROADMAP.md](ROADMAP.md).
