# Expert cache ↔ copy-stream seam

Status: **design only, 2026-11-12. This defines the narrow interface between `kindi/expert_cache.py`
and a future copy stream. There is no copy-stream implementation, GPU allocation or MoE backend in
Kindi. The cache state machine and its eviction policy are unchanged.**

The cache owns slot allocation, pinning, LRU/transient selection and all slot-state transitions. A
future copy stream owns only the movement of already-requested bytes. It must not see a free-slot
count, choose a victim, reserve capacity, release a pin or call `plan()`.

This seam is generic. The Gemma 4-specific slot image, staging layout and forced-eviction trace are
in [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md); the backend contract is in
[MOE-TANDEM.md](MOE-TANDEM.md).

## 1. Policy firewall

| owner | allowed to decide | must not do |
|---|---|---|
| `ExpertCache` | slot count, free/transient choice, LRU victim, pins, `LOADING` → `READY`, failed-slot discard | perform an H2D copy, know a CUDA stream, expose an eviction hook |
| copy stream | staging, H2D scheduling, copy queue depth, completion/error detection | call `plan`, `_evict`, `mark_ready`, `fail`, `release`, `begin_use` or `end_use`; choose or free a slot |
| backend scheduler | call `plan`, build jobs for misses, call `mark_ready`/`fail` after completion, run expert kernels, combine, release | move victim selection out of `ExpertCache`; retry a copy by taking a different slot |

The **only** cache API a backend needs is the existing one:

```text
plan(keys, plan_token) -> ExpertPlan
acquire(key, plan_token) -> Slot
mark_ready(key)        # after one successful copy event
begin_use(key)         # before the expert kernel
end_use(key)           # after the expert kernel
release(plan_token)    # after combine or on cancellation
fail(key)              # after release and a terminal copy failure
slot_for(key)          # read-only lookup for completion validation
```

No `begin_copy`, `cancel_copy`, `reserve_slot` or `evict_for_copy` method is added. `plan()` already
pins every selected expert and puts each miss in a `LOADING` slot. That plan entry is the copy
stream's authorization to write exactly one slot.

## 2. Wire objects

The copy stream receives plain descriptors, not an `ExpertCache`, `Slot` or `PlanEntry` object. This
keeps policy state out of the copy worker.

```python
# All fields are immutable for the lifetime of one job.
CopyTicket = namedtuple("CopyTicket", "request key slot_index generation job_id")

CopySegment = namedtuple("CopySegment", "source slot_offset length")

CopyJob = namedtuple(
    "CopyJob",
    "ticket destination segments",
)

CopyCompletion = namedtuple("CopyCompletion", "ticket status error bytes_copied")

COPY_OK = "ok"
COPY_FAILED = "failed"
COPY_CANCELLED = "cancelled"


class CopyQueueFull(RuntimeError):
    """The bounded copy queue cannot accept another job yet."""
```

`source` is a stable host range owned by the resident pack; the backend must not unmap or reuse it
while the job is live. `destination` is an opaque backend-owned device buffer for the assigned slot.
`slot_offset` is relative to that buffer. The copy stream does not need the cache's `slot_bytes` or
its eviction state.

`CopyTicket.request` is the plan token (a backend-defined request/layer key). `key` is
`ExpertKey(layer, expert)`. `slot_index` is the cache slot, including a negative transient index.
`generation` is `entry.slot.loads` at the time the job is submitted. `job_id` is a scheduler-owned
unique integer used for diagnostics and completion matching.

The ticket is the fence. A completion is valid only if the cache still maps `ticket.key` to the same
`slot_index` and `slot.loads` still equals `ticket.generation`. This is necessary because a cancelled
or failed copy can complete after its plan was released and the slot was reassigned.

## 3. Copy-stream protocol

```python
class CopyStream:
    def submit(self, job: CopyJob) -> None:
        """Enqueue one complete slot fill.

        Raises CopyQueueFull without touching any cache state. The caller
        holds the plan pins while it waits for room. It never asks the cache
        for another slot.
        """

    def poll(self, limit=None) -> tuple[CopyCompletion, ...]:
        """Return zero or more terminal completions.

        The copy worker never calls back into the cache. The scheduler thread
        is the only thread that calls mark_ready/fail.
        """

    def cancel(self, request) -> None:
        """Best-effort cancel of every job owned by request.

        Cancellation is not completion: the scheduler must still poll/drain
        until each ticket is terminal before accounting for the slot.
        """
```

`submit()` is called only for a miss that has no active job for its
`(slot_index, generation)`. A cache `LOADING` slot may already be backed by a job submitted by
another plan; that is a coalesced wait, not a second copy. The backend keeps:

```text
jobs_by_generation[(slot_index, generation)] -> CopyJob
jobs_by_request[request] -> set[(slot_index, generation)]
```

A `plan()` entry with `hit=False` therefore has two cases:

1. **new miss**: no active job exists; build and submit one.
2. **coalesced miss**: a job already exists; do not submit another. Wait for its completion and use
   the same ticket.

A `hit=True` entry needs no job.

`submit()` raising `CopyQueueFull` is backpressure, not `ExpertCacheFull`. The scheduler keeps the
plan pinned and retries after `poll()` frees queue space. It must not evict a different cache slot to
make the queue shorter.

## 4. Scheduler sequence

```text
plan = cache.plan(keys, plan_token)          # cache chooses the slots and pins them
jobs = []
for entry in plan:
    if entry.hit:
        continue
    generation = entry.slot.loads
    if (entry.slot.index, generation) in jobs_by_generation:
        continue                              # coalesced; wait for existing job
    job = CopyJob(
        ticket=CopyTicket(plan_token, entry.key, entry.slot.index,
                          generation, next_job_id()),
        destination=slot_buffer(entry.slot.index),
        segments=source_segments(entry.key),
    )
    stream.submit(job)
    jobs_by_generation[(entry.slot.index, generation)] = job
    jobs.append(job)

# Concurrently: poll completions and launch expert kernels for ready slots.
for completion in stream.poll():
    job = jobs_by_generation.pop(
        (completion.ticket.slot_index, completion.ticket.generation), None)
    if job is None:
        continue                              # duplicate/stale completion
    slot = cache.slot_for(completion.ticket.key)
    if (slot is None or slot.index != completion.ticket.slot_index
            or slot.loads != completion.ticket.generation):
        continue                              # stale; never touch a new occupant

    if completion.status == COPY_OK:
        cache.mark_ready(completion.ticket.key)
        cache.begin_use(completion.ticket.key)
        launch_expert(completion.ticket.key)
        cache.end_use(completion.ticket.key)
    else:
        failures[completion.ticket.request] = completion.error

# Wait for all expert kernels; combine in the backend's reference order;
# then release every plan.
```

The exact kernel/combine order is backend-specific and is not part of this seam. For Gemma 4 it is
ascending expert-id order; see [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md).

### One plan, one owner

The scheduler is the only thread that calls the cache. The copy worker may run concurrently, but it
only fills `destination` buffers and posts completions. This avoids locks around `ExpertCache`, whose
docstring already says it is not thread-safe.

For batch size 1, one layer plan is active at a time. The plan is released only after all eight expert
kernels and the combine are done. A future batch or prefetch path must preserve the same single
scheduler-thread rule; it must not let a copy completion call `mark_ready` directly.

## 5. Lifetime and generation

A slot's `loads` counter increments on every admission. A ticket carries the value observed after
`plan()` assigned the slot. The backend must validate all four parts before acting:

```text
cache.slot_for(ticket.key) is not None
slot.index == ticket.slot_index
slot.loads == ticket.generation
slot.state == LOADING
```

If any check fails, the completion is stale and is ignored. In particular, the backend must never
call `fail()` on a key that now refers to a different generation.

A live job keeps its slot `LOADING` and pinned. The cache never evicts it. The scheduler must not
release the plan before the job is terminal unless it deliberately cancels the request; even then,
the slot remains `LOADING` until `mark_ready()` or `fail()` is called after the event is accounted
for.

The slot buffer is not reused until the cache has discarded the slot. The copy stream must therefore
not write to a destination after reporting the terminal completion for that ticket. A cancellation
that cannot stop an in-flight copy still ends with the scheduler draining the event and discarding
the slot.

## 6. Failure and cancellation

| event | copy stream | scheduler/cache | arithmetic |
|---|---|---|---|
| queue full | raise `CopyQueueFull`; no cache call | keep plan pinned; poll and retry; timeout aborts | none |
| stage/source failure | terminal `COPY_FAILED` | stop new jobs; release plan; drain; call `fail()` for the affected LOADING slot | no expert, no combine |
| H2D launch/event failure | terminal `COPY_FAILED` | same; never call `mark_ready()` | no expert, no combine |
| partial segment arrival | no success; discard the whole slot | release then `fail()`; never expose a partial slot | no expert, no combine |
| cancellation | best-effort `cancel(request)`; later terminal result | release, drain, account for every ticket; no prompt-cache commit | no token |
| stale completion | terminal completion for old ticket | ignore by `(key, index, generation)`; never fail a new occupant | none |
| copy worker dies | no more completions | abort request, stop submissions, drain/reset; fail assigned LOADING slots after release | request fails |
| `ExpertCacheFull` | never called | no job is submitted; plan is unchanged | request fails before copy |

Failure is a hard request failure. The copy lane has no CPU-compute fallback and no "use the old
expert in that slot" behavior. A failed copy may discard a newly assigned slot, but it does not
choose a different victim or change the cache's LRU/transient policy.

### Ordering on failure

1. Stop submitting jobs for the request.
2. Call `stream.cancel(request)` if the stream is still healthy.
3. Release the plan; this removes pins but leaves `LOADING` slots unavailable.
4. Drain/poll until every ticket for the plan is terminal (or reset the device if it cannot be
   drained).
5. For each current `LOADING` ticket with no pins, call `cache.fail(key)`.
6. Do not launch an expert kernel, combine or emit a partial token.

A coalesced slot may have more than one waiting plan. All waiters must be released before its final
`fail()`; the backend must not discard a slot still pinned by another request.

## 7. Invariants of the seam

1. The cache is the only component that selects or frees a slot.
2. No copy job exists for a `hit=True` entry, and at most one job exists for a given
   `(slot_index, generation)`.
3. A job destination is exactly the slot assigned by `plan()`; the stream never searches for another
   slot.
4. Every `LOADING` slot with an active job stays pinned until the plan is released.
5. `mark_ready()` is called only after a successful terminal completion whose generation still
   matches.
6. `fail()` is called only after release and only when no pins remain; a stale completion is ignored.
7. `release()` does not make a `LOADING` slot reusable; only `mark_ready()` or `fail()` does.
8. A copy-stream failure does not change victim selection, slot count, transient count or LRU order
   except through the cache's normal admission/discard transitions.
9. The scheduler, not the copy worker, calls every cache method.
10. All jobs for a request are terminal before the prompt checkpoint is committed or a saved opening
    is written.

## 8. Contract tests

The interface can be tested with a fake copy stream before any CUDA code exists:

1. **Policy equivalence:** run the same seeded plan/release trace once with direct `mark_ready()` and
   once through the fake stream; assert identical slot assignments, evictions, transient admissions
   and metrics.
2. **Coalescing:** two plans select the same `LOADING` key; exactly one job is submitted and both
   waiters see one completion.
3. **Stale ticket:** complete a job after its slot was failed/reassigned; the new occupant is
   untouched and no cache state changes.
4. **Queue full:** fill the fake queue; assert the plan remains pinned, no extra slot is selected,
   and a retry after `poll()` progresses without changing eviction order.
5. **Copy failure:** fail one segment or the whole job; assert release, `fail()`, no expert launch,
   no combine, and no prompt checkpoint.
6. **Cancellation:** cancel with a job in flight; assert the slot stays `LOADING` until the terminal
   result and is then failed, not reused.
7. **Forced eviction:** run the seven-eviction-plus-transient trace through the fake stream; assert
   the pinned slot is untouched and the copy stream never receives an eviction instruction.
8. **Worker death:** stop the fake worker; assert the scheduler aborts and does not reuse any
   assigned `LOADING` slot.

These are contract tests for the seam. They are not a copy-stream implementation and do not replace
the Gemma 4 G2 exactness traces.

## 9. Non-claims and open items

This design does **not**:

- add a copy stream, CUDA stream/event code, GPU allocation or a MoE backend;
- change `kindi/expert_cache.py`, its `slot_bytes` accounting or its eviction policy;
- add an eviction/reservation API to the cache;
- claim a copy path, throughput, hit rate or peak-VRAM result;
- make `ExpertCache` thread-safe.

Before implementation, G0 must pin the exact source ranges, the device slot arena layout and the
reference's failure semantics. The generic seam above intentionally leaves those model-specific
details to the backend and to [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md).

Related records: [MOE-TANDEM.md](MOE-TANDEM.md), [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md),
[GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md),
[RAON-1-MINI.md](RAON-1-MINI.md), [ROADMAP.md](ROADMAP.md).
