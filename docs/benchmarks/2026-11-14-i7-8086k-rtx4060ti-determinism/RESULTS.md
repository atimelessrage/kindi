# Determinism gate, first run (2026-11-14, roadmap item 4)

Host: i7-8086K + RTX 4060 Ti (8,188 MiB), `KINDI_TC=1`, exact head, speculation off, greedy,
one prompt ("Explain photosynthesis in two sentences.", `max_tokens: 120`), 10 requests against
one server process per model; output compared by sha256.

| model | unique outputs | hash |
|---|---|---|
| 2B (`~/.kindi/packs/bonsai-2b`) | 1 of 10 | `180e7392dce5` |
| 9B (`bonsai-9b`) | 1 of 10 | `d5b027f65ade` |

**Gate met: 10/10 identical on both.** This replaces the Strata-class vendor-matmul suspicion with a
run the project performs: same prompt, same tokens, every run, on the only two cards' models
measured here (2B/9B). The historical 9B int8-activation flip is not reproduced under default
settings on this host.

## Settings that legitimately break determinism (the other half of the gate)

These are documented features or configurations whose output is *expected* to differ run-to-run or
setting-to-setting; they are not bugs:

- **Lossy K/V caches** — `KINDI_KV=k8v4` (lossy 4-bit V; `kindi kld` measures its drift), and any
  non-default cache format vs the fp16 reference.
- **Dynamic offload** — tandem placement changes *when* layers are recomputed/reloaded (a cache
  eviction or a different CPU-layer budget uses a different order); the default static placements
  are exact (verify 4/4), while `auto` re-derives the placement from runtime free VRAM.
- **Speculation** — drafted/verified and MTP paths change token boundaries in principle; on this
  project they are exactness-checked (output byte-identical in every speculation measurement), but
  a draft head that degrades to off changes the timing, and the TC_HP=T=4 cap changes draft length.
- **Sampling** — any temperature > 0 / random sampling is non-deterministic by design.
- **Chunked/pipelined kernels with nondeterministic reductions** — if a future kernel introduces a
  vendor reduction whose order varies per launch, this gate is what catches it (that is what it is
  for). A failure here, under default settings, is a bug until reclassified into the list above.
