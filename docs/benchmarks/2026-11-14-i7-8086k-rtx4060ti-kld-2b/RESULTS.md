# kld quality gate, first numbers: 2B (2026-11-14)

Host: i7-8086K + RTX 4060 Ti (8,188 MiB), tensor-core plans (`KINDI_TC=1`). Pack: 2B
(`~/.kindi/packs/bonsai-2b`). One `kindi kld` run (roadmap gate 3): teacher-forced against the
fp16 plain-engine reference, warmup 64, exact head, fixed repeated-sentence ids
(`kindi/kld.py`, `--json kld-2b.json`). The reference pass is rerun per variant; each row is one
lockstep pass of reference + variant at that context.

| ctx | variant | KLD (nats) | same-top | PPL ratio |
|---|---|---|---|---|
| 2K | plain-q8 | 2.74e-07 | 100% | 1.0000024 |
| 2K | tp-q8 | 2.74e-07 | 100% | 1.0000024 |
| 2K | tp-k8v4 | 1.29e-04 | 100% | 1.0009 |
| 8K | plain-q8 | 2.07e-07 | 100% | 1.0000097 |
| 8K | tp-q8 | 2.08e-07 | 100% | 1.0000087 |
| 8K | tp-k8v4 | 2.27e-04 | 100% | 1.0011 |

Steps: 1,983 (2K), 8,127 (8K). PPL of the reference itself ~1.01 (the eval text is a repeated
sentence, so this gate ranks formats against each other, not model quality).

Reading:

- **q8_0 is distribution-lossless for practical purposes**: KLD ~2e-7 nats (three orders below any
  greedy-flip risk), top token never flips in 10,110 scored positions, PPL cost under 0.0015%.
  Plain and plan engines measure the same q8_0 drift, so the gate is not picking up engine noise.
- **k8v4's drift is real and context-growing**: ~470x q8_0's KLD at 2K, ~1,090x at 8K, PPL cost
  +0.09%/+0.11%. Its top token never flips teacher-forced, but `kindi verify` still showed greedy
  output differing on 2 of 4 prompts - teacher-forced share does not capture 40-token compounding.

Decision re-derived from the measurement: **the k8v4 quarantine stands** - not because of the old
"2 of 4 prompts differ" count, but because its drift is 2-3 orders of magnitude above q8_0's and
grows with context (1.75x from 2K to 8K), while q8_0 stays at ~2e-7 nats flat. It stays an
opt-in experimental flag (`KINDI_KV_EXPERIMENTAL=1`), and any future lossy cache (e.g. FP8,
roadmap gate 8) must beat the q8_0 row here, not just fit a bar.
