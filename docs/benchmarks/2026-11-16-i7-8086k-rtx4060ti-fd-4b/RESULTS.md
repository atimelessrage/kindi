# DEC-0003 re-measurement: KINDI_FD=1 on the 4B (RTX 4060 Ti, 2026-11-16)

DEC-0003 (2026-11-11) guards the 4B's 2560-wide tensor-core plan against
`KINDI_FD` split flash-decode, because FD-on diverged at prompt 2 token 6 and
prompt 3 token 37. This round re-measured on the current tree to decide whether
the guard can be lifted. Reproduction: run the Bonsai verify with the
`fd_supported` guard monkeypatched to always-True (a one-off harness, since
removed) under `KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1 KINDI_TC_TMAX=4`, plan
engine, ctx 1024, exact head, against the shipped `kindi/reference-4b.json`:

| prompt | result | matched | first diff |
|---|---|---|---|
| 1 'The sky is blue because' | PASS | 40/40 | — |
| 2 'The capital of France is' | FAIL | 6/40 | token 6, llama.cpp top-2 gap 0.0447 (not a tie) |
| 3 'Water boils at a lower temperature on mountains because' | FAIL | 37/40 | token 37, no gap recorded |
| 4 'def add_two_numbers(' | PASS | 40/40 | — |

2 exact, 0 ties, 2 fails — the same divergence points as the 2026-11-11
measurement. Verdict: **the guard stays.** FD-on 4B is not exact, so no speed
gain is recorded (the FD-on run was not benched: a gain on a diverging path is
not a shippable gain).
