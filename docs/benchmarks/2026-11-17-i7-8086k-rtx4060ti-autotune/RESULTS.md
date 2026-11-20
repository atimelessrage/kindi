# 2B autotune on the i7-8086K + RTX 4060 Ti

Measured 2026-11-17 with `--ctx 8192` on the Bonsai 2B pack, one card.

`kindi autotune` proposed `gpu0` because one card holds the whole model. `tandem-auto` was refused
by the planner (the model already fits at 8,192 tokens; a split would only make it slower), and
`tandem-gdn:8` failed because no compiler was on `PATH`. Both failures are visible and unscored,
not zeros in the table.

| config | decode | read | peak | outcome |
|---|---|---:|---:|---|
| `gpu0` | 97.4 tok/s | 111.7 tok/s | 3,050 MiB | winner |

Raw report: [autotune-2b.json](autotune-2b.json). The page range `97.4–97.8 tok/s` in
`docs/AUTOTUNE.md` also includes the adjacent 97.6 tok/s run; this tracked report records the
final selected run. No exactness claim is made here: `kindi verify` owns that.
