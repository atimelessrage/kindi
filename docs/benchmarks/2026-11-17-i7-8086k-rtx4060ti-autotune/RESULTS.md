# 2B autotune on the i7-8086K + RTX 4060 Ti

Measured 2026-11-17 with `--ctx 8192` on the Bonsai 2B pack, one card.

`kindi autotune` proposed `gpu0` because one card holds the whole model. `tandem-auto` was refused
by the planner (the model already fits at 8,192 tokens; a split would only make it slower), and
`tandem-gdn:8` failed because no compiler was on `PATH`. Both failures are visible and unscored,
not zeros in the table.

| config | decode | read | peak | outcome |
|---|---|---:|---:|---|
| `gpu0` | 97.4 tok/s | 111.7 tok/s | 3,050 MiB | winner |

Raw report: [autotune-2b.json](autotune-2b.json). The repeat reports are
[autotune-2b-r2.json](autotune-2b-r2.json) (97.6 decode / 111.7 read) and
[autotune-2b-r3.json](autotune-2b-r3.json) (97.4 decode / 113.0 read). Those are the endpoints
behind the page's `97.4–97.6` decode and `111.7–113.0` read ranges; there is no 97.8 artifact, so
that former endpoint is not retained. The reported peak comes from the selected run, which used
`--sample-vram`; the repeats did not. No exactness claim is made here: `kindi verify` owns that.
