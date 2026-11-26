# AGENTS.md - working on Kindi

Kindi is a custom inference engine for ternary Bonsai models (plus Gemma 4 and Nanbeige), with a
CUDA/CuPy backend, a CPU backend, an OpenAI-compatible server and a one-command installer.
`docs/AGENTS.md` is the guide for agents that *use* the API; this file is for agents that
*change the repository*.

Keep this file current: it is the handoff document. Tick a step when it is done and add one
line of evidence (command + result). A task must be resumable from this file alone.

## Current state (live — read this first)

### Handoff

- **HEAD:** `f9182bb` on `main` (2026-11-26: **the Arc packed GEMV's read pattern was not the limiter.** The 26.87 ms/token arm left 253 GB/s against a 385 GB/s streaming read, and the sweep's verdict named the access pattern as the next lever. A 256-row-blocked layout (`[tile][phase][slot][row]`, so a wave's 32 slot loads per phase are one contiguous 8 KB run instead of 32 loads a `rows`-byte stride apart) was built and A/B'd over the 401-matrix 27B list with distinct bytes in two 3.76 GiB chunks: 27.44 -> 26.31 ms/token, 247.8 -> 258.5 GB/s, faster in 13 of 14 paired samples, for a one-time ~75 ms repack pass. The attribution experiment is the finding: with the decode deleted and the same addresses, the *shipped* layout reads at 405.0 GB/s and the blocked one at 376.7 GB/s, so the read pattern was already saturated and the blocked arm's 4% is cheaper address arithmetic (a shift instead of a `rows` multiply in a 64-bit address chain). The remaining ~8 ms of the token is `TRI_DOT`, and 16.8 ms/token is the measured ceiling if the decode were free, so the next arm cuts the decode, not the layout; sixteen rows per work-item is worse (33.6 ms, register pressure). Evidence `docs/benchmarks/2026-11-26-i7-8086k-arc-a770-layout/`, claim `arc-a770-layout-20261126`, `docs/INTEL-ARC.md` corrected and the overhead page now points forward at the follow-up; the harness stays untracked in `bench-logs/`. 2026-11-26: **`main` re-checked as the latest**, by time and by content: no branch or remote ref in the repo carries a commit dated after its tip - 42 local branches hold commits the tip cannot reach, and every one of them is older, the newest unmerged commit anywhere being the open PR #85 delta - `myfork/main` and both worktrees are ancestors of the tip (the worktree is 34 commits behind), `origin/main`'s three v0.7.2 commits are still ported-by-content only, and the full suite is green at the tip, so what sits outside `main` is the open PR branches and the archival snapshots and nothing else. Before that: 2026-11-26: **the Arc kernel work list is written down**, `docs/INTEL-ARC-KERNELS.md` - the per-token op graph read out of `kcpu.c` at the 27B's own shapes, the 12 kernels that do not exist yet grouped by dependency, and a four-phase cheapest-first order (norm/fold + embed + GEMV wiring first, GDN next, attention last) plus the three corrections that reshape it: no f32 head to build, the one-launch work list is not a runnable decode, and sampling needs no kernel. Its estimates are labelled and its measured figures come from the linked pages; `INTEL-ARC.md` links to it. Before that: 2026-11-26: **the 2.5 27B's CPU load is 5.5x faster, byte for byte.** The one working 2.5 path here is the CPU engine, and its first cost was 339.9 s of load; cProfile put 96% of it in `kindi.cpu.paircodes`, which derived four base-3 trits per weight byte with `% 3` / `// 3` and gathered a uint32 code per column pair. It now decodes the whole byte through one 256-entry `uint16` table (`_PAIR_LUT`) and assembles each word from four bytes - a gather, a shift-or and a byte transpose - measured side by side in one process against the frozen old decoder on the packs that fit here: 21.13 -> 3.84 s (2B, 5.50x), 47.15 -> 8.16 s (4B, 5.78x), 109.97 -> 19.91 s (9B, 5.52x), 296-366 MB/s of pack; the 27B's 339.9 s predates this and its pack is not on this host, so it is a projection (20-25 s of decode work against the 326 s it spent) and docs/CPU-SUPPORT.md now says which figures are measured and which is projected. Weights are unchanged: `tests/test_cpu_paircodes.py` compares new against old over the real code range and all 256 byte values, six shapes, both chunk paths and a memmap slice, and `KINDI_DEVICE=cpu kindi verify` is still ALL PASS token-exact with llama.cpp on the 2B (40/40 x3, 13/13) and the 9B (same); evidence `docs/benchmarks/2026-11-26-i7-8086k-cpu-pack-load/`, tool `tools/cpu_pack_load_check.py`. Before that: 2026-11-26: the batched pass is covered in CI without the pack download.** `tests/test_cpu_batch.py` only runs with `KINDI_CPU_PACK` set to a multi-gigabyte pack, so nothing in CI executed `kc_step_batch`, `kc_batch_rollback` or the state block - the two places that regressed already. `tests/test_cpu_batch_synth.py` writes a two-layer pack from a seed inside the test (`_write_pack`: 128 wide, FFN 256, 8 GDN gates, hadamard block 128, 256-token head, ~300 KiB in 0.5 s, no GGUF and no converter) and pins the contract: the pack loads and plain decode runs on it, `step_batch` returns one in-range token per row with a head-width logits view and a width-wide hidden view, rows past the context are refused while one row is plain decode, `batch_rollback(0)` restores the pre-batch state bit for bit and `batch_rollback(n)` changes nothing, and `save_state`/`load_state` round-trips the state block a drafter rewinds with; 6 tests, 3 s, skipped where no C compiler exists. Bit-exactness stays with `tests/test_cpu_batch.py` on purpose: on every hand-built pack tried (128 to 2048 wide, attention-only and GDN, one to four layers, fresh and fed state) the batch's GDN block disagrees with plain decode - `batch_hidden(0)` can read 1e28 where plain decode is finite, and the state block gains NaN blocks depending on what ran before it in the process - while the real 2B and 27B packs stay byte-exact, so either the batch reads a buffer it never wrote for some pack layouts or the two paths differ by weight-magnitude sensitivity; unattributed, and the new file's docstring records it rather than asserting it. Before that: 2026-11-26: **the batch arena's reuse key now covers the whole geometry it sizes from.** The previous commit keyed reuse on the `Model*` plus `NL`/`D`/`ctx`; that still let a freed Engine's recycled heap address match a *different* model, so `batch_prepare` now compares every field it allocates from (`NL`, `D`, `NFF`, `CONV_CH`, the widest GDN gate row, `NH`/`NHKV`/`HD`/`NK`/`HK`/`NV`/`HB`, `ctx`, the head rows), the derived gate width moved into `batch_gate_rows()` so the key and the sizing cannot drift, and `kc_batch_prepare`/`kc_batch_reuse_ok` are exported as probes. Pinned by `tests/test_cpu_batch_arena.py`: two synthetic Models (one more GDN layer, a wider SSM, a bigger vocabulary, a wider head - each mutation reproduced the hazard red before the key was widened), the same arena reused for its own model, more rows than allocated refused, and release/re-prepare; 6 tests, no pack load. Before that: 2026-11-26: the review-audit follow-up - `kindi/kcpu.c`'s batch arena is keyed on the `Model*` plus `NL` instead of `D`+`ctx` alone (a second same-width Model in one process could reuse a SNAP table sized for the first), `kc_batch_release` + `kc_state_load_len` exist, `Engine.save_state(into=)` reuses a buffer (the drafter snapshotted 157 MB per round on the 27B) and `Engine.close()` frees the arena, roadmap CP3 names the byte-lane reopen, `kindi models` says `CPU engine (~32 GiB RAM)` for the 27B rows, and `docs/MTP.md` says the 2.5 CPU numbers are ctx 2048 only; focused gates `tests/test_cpu_batch.py` + `tests/test_catalog_scheduling.py` + `tests/test_cpu_build.py` = 50 passed. Before that: **the 2.5 27B's cardless CPU path stops pointing at a card.** `kindi plan` ended with `bench --tandem cpu-first` follow-ups that exit 1 without a CUDA/ROCm device - the exact path a 2.5 27B user without a card takes - and `kindi models` listed `bonsai-2.5-27b`/`-mtp` (and the 27B) as `GPU only` although the CPU engine is what loaded them; the plan now says the split needs a device and prints the CPU engine's own `bench`/`verify` lines with `KINDI_DEVICE=cpu` in the host shell's syntax (`set "KINDI_DEVICE=cpu" &&` on cmd.exe, the bare prefix elsewhere; a cardless `--portfolio` gets the same recommendation), the three 27B catalog rows carry `cpu`, and `docs/CPU-SUPPORT.md` names the 2.5 row (339.9 s load, 4.02 tok/s best, `verify` refusing for lack of a reference). Tests: three CPU follow-up cases in `tests/test_plan.py` (one through the real parser) and the catalog labels. Before that: **roadmap CP1 measured - the pass is exact and the gate is a miss.** `kindi/kcpu.c`'s `kc_step_batch` runs T rows through one weight stream, `kc_batch_rollback` rewinds the GDN state from the vectors it stored, and `kindi/mtp_cpu.py`'s `speculative_decode` drafts k with the 2B pack and confirms them in one pass: tokens and every row's logits are bit-identical to plain decode, the rollback lands exactly, and the four-prompt 27B gate is 4/4 token-exact - but **0.51x mean** against the >=1.25x bar (acceptance 36/128, 28.1%), because a 5-row pass costs 4.4 plain rows in isolation and 0.59 plain tokens per row in the loop: the pair-code GEMV is permute-bound (T rows need 2T `vpermps` per weight word), so break-even needs m >= 2.0 and the bar needs 2.7 of k=4 drafts. No speedup claimed, CP2 blocked, and what would reopen it is named (a ~0.2-row kernel or a bandwidth-saturated host); `tools/cpu_batch_check.py`, `tools/spec_cpu_bench.py`, `tests/test_cpu_batch.py`, [evidence](docs/benchmarks/2026-11-26-i7-8086k-27b-cpu-batch-verify/RESULTS.md), claim `mtp-cpu-batch-20261126`. Two bugs fixed on the way: the loop fed the drafter the prefix without the pending row it was about to verify (2.6% acceptance; 28.1% once included) and the incremental drafter was verified exact against the replaying one at half the cost. Before that: the 2026-11-25 27B CPU MTP evidence page now claims every number both docs quote - the plain/drafted tok/s pairs, the depth histograms and the head's per-prompt 1/54 and 6/54 - so `check_claims` re-derives 20 numbers from the tracked `results.json` instead of 6; that artifact's tok/s and seconds are stored at full precision so the docs' 2-decimal rounding is re-derivable. Before that: three CPU drafter routes compared on the same 2.5 27B weights - the 2B pack accepts 33/108 (30.6%) on both the base and the `-mtp` packs, so the `-mtp` file buys a `draft2b` deployment nothing and costs 239 MB, while the head as `Head()` loads it (raw hidden, no seed) accepts 9/108 (8.3%) at 0.10 tok/s - tracked at `docs/benchmarks/2026-11-26-i7-8086k-bonsai25-27b-cpu-drafter-routes/`, claim `mtp-cpu-routes-20261126`, and `docs/MTP.md` carries the table. Before that: the MTP variant ladder is tracked evidence - the `eh`/`he`/`eh-normed`/`eh-normed-seeded` figures (8.3% -> 10.2% -> 13.9%, `he` 0.0%) plus the whole-prompt and trace numbers were sanitized from the untracked `bench-logs/probe-out*.jsonl` into `docs/benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp-variants/` with claim `mtp-cpu-variants-20261125` (14 attempts, five loads, `trace_summary` from the seedfull trace); `docs/MTP.md` and ROADMAP CP1/CP2 cite it instead of saying untracked). Before that: `kindi verify --reference FILE` verifies a release with no bundled llama.cpp reference, which is the one gate Bonsai 2.5 27B could not pass here - the file's shape is checked before the engine loads (`verify.load_reference`), the context is sized from its longest row instead of a hard-coded 1024, the verdict prints `user-supplied`, and the no-reference refusal names the flag; exercised end to end on the 2.5 pack with a reference that pack produced, so the route and matrix are checked and exactness still is not). Before that: the MTP head's whole-prompt and grown-window candidates measured in one 27B load and the answer is still no - seeding the prompt's last row is a no-op by construction (its row is bit-for-bit the first draft row, max|du| = 0) and repeats 15/108, a window grown from 64 to 132 on a 128-token prompt repeats 4/54, and the new per-row `u`/`out` dump shows why: the head's self-chained row and the verified row at the same position have cos(u) median 0.12-0.22 and cos(out) 0.16-0.33, so `blk.64` is not reproducing the model's hidden and no seed or window choice can lift it (`kindi/mtp_cpu.py` gained `seed_last`, `grow_window`, `last_u`; `tools/mtp_cpu_probe.py` gained the `-full` variants, `window` and `--trace`; docs/MTP.md records the result; probe runs stay untracked). Before that: the Arc pack probe runs the persistent work list on a real pack - the sweep's eight-rows-per-item, one-launch kernel is ported into `kindi/intel_pack.py` beside the per-shape GEMV (`--no-persist` skips it), the upload chunk is sized from `CL_DEVICE_MAX_MEM_ALLOC_SIZE` instead of a guess (two 3.76 GiB chunks on the 27B, 1 GiB fallback; seven 1 GiB chunks measured 154 GB/s against 235 at two), and on real bytes at their real offsets six runs read 219-228 GB/s against 138-149 GB/s per shape (1.489-1.658x, 31.3-32.5 ms per pass against 47.8-51.8 ms) with 0 mismatches in both validation paths, so the sweep's 253 GB/s is a per-launch same-offset dummy figure and the honest pack number is ~1.5x; evidence `docs/benchmarks/2026-11-25-i7-8086k-arc-a770-pack/` (six tracked run JSONs), claim `arc-a770-pack-persist-20261125`, `docs/INTEL-ARC.md` corrected. Before that: `_lib()`'s cache is keyed on the library's source stamp - the filename now carries a hash of `kcpu.c`, so an edited kernel rebuilds to a new path and is actually loaded, instead of rebuilding over the same path and returning the old cached handle (on Windows the rename is also the only way the new code can take effect, since `LoadLibrary` on a mapped path returns the old module). `tests/test_cpu_build.py::test_rebuilt_library_is_picked_up` proves the rebuild is picked up. Before that: the 2.5 MTP head's acceptance ladder: the head's first row was
  being handed the raw residual where the GPU head (`kindi/mtp.py`, `XN`) and llama.cpp's Qwen3.5
  `graph_mtp` both read the output-normed row, so `kindi/mtp_cpu.py` now norms it - 8.3% -> 10.2% accepted
  (13.3% vs 3.3% at k=1) - and `Head.seed()` writes the **verified** rows into the head's K/V window, the
  way the reference's MTP cache holds every verified position: 10.2% -> 13.9%, depth [2, 3, 4, 3] on one
  prompt. `eh_proj`'s order is confirmed the other way ([hidden, emb] accepted 0/108); folding the embedding
  row and RoPE at p+1 changed nothing (11/108 each). `tools/mtp_cpu_probe.py` runs several head variants
  against one six-minute 27B load (JSONL attempts, tokens asserted equal to plain decode), which is what
  made the ladder affordable; embedding rows are now cached per token. **Still not a speedup and still
  behind the 2B drafter (30.6%), so CP1 is built against the 2B**; merged into `main`
  (unproven, no speedup claim) and `docs/MTP.md` carries the ladder. Before
  that: 2026-11-25 (CPU MTP head, drafting and scoring only): `kindi/mtp_cpu.py` loads the 2.5 `-mtp` file's blk.64 (15 tensors, Q4_0) and drafts with the main model's own folded head; `kcpu.c` gained `kc_q4gemv` (`cpu.q4_gemv`, bit-identical to the dequantised matrix in tests) and `_lib()` caches the built library per path. `cpu.py` verifies one row per pass, so a draft cannot shorten a run: `score_drafts` decodes exactly `Engine.generate` and scores the head, 7 of 108 drafts (6.5%%) with the drafted run slower (1.11 vs 1.69 tok/s) on 2026-11-25 - no speed claim, tracked at `docs/benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp/` (claim `mtp-cpu-20261125`), and road map CP1 (batched T-row verify) / CP2 (2B drafter) hold the gate; the drafter switch is one interface (`score_drafts` takes `Head` or `PackDrafter`, the tool picks with `--gguf` or `--drafter-pack` and refuses both), and the external 2B drafter measured **33/108 accepted (30.6%)** against the head's 7/108 while still running ~0.5x plain decode, so CP1 should be built against the 2B. Before that: 2026-11-25 (vision projector tier, no encoder): `kindi/mmproj.py` prices a standalone `mmproj-*.gguf` against a fail-closed ggml type table and places it by policy - `kindi mmproj PATH [--place ram|vram|auto] [--max-mb N] [--json]`, where `ram` (the default, `KINDI_MMPROJ_PLACE`/`mmproj_place`) holds the weights in one host arena and reports resident bytes while `vram` only accounts them - and the policy now decides outside that command: `kindi mmproj FILE --record PACK` writes the projector's size, sha256 and policy into the pack manifest, `kindi plan` prints a `projector:` line from it (`vram` adds the bytes to the reserve, `ram` leaves them with the card's K/V cache), and `/health`'s `projector` block is the same `mmproj.policy` dict, so plan and health cannot disagree. 13 tests in `tests/test_mmproj.py` and `tests/test_plan.py`, 157 in the plan/server/config/docs set, claims and both generated references clean; `docs/ROADMAP.md` keeps the V0-V4 ladder for the encoder itself, because no build has one and image parts still get a 400. Before that: `docs/MTP.md`, `docs/BONSAI-2-SUPPORT.md` and `docs/development/KNOWLEDGE.md` carry the 2.5 facts re-read from both GGUFs on disk (851/866 tensors, `blk.64`'s 15 tensors and 238,983,168 B, the 16/48 block split). Before that: `main` re-checked as the branch of record - `CURRENT-VS-KINDI-0.7.md` measures the fork against the newest release, v0.7.2 (`22af8d4`): 3 commits behind by history and 0 by content, 807 ahead, 1,094 files (+158,609/-2,325), 1,224 tracked files against 235; the two upstream paths with no counterpart are named (`kindi/apierr.py`, superseded by `kindi/api/errors.py`, and the root `test_inkling2_engine.py` harness, which needs an Inkling pack this host does not have). A branch-state section records the audit: `main` = `merge/upstream-0.7.0` = this worktree at one clean commit, `myfork/main` fully contained, and every `pr/*`/`wip/*` branch either an open PR's own delta or work already here by content (backend adapters, dense tandem, plan placements) - so nothing on a local branch is missing from `main` except the open PRs' diffs. Before that: the round-4 audit triage - a forward-slash drive root with a space is caught now, the three tandem-cache scripts take `--clang`/`--llvm` or `KINDI_CC`/`KINDI_LLVM`, `kindi backends` prints a `limits=` column, three Intel help strings say "no backend", and the Bonsai 2.5 download advice is explicit; the cold reader's close was already wired (finding stale), Ruff debt 207 -> 189; before that the Arc pack probe (`kindi intel-pack`): 7.14 GB -> 49.9 ms -> 20.0 tok/s at 143 GB/s on the A770, the same for the plain and `-mtp` pack, no Intel converter needed, eight gate-only rows; before that the sampling page as a settings-and-engine check, the 2.5 publisher sampling sets, and the `-mtp` intake (head recorded, not drafted); 0.7.2 port (LoRA adapters, one OpenAI error body); 0.7.0 merge `35e4697`; Arc GEMV rows4 (1.357x); the constrained mask's accepted-character skip (+41.37x); llama.cpp proxy stays engine-gateway only; the 2.5 27B CPU run (4/4 token-exact) and `chat._tojson` parity 26 of 26; GPU cells still `not run`, no exactness claimed; guarded by `tests/test_agents_handoff.py`).
  **2026-11-24:** the llama.cpp proxy backend landed (wildcard; 12 CPU tests, live gate open, no
  arbitrary-GGUF claim) with REQ-0030's native-CPU 2B profile, and `Backend.capabilities` was declared
  instead of `None` (six adapters' mypy `assignment` finding gone, 59 -> 53). Suite: 1,633 / 47 / 0.
  **2026-11-25 (round-5 audits, MiniMax round 5 + GLM-5.3-Flash):** `intel-pack` fails cleanly on a
  pack without `bcodes.bin` and reports `mean_ms` (both findings real); the 2.5 template accepts `high`
  as the `xhigh` alias and the 400 names `param: reasoning_effort` (DEC-0013 amended); the matrix says
  `PTQ1_0 only` for the 2.5 rows; the docs record once that no 2.5 7B exists. **Open:** the CPU-MTP work is
  uncommitted in this worktree (`kindi/mtp_cpu.py`, its test and tool, `kindi/cpu.py`, `kindi/kcpu.c`,
  `docs/MTP.md`, `docs/ROADMAP.md`); its measured acceptance is low, so CP1/CP2 gate any speedup claim.
  **2026-11-25: the constrained mask's walk, cut exactly - and a memo withdrawn.** The walk's
  cost is the automaton, not the trie (8,303 `_step` calls per schema miss, 8,219 of them one per
  distinct token-start character, 91% rejected). An interim per-class edge memo measured 1.486x and
  was **withdrawn the same day**: cached transitions embed the schema stack, so warm tables returned
  masks the 248,320-entry sweep disagreed with. `JsonGrammar._accepted_chars` now returns a superset
  of the characters a state accepts (None for `str`/`akey`, already safe-set pruned) and
  `TokenClassTable` looks up only those per node: median miss 4.923 -> 0.119 ms (41.37x), steps
  8,303 -> 38, the document's 23 first-visit masks 121.1 -> 24.9 ms, every mask equal to the sweep.
  Served 23.11 -> 20.82 ms/token (9.9%), constrained gap 3.83 -> 1.65 (56.9%), control flat. A6's
  mask and end-to-end bars are now met in Python, so no native variant was started. Tests: superset
  check, hook-on/off mask parity, and a warm-table sweep guard that fails on the memo commit. Run page
  `docs/benchmarks/2026-11-25-i7-8086k-cpu-key-chars/`, claim `req0030-key-chars-20261125.json`; the
  memo page stays retracted. (2026-11-24's mask/pick -8.4% stands; no GPU run; suite below.)
### Python quality ratchet

Audit F9's Python quality ratchet is `tools/check_python_quality.py`: Ruff 0.16.0 (`F`, `E9`, `B`, `I`)
over the Python tree, mypy 2.4.0 over first-party production
code, Pyright 1.1.414 (`basic`, all platforms) and Pylint 4.1.3 (`E`, `F`, import-error
disabled) at Python 3.10 over the same production targets, recorded as **189 Ruff findings / 53 mypy
errors / 829 Pyright errors / 245 Pylint findings** in
`tools/python_quality_baseline.json`. CI runs it; a new or increased file/rule count fails, while
existing debt is tolerated and `--update --reason "..."` lowers the baseline after review; it refuses growth and records the reason and delta, so it cannot launder a new finding. Pyright and Pylint skip `VENDORED_UNWIRED` (qwen4exp/qx) and `UPSTREAM_UNVERIFIED` (`kindi/inkling2*`, merged but never run here); `kindi/server.py` stays in scope.
`tests/test_python_quality.py` + `tests/test_ci_config.py` **15 passed**; no GPU run. Dependency scanning is report-only in CI (`tools/check_dependencies.py`, `requirements-runtime.txt`); the gate waits for a reviewed baseline. The prior test task is below.

### Full CPU gate

The 2026-11-25 proxy-boundary gate is **1,658 passed, 47 skipped, 0 failed** in 247.63 s; the trie-walk gate is 1,644 / 47 / 0 in 247.32 s, the Bonsai 2.5 gate 1,640 / 47 / 0 in 242.39 s (the 2026-11-24
mask/pick gate was 1,635 / 47 / 0 in 238.23 s; 1,633 before that). The 27B-CPU gate at `32075c4` is **1,598 passed, 47 skipped, 0 failed** in 226.48 s (the CPU build checks now compile and load every tier here instead of skipping); the 2026-11-24 grammar-mask gate at `ebf871f` is 1,513 passed, 58 skipped, 0 failed in 180.66 s (plain CPU suite, no coverage rerun); the 2026-11-22 prewarm gate at `4bab483` was 1,423 / 58 / 0 in 138.50 s and the 2026-11-20 integration gate at `82a72bf` 1,412 / 28 / 0 in 250.33 s. The last coverage-measured run remains `539a504`: **1,241 passed, 28 skipped, 0 failed at 46.36%** line+branch (23,504 statements, 8,394 branches; plain 1,241 / 28 / 0 in 227.62 s), and the consolidated 2026-11-19 commits (`f75253b` + `84b6686`) 1,357 / 28 / 0 in 229.53 s. Every skip is inventoried by node ID and reason in `tests/README.md`.

### Upstream v0.7.0 and comparison

**2026-11-23:** upstream `v0.7.0` (`8298a0f`) is merged into this branch; the current comparison is
`CURRENT-VS-KINDI-0.7.md` (refreshed 2026-11-24 vs `v0.7.1`: 0 behind, 778 ahead, 1,027 files, 1,159 tracked vs 232;
PRs and the no-PR gap in §3a). PRs rechecked then: **33 fork PRs (#47-#79) open, 0 merged, 7 drafts**, external #37 open and dirty; #48, #49, #72 and #76 are in v0.7.0 by
content and can be closed as included. The paragraphs below are the 0.6.1-era record that produced this tree.

Stock moved to upstream `v0.6.1` (`7fececb`, fetched 2026-11-17) and one of its fixes was a bug this fork had: `kindi/mtp.py` never passed `k_rmsfold_qT`'s
`hp`, so the MTP head's CUDA graph did not build and mode A speculation was dead on the tensor-core layout (fixed in `f934a55`, with
`tests/test_kernel_arity.py`); measured after the fix, mode A at the old default `KINDI_SPEC_K=4` costs 0.66x on the 2B; at
`KINDI_SPEC_K=1` it is 1.12-1.13x at 512 and 24K tokens of context and a wash at 8K (4-run medians, 2026-11-17, `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-mtp-context/RESULTS.md`), so mode A now
defaults to k=1 (`docs/MTP.md`), and `CURRENT-VS-KINDI-0.7.md` records the current official v0.6.1 comparison (664-commit / 827-file
delta at `53198e5`; the 2026-11-19 round adds request-shape validation, Windows secret-scan batching, MTP k=1, the Q2 expert lane, the controlled pairs and the chunk sweep; upstream `main` now points at the release commit `7fececb` (no post-release commit to port); round history in
`detailed-fork-changelog.md`); §4.6 closes the same-settings A/B with stock `v0.6.1` (tag source, loader shim) and this
tree at `a79137f` on the same 2B/9B packs at ctx 8192, plain engine and exact head, five passes each: every decode median is within 2.6% and the ranges overlap,
while the 9B prompt-reading gap (83.9 vs 72.4) was rechecked at 20 passes and closed as noise (70.7 vs 75.7; paired -1.4, p=0.648); a separate stock harness
adds literal position-512 medians (190.3 on the 2B, 84.5 on the 9B) outside the bench sweep, after a 2026-11-18 paired run showed its position-100 anchor
matching the immediately preceding bench (+1.2%/0.0%) and replaced the earlier cold-clock-heavy 171.2/80.7 values, and the four sweep cells now carry three-pass
medians with ranges (the pages note that the 9B prompt-reading ranges overlap, so no reading shape is claimed). §4 is re-based at `0184bb8` into separate worse
/ missing / unverified tables with a closure condition per row; it distinguishes the 16-file `v0.6.0..v0.6.1` range from the 15-file 0.6.1 commit (the compose
surface is present from `dce89df`) and records the MoE backend's status (exercised on GPU through Lightning). Upstream `v0.6.0` (`179710c`) + `origin/main` head
(`dce89df`, v0.6.1's parent) both contained, while `v0.6.1` itself is one commit past `dce89df`; vs `v0.6.0` the tree is a strict superset; comparison report in
`CURRENT-VS-KINDI-0.7.md`.

Dated notes (MTP smoke guard, bench split, audit follow-up, commit-history redaction, Ollama links) moved to
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) on 2026-11-23.

### Engine-liveness roadmap

The general lessons from that bug are on the roadmap, not started: `docs/ROADMAP.md` "Engine-liveness follow-ups (2026-11-17)" ranks eleven follow-ups and takes
four with their gates first — a launch-shape contract for every kernel call site (REQ-0023 — A1+A3 done 2026-11-17: 105 launches measured across
`spec.py`/`dspine.py`/`mtp.py`/`tp.py`, 37 unresolvable sites pinned with reasons, every measured site proven by mutation; the dspine `kf`/`self.k`/`self.k8`
tables are resolved from their literals and only two dspine sites remain pinned; the grid/block half stays open), report the effective speculation state plus a
`mtp_heads.json` digest check (REQ-0024 - A1+A2+A3+A4 done 2026-11-17: one mode resolver behind `/health`, the startup log and `kindi info`, the head's sha256
and size checked before `np.load`, and `/metrics`' `kindi_spec_mode_effective` naming the effective mode), `kindi verify --spec` as an exactness gate (REQ-0025
- A1/A2/A4 done and A3 four of five, 2026-11-17: the mode is printed first, a missing drafter exits 2, and the 2B is token-exact 4/4 in plain, `ngram`, `mtp`,
`draft2b` and `both`; `dspine` waits on weights, `docs/MANUAL-HELP-NEEDED.md` row 7) and `kindi bench --spec` (REQ-0026; REQ-0025 A6 now also runs the four
modes as `tc-<spec>` cells in `tools/verify_matrix.py`, scoring requested/effective and drafts and skipping absent files so a baseline gates them); the seven
parked ones name the measurement that would reopen them.

### I/O and cold tier

`kindi io` now measures this host's memory tiers (RAM copy, unbuffered disk read, pinned PCIe both ways) so a cache or streaming budget is an input, not an
assumption; the mixed-precision tiers stay parked, while the CPU-compute lane has an exact expert-level gate for the Lightning Q2_0 format and is still off by default (the copy lane won on this host, and the model row needs the absent pack) in `docs/ROADMAP.md` "Memory-tier follow-ups (2026-11-17)"; I/O-aware slot sizing now
consumes those rates in `kindi plan`/`kindi autotune`; the plan's follow-up bench carries the planned context and count with the global `--ctx` and portable
`--dcache` flags and a test pins both, the printed `bench` and `verify` commands quote spaced interpreter and pack paths, `kindi autotune` adds an `expert-cache-<MiB>` cold-tier candidate that sets `KINDI_LT_COLD=1` + `KINDI_LT_HOST_MB` from
the same rate-sized working set and caps it by the host experts, and `kindi io --json FILE` reports can be reused by either command through
`--io-report`/`KINDI_IO_REPORT` (a missing or unusable report names itself and falls back to one probe), with the arithmetic unit-tested and its Lightning-pack
comparison still open; the NVMe cold-expert tier is **started 2026-11-17** behind `KINDI_LT_COLD=1` (`kindi/cold_experts.py`: the hottest host experts stay
pinned under `KINDI_LT_HOST_MB` (default available RAM less 10%/1 GiB), only the overflow is file-backed, unbuffered aligned reads, coalesced gaps, a bounded
lookahead prefetch queue (a daemon thread fills it while the current layer's kernels run; the next cold layer joins before its demand read), Q2_0 de-interleave,
requested/transferred and exact pinned/disk byte accounting; prompts stage the host block with each pinned expert at its host-tier slot, `cold_stats()` counts
each pinned expert once while its byte totals stay per matrix, decode keeps a per-layer pinned slot cache (`cold_slot_plan`: a routed expert's slot is never
evicted and a repeated expert takes no read; `cold_stats()` reports `slot_hits`), the decode graph is disabled on this opt-in path, and the per-layer stream
sync plus blocking current-layer read stay open (C9; no tok/s claimed), unit-tested only with **no larger-than-RAM pack, verify, GPU run or tok/s claimed**, and
the Strata/KTransformers/BigMoeOnEdge triage is in `docs/EXTERNAL-ENGINES.md`.

### Suite and recent additions

The coverage-measured full suite at `539a504` is **1,241 passed, 28 skipped, 0 failed** in 277.17 s at
**46.36%** (23,504 statements, 8,394 branches; the plain run is 1,241 / 28 / 0 in 227.62 s). The per-round
list of new tests (kernel-arity resolvers, CLI comma values, saved I/O reports, autotune/cold-expert checks,
effective-mode metrics, speculation cells, I/O probes, MTP registry, secret-history guards) moved verbatim to
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) on 2026-11-24.

### Secret guards

The secret guards are now three: entropy baseline over tracked files, pattern scan over tracked+untracked files, and a history gate against
`tools/secret_history_baseline.json` (163 rows / 3,319 legacy hits from an old benchmark dump, old AGENTS/log revisions, the 27B median transcripts and one commit message that names a
cloud storage folder, recorded not endorsed; the gate scans commit messages - subject and body - with the same rules as file content, so a path in a message is
caught instead of invisible).

### Plan slot arithmetic

Thirteen tests pinning `kindi plan`'s decode-slot arithmetic (the fit subtracts the slots before dividing over `n_moe - 2` layers, so 10 slots cost 20 experts
per layer on the 256-expert test pack; the line used to claim one per slot, `kindi/ltfit.py` is now the one implementation both `plan._lt_fit` and
`lightning._plan` call, and the weighted branch prices the slots against the same room with `dcache=0` - 9 of 128 card experts and 5.8 points of served share on
the test pack's tight card - in the line and in `slots_cost`).

### Coverage gate

The coverage gate `python tools/check_coverage.py` passes on 46 modules from a measured run (`coverage run --branch --source=kindi -m pytest` -> `coverage
json`, since `pytest-cov` hangs on this host; the 2026-11-18 historical run 1,196 passed / 28 skipped / 45.4% total is superseded by the final 539a504
run: **1,241 passed, 28 skipped, 0 failed at 46.36%**, 23,504 statements and 8,394 branches), and `tests/test_no_secrets.py` now scans untracked files too (pinned by
`test_untracked_files_are_scanned`), with `project_memory --check`, `config_reference --check` and `support_matrix --check` all clean.
The per-module rows in `tests/README.md` were checked against the final run's `.coverage` data; their rounded figures are unchanged, and the total and
per-module floors are untouched. Its tracked revision, command and counts are in
[docs/benchmarks/2026-11-18-i7-8086k-windows-cpu/full-suite.json](docs/benchmarks/2026-11-18-i7-8086k-windows-cpu/full-suite.json).

### Grammar and claims

[REQ-0027](docs/development/requirements/REQ-0027.md) passed its 10x gate on 2026-11-22 (token-class table + startup prewarm) and again on 2026-11-24 (edge-derived
safe sets, mask unions); its A4/A5 GPU rows stay open. The CPU set-equality guard against the pre-token-class `vocab_allowed` sweep is what keeps the accepted
language fixed. GPU grammar/JSON-schema gate closed on the 2B (§39); recursive GBNF is opt-in (`KINDI_GBNF_MAX_DEPTH`, bounded unroll cross-checked against an
independent matcher - see the newest rows in `docs/development/AGENTS-HISTORY.md`) and measured as a diagnostic only: it once cost 2,720 ms/token against 231 for
`json_schema` and *fell* to 463 at 200 tokens (cache hits; a whitespace-permitting constraint lets the greedy 2B spend the budget on 9 characters) (`docs/API.md`,
`docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-grammar-cost/RESULTS.md`). The harnesses are tracked as `tools/grammar_cost.py` and
`tools/grammar_cost_serve.py`; every quoted number is re-derived, not prose:
`python tools/check_claims.py` re-derives every one of them from the tracked artifact of its run (49 gates, 416 numbers across `docs/claims/*.json`, six
claimed docs and zero citations listed as debt), and refuses a benchmark citation no claim accounts for - schema and procedure in `docs/claims/README.md`.

### Dated evidence

The 27B A/B, fail-closed gate, startup-warning, MTP-server and cold-expert bullets
(`Latest` through `Older still`) were moved verbatim to
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) under
"Current-state bullets moved from AGENTS.md (2026-11-18)".

- **Shipped on main:** dense tandem runtime (2B/4B/9B/27B rate, peak and exactness at
  512/4K/8K; 2B CPU-layer sweep; joint checkpoint 149.6 MiB on the 27B at 54.5-61.5 ms
  save), owned-slice CPU load, `kindi bench --sample-vram` sampled-peak VRAM, multi-ISA CPU
  dispatch, FP8 `mma.sync` kernels (no real model), MoE expert-cache prototype (no backend),
  rate limiting, JSON access log, coverage floor (per-module, including `engine.py` at its
  CPU-reachable coverage and every `kindi/backends/*` module), perf-baseline gate, CI static analysis.
- **Open gates:** installer pin/version (release owner), tandem-beats-GPU (tuning), a
  device where full-GPU cannot run yet a split can (on the 4060 Ti the loader reduces the 27B to
  ~37K full-GPU / ~30K tandem tokens, so the split buys headroom, not context; the 9B never
  reduces — measured 2026-11-17 up to 131,072 tokens on both placements, so a split there is
  2.8-4.6x cost for <0.9 GiB), GPU CI runner registration (steps in `docs/CI.md`), mixed-card/P106/Arc
  hardware runs, whether the tandem helps pre-tensor-core cards (Pascal, M4000 — measured cells
  defined in `docs/ROADMAP.md` Cards), the remaining eight prior-art items in
  `docs/EXTERNAL-ENGINES.md` (the first - the measured CPU/PCIe ratio and the `rate` planner - is
  done; its beat clause measured 8.0 vs 7.9 tok/s, not met),
  real FP8/BF16 models, Dory D1, Raon/Gemma 26B backends, AVX-512. See
  `docs/ROADMAP.md`. 27B boundary: 24K needs >=12 CPU layers here with the plan owner (the `rate`
  pick); 24-32 layers buy memory at a 35-50% rate cost.

### Owner actions

Owner actions pending: Nanbeige 33B GGUF replacement, Gemma 4 E4B pack, DeepSeek V4.1 shards/gate data, PQ2_0/Q2_0 quant downloads, a pinned llama-server, a Linux A770 host for the M2 target, other-hardware test rigs, push/installer/CI publications, T4.4/T4.5 + D3/D5 decisions — `docs/MANUAL-HELP-NEEDED.md`.
**A770 (2026-11-24, no claim):** `kindi intel-ternary` splits the packed GEMV over the weight phases (0 mismatches
at six shapes, 5.75x-14.14x) and `kindi intel-gemm` runs the engine's q8 GEMM bit-exact; pinning its work-group to `(256,1)` bought 1.71x (555->323 ms at 512x512x4096, and it was wrong across tiles before), `--staging local` another 1.02-1.05x, and all three OpenCL probes now take `--device-name Arc` because the driver's device order is not stable
(the iGPU has come back first). **`kindi intel-sycl` now passes on Level Zero**: oneAPI 2026.2 `icpx`, backend `ext_oneapi_level_zero`, driver `1.17.39894+6`, 0 mismatches at 1,048,576/1/1,000,003 elements, and `--selector opencl:gpu` exits 1 instead of falling back (`docs/benchmarks/2026-11-24-i7-8086k-arc-a770-sycl/`). The earlier "no L0" was PATH: the adapter loads only from oneAPI's `bin` dirs, `intel-info` finds the install via `KINDI_ONEAPI_ROOT`, and the `sycl-ls` parser now accepts both bracket layouts. M2 still needs the Linux target; no SYCL backend or support claim.
- **Shared-file owners:** the newest lease row in the table below owns its files until
  marked done; `T:/kindi/AGENTS.md` is the main worktree's bridge note, not this file.

## Project memory (permanent)

Requirements, architecture, design decisions, lessons and operational knowledge live in
`docs/development/`; start at `docs/development/README.md`. The project-local workflow is
`.agents/skills/project-memory/SKILL.md`. Each requirement is one stable
`docs/development/requirements/REQ-NNNN.md` file. The external local-AI knowledge base at
The external knowledge base (its path comes only from `$KINDI_AI_INVENTORY`, because a synced-folder
path is personal data) is read-only inspiration (`.agents/skills/ai-inventory/SKILL.md`):
its numbers are hypotheses about someone else's engine, so an idea from there becomes roadmap work
only with a gate, and is a win only once measured here; what was read is recorded in
`docs/development/AI-INVENTORY-SCAN.md`; `python tools/project_memory.py` records
its added, last-modified and deleted commit hashes in
`docs/development/REQUIREMENTS-HISTORY.md`.

Update the internal record and the relevant user-facing docs/examples when behavior changes.
For roadmap work, reconcile `docs/ROADMAP.md` against requirement status and
`REQUIREMENTS-HISTORY.md`: an item is not done until its requirement evidence and commit
hashes are recorded; keep deleted/retired history instead of rewriting it. After a context
reset, start at `docs/development/README.md`, read `REQUIREMENTS-HISTORY.md`, inspect old
requirement states with `git show <hash>:docs/development/requirements/REQ-NNNN.md`, and run
`python tools/project_memory.py --check` before making new claims. Before handoff run
`python tools/project_memory.py --check`,
`python -m pytest tests/test_project_memory.py tests/test_docs_links.py -q`, the normal
CPU gate, and `git diff --check`. Do not leave durable knowledge only in chat.

## Worktree and shared-file protocol (2026-11-08)

This session runs in a **linked worktree**, not the main checkout, and its `.worktrees/pr-*` and `wip/*` entries are archival PR branches, never a merge source for this branch. The two working trees share the
Git object database, not the index or the files on disk:

| worktree | branch | HEAD | role |
|---|---|---|---|
| `T:/kindi` | `main` | `f9182bb` | main checkout, this session's branch of record; the CPU MTP head and its evidence, the batched T-row verify (CP1, a measured miss), the Arc persistent work list, the projector tier and the branch audit are committed here |
| `<worktree>` | `merge/upstream-0.7.0` | `f9182bb` | the linked worktree, read-only this round; 2026-11-25 the CPU MTP head, before that the vision projector tier (`mmproj.py`, `kindi mmproj`, `mmproj_place`, plan/health policy) and the 2.5 facts re-read from the GGUFs, and before that the branch-of-record check (v0.7.2 comparison, branch audit) and before that the round-4 audit triage (secret-rule width, toolchain flags, backend limits column) and before that the Arc pack probe (real 2.5 27B shapes, no converter needed) and its gate-only roadmap rows; before that the 2.5 publisher sampling sets and low->xhigh, checked on both 2.5 27B packs (settings only: the host was busy, so no rate is quoted); before that the -mtp intake, the official 0.7.2 port (LoRA, one error body) and 0.7.2-fork, before that Arc A770 GEMV rows4 +1.357x (dispatch floor 0.0410 ms), exact accepted-character skip +41.37x median miss (class memo withdrawn), before that proxy serving boundary decided, before that Bonsai 2.5 27B CPU run + template parity, the v0.7.1 merge, grammar-mask edges, Anthropic endpoint + A4 dialects, Arc SYCL smoke + Arc GEMM, llama.cpp proxy backend, constrained mask/pick -8.4% structured token, REQ-0030 CPU profile, bench --spec, 27B CPU verify + compiler detection |

Read its copy only while that worktree has uncommitted changes; otherwise this file is the source of truth.

**Shared files:** `AGENTS.md`, `README.md`, `WHY.md`, `CHANGELOG.md`, `install.sh`,
`kindi.conf.example`, `docs/**`, `.github/**` and every file named in the lease table below.
`AGENTS.md` is per-worktree until it is committed; the main agent cannot see an untracked copy here.

**Rules**

1. Before editing a shared file, run `git worktree list --porcelain`, `git status --short -- <file>`
   and `git -C <other-worktree> status --short -- <file>`. Read the other copy if it differs.
2. A task owns the shared files in its row. Only the owner edits them; other tasks read only. Add a
   row before the first edit of a new task.
3. If the other worktree has uncommitted changes to a shared file, do not overwrite it, copy over
   it or merge it. Wait for the owner to commit or hand off. When both worktrees are clean,
   synchronize with `git merge` or `git rebase` on the owning branch; never copy working files
   between worktrees.
4. Do not run `git reset --hard`, `git clean`, `git checkout --`, `git stash` or a rebase in a
   worktree you do not own. Do not use `git add -A`; commit only the task's files.
5. Re-read the file immediately before writing and after another agent's commit. Git is the source
   of truth; a chat message is not a lock.
5a. Every commit that touches `AGENTS.md` refreshes the worktree table hashes (to HEAD after the
   commit, i.e. the commit's own parent) in the same commit. A bump commit cannot name itself, so the
   parent is the value; a content commit must not leave the table older than its parent.
   `tests/test_agents_handoff.py` enforces both the table value and the bump in the diff.
6. At handoff: commit the task, record its branch, HEAD and remaining files here, and leave the
   worktree clean or explicitly listed. Do not delete another worktree's files.

### Reconciliation (2026-11-09)

The main worktree was interrupted mid-edit: `install.sh` was empty, `kindi/tandem.py` did not parse,
and `tandem_new.py`/`hostram.py` were placeholders. Its full tracked and untracked state is preserved
in `bench-logs/main-backup-20261109-045352/` and on the branches
`wip/main-interrupted-snapshot-20261109-045352` and `wip/main-interrupted-backup-20261109-045352`.
The main worktree is clean now; `main` was fast-forwarded to this branch after its three Turing
commits were merged. The corrupt tandem WIP was not merged into `main`; restart it from
`docs/CPU_GPU_TANDEM_DESIGN.md` and the WIP branch if it is resumed.

### Consolidation audit (2026-11-17)

`main` and this worktree are the same commit and both clean, the local untracked comparison render is
now gitignored, and every remote ref is already contained: `origin/main` (`dce89df`), `myfork/main`
(`fd54148`) and `myfork/freebuff/tandem-cpu-recovery` (`437891e`) are ancestors of `main` (0 ahead,
358-415 behind). Two branches stay outside `main` on purpose, audited rather than merged because
merging either would import superseded or corrupt code:

- `wip/main-interrupted-snapshot-20261109-045352` — and the stash-format
  `wip/main-interrupted-backup-20261109-045352`, whose files are a subset of it — holds 38 paths that
  exist nowhere else: a doorbell/async-copy `tandem_kernels.cu` plus a `tandem_new.py` placeholder, an
  AVX-512/VNNI/AMX `cpuid` detection draft, the 2026-11-07/08 planning set (`GOALS.md`,
  `IMPLEMENTATION_PLAN.md`, `docs/GPU_PROFILE_ANALYSIS.md`, `docs/PAIR-SETUPS.md`,
  `researchAndComparisons.md`, `docs/dev/spec-measurement-plan-2026-11-08.md`) and one-off
  `fix_*.py`, `update_*.py`, `test_brace*.sh`, `test_pair_ok.sh` scripts. The async design it
  implements was measured unnecessary on 2026-11-15 (no dispatch gap; see
  `docs/TANDEM-OVERLAP-DESIGN.md`), the ISA draft has no AVX-512 kernels behind it and AVX-512 is
  already a named gate in `docs/ROADMAP.md`, the `pair_ok` bugs its analysis targets were fixed by
  `pair_fits` on 2026-11-14, and the planning docs are 0.4.0-era drafts superseded by
  `docs/ROADMAP.md`, `docs/GOALS.md`, `docs/CARDS.md`, `docs/MIXED-CARD-ROADMAP.md`,
  `docs/OLD-CARD-RIG.md` and `docs/EXTERNAL-ENGINES.md`. Nothing was merged; both stay archival.
- `clean/kindi-0.5` (1 commit, 633 behind) is the 0.5 comparison baseline, not a candidate. A
  2026-11-18 re-check at `f6e73ae` found 38 snapshot-only paths, 13 in the backup (a subset), and
  none in `clean/kindi-0.5`; the candidates are the same broken/unbuilt drafts, so nothing was
  extracted and no new test was added.

There is no TODO/FIXME or unfinished code path in `kindi/` or `tools/`; the open work is the
owner/hardware-bound gate list above. **2026-11-26 recheck:** `bonsai-2.5-support` (`72538e3`) is not
contained in `main` (4 ahead by merge-base, still carrying `kindi/lora.py`/`kindi.cu`/server work on a
0.7.2 fork point) - delete as superseded or move to `pr/*`; it is not a merge source.

#### Verified again and finished where possible (2026-11-17, second pass)

`main`, this session branch and both worktrees are the same commit and clean; `git stash list` is
empty; `origin/main` (`dce89df`), `myfork/main` (`fd54148`) and
`myfork/freebuff/tandem-cpu-recovery` (`437891e`) are all ancestors (423/366/367 behind). Nothing
was left to merge: the session branch had no commit that `main` did not have.

The one piece of the interrupted snapshot worth taking over was taken: its CPUID/feature detection
idea (`kindi/kcpu_cpuid.c`, which is where the AVX-512/AMX reporting came from) now answers
`kindi caps` on Windows through `kernel32.IsProcessorFeaturePresent` (39/40/41) with no compiler
needed, merged only where the engine probe has no answer. Measured on the i7-8086K:
`AVX=yes, AVX2=yes, AVX-512=no` instead of three `unknown`s. Its AVX-512 kernels
(`fwht_block_avx512`, `rows16_avx512`) stay archived: no host here has AVX-512, so they cannot be
compiled or verified, and shipping an unverifiable kernel would be a claim rather than a feature
(recorded in [docs/AUTOTUNE.md](docs/AUTOTUNE.md)). `clean/kindi-0.5` remains the 0.5 baseline.
Nothing was deleted, rebased or force-moved; both `wip/` branches stay as the audit-sized archive.

### Current leases

The full dated lease table (one row per task since 2026-11-08, with its evidence) is in
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) under "Leases moved out of the
live table (2026-11-17)". **Keep this table to open work.** A new task adds one row here with its
worktree, branch, shared files and state; when it finishes, update the state and move the row to that
file. This file is capped at 450 lines by `tests/test_agents_handoff.py`, so a finished row left inline
is what makes it trip.

| task | worktree | branch | shared files | state |
|---|---|---|---|---|
| 2.5 MTP head acceptance (2026-11-25) | `T:/kindi` | `main` (`1009bfd`) | `docs/MTP.md`, `docs/ROADMAP.md`, `CHANGELOG.md` | **merged into `main`, unproven.** `kindi/mtp_cpu.py` normed hidden + `seed()`, `tools/mtp_cpu_probe.py`, one plumbing test; acceptance 8.3% -> 13.9% vs the 2B drafter's 30.6%, so no speedup claim, and CP1 builds against the 2B. Next measurement: seed the whole prompt and dump `u`/`out` per row, then re-run the probe. Evidence: tracked at `docs/benchmarks/2026-11-25-i7-8086k-27b-cpu-mtp-variants/` (claim `mtp-cpu-variants-20261125`), sanitized from `bench-logs/probe*.log`/`probe-out*.jsonl`; `docs/MTP.md` has the ladder. |

## Layout

| path | what |
|---|---|
| `kindi/` | Python package: `engine.py`, `server.py`, `cache.py`, `spec.py`, `cpu.py`, `convert.py`, `config.py`; CLI in `__main__.py` |
| `kindi/*.cu` | CUDA kernels, compiled at runtime; `kcpu.c` is the AVX2 CPU engine |
| `install.sh` | the installer: machine check, Docker, models, `kindi verify`, services |
| `docs/` | user docs; `docs/CONFIGURATION.md` is the key reference |
| `docs/SUPPORT-MATRIX.md` | generated by `python -m kindi.support_matrix`; models, formats, K/V modes and CPU ISA from the code |
| `docs/ROADMAP.md` | the single index of archived no-go work, its gates and the live work |
| `tests/` | CPU-only unit tests (config, chat, tools, tokenizer, sampling, cache), the installer GRUB fixture test and the plan-engine contract test (`test_plan_engine.py`, stand-in `cupy`) |
| `docker/` | `Dockerfile` (CUDA 10.2) and `Dockerfile.rocm` |

## Setup

Native, on Linux with the driver and the CUDA 10.2 toolkit:

```sh
pip install "numpy==1.19.5" "fastrlock==0.8.3" "cupy-cuda102==9.6.0" \
            "fastapi==0.79.1" "uvicorn==0.16.0" "pydantic==1.9.2"
```

Docker, the recommended path for the old cards:

```sh
docker build -t kindi:0.5.0 -f docker/Dockerfile .
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.5.0 info
```

The image tag follows `kindi/__init__.py:VERSION`; `docs/DOCKER.md` and the Dockerfiles are
 authoritative when they disagree.

Models and packs live in `~/.kindi` (`pack`, `state_dir`). See `docs/INSTALL.md` for the pinned
GGUF files and hashes.

## Build and run

There is no compile step for the engine. "Build" means: convert a model once, optionally build
the fast head, and start the server.

```sh
python3 -m kindi convert ~/.kindi/Ternary-Bonsai-2-2B-PTQ1_0.gguf   # once, ~1 minute
python3 -m kindi fast-head                                        # optional, ~2 minutes
python3 -m kindi serve --host 127.0.0.1 --port 8093
curl -s localhost:8093/health
```

Docker build/run commands are in `README.md` and `docs/DOCKER.md`; the installer wraps all of
this (`bash install.sh --dry-run` prints every step without changing anything, Linux only).

## Test

Run from the repository root. The no-GPU gate is cheap and must pass before any handoff.

| command | needs | proves |
|---|---|---|
| `bash -n install.sh` | bash | the installer parses |
| `python -m compileall -q kindi` | python | every module parses |
| `python -m pytest -q` | `pip install -r tests/requirements-test.txt` | the whole CPU suite (`pytest.ini` sets `testpaths = tests`, so bare `pytest` is safe) |
| `python tools/check_python_quality.py` | python, `ruff==0.16.0`, `mypy==2.4.0`, `pyright==1.1.414`, `pylint==4.1.3` | no new or increased Ruff, mypy, Pyright or Pylint findings against the measured baseline |
| `python -m pytest -q tests/test_installer_grub.py` | python, `sed` | the installer's GRUB sed keeps existing kernel parameters |
| `python -m kindi.support_matrix --check` | python | the generated support matrix matches the code |
| `python -m kindi.config_reference --check` | python | the generated configuration reference matches the code |
| `python tools/project_memory.py --check` | python | the requirement history matches git |
| `python tools/check_perf_baseline.py` | python | the committed CROSS-MODE rows are within 15% of the perf baseline |
| `python tools/check_claims.py` | python, git | every number the six claimed docs quote is re-derived from its tracked run, and every benchmark citation has a claim |
| `python tools/check_warp_masks.py` | python | no new unmasked warp shuffle/barrier in `kindi/*.cu` (7 reviewed sub-warp masks in `tools/warp_mask_baseline.json`) |
| `python tools/check_secret_history.py` | python, git | no new credential or personal-data row in git history (tracked-file scans cannot see a deleted line) |
| `python tools/check_secret_baseline.py` | python, `detect-secrets==1.5.0` | `.secrets.baseline` still matches the tracked tree |
| `git diff --check` | git | no whitespace damage |
| `python -m kindi --help` / `python -m kindi info` | python | the CLI imports without a GPU; config, pack and GPU report |

The GRUB test extracts the real `sed` script out of `install.sh` and runs it against
`tests/fixtures/grub.default`; it never touches `/etc`. If you change the edit, reintroduce the
old bug on a copy and confirm the test fails, so the test is not vacuous.

## Verify (GPU, model pack and reference)

```sh
python3 -m kindi convert MODEL.gguf     # once; the pack must match the reference file
python3 -m kindi verify                 # greedy output vs llama.cpp, token for token; must PASS
python3 -m kindi bench                  # tokens/s at several positions, prompt read, memory
KINDI_DEVICE=cpu python3 -m kindi bench # CPU engine only, no GPU
curl -s localhost:8093/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-2b","messages":[{"role":"user","content":"Say hello."}]}'
```

Without a GPU you can run the no-GPU gate only. Say so in your report; never present a compile
or a syntax check as a runtime verification.

## Definition of done

- The no-GPU gate passes: `bash -n install.sh`, `compileall`, `pytest -q` and the GRUB fixture test
  (the rows above that need no GPU).
- If the server or cache changed: an invalid request returns 400 (not 500) and does not clear
  the prompt cache; a valid request right after it still hits the cache.
- If `install.sh` changed: `bash -n`, the fixture test, and `--dry-run` on Linux. Kernel edits
  are never tested against the real `/etc/default/grub`.
- Docs updated when a config key or environment variable is added (`docs/CONFIGURATION.md`,
  `kindi.conf.example`, and the regenerated `docs/CONFIGURATION-REFERENCE.md`).
- `python tools/project_memory.py --check` passes, and any new requirement or decision is recorded in
  `docs/development/` rather than only in chat.
- The AGENTS.md worktree table names the current HEAD (guarded by `tests/test_agents_handoff.py`).
- The report names what was not verified.

## History

The dated Snapshot, Handoff, lease history, Current plan and Resume protocol that used to live here were
moved verbatim to [docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) on 2026-11-14
(this file had grown to 1,057 lines and the audit flagged it). Read them when you need the reasoning behind
an older decision; `docs/development/` is the durable record and this file is the live handoff. The
finished lease rows were moved there again on 2026-11-17 (its last section), leaving the live table with
no open lease and this file well under the 450-line guard.
