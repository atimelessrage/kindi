# Changelog

## Unreleased

### Added

- **The kernel-arity check reaches the DSpine graph** — the checker now resolves the 31
  `kf["..."]` launches from the literal iterable that builds `self.kf`, and `self.k` /
  `self.k8 if n <= 8 else self.k16` from their `get_function(...)` assignments. DSpine moves from
  4/39 measured to 37/39; across the four graph builders the coverage is 105 measured / 37 pinned
  (spec 13, dspine 2, mtp 2, tp 20 unresolvable), with every measured site still mutation-proven.
- **`/metrics` names the effective speculation mode** — `kindi_spec_mode_effective{mode="..."} 1`
  comes from `drafting()["method"]`, so a refused head reports `none` and a missing 2B drafter
  reports `ngram`, never the requested `KINDI_SPEC` mode. The accepted/drafted counters stay global
  and their help points at the gauge; three test cases cover the loaded, refused and fallback states.
- **The verify matrix runs and scores speculation** — `tools/verify_matrix.py` gained a second axis:
  `--spec-modes` (default all four) on `--spec-models` (default 2b) builds `<spec-base>-<spec>`
  cells (default `tc-ngram`, `tc-mtp`, `tc-draft2b`, `tc-both`), passes `verify --spec` plus the
  mode's files, and records requested/effective and accepted/drafted counts in the table, CSV and
  JSON. A fallback is `refused` (checked before the generic exit-2/no-reference branch), and a
  missing head or pack is a printed skip, so a host without them still runs the plain matrix; a
  baseline containing a cell makes `--fail-on-regression` fail when it disappears. `--spec-base`,
  `--mtp-head`, `--spec-pack` and `--draft-head` set the base and files. Eight tests cover the
  environment chaining, skip reasons, parsing, refusal ordering and dry run.
- **`kindi io` measures this host's memory tiers** — a RAM copy, an unbuffered disk read and pinned
  PCIe copies both ways, as text or `--json` (`--mib`, `--disk-dir`, `--skip`). The disk probe writes
  a temporary file and reads it with `O_DIRECT` (Linux) or `FILE_FLAG_NO_BUFFERING` (Windows) so it
  measures the device instead of the page cache, and says `buffered` rather than pretending when the
  OS refuses; the PCIe probe uses pinned memory like the engine's staging buffers and reports H2D and
  D2H apart. It loads no model. Measured here on 2026-11-17 with `--mib 64`: 13,456.7 MiB/s RAM,
  2,509.8 MiB/s disk, 6.0/6.1 GiB/s H2D/D2H on the 4060 Ti. The rates feed slot sizing and the
  parked NVMe cold-expert tier, whose gate is in `docs/ROADMAP.md` "Memory-tier follow-ups".
- **`kindi plan` and `kindi autotune` now consume the `kindi io` rates** — the Lightning report
  normalizes the measured RAM, disk and PCIe numbers into bytes/s, predicts one token's slot
  working set plus `ceil((RAM copy + disk read) / PCIe H2D)` lookahead, caps it by `n_exp` and by
  the room that still leaves `n_used` experts resident, and prints the three assumptions in a
  `rates:` line and in the slots explanation. Autotune tests that count with half and double it.
  Unit-tested on synthetic rates; the within-10%-of-measured-best Lightning-pack gate is still
  **not measured** (`docs/ROADMAP.md`, memory-tier #3).
- **`tools/ollama_links.py` links the GGUF models Ollama already downloaded under readable names**
  — Ollama keeps its weights as `blobs/sha256-<hash>` and describes them in manifests, so a model
  you already have could not be named on a Kindi command line. The helper reads those manifests
  (`$OLLAMA_MODELS`, else `~/.ollama/models`; `-m` for another directory), picks the model layer
  (not the projector, template or license) and makes one link per model into
  `~/.kindi/models/ollamaLinks` (`-o` to move it), named after the manifest: `gemma3/4b` becomes
  `gemma3-4b.gguf`. A symlink is used when the host allows it and a hard link otherwise, so nothing
  is copied; a cross-volume failure explains both ways out, `--dry-run` lists without writing, and
  `--force` replaces a file that is not the same blob. If there are no manifests, the blobs
  themselves are scanned for the `GGUF` magic, so a bare `blobs` folder or a folder of `.gguf`
  files works too. `tools/ollama_links.ps1` is now a thin wrapper over the same script, and twelve
  tests cover manifest naming, the blob fallback, the environment variable, dry run, idempotence,
  the symlink-to-hardlink fallback and the failure hints.
- **`kindi bench` reports prefill vs decode as a share of one request's wall clock** — two separate
  tok/s numbers left the reader to combine them, and the printed `prompt reading` rate was measured
  one token per `feed` call, so it was not a request's prefill rate at all. `--split-at` (default
  4096,8192) and `--split-generated` (default 128) now time a real chunked prefill followed by that
  many decode steps, sync, and report each phase's milliseconds, tokens/s and share. Measured on the
  2B (i7-8086K + RTX 4060 Ti, two runs per context): 512+128 is 13.6% prefill / 86.4% decode,
  4,096+128 is 64.1% / 35.9%, 8,192+128 is 82.2% / 17.8%; chunked prefill runs at 4,349 tok/s at 4K
  against the 302 tok/s read line, so the split is not derivable from the old numbers. The
  AI-inventory row that asked for this is closed for the 2B, with the 27B half and the multi-request
  share still open: `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-bench-split/RESULTS.md`.
- **The secret guards now cover git history** — both existing scans look only at the files that exist
  now, so a key that was committed and deleted lived on in history, which is exactly what a push
  publishes. `tools/check_secret_history.py` walks every commit's diff once (`git log --all -p -U0`),
  shares the detector patterns with the file guard (`tools/secret_patterns.py`), reports only the
  commit, path and detector, and gates against a counted snapshot
  (`tools/secret_history_baseline.json`) so any new row or a row whose count grew fails. CI runs it in
  the `secret-scan` job with `fetch-depth: 0`; `tests/test_secret_history.py` proves a key deleted from
  the tree is still found there. The first baseline records the legacy exposure (3,430 hits: an old
  gitignored benchmark dump, old benchmark evidence with drive paths, and old revisions with home
  paths) as a reviewed snapshot rather than an endorsement - removing it needs a history rewrite,
  which is now a decision instead of an accident.
- **A synced-folder path counts as personal data** — the file scan gained a cloud-storage rule
  (Dropbox, OneDrive, iCloud, Nextcloud, ...) and validates addresses against a no-reply/placeholder
  domain list. It immediately caught the knowledge-base path this repository had just gained in
  `AGENTS.md`, the skill and the scan ledger; that path is now read from `$KINDI_AI_INVENTORY` and is
  no longer written down anywhere in the tree.
- **`kindi caps`: what each card and this CPU can actually do** — card names alone were not
  enough to pick settings. `kindi caps [--json]` reports every card the driver sees (index, name,
  UUID, PCI bus id, compute capability, total/free VRAM, driver) from nvidia-smi, with pynvml and
  CuPy fallbacks, so it still answers when the CUDA runtime is missing; the CPU half reuses the
  engine's own probe (same tier/build the engine would select) and adds the marketing model name,
  core counts, L2/L3, AVX-512/AMX and the CPU-engine toolchain. Hardware facts (tensor cores, BF16,
  FP8, FP4) are kept apart from the Kindi paths those capabilities enable, a missing compute
  capability is reported as unknown rather than inferred from the name, and a missing compiler is
  named instead of implying the CPU engine or tandem can run.
- **`kindi autotune`: measure the placements for this model and rank them** — candidates are
  derived from the hardware report and the pack manifest (full card, two-card tensor-parallel split,
  tandem `auto`/`gdn:N`, Lightning decode-cache slots, CPU engine), each one a real
  `kindi bench --sample-vram` child run, ranked by decode tok/s with a reduced-context run ranked
  below a full-context one and a failed run reported with its own error text. `--dry-run` prints the
  plan without loading weights, `--json` records every run. Measured on the RTX 4060 Ti with the 2B
  pack at ctx 8192: 97.4-97.8 tok/s decode, 3,050 MiB sampled peak; the two tandem candidates were
  refused (already fits) and blocked (no compiler) and were reported as such, not scored.
- **Recursive GBNF compiles on request, up to a stated depth** — the canonical
  `json.gbnf` was refused outright, which made the standard recursive grammar unusable.
  `KINDI_GBNF_MAX_DEPTH=N` (default 0, i.e. still refused) unrolls the recursion N levels
  deep, compiling a fresh copy per reference and pruning any branch that would need level
  N+1, so deeper nesting is refused rather than silently truncated. The language at the
  allowed depths is checked against an independent AST matcher over every string up to
  length 8-9 for four recursion shapes, and the refusal paths keep their tests. The cost
  is states, not runtime: with no sharing each level re-expands every reused rule, so
  llama.cpp's `json.gbnf` is 4,000 states at depth 1, 21,252 at depth 3 and 398,268 at
  depth 7 — the default 20,000-state limit allows nesting 2, and `json_schema` remains the
  unbounded path.

### Changed

- **The kernel launch contract now covers every graph call site, not one kernel** — `k_rmsfold_qT`'s
  missing `hp` was caught for one kernel only after four releases of silent breakage. `tests/test_kernel_arity.py`
  now reads every `__global__` declaration in `kindi/*.cu` (including `__launch_bounds__` forms and the
  generated `k_*_T` token-batched wrappers from `kindi/chunk.py::_wrappers`, whose signature is
  `params + int T_ + one stride per pointer`) and checks **71 launches** across `kindi/spec.py` (26),
  `dspine.py` (4), `mtp.py` (3) and `tp.py` (38). **71** further sites cannot be resolved statically
  (a kernel held in a variable, list element, dict lookup, `getattr` or conditional; an argument tuple
  assembled at run time) and are now pinned in `UNRESOLVED_SITES` with a reason each, so a new dynamic
  launch fails the test until it is classified, and the measured count is pinned so a silent coverage
  drop fails too. Every measured site is proven by mutating a copy of its module and requiring the
  checker to catch it (drop-last argument first, empty tuple where a dual-arity kernel such as
  `k_tc_q8` — 7 parameters on HIP/Pascal, 8 on CUDA — makes the drop land legally).
  Tracked as [REQ-0023](docs/development/requirements/REQ-0023.md); the grid/block half stays open.
- **A structured request compiles its grammar once, not twice** — validation has to build the
  automaton to answer an invalid schema or grammar with a 400 before the engine runs, and the
  decode path then built the same one again. Validation now parks the compiled object on the
  request and generation reuses it, so a large GBNF is paid once per request instead of twice:
  llama.cpp's canonical `json.gbnf` at nesting 2 is 7,936 states and a 7.8 ms median build (five
  runs, this host), so a structured request with that grammar drops from ~15.6 ms to ~7.8 ms of
  compile before its first token.
- **A refused GBNF grammar now says why it is large and what to write instead** — the rule cache
  dropped on 2026-11-17 (it let every use share one copy of a rule, which made an "exactly 3"
  repetition accept 0-5 copies) was 20x smaller because it changed the language. With exact
  per-use expansion the price is states: a flat 40-pair record referencing `ws` 82 times is 5,915
  states / ~6 ms, against 302 for the incorrect build, and the loop form of the same constraint is
  ~330 states at any document length. The 400 now names `*`/`+` and `json_schema` instead of only
  the state count, `Grammar.states` exposes the compiled size, and `docs/API.md` records the
  numbers and the default limit's practical reach (~130 enumerated copies).

### Added

- **`kindi verify --spec <mode>` gates speculation against the reference** — verify always built an
  engine with speculation off, so `docs/MTP.md`'s "the output of every mode was the same as without
  speculation" had never been checked by anything. The flag sets the mode for the run (its own
  variables, `KINDI_MTP_HEAD` / `KINDI_SPEC_PACK` / `KINDI_SPEC_DRAFT_HEAD`, stay as set), the first
  line reports `speculation: requested …, effective …, head …`, every prompt line adds
  `N/M drafts accepted`, and the summary names the mode. A mode whose drafter did not attach exits 2
  with the reason instead of quietly verifying the plain engine; `auto` alone may settle elsewhere.
  Measured 2026-11-17 on the 2B / RTX 4060 Ti (4 prompts x 40 greedy tokens): plain 4/4 at 279.0
  tok/s, `ngram` 4/4, `mtp` 4/4 at 203.5 (mode A is token-exact, 98/248 drafts accepted), `draft2b`
  4/4, `both` 4/4 - the last two with the same 2B as target and drafter, so they prove the path but
  not the 27B pairing. `dspine` still has no cell (no weights on this host;
  `docs/MANUAL-HELP-NEEDED.md` row 7). The plain output gains only the mode line and a summary
  suffix; the `PASS` and `tokens/s` lines `tools/verify_matrix.py` parses are unchanged.

### Fixed

- **`kindi io --skip` rejects unknown probe names** — a misspelled name used to match nothing,
  so every probe ran, including the disk probe the user had asked to skip. The names are now
  validated before any probe starts; an unknown one exits 2 and prints the valid set (`ram`,
  `disk`, `pcie`). Pinned by `tests/test_io_probe.py`.
- **`kindi plan`'s `slots:` line now prices the decode slots in the weighted branch** — once usage
  counts exist the line reported only the served share, so the slots' cost was invisible exactly
  where the plan had already paid it. It now runs the same usage-weighted placement twice, once with
  the slots and once against the same room and `--moe-cache-mib` cap with `dcache=0`, and states the
  difference: on the test pack at 0.077 GiB of free VRAM, 4 slots per layer cost 9 of 128 card
  experts and 5.8 points of served share (94.2% against 100%). The report dict carries the same
  numbers as `slots_cost` (`served`, `served_without`, `cost_pp`, `experts`, `experts_without`), so a
  room or cap where the slots are free reads as 0.0 points instead of an assumed one expert per
  layer. Pinned by `tests/test_plan.py` on a synthetic two-layer spec and on the fake pack's
  tight-card case.
- **`kindi plan` and the Lightning engine now plan from one helper, so the slots line cannot drift**
  — `plan._lt_fit` and `lightning._plan` carried the same fit arithmetic twice, which is how the
  report's slots line fell behind the engine in the first place. `kindi/ltfit.py` now holds
  `dcache_default` and `fit_experts`: the report builds the room and calls them, the engine calls them
  too, and the module-level `DCACHE` snapshot that could disagree with a `KINDI_LT_DCACHE` set later is
  gone (both read the environment at plan time). Sharing also removes a latent division by zero for a
  model with two MoE layers, where the engine copy divided by `n_moe - 2`. `tests/test_lt_fit_equivalence.py`
  drives both callers over synthetic specs (all resident, slots priced, two layers, nothing fits), pins
  exact equality and the identical refusal, and fails if either side grows its own formula again.
- **A commit message is now scanned like file content, so a path in one is a caught leak** — the history
  gate read each commit's subject for credential shapes only and never read a body at all, which is
  exactly where a path survives a push with no file left to find it in. `tools/check_secret_history.py`
  now scans the whole message (subject and body) with the same detectors as a diff line, reporting
  `(commit message)` and the detector, never the text. Re-recording the baseline found one such row in
  this history: a legacy commit body names a cloud storage folder, so the baseline is 128 rows / 3,247
  hits and that row is visible instead of invisible. Two tests pin the behaviour — an empty commit
  whose subject carries a home path, with no file containing it, and a body carrying a drive path.
- **`CURRENT-VS-KINDI-0.6.md` §41's delta figures were 7 commits stale, and the harness now gates
  repository figures the way it gates runs** — re-derived from the tags on 2026-11-17: the v0.6.1
  release delta (16 files, +336/−43) reproduced exactly, but "433 commits, 467 files, +67,049/−1,128,
  578 tracked files (169 upstream)" at `f934a55` was measured at an earlier commit: the real values
  are 440 commits, 468 files, +67,156/−1,125, 579 tracked files, and 172 upstream files at v0.6.1
  (169 is v0.6.0's tree). The v0.6.0 comparison is 442 commits, 467 files, +67,420/−1,096. `v0.6.1`
  turns out to be one commit past the shared base `dce89df` and not an ancestor of the fork, which
  the section now says. `tools/check_claims.py` grows a `git` check kind (`commits`, `diffstat`,
  `deleted`, `treefiles`) and `ref:` evidence, so all 22 of these figures - including §1's baseline -
  are recomputed from the tags on every run: 14 gates, 75 numbers, four citations as debt.
- **Mode A is a win at `KINDI_SPEC_K=1`, and the docs now say so instead of "keep it off"** — the
  0.66x recorded on 2026-11-17 was at the default `KINDI_SPEC_K=4`. Re-measured at depth on the 2B /
  RTX 4060 Ti with one harness (`tools/mtp_context_bench.py`: same engine, plain and mode A
  interleaved, prompt reading excluded): k=1 gives 1.12x at 512 tokens and 1.13x at 24K (4 runs a
  side, 72% acceptance), a wash at 8K (1.01x), while the default k=4 still loses (0.89x / 0.66x /
  0.87x) and k=6 at 24K is 0.66x. `KINDI_SPEC_K=1 kindi verify --spec mtp` stays 4/4 token-exact.
  `docs/MTP.md` carries the table and the corrected advice; `auto` still cannot reach the win (it
  caps mode A at k=2 below 8K and switches it off above). Results:
  `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-mtp-context/RESULTS.md`.
- **`/health`, the startup log and `kindi info` report the speculation that is running, not the one
  that was requested** — `drafting()` answered `enabled: true, method: mtp` from `KINDI_SPEC` alone,
  so a head whose graph never built still looked like drafting for four releases. One resolver
  (`spec.resolve_mode`) now answers for `spec_begin`, `/health` and the startup log, which prints
  `speculation: requested <mode>, effective <mode> (<reason>)`; a head that refused to load is a
  `warning: speculation off: ...` line rather than a debug note, the refusal reason is folded into
  `/health`'s `reason` (with `requested`, the effective `method`, the loaded `head` and its
  `head_sha256` prefix), and `kindi info` prints the requested mode with the head's registry verdict
  instead of a mode it cannot know before an engine loads.
- **The MTP head is checked against `kindi/mtp_heads.json` before its weights are loaded** — the
  registry's `sha256` and `bytes` were used by hand once and never by the code, so a truncated or
  patched 60 MB download would load and draft. `kindi/mtp_heads.py` hashes the file in 1 MiB blocks
  and `MTPDraft` calls it before `np.load`, refusing a mismatch with the expected and found digest
  (and both sizes) or an unregistered file, naming the local-development escape hatch
  `KINDI_MTP_HEAD_ALLOW_UNREGISTERED=1` (documented in `docs/CONFIGURATION-REFERENCE.md`). The head's
  startup line now carries its `sha256` prefix.
- **The MTP head's CUDA graph never built, so mode A speculation was silently dead** — ported from
  upstream `v0.6.1`. `k_rmsfold_qT` gained a trailing `hp` argument in 0.5.0; `kindi/spec.py` and
  `kindi/dspine.py` were updated and `kindi/mtp.py` was not, so with `KINDI_TC=1` and
  `KINDI_MTP_HEAD` set the head failed to build (`cuGraphAddKernelNode failed (1)`) and the engine
  logged `speculation off` — a documented feature that had not run on the tensor-core layout for four
  releases. A short argument tuple is not a Python error, so nothing caught it; `tests/test_kernel_arity.py`
  now reads each kernel's parameter count from its `.cu` source and each Python call site's argument
  tuple, and fails on the pre-fix `mtp.py`. The 0.6.1 graph-build diagnostic (node, grid, block,
  argument count) is ported with it. Measured after the fix on the 2B / RTX 4060 Ti: the head attaches,
  and mode A costs 0.66x (236.9-239.6 vs 361.2-363.7 tok/s, 22.5% acceptance) — the feature works,
  the card still does not want it (`docs/MTP.md`).
- **`kindi plan`'s `slots:` line priced the decode slots wrong** — it said each slot displaces one
  expert per layer. The engine's fit (`lightning._plan`, mirrored by `plan._lt_fit`) takes the slots
  off the room *before* dividing it over the remaining `n_moe - 2` layers, so `dcache` slots per
  layer cost `dcache * n_moe / (n_moe - 2)` experts there, and the residency percentage was computed
  against `n_card + dcache` rather than against what the same room fits with no slots. The line now
  runs the fit twice (`dcache=0` for the counterfactual), reports the measured difference and the two
  coverage shares, and the report carries `slots_resident` for machine readers; on the 256-expert
  test pack 10 slots cost 20 experts per layer (162 resident against 182, 86.4% of measured picks
  against 91.6%). `--dcache` is also passed into the fit now instead of only being printed.
- **An image request can no longer reach a text-only engine through a manifest flag** — the server
  accepted image content parts and Ollama `images` for any pack whose manifest named `vision` or
  `mmproj`, but no converter writes that field and no build implements a projector, so such a
  request would have been answered about an image the engine never saw. Image input is now always
  a 400: the message is the existing "no vision weights" one for a text pack, and a new one naming
  the missing projector (`mmproj`) for a pack that claims vision. Pinned by a test that builds an
  app with a vision-declaring manifest and still gets the 400.
- **The tokenize and detokenize routes ignored the documented body cap** — they called
  `await request.json()` instead of the capped reader the chat routes use, so on a `--lan`
  bind a client could make the process allocate an unbounded body before the API key or the
  shape was checked. Both routes now read through `_read_json_body` and answer 413
  `request_too_large` over the limit, with the same 400s as before for a bad body.
- **A dry run could answer "fits" for a prompt it failed to count** — an unexpected
  exception from the tokenizer or counter was swallowed, `ids` became empty and the preflight
  reported `fits: true, prompt_tokens: 0`. A caller acts on that answer, so a counting
  failure now returns a 500 naming the exception instead of a fit.
- **A Nanbeige or DeepSeek V4.1 pack reported a Bonsai model id** — `model_name: auto`
  inferred the id from a literal tuple of backends that forgot those two families, so every
  response, `/v1/models` entry and access-log line said e.g. `bonsai2-33b`. `kindi/modelids.py`
  now resolves through the adapters' own `model_id_for_manifest` hooks, which config and the
  engine gateway share; a manifest two adapters claim is still refused.
- **A non-numeric config value raised a traceback instead of an actionable error** —
  `KINDI_CTX=abc` now says "ctx must be an integer (got 'abc')" like the rest of the
  config validation, instead of an unhandled `ValueError` at startup.
- **Two sessions with the same title on the same day overwrote each other's export** —
  the export name was date + slug with no uniqueness, so the second write won; a name that
  already belongs to another session now gets the session id appended, while re-exporting a
  session still rewrites its own file.
- **`--api-key` is validated as a single line** before it is written to the container env
  file: a value with a newline would have injected extra variables into that file.
- **`--moe-cache-mib` was silently ignored by the usage-weighted plan** — with a usage file that has counts, `kindi plan`'s weighted scan spent free VRAM instead of the cap and reported every expert on the card (a 1 MiB cap printed "32 of 32 experts per layer"). The scan's budget is now bounded by the cap, so the report and the flag agree. The same line also stops quoting the top-n share a uniform split would get when a usage-weighted placement exists: it reports the share that placement actually serves. Tests: a capped plan with counts reports at most the cap's experts and still prints the measured share.
- **A GBNF rule used twice lent its successors to the other use** — the compiler cached a
  rule's NFA pair and every use attached its successors to the same end state, so the
  epsilon closure after the first use reached a later use's successor and the automaton
  accepted strings that skipped the symbols between them (a JSON grammar accepted
  `{"a": }`, since `ws` is reused everywhere). Each reference now compiles its own copy;
  the state limit bounds the duplication. Found by running llama.cpp's own `json.gbnf`.
- **GBNF bodies may start on the next line and continue inside parentheses or after `|`** —
  llama.cpp's documented layout, and the one its `grammars/json.gbnf` is written in, used
  to fail with "expected a rule name or a reference" before any semantic check ran. A
  newline anywhere else still ends the rule.
- **A GBNF grammar is bounded before it is compiled** — the automaton is one state per literal character, so `root ::= "a"{1,100000}` (or a megabyte literal) was a memory request from whoever sent the request. `{m,n}` now refuses a bound above 10,000 copies, and the compiler refuses to exceed 20,000 states (`KINDI_GBNF_MAX_STATES`, read per grammar; a bad value falls back to the default). Both are `ValueError`s, so the server returns an enveloped 400 with `param: response_format` instead of failing the process. Tests: the bound, the state limit, the override and a bad override.
- **`docs/API.md` still said `grammar` was rejected** — true before the GBNF slice landed, stale afterwards; the row now lists all three structured types as accepted and documents the compile bounds.
- **The dense tandem no longer accepts a MoE pack** — `TandemEngine` only refused Gemma 4 and Nanbeige, so `--tandem cpu-first` on a `kindi_lightning` or `kindi_dsv41` pack reached the GDN layer split with a manifest that has no such layers (wrong dims, or a crash) while the flag looked honoured. It now refuses any `backend` outside the dense Bonsai family before the GPU is inspected, and for a MoE pack the message names the split that does apply (expert tiers, not layers). `make_engine` (the one gateway) turns a `--tandem` flag on a MoE family off and logs why, pointing at `kindi plan`, so a config that sets tandem globally gets the MoE engine rather than the wrong one. Tests: the MoE refusal, the non-dense refusal, the gateway log and the untouched `off` path.
- **Adapter generation loops: the O(n^2) stop scan and an empty-prompt token filter** — the five T4.2 adapters rescanned the whole decoded reply for every stop string on every token, the bug already fixed in `server.py` and the `{x}_gen` loops; they now share `base.StopScanner`, which keeps only the longest-stop overlap (a match across two pieces is still found). The same loops passed `{"prompt_ids": []}` to `token_filter`, so a prompt-dependent filter decided on the wrong state; all of them now pass the real prompt. An over-context prompt raises `BackendUnavailable` naming prompt and context instead of silently returning an empty reply finished for length. Tests: scanner boundary/long-reply, real prompt ids, over-context refusal.
- **`kindi use` no longer re-hashes a verified multi-GB GGUF on every run** — a size+mtime marker next to the file skips the second full read (40-90 s on a 9.5-13 GB model) while a changed stat still forces a re-hash. The KINDI_PACK symlink update is also wrapped: Windows without Developer Mode/admin raised WinError 1314 after the config file was written, failing the command for no reason; it now notes the failure and continues. Tests: cached hash, changed-file rejection, symlink refusal.
- **Shipped backends no longer register with `replace=True`** — every adapter opted out of the registry's "two backends claiming one name is a bug" guard, which let a `KINDI_BACKEND_MODULES` module silently shadow a shipped family. They register once, and a second registration is now refused.
- **The engine gateway can no longer build the wrong engine silently** (REQ-0019 T4.2) — `make_engine` now refuses a pack whose manifest names a family no adapter serves (`backend: kindi_cascade`, an unknown one) or that has neither a `backend` nor the dense weight fields, instead of handing it to the dense legacy builder; a manifest two adapters claim is refused as contradictory rather than resolved to the first hook to fire; and `registry.resolve` refuses an equally-specific two-backend match instead of picking by registration order. Every pack now logs the path it took (`-> the bonsai backend (bonsai2-4b)`), and the one legitimate fallback — a dense pack whose own adapter does not claim it, the "Hf" placeholder-`model` case — says so in the log with the manifest field to check. New harness test over the packs on the host, plus gateway tests for each refusal.

- **Installer name fallback: RTX 50 (Blackwell) and RTX 4000 (Turing)** — computed capability now derived from the card name when the driver does not report it (`cc_from_name()` in `install.sh`). RTX 50 series maps to 12.0 (Blackwell sm_120); previously it fell through to the 6.1 default. The existing `RTX 40` rule also matched `RTX 4000` and reported Turing's sm_75 card as Ada 8.9; the separate `RTX 4000` rule now wins.
- **Config reference regenerated** after the sessions round shifted source lines (191 env reads, same content).
- **`test_server_cpu` skips instead of erroring** when the libkcpu cache is cold and no gcc/clang is available.

### Added

- **`kindi plan --moe-chunk` reports what the MoE tandem would do for a prompt chunk** — the scheduler was reachable only from tests, so nothing outside them could answer "what does a prefill of N tokens cost in copies?"; the flag builds the placement from the pack header and `--usage` (honouring `--moe-cache-mib` / `--dcache`) and prints the lane split (card / copy / cpu), the deduped copy list and its bytes for an N-token sample chunk, with `--moe-lane cpu` for the host lane. The sample spreads over the measured order so cold experts appear, and adjacent tokens overlap so the dedupe shows. No weights are read and no bytes are moved.
- **MoE tandem lanes, as a scheduler** — the dense tandem cannot split a routed layer, so `kindi/moe_tandem.py` writes the MoE split down for one execution unit: every selected `(layer, expert)` pair goes to `card` (resident, no slot), `copy` (a complete expert into an `ExpertCache` slot via the chunk planner, so an expert two tokens share is copied once) or `cpu` (a host worker computes it; opt-in, `lane="cpu"`, throttled by a deterministic `cpu_share` that offloads an even, repeatable share of host pairs rather than a sample). Disk-tier experts are always copied even on the CPU lane, because their bytes have to come through anyway. Scheduling order (router order) and arithmetic order are separate: `combine_order="router"` for Raon, `"expert"` for Gemma 4's ascending-expert-id loop, with router ranks kept on every pair. A token that cannot fit rolls the whole forward back, so no pin or LOADING slot survives a refusal. Plan metrics are JSON-safe and split card/copy/cpu pairs plus copies, bytes and hits. `tests/test_moe_tandem.py` (9 cases) pins the lanes, cross-token dedupe, the CPU lane and its thinning, disk experts on the CPU lane, both combine orders against scheduling order, the all-or-nothing rollback, metrics and input validation. Scheduler only: it moves no bytes, runs no kernel and no backend uses it yet.
- **MoE prefill chunks get one copy list, not one per token** — a plan covered one router token, so a prompt chunk that routes several tokens to the same expert would copy that expert once per token. `kindi/expert_chunk.py` adds `plan_chunk(cache, [(token, keys), ...])` on the model-free cache: each token is still reserved by the cache's own atomic `plan()` (so one token's all-or-nothing guarantee is unchanged), an expert selected by several tokens is copied once, and the chunk returns its copy list in first-need order (chunk order, then each token's router order), plus hits, coalesced selections and misses. `ChunkPlan.metrics()` is JSON-safe and `saved_copies` is the traffic the dedupe removed. If a later token does not fit, the chunk is rolled back: this call's pins are released and the copies it started are failed, while a slot already resident or already loading for another token is left alone. Tests: `tests/test_moe_chunk.py` (10 cases) pins dedupe, run-to-run copy order, hits vs coalesced, per-token router order, rollback (no pins, no LOADING slot, unchanged slot state), the untouched out-of-chunk copy, and input validation. Planner only: no backend, engine or GPU copy path is wired.
- **MoE placement report: the decode-slot trade is now stated** — `kindi plan` on a `kindi_lightning` pack gained a `slots:` line: the bytes the device slots take, the experts per layer the engine's own fit gives up for them (the fit takes the slots off the room the remaining `n_moe - 2` layers share, so `dcache` slots cost `dcache * n_moe / (n_moe - 2)`), and, when a measured usage file has counts, the share of routed picks the card serves with the slots against the same room with no slots (`Usage.coverage()`, the top-n share over all MoE layers). `kindi plan`'s MoE report also returns `dcache` for machine readers. `kindi/moe.py` had no direct unit tests; `tests/test_moe_policy.py` now covers coverage, the full-fit and host-budget placements, the one-token minimum and the measured share the placement claims.
- **GBNF grammars: `response_format` type `grammar`** (REQ-0018) — decode-side constraint for any regular grammar, not only JSON schemas. `kindi/gbnf.py` parses llama.cpp GBNF (rules, alternation, groups, literals with C escapes, character classes with ranges/negation, `.`, `* + ? {m} {m,} {m,n}`, references, comments) and compiles it to an NFA whose state is the epsilon-closure of live states; a token is allowed only when its state can still reach acceptance, so constrained decoding cannot walk into a dead end with an empty allowed set. Syntax errors, unknown/duplicate rules, a missing `root` and recursive rules are 400s (`param: response_format`) before the engine runs; recursion is refused because the automaton is regular (`json_schema` is the recursive-JSON path). The three generation paths now ask one predicate (`kindi/constrained.py::is_structured`) instead of repeating a type tuple. Tests: `tests/test_gbnf.py` (language, dead ends, `vocab_allowed`, refusals) and a dense-loop end-to-end test.

- **T4.2 Gemma 4 adapter ported, GPU gate blocked** (REQ-0019) — `kindi/backends/gemma4.py` registered (`gemma4-e4b`, the GGUF chat template through `gemma4_chat`, the tokenizer's stop ids, batched prefill at the KV ring's 512, greedy or sampled from the full logits) over `gemma4.Engine`/`gemma4_tp.TPEngine` (the two-card check moved to `backends.gemma4.two_cards()`, shared with the server); `make_engine` resolves `kindi_gemma4` packs through it. No parity claimed: no Gemma 4 pack on this host.
- **JSON-schema constrained decoding on the Lightning/Nanbeige paths** (REQ-0018) — `lightning_gen.py` and `nanbeige_gen.py` now wire the same `JsonGrammar` used by the dense/Gemma paths (whole reply, no thought channel; Nanbeige drafting disabled while structured output is active; stop tokens only once the automaton is done). `supports_json_grammar` now names dense, Gemma 4, Lightning and Nanbeige. CPU-verified with fake engines (a scripted illegal top logit is skipped and the reply parses); no GPU gate claimed.
- **T4.2 DeepSeek V4.1 adapter ported, GPU gate blocked** (REQ-0019) — `kindi/backends/dsv41.py` registered (`deepseek-v41-flash`, CPU engine) over `kindi.dsv41.Model`; a pack is a hand-written `kindi_pack.json` (`backend: kindi_dsv41`, `model_path`, optional `encoding`) because no converter exists. The DSpark drafter, RAM slots, prompt cache and turn saves stay in the standalone `dsv41_server`. Gate blocked on weights and on the release gate data; no parity claimed.
- **T4.2 Nanbeige adapter ported, GPU gate blocked** (REQ-0019) — `kindi/backends/nanbeige.py` registered (`nanbeige-33b`, SPM + Nanbeige chat template, sampled generation mirroring `nanbeige_gen._pick`); `make_engine` now resolves `kindi_nanbeige` packs through it (previously a Nanbeige pack built a wrong dense engine). No verify/bench parity claimed: the catalog's ahtx GGUF is corrupt as served by Hugging Face (SHA mismatch, garbled tensor infos); owner action in `docs/MANUAL-HELP-NEEDED.md`.
- **T4.2 Bonsai adapter 27B gate closed** (REQ-0019) — the 27B pack (manifest `model` is the GGUF's placeholder `Hf`) never resolved to the adapter; `model_id_for_manifest` now falls back to the `source_file` size, like `config.py`. 27B `kindi verify` is 4/4 on both the adapter and `KINDI_LEGACY_ENGINE_PATH=1` paths, and bench agrees within ~1% — the Bonsai family gate is now complete for all four packs on this host.
- **T4.2 Nemotron Lightning adapter ported, GPU gate closed** (REQ-0019) — `kindi/backends/lightning.py` registered (`nemotron-lightning-30b`, the Lightning tokenizer + the GGUF chat template, sampled generation via the shared `lightning_gen.pick_candidates`); `make_engine` now resolves `kindi_lightning` packs through it. The first local catalog copy was unusable (catalog `38854de6…` vs local `bc22bc…` vs pack manifest `2ec6d074…`; zero first megabyte, rejected as not a GGUF). **Update (same day):** a clean re-download matched the catalog size/SHA-256/GGUF magic, and a fresh pack passed the RTX 4060 Ti gate — verify 5 exact + 7 llama.cpp ties, 0 fail; bench ctx 8192 read 788.3–821.3 and write 165.96–167.41 tok/s. Evidence: `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-lightning-30b/RESULTS.md`.
- **T4.2 Bonsai adapter, 4B/9B gate** (REQ-0019, measurement only) — extended the adapter's GPU gate beyond the 2B to the other two packs on the RTX 4060 Ti host: verify 4B 3+tie and 9B 4/4 on both the adapter and legacy-valve paths, bench rows through the adapter agreeing with the valve within ~1%. Evidence: `docs/benchmarks/2026-11-15-i7-8086k-rtx4060ti-t42-bonsai-4b-9b/RESULTS.md`. No 27B pack on host; no new support claim.

- **Doc-consistency fix** (no code) — `docs/BACKENDS.md`'s intro now describes the actual registry (fake + bonsai adapter + external; T4.2 Bonsai shipped 2026-11-15, T4.3 shipped the same day) instead of "only the fake backend resolves". `docs/AGENTS.md` documents `response_format` structured output (carrying backends, loud 400 elsewhere); ROADMAP's T5.1 bullet no longer lists structured output as planned for the dense/Gemma paths.

- **OpenAI-client wire coverage** (REQ-0018/REQ-0016, tests only) — the live-HTTP conformance suite (`tests/api/`, real app over uvicorn driven by the official `openai` client) now sends a structured-output request (accepted, recorded) and a backend-without-the-grammar request (loud 400, `param: response_format`), plus a `/sessions` HTTP round trip (list/show/export/status/tags/read, malformed-id 400) over the same harness. GPU-free against the stub engine; `KINDI_API_URL` points the suite at a real server.

- **JSON-schema constrained decoding** (REQ-0018) — `response_format` `json_object` and `json_schema` now constrain the dense decode instead of being rejected: a schema-ordered JSON automaton (`kindi/jsongrammar.py`) intersected with the sampler's candidate set (`allowed` in `kindi/sampling.py`), so the reply parses against the schema by construction. Malformed schemas and `tools` with structured output are enveloped 400s; `grammar` stays rejected. Covered by grammar conformance tests and a CPU-only end-to-end loop test.
- **JSON-schema constrained decoding on the Gemma 4 path** (REQ-0018 slice) — the native Gemma 4 server now honors the same constraint (`Kindi._generate_gemma` + `_g4_json_token`): structured output takes over the whole reply (no thought channel is entered), and stop tokens are admitted only once the automaton is done. At the time, the `supports_json_grammar` flag gated only the dense + Gemma 4 backends at the HTTP boundary (Lightning/Nanbeige got a loud 400); the entry above now extends the constraint to those two paths too. CPU-verified end to end (fake Gemma engine/tokenizer); the GPU gate for the Gemma path is not run (no Gemma pack on the host).
- **T4.2 Bonsai adapter** — the dense Ternary-Bonsai-2 engine now lives behind the backend contract (`kindi/backends/bonsai.py`, registered as `bonsai`). `kindi.engine.make_engine` resolves dense Bonsai packs through the registry by default; `KINDI_LEGACY_ENGINE_PATH=1` restores the pre-port builder. Verified 2026-11-15 on the RTX 4060 Ti: `kindi verify` 2B 4/4 on both paths (103.0 vs 103.2 tok/s, timing jitter); `kindi bench` 2B 97.3/91.5/86.3/81.3 tok/s at 2K/8K/16K/24K through the adapter. This was true when the Bonsai adapter landed; Nanbeige, Nemotron Lightning, Gemma 4 and DeepSeek V4.1 now also resolve through the T4.2 adapters above (REQ-0019 T4.2, partial).
- **Cross-ISA parity test** — all five tier libraries run a fixed decode: greedy tokens identical across tiers, hidden states within summation-order tolerance (measured ~3e-4 max diff).
- **Sessions protection + unread flags** — records carry `protected` (prune keeps, `kindi sessions protect SESSION_ID [--off]` sets) and `unread` (raised by record/append/import, cleared by `show` or `POST /sessions/<id>/read`). `list`/`triage` show `[unread]`/`[protected]` markers, `stats` counts both, and server summaries expose both.
- **Legacy Fermi path** — Fermi / pre-Kepler NVIDIA cards now classify as `fermi` in `install.sh`; they offer no GPU models or pairs, skip the NVIDIA container toolkit setup, and the installer offers the CPU engine on an AVX CPU. No CUDA support is claimed (no code path compiles for sm_21).
- **Support-matrix slugs for the 5060 Ti, GeForce 940M and GT 1030** — rendered as not measured (no fabricated rows); the Fermi GT 430 stays unsupported (below the installer’s sm_30 floor, the CUDA 10.2 image cannot run it).
- **Fake-GPU tests for the full ten-card rig list** in `tests/test_installer_cards.py` — classification and model offers for every card, plus the name-fallback cases.
- **Tandem split timings on `/metrics`** (`kindi_tandem_*` gauges: cpu,
  gpu, transfer, head and total ms/token plus tokens and handoffs per
  token) — visible whenever the `KINDI_TANDEM_DEBUG_TIMING` probe ran.
- **`kindi chat --resume SESSION_ID`** continues a saved session: the REPL
  is seeded with the stored turns and new turns append back to the same
  record on exit (`SessionStore.append_messages`).
- **`kindi sessions prune --days N [--all] [--dry-run]`** retention:
  deletes sessions not updated for N days (resolved-only by default).
- **`kindi sessions stats [--json]`** reports sessions by status,
  message totals, bytes on disk, and oldest/newest updates.

### Changed

- **Stop-sequence checks are O(1) per token.** The generation loop
  scanned the whole accumulated reply for stop strings every token
  (O(n²) over a generation); it now scans only the not-yet-checked tail
  via a `stop_from` cursor. Same truncation semantics, including stops
  that accumulated during thinking. Isolated check at 20K tokens:
  0.088 s → 0.018 s (4.8×). The same fix is applied to the Gemma,
  Lightning and Nanbeige generation loops, which had the identical scan.
- **Tandem decode dispatch resolved once.** `TandemEngine` now iterates a
  per-layer owner table built at construction instead of rebuilding a
  `frozenset` and re-resolving the owner per layer per token. Same
  routing; dispatch overhead measured on the fake backends at 19.34 µs →
  17.57 µs per token (×0.91, Python overhead only — no GPU claim).

### Fixed

- **Sessions export carries its filename.** `GET /sessions/<id>/export`
  returns a `Content-Disposition` filename (`<date>-<slug>.md`, matching
  the CLI export convention).
- **Config reference also scans `os.environ.setdefault` reads**, so a var
  introduced only via setdefault cannot silently vanish from the
  reference.
- **Tandem `load_ratio` robustness pinned by tests**: wrong-kind and
  corrupt calibration files return `None` (fit-only fallback) instead
  of raising or being priced.
- **Sessions mutations are serialized.** Each `/sessions` request builds
  its own `SessionStore`, so concurrent status/tag posts raced their
  read-modify-write and one update could silently win; the routes now
  share one lock, `_save` writes unique tmp files, and `list()` closes its
  file handles.
- **Validation errors now name the field.** `RequestError` carries an
  optional `param` and the handlers surface it, so the `n`,
  `response_format` and `logprobs` 400s return the field in the envelope
  instead of `null`.
- **Tandem `rate` log no longer prints a fake prediction.** When the
  split-ratio calibration has no row for the picked layer count, the log
  now says the pick is fit-only instead of `predicted ~0.0 tok/s`.
- **Config reference also scans `os.getenv` reads**, so a future spelling
  cannot silently drop an env var from the reference.
- **Config reference now scans subpackages.** `kindi/config_reference.py`
  `env_vars()` only read `kindi/*.py`, so `KINDI_BACKEND_MODULES` (read in
  `kindi/backends/registry.py`) was missing from the generated
  `docs/CONFIGURATION-REFERENCE.md`. The scan is now recursive with
  subpackage-aware site paths, the reference is regenerated, and
  `test_the_scan_covers_subpackages` pins it.

### New

- **Loud rejection of unsupported `n` and `response_format`**
  ([REQ-0018](docs/development/requirements/REQ-0018.md), decision D2).
  `n` values other than 1 and structured-output `response_format` types
  (`json_schema`, `json_object`) now return an enveloped 400 before the
  engine or prompt cache is touched, instead of being silently ignored.
  `n = 1` and `response_format: {"type": "text"}` are accepted.
- **Server-side `/sessions` endpoints** ([REQ-0016](docs/development/requirements/REQ-0016.md)).
  `GET /sessions` (status/tag filters), `GET /sessions/<id>`,
  `GET /sessions/<id>/export` (raw markdown), `POST /sessions/<id>/status`
  and `POST /sessions/<id>/tags` expose the chat session history the CLI
  already records, over HTTP. Host-only (no engine, cache or GPU), behind
  the API key when one is set, with enveloped errors and a strict id check.
- **Backend plugin interface, foundations (Feature 4, T4.1)** ([docs/BACKENDS.md](docs/BACKENDS.md),
  [REQ-0019](docs/development/requirements/REQ-0019.md)). `kindi/backends/` defines the
  backend contract (capabilities, sampling params, generation events, cancellation), a
  registry that resolves by model id and by GGUF `general.architecture` metadata and
  loads external backends through `KINDI_BACKEND_MODULES`, and a deterministic
  `FakeBackend` for CPU-only tests. No existing engine is an adapter yet (T4.2), so
  there is no arbitrary-GGUF claim; serving behavior is unchanged.
- **`kindi backends` listing (Feature 4, T4.3)** ([docs/BACKENDS.md](docs/BACKENDS.md),
  [REQ-0019](docs/development/requirements/REQ-0019.md)). `kindi backends` lists the
  registry without touching a pack or GPU, and `--json` emits the registry rows for
  scripting. External backends loaded through `KINDI_BACKEND_MODULES` appear too.
- **Agent-grade API foundations (Feature 5, T5.1)** ([docs/API.md](docs/API.md),
  [REQ-0018](docs/development/requirements/REQ-0018.md)). Every error the server
  returns now uses the OpenAI envelope (`message`, `type`, `param`, `code`), including
  404/405 and unhandled failures; `context_length_exceeded` keeps its promised code.
  `tests/api/` is a conformance suite that drives the real app with the official
  `openai` Python client (non-streaming, streaming, usage, models listing, error
  shapes); it runs GPU-free against a stub engine and can be pointed at a real
  server with `KINDI_API_URL`. No request behavior changes: `n` and `response_format`
  are still ignored (decision D2 pending), and the queue default is unchanged.
- **Repo map for the delegation plans** ([docs/dev/REPO-MAP.md](docs/dev/REPO-MAP.md)):
  every module by role with the FROZEN list, the `/v1/chat/completions` request trace,
  per-family chat-template and tool-call locations, and the resolution of every
  `[LOCATE]` in the Feature 4/5 plans.
- **Intel integrated-graphics G0 slice** ([docs/INTEL-IGPU-TANDEM.md](docs/INTEL-IGPU-TANDEM.md),
  [REQ-0021](docs/development/requirements/REQ-0021.md)). `kindi/intel.py` discovers iGPUs from
  Linux DRM sysfs, checks them against the oneAPI floor (unknown PCI ids refused), models the
  shared-memory budget (half of RAM; Core Ultra override), and checks the tandem GPU-side
  contract a future SYCL engine must expose. `KINDI_DEVICE=intel` now refuses loudly naming
  REQ-0020/0021 instead of silently running the CPU engine. No SYCL kernel exists yet — that
  needs a 11th-gen+ iGPU to compile and verify token-exactness — so this is not a support claim.
- **Intel Arc A770 purchased** ([docs/INTEL-ARC.md](docs/INTEL-ARC.md),
  [REQ-0020](docs/development/requirements/REQ-0020.md)). One A770 16 GB is in transit
  (not installed); the SYCL/oneAPI route is planned (M0 audit done) and M1 hardware
  qualification can start on arrival, with an arrival checklist in INTEL-ARC.md. The
  card is registered in the support matrix as not measured; no backend exists and
  this is not a support claim.

## 0.6.0 (2026-11-12)

### New

- **DeepSeek V4.1 Flash is a supported model** ([docs/DEEPSEEK.md](docs/DEEPSEEK.md)). It uses one CUDA card with
  12 GB or more and host RAM for the experts. A second card can hold hot experts. It drafts with DSpark and saves its
  prompt cache.
  - On one RTX 3060 12 GB, a Quadro M4000 as the expert card and two Xeon E5-2673 v4 CPUs, Kindi writes 18.7
    tokens/s with DSpark (the default). It reads a real-text prompt at about 95 to 100 tokens/s.
  - `tools/install_dsv41.sh` checks the host and picks a profile. The big-RAM profile is measured. The NVMe profile is
    experimental, and we did not measure it yet.
  - The script downloads the files and checks their sha256. It writes a systemd service. It does not replace an
    existing service without `--force`.
  - `python -m kindi.dsv41 verify` checks 20 prompts of 64 tokens against the reference. The gate is 98.5% top-1
    agreement. Each miss must be a near-tie, with a top-2 gap of 0.75 or less.
  - The default numerics are DeepSeek's own FP8/FP4 rounding. `--flags 160` uses the rounding of the reference.
- **Nemotron 3.5 Lightning 30B-A3B (ternary).** Kindi has a backend for Nota AI's ternary version of NVIDIA
  Nemotron 3.5 Lightning 30B-A3B (GGUF by boro64, Q2_0). Use `kindi use lightning-30b`. The model has Mamba-2,
  mixture-of-experts and attention layers. On an RTX 3080 at 190 W, Kindi writes 310 tokens/s at 512 tokens and
  277 at 32K (llama.cpp 202 and 182). It reads a 32K prompt at 3,777 tokens/s (llama.cpp 2,138). On an RTX 3090
  at 220 W, it writes 341 tokens/s at 512 tokens (llama.cpp 284). On a GTX 1080 with 8 GB, 95 of the 128 experts
  of each layer are on the card, and Kindi writes 75 tokens/s after a text prompt of 512 tokens (llama.cpp, with the experts of 9 layers on
  the CPU: 4.7). On a Quadro K6000, it writes 61.9
  tokens/s and reads 206 tokens/s at 512 tokens (llama.cpp 35.1 and 165). The server supports chat, streaming, the thinking block and tool calls.
  [docs/CARDS.md](docs/CARDS.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) give the numbers. The weights are
  not part of Kindi; their licence is OpenMDW 1.1.
- **Host-memory experts.** On cards with less than 10 GB, the least used experts of the Lightning model stay in
  host memory. The card reads them over PCIe. During generation, each layer keeps its last used host experts in 10
  slots on the card (`KINDI_LT_DCACHE`), so an expert that the next token uses again is
  not read again. On an RTX 2070 SUPER (8 GB, PCIe 3.0 x16), Kindi writes 116.4 tokens/s after a 512-token text
  prompt and 120.3 after 8K (llama.cpp on the same card, with the experts of 12 layers on the CPU: 113.5 and
  111.8), and reads 2,121 tokens/s (llama.cpp 1,137).
- **Nemotron 3.5 Lightning is a supported model.** The installer offers it (`--model lightning`) on NVIDIA cards
  with 6 GB or more from the GTX 10 series on, on the Tesla P100 and on the 12 GB Kepler cards, after the 27B. The
  default context is 32K.
- **One mixture-of-experts placement for every MoE engine** (`kindi/moe.py`). It decides which experts of which
  layer go to the card, to host memory and (for the DeepSeek engine on hosts with little RAM) to the disk, from
  measured pick counts (`kindi/lightning-experts.json` now has them). With host experts, the card slots go to the
  most used experts over all layers. `KINDI_MOE_WEIGHTED=0` keeps the same count in each layer.
- **The DSpine drafter for the 27B** ([docs/DSPINE.md](docs/DSPINE.md)). It is a block drafter that reads five
  layers of the 27B. Its weights are Apache-2.0. The output is the same as plain decoding.
  - RTX 3080 at 190 W, context 32768: DSpine writes 115.5 tokens/s and accepts 3.10 tokens per pass.
  - The installer downloads DSpine for the 27B on tensor-core cards. `KINDI_SPEC=auto` uses it. `--no-dspine` keeps
    the 2B drafter.
- **Exact token counts and the real context.** New endpoints: `/v1/tokenize` and `/v1/detokenize`. With
  `"dry_run": true`, a request gives the exact `prompt_tokens` and does not generate. `/v1/models` gives the context
  limit (`max_model_len`, `meta.n_ctx`). A prompt that does not fit gets an HTTP 400 at once, with the code
  `context_length_exceeded`.
- **Progress during long reads.** A stream gets a `: kindi progress {...}` comment every 10 s. A client that does not
  stream can ask for keep-alive spaces with `kindi_keepalive`.
- **A save after each turn.** The server writes the state to the disk in the background after each answer. A restart
  does not lose the turn. DeepSeek in thinking mode saves the state before `<think>`.
- **Conversations by session.** Send the `X-Kindi-Session` header, or `kindi_session` or `user`. When the cache is
  full, older states of a session go before the newest states of other sessions. The DeepSeek server keeps 2 to 4
  conversations in RAM with `--slots N`.
- **Priority.** A request is chat (the default) or background (`kindi_priority`, `X-Kindi-Priority`). Chat goes
  first. A background read stops between chunks for a chat request. Then it continues from the cache, and the answer
  does not change.
- **DeepSeek sampling.** The server uses temperature, top_p, top_k and min_p. DSpark drafts for sampled requests too,
  and the output has the distribution of plain sampling. A repeat guard is on by default.
- **`/health` reports drafting:** the method, the drafter, and the drafted and accepted tokens of the last request.
- **`install.sh --lan`** serves the local network. It makes an API key in `~/.kindi/api-key` and gives the key to the
  container in an env file. `--api-key K` sets the key. `--lan` is off by default.
- **Kindi Bridge in the installer.** It is optional: `--with-bridge`, `--no-bridge` and `--bridge-source D`, and
  `-WithBridge` on Windows. Bridge 0.6.0 shows the Kindi servers and their features. Its menus are Agents, Schedule,
  Kindi, Files, System, View and Help. The manual has a Kindi Bridge chapter. Bridge is AGPL-3.0, as Kindi is.
- **AH-Nanbeige4.3-33B with an 8-bit K/V cache.** The new default is `kv_cache auto`. It uses f16 when the model and
  the f16 cache fit at the chosen context. Otherwise it uses q8_0. Both caches pass `kindi verify`. The q8_0 check
  counts the tokens up to the first difference from the f16 reference. Kindi keeps 475 of 586 tokens.
- **Kindi on Windows 10 and 11 with one PowerShell command** (`install.ps1`, [docs/WINDOWS.md](docs/WINDOWS.md)).
  - The installer checks the PC: x86-64, the Windows build, virtualization, RAM, the GPU driver and the proxy.
  - It turns on WSL2 if it is off. After the restart, it continues by itself.
  - It makes a WSL2 distribution only for Kindi (Ubuntu 24.04 from Canonical, sha256-checked). It does not change your
    other distributions.
  - It uses Docker Desktop if you have it. If not, it installs Docker Engine and the NVIDIA container toolkit in the
    Kindi distribution.
  - It checks that the GPU reaches WSL2, and it runs `install.sh` inside. Then it checks the server on
    `http://localhost:8093` from Windows.
  - It adds the Kindi Console (status, chat, logs, API settings), a shell item, an uninstall item and a Settings > Apps
    entry.
  - AMD GPUs use the CPU engine on Windows, because WSL2 has no `/dev/kfd`. A native Windows build is not in 0.6.0.
- **`install.sh` under WSL2.** It finds `nvidia-smi` in `/usr/lib/wsl/lib`. It says that the RAM is the RAM of the
  WSL virtual machine. It does not make kernel changes (gpu-unlock, the 780M option). It uses the GPU support of
  Docker Desktop.
- **The model on the first start of a Kindi container.** Set `KINDI_MODEL` to a model id or to a GGUF file in the volume. The container
  downloads the model and checks its sha256. It builds the pack once into `/data/packs/<name>` and uses it again at
  the next starts. Two containers on the same volume build it once. `KINDI_VERIFY_ON_START=1` runs `kindi verify`
  once for each pack and kind of card.
### Changed

- DeepSeek: the card does its part of each layer in fewer steps, and some kernels run at the same time. The card
  time for each token goes from 34.6 to 24.1 ms. The output does not change, bit for bit.
- DeepSeek: DSpark calculates the cost of a draft from the experts that all the drafted tokens use together. This
  is on by default. The output does not change.
- The images run as the user 1000:1000, not as root. With rootless Docker, the installer now uses `-u 0:0`.
- The CUDA image does not have the CUDA static libraries, NPP and nvJPEG. Kindi does not use them.
- Drafting on the tensor-core path uses `KINDI_FD=1` when `KINDI_FD` is not set. With `0`, a drafted answer could be
  different from plain decoding at near-ties. 0.5.0 had this too. The profiles of the installer already set `1`.
- The installer names more cards correctly with the 470 driver. The Maxwell Quadro M cards are compute 5.2 (it
  printed sm_50). The Tesla P100 is sm_60 (it guessed 6.1). The A100 is sm_80 (it matched a Maxwell pattern).

### Fixed

- DeepSeek: when `max_tokens` or the context stopped a reply in its tool calls, the server sent the raw tool-call
  markup as text. Now the reply has `finish_reason` "length", the complete calls, and the cut call with the start
  of its arguments, as in the OpenAI API. A request without `max_tokens` can use the remaining context (the limit
  was 4096 tokens).
- Lightning on 8 GB cards with host experts stopped with "out of memory" at the decode slots (an RTX 2070 SUPER):
  the plan's reserve of 350 MB did not cover what the plan does not count. The engine now frees everything, keeps
  free what the failed allocation was short of (plus 96 MB), and plans again (up to 3 times). `kindi verify` passes
  on the RTX 2070 SUPER.
- Tool calls of the Lightning server: one call per reply unless the request asks for `parallel_tool_calls` (the
  model could write the same call 20 times, 3,000+ tokens), a repeated call ends a parallel reply, a call cut by a
  stop token is closed and parsed (a call without a required parameter is still dropped), the slips
  `<parameter=name=value>` and `</function_call>` are read, a `<tool_call>` quoted in backticks is text, and a call
  of a tool that the request did not offer (the chat template's example call) is not returned. 57 of 60 test
  requests come back as correct calls (`tools/lt_tooltest.py`, 5 rounds), without the long repeated-call replies.
- Sampled speculation used the temperature before top-p, and it did not use min_p. So the server turned drafting
  off for min_p requests. Now plain and speculative sampling use one distribution (#32).
- On native Windows, the host RAM read 8 GiB, so the checkpoint budget was too small (#4).
- A parked conversation was removed at once when the cache was full (#26).
- `install.sh --dry-run --yes` waited for a sudo password on a computer without Docker.
- The sampled DeepSeek path gave HTTP 500.

### Thanks

- The Nemotron 3.5 Lightning backend was started by the work of slavalevski-tech
  ([github.com/slavalevski-tech/kindi](https://github.com/slavalevski-tech/kindi)). slavalevski-tech first ran this model on Kindi
  and shared that backend under the Kindi CLA 1.1. That work set the scope of the backend. The Kindi code is a new
  implementation ([docs/dev/LIGHTNING-CLEANROOM.md](docs/dev/LIGHTNING-CLEANROOM.md)). Thank you.

### Not in 0.6.0

- Gemma 4 31B Dense: Kindi was not faster on the same cards (two RTX 3080s).
- Qwen3.8-Flash-Next on many cards: moved to 0.7.0.
- The RTX 20 speed work: moved to 0.8.0.

### Container images (added on 2026-11-12)

- Ready-made images are on GitHub's container registry ([docs/CONTAINERS.md](docs/CONTAINERS.md)): `ghcr.io/ahtx/kindi:0.6.0-cuda`, `ghcr.io/ahtx/kindi:0.6.0-rocm6`, `ghcr.io/ahtx/kindi:0.6.0-rocm7`, and the
  moving tags `cuda`, `latest`, `rocm6`, `rocm7`. Each image passed `kindi verify` on a real card before it was published: `0.6.0-cuda` on NVIDIA GeForce RTX 3080; `0.6.0-rocm6` on AMD Radeon Graphics (with HSA_OVERRIDE_GFX_VERSION=11.0.2); `0.6.0-rocm7` on AMD Radeon 780M Graphics (RDNA3, run with HSA_OVERRIDE_GFX_VERSION=11.0.2).
- `compose.yaml` and `compose.amd.yaml` start a server with Docker Compose or podman-compose, without a clone or a build.

## 0.5.0 (2026-11-09)

### New

- **The NVIDIA CMP 170HX (GA100, sm_80) is a supported card.** The board that we measured has 40 GB of HBM2e
  (device id 0x2082) on PCIe gen4 x16, and its FP32 FMA rate is not limited. Kindi writes faster than llama.cpp on
  the same card for the 27B (one and two cards) and for AH-Nanbeige4.3-33B IQ3_XXS (one and two cards, up to 32K
  tokens of context). We did not test the CMP 170HX board with 8 GB.
  [docs/CARDS.md](docs/CARDS.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) have the numbers.
- **The b2 weight copy (sm_80).** On sm_80 cards with 20 GB or more, the decode products read a second copy of the
  ternary weights with 2 bits for each digit. Each mma register then needs one logic operation instead of the
  five-trit decode. The products are bit-identical. The copy is 1.29 times the size of the weights. `KINDI_TC_B2=0`
  turns it off.
- **The 4B on the tensor-core path.** Widths that are not a multiple of 1024 (the 4B: 2560) use padded units. On an
  RTX 3080 at 190 W, the 4B writes 258 tokens/s at 512 tokens (the single-card engine: 118).
- **The AMD Radeon 840M (RDNA 3.5, gfx1152) is a supported graphics unit.** It is the graphics unit of the Ryzen AI 5
  340. ROCm runs it natively, without `HSA_OVERRIDE_GFX_VERSION`. On the same computer and with the same model files,
  Kindi writes faster than llama.cpp for the 2B, 4B, 9B and 27B, at 512 and 8K tokens, and it reads the 27B faster.
  [docs/CARDS.md](docs/CARDS.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) have the numbers.
- **AMD Radeon 8060S (Strix Halo, Ryzen AI MAX+ 395, gfx1151).** Kindi runs the 2B, 4B, 9B and 27B on it, and
  `kindi verify` passes on all four. On the same computer and with the same model files, Kindi writes and reads
  faster than llama.cpp for all four models, at 512 and 8K tokens. [docs/BENCHMARKS.md](docs/BENCHMARKS.md) has the
  numbers.
- **The `kindi-rocm7` image for RDNA 3.5.** The installer finds gfx1150, gfx1151 and gfx1152 graphics and builds
  `kindi-rocm7:0.5.0` (`docker/Dockerfile.rocm7`: ROCm 7 with HIP 7.16 and CuPy 14.2.0). These graphics units do not
  need `HSA_OVERRIDE_GFX_VERSION`. The installer offers the 27B, 9B, 4B and 2B on them. We did not measure Strix Point
  (gfx1150). RDNA3 graphics (gfx1100-gfx1103, the Radeon 780M) keep the `kindi-rocm` image (ROCm 6.3.3).

### Changed

- The new decode defaults (128-position flash-decode chunks, 4-warp product blocks and the one-pass RMS fold on
  cards with 80 SMs or more, the thread-per-element SiLU fold, NVRTC's own `expf` in the tensor-core kernels) are on
  for sm_80 and sm_86 only. Other cards keep the 0.4.0 behaviour until we measure them. `KINDI_TESTED_CC` adds a
  compute capability (for example `8.9`).
- The bank-free IQ3_XXS codebook of the 33B decode is on for sm_80 only (on an RTX 3090 it was 14% slower).
- `kindi verify` counts a first difference as a tie when llama.cpp's own top-2 gap there is under 0.01 nats, or when
  another llama.cpp run of the same file gives Kindi's token. This is the rule of the 33B backend. The 4B and 27B
  references have the gaps (`tools/bonsai_gaps.py` makes them from a running llama-server).
- Weights go to the card with a plain synchronous copy (a large pinned staging buffer failed on hosts with little
  free memory).
- On AMD graphics, the GEMV configurations are set for each graphics unit (the Radeon 780M and the Radeon 840M).
  Kindi times the configurations at start on other AMD graphics units. On AMD, a row-parallel product uses 8 column
  splits at most, because 16 splits changed the 27B output on the 840M.
- On the Radeon 8060S (gfx1151), the 2B, 4B and 9B use the plan engine, with GEMV kernels and settings for this
  unit. Kindi waits for the graphics unit in a loop instead of a blocking wait (the blocking wait woke late at
  random). `KINDI_BLOCKING_SYNC=1` keeps the blocking wait.
- The installer finds AMD graphics on computers and containers without `lspci`.
- Kindi copies the weights to the card in pieces, one tensor at a time. The host does not keep a full copy of the
  weights. A computer or container with 2 GB of free memory can load the 9B and the 27B.

### Fixed

- The 9B on the tensor-core path stopped with an illegal memory access (the embedding fold had a 1024-value buffer
  for Hadamard blocks of 4096).
- The 9B on the tensor-core path differed from llama.cpp at prompt 1 token 36 with Kindi's own `expf`; it is
  token-exact with NVRTC's `expf`.
- The 9B on AMD graphics gave wrong output (`kindi verify` FAIL). A barrier in the embedding kernel was inside a
  condition, so on AMD the waves did not wait for each other. The 9B is now token-exact on the Radeon 780M and 840M.
- On AMD graphics, prompts were read one token at a time when the weight cache had no lane-layout copy (Radeon 780M,
  27B: 10.8 tokens/s). Kindi now makes the missing copy (105 tokens/s).

## 0.4.0 (2026-11-04)

### New

- **Speculative decoding by default on tensor-core cards.** On NVIDIA sm_80 and later, the installer sets
  `KINDI_SPEC=auto`. The 27B uses the 2B as its draft model: k=2 below 1K tokens of context, k=3 from 1K. On an
  RTX 3080 at 190 W, the 27B writes 1.46 times as fast at 512 tokens of context and 1.82 times as fast at 8K. The
  output is the same as without speculation. [docs/MTP.md](docs/MTP.md) gives the measurements.
- **An MTP head for the 2B.** `kindi-mtp-2b-v1.npz` drafts tokens for Ternary-Bonsai-2-2B, up to 8K tokens of context
  (about 1.1 times as fast). The installer downloads it from
  [Hugging Face](https://huggingface.co/ahtx/kindi-mtp-heads) and checks its sha256.
- **The Tesla P100 (sm_60) is a supported card.** The 27B writes 36.0 tokens/s on one P100 (SXM2, 300 W) at 512 tokens, against
  18.7 for llama.cpp on the same card. A new fp16 prompt path is on by default for sm_60 (`KINDI_P100_FP16_PROMPT=0`
  turns it off). With it, Kindi reads prompts 12% to 14% faster than llama.cpp. Speculation is off on the P100. The
  33B runs on two P100 cards at about 2.2 times the speed of llama.cpp. [docs/CARDS.md](docs/CARDS.md) has the numbers.
- **The GeForce RTX 20, 30, 40 and 50 series.** All four series are supported. `kindi verify` passes on them. The
  speed numbers are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the cards that we measured.
- **AH-Nanbeige4.3-33B IQ3_XXS.** Kindi has a backend for AH-Nanbeige4.3-33B IQ3_XXS (our IQ3_XXS quantization of Nanbeige4.3-33B, made with bartowski's imatrix). It puts the layers on one or more NVIDIA
  cards (two RTX 3080, two Tesla P100, one RTX 3090), and on a Quadro K6000 with layers in host memory (3.56 tokens/s at 8K, 1.52 for llama.cpp). Use `kindi use nanbeige-33b` or `install.sh --model nanbeige`. On two RTX 3080 cards, it writes 45.5
  tokens/s, and llama.cpp writes 32.3 with the same file. Prompt lookup is on by default (2 times as fast on copy and edit
  work). In an initial test of 115 tasks, the 33B solves 98 and the 27B solves 96. Model card:
  https://huggingface.co/ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF
- **A start-up self-check for the 33B.** The AH-Nanbeige4.3-33B server checks itself at start-up. If the check
  fails, the server stops, so the service restarts it before it gives a wrong reply.
- **Sampler settings.** Requests can set `min_p`, `repeat_penalty` and `repeat_last_n` (#4).
- **Pinned checkpoints.** `KINDI_CKPT_PINNED` keeps the host copies of the prompt-cache checkpoints in pinned
  memory. It is on by default under WSL2 (#4).

### Changed

- The speculation guard works from 32K tokens of context (`KINDI_SPEC_GUARD_FROM`). Below 32K, speculation was
  faster on each card that we measured.
- The installer works with rootless Docker, and it checks `--gpus` before `kindi verify` (#8).
- The README tells what to do after the install: the port, `/health`, clients, curl and Python (#11).

### Fixed

- The CPU engine restores a cached state in place (#9).
- `kindi bench` times the CPU engine with the host clock when there is no CuPy (#10).
- A streaming request with a prompt that is too long gets HTTP 400 (`context_length_exceeded`) before the first
  byte. Before, the server sent HTTP 200 and then an error in the stream (#18).
- `kindi fast-head` works on a model with more than 4,096 hidden values, for example the 27B (1,280 threads in one
  block was more than the limit of 1,024) (#4).
- `/health` shows the head that runs. A model that has no fast head (the 27B, Nanbeige) shows `exact` also when
  `kindi head fast` was set (#4).
- The installer finds the compute capability of the Tesla K10 and K20 (Kepler) and of the GTX TITAN X (Maxwell,
  sm_52) from the card name. The Quadro K2000 and K4000 are no longer taken for a K20 and a K40 (#19).

## 0.3.0 (2026-10-28)

### New

- **AMD Radeon 780M graphics.** A ROCm image (`docker/Dockerfile.rocm`) runs the 2B, the 27B and Nemotron on the
  780M. The installer finds the card and builds the image. [docs/AMD.md](docs/AMD.md) gives the settings.
- **Quadro K6000.** The installer runs the 27B on this 12 GB Kepler card. [docs/CARDS.md](docs/CARDS.md) gives the
  settings.
- **NVIDIA Nemotron 3.5 Bolt E4B (experimental).** Kindi has its own decoder for this model, with tool calls and
  thinking. It runs on one card, or on two cards of the same kind in tensor parallel. Use `--model nemotron` with
  the installer.
- **More prompt cache.** Nemotron has its own prompt cache (`KINDI_G4_PROMPT_CACHE`). With the Bonsai models, the
  server can also resume a request inside a parked conversation (`KINDI_PARK_BRANCH`, on by default on ROCm).
  `KINDI_PROMPT_CACHE=0` turns the Bonsai cache off. [docs/PROMPT-CACHE.md](docs/PROMPT-CACHE.md) gives the
  results, and `examples/prompt_cache_bench.py` is the test.
- **A setup menu.** The installer shows each setup that the machine supports, the recommended setup first.
  `--auto` takes the recommended setup. `--config N,M` runs several setups side by side, and `--config NxK` runs
  one setup on K cards of the same kind. Each server gets its own container and port.
- **CPU servers next to GPU servers.** The installer offers a 2B server on the CPU on each machine.
- **gpu-unlock.** The installer finds the GeForce RTX 30 cards that NVIDIA limits to half-rate 16-bit arithmetic.
  It offers to unlock them with the [gpu-unlock](https://github.com/Zagitta/gpu-unlock) module by Zagitta. It never
  does this without a question.
- **`kindi models` and `kindi use`.** A model catalog, the models that fit this machine, and a resumable download
  with a sha256 check.

### Changed

- Sampling uses the order of llama.cpp by default: top-k, top-p, then temperature.
- The server returns OpenAI `logprobs` and `top_logprobs` for Nemotron.
- With the 2B draft model, a guard stops speculation when it is slower than plain steps (`KINDI_SPEC_GUARD=0` turns
  the guard off).
- The installer detects a server that restarts in a loop, and it checks that the Kindi modules load after the build.
- The installer makes one pi provider for each server.
- On a 2 GB Kepler card, the installer asks for a context of 23,808 tokens for the 4B. This is the context that fits.

### Fixed

- `kindi use` now verifies an already-downloaded catalog GGUF against its catalog size and SHA-256 before
  converting it. Before, a truncated or damaged local file was silently trusted because it merely existed.
- `model_name = auto` also reads the model size from the GGUF file name. Before, a 27B pack could report
  `bonsai2-2b`.
- `/health` shows a git commit only for a Kindi checkout. Before, a copy of Kindi in another git folder showed the
  commit of that folder.
- The installer works on machines with one GPU and with Kepler workstation cards.
- The cache brings a parked conversation back in place, and the server continues after an engine error.
- The tensor-core path starts on cards after the RTX 30 series (RTX 40, RTX 50 and Hopper). Before, the 27B
  with the installer's tensor-core profile and the Nemotron model stopped at start-up with `CUDA_ERROR_INVALID_PTX`.
- Kindi starts on RTX 50 cards. Before, no kernel compiled on these cards, because CuPy compared the card versions
  as text.

## 0.2.0 (2026-10-22)

### New

- **More models and cards.** Kindi runs Ternary-Bonsai-2-2B, -4B, -9B and -27B (the PTQ1_0 GGUFs). Tested on the
  Quadro K420 (1 GB and 2 GB), the Quadro K1200 (one card, or two in tensor parallel), the GeForce GTX 1080, the
  GeForce RTX 3080 and the CPU. [docs/CARDS.md](docs/CARDS.md) gives the settings for each card.
- **A one-command installer** (`install.sh`, [docs/INSTALL.md](docs/INSTALL.md)). It finds the cards, selects the
  model and the settings, downloads the model and checks its sha256, converts it, runs `kindi verify` and starts the
  server as a Docker service. It works with the NVIDIA Container Toolkit or without it. `--dry-run` shows each
  command. A re-run upgrades or repairs, and `--uninstall [--purge]` removes Kindi.
- **The 27B.** A plan engine (`kindi/tp.py`) runs it on one card (GTX 1080 8 GB, RTX 3080) or on two cards (2x
  Quadro K1200, tensor parallel). The K/V cache is q8_0, as llama.cpp `-ctk q8_0 -ctv q8_0`.
- **Prompts in chunks.** The engine reads up to 256 prompt tokens in one pass. The tokens are the same as with one
  token at a time. `KINDI_CHUNK=0` turns it off.
- **5-trit weights** on Pascal and later (1.75 bits per weight). The GTX 1080 holds the 27B with a context of 65,536
  tokens.
- **RTX 3080 (sm_80 and later):** int8 tensor-core products and CUDA graphs (`KINDI_TC=1`), split flash-decode
  attention (`KINDI_FD=1`) and tensor-core attention for the prompt chunks (`KINDI_ATT_TC=1`). `KINDI_SPEC=draft2b`
  uses the 2B as a draft model; the greedy output is the same as without it.
- **A CPU engine** (`KINDI_DEVICE=cpu`) for the 2B, 4B and 9B. It is token-exact with llama.cpp and needs no CUDA.
- **Checkpoints in card memory** (`KINDI_CKPT_VRAM=1`, one card): the copy to the host waits until the server is
  idle.
- **`/health` shows the version and the source commit.** `kindi verify` uses the reference of the pack's model.
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** compares Kindi and llama.cpp on the same card, with the same model
  file and the same context.

### Changed

- The context length fits the free card memory (`KINDI_MEM_RESERVE_MB`).
- The prompt cache: the memory budget counts every checkpoint and every parked conversation, and there is a
  checkpoint at the end of the opening. Saved openings belong to one engine configuration, and the server does not
  resume an opening of another configuration. A damaged opening file does not stop a request.
- Blocking sync: while it waits for the card, the host thread sleeps and does not keep a CPU core busy. It is on by
  default on CUDA (`KINDI_BLOCKING_SYNC=0` turns it off). It costs 1-3% of the decode speed, so the installer turns
  it off for one server on its own card.
- Before the 27B loads on two cards, the engine checks the free host RAM and stops with a clear message when it is
  too small (`KINDI_HOST_RAM_CHECK=0` skips the check).
- `model_name = auto` (the default) takes the model id from the pack.
- The K420 comparison in the documentation now uses the same model file for llama.cpp (PTQ1_0). The 0.1.0 page used
  llama.cpp's Q2_0 file. With the PTQ1_0 file, llama.cpp generates 1.65 tokens/s on the K420; Kindi generates 20.8.

### Fixed

- The fast head always scores the special tokens (`</think>` and the end tokens). Before, it could fail to close
  the thinking.
- Host memory: the server frees the CuPy pinned-memory pool after the load, and freed checkpoints go back to the
  system. Before, a server grew with each request.

## 0.1.0 - first public release

- License: GNU AGPL-3.0. Commercial licenses are available (see COMMERCIAL-LICENSE.md).
- An engine for Ternary-Bonsai-2-2B on the NVIDIA Quadro K420. It generates 20.8 tokens/s with the exact head and
  23 tokens/s with the fast head. It reads prompts at 26.5 tokens/s. The context is 24,576 tokens in 1 GB.
- With the exact head, the output is token-for-token identical to llama.cpp (`kindi verify`).
- `kindi convert` makes an exact conversion from the PTQ1_0 GGUF.
- An OpenAI-compatible server with chat completions, streams, tool calls, an optional API key and live progress in
  `/health`.
- Thinking mode: the server sends the thinking as `reasoning_content` (in a stream, as `delta.reasoning_content`).
  `thinking_budget_tokens` sets a limit. At the limit, the server closes the thinking and the model gives its answer.
- When `max_tokens` stops a reply in the middle of a tool call, the server returns that text. The client can then
  see that the reply is not complete.
- A prompt cache: checkpoints of the recurrent state, parked conversations in RAM and saved openings on disk.
- An optional fast (shortlist) output head. You can change the head at run time (`kindi head`).
- A Docker image with the CUDA 10.2 / CuPy 9.6 stack that Kepler needs.
- When a client disconnects (timeout, Ctrl-C), the server stops the request at the next token. This is also true
  during a long prompt read.
