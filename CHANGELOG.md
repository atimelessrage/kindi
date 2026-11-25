# Changelog

## Bonsai 2.5 27B publisher sampling (2026-11-25)

- **The card's two sampling sets are the profile, in full.** `sampling_profile = publisher` on
  `bonsai2.5-27b` now states every value the card gives: thinking temperature 1.0 / top-p 0.95 /
  top-k 20 / min-p 0.0 / presence 0.0 / repetition 1.0, instruct 0.7 / 0.80 / 20 / 0.0 / 1.5 / 1.0.
  The off values were already the defaults, so behaviour does not change; they are now part of the
  profile so a user override cannot silently leave one out and the tests pin the whole set.
- **`reasoning_effort = low` renders the `xhigh` instruction.** The 2.5 card says `low` is not
  supported and the model then behaves close to `xhigh`; the template's own `low` text is not what
  the model was trained on, so Kindi maps `low` to `xhigh` (`kindi/chat.py`, DEC-0013). `xhigh`
  stays the default and `medium` the shorter-response setting.
- **A repeatable benchmark covers both packs.** `tools/bonsai_sampling_bench.py` renders the 27B
  template, asks `kindi.profiles` for the values (so the benchmark cannot drift from the server),
  samples with `kindi.sampling.Sampler` and records settings, ids, text and timings for instruct and
  for thinking at `xhigh`, `medium` and `low`. Its 2026-11-25 run on the CPU engine covers both the
  plain and the `-mtp` Bonsai 2.5 27B packs; that host was busy with other work, so the run is a
  check that the settings apply and the engine answers, not a performance comparison - no rate from
  it is quoted. No GPU run (the Arc A770 has no backend).

## Bonsai 2.5 27B `-mtp` files (2026-11-25)

- **The `-mtp` GGUF converts to the base model's pack.** Prism ships the 2.5 27B draft head as a
  65th block (`blk.64`: one decoder layer plus its `nextn.*` tensors, stored Q4_0) after the
  64-block language model, so `kindi convert` refused the file: the block count did not match the
  27B profile, and the head's Q4_0 tensors failed the format gate. The converter now recognises
  the shape (a profile one block past a known model whose extra block carries `nextn.eh_proj`),
  converts the language model, and records the head's block and tensor names in `kindi_pack.json`
  (`mtp_head`). The catalog gains `bonsai-2.5-27b-mtp` with the file's pinned size and SHA-256.
  The head is not used to draft: Kindi's drafter is the 2B's registered head on the tensor-core
  path, and the 27B head is not wired to it. Verified on the CPU engine only - full conversion,
  then a greedy load-and-generate run of that pack; no GPU run (no CUDA device here, the Arc A770
  is probe-only).

## 0.7.2-fork (2026-11-25)

Official Kindi 0.7.2 (`22af8d4`, released 2026-11-24) is merged by content on top of the fork's
`v0.7.1` merge (`1cb59d2`). Its `load_json` fix (PR #50) was already in this tree, so the rest of
0.7.2 is what this round adds: LoRA adapters, and one OpenAI error body on the router.

### New

- **LoRA adapters for the dense Ternary-Bonsai GPU engine.** `kindi --lora FOLDER serve` (also
  `--lora`, `KINDI_LORA`, and `lora=` in `kindi.conf`) loads a PEFT folder (`adapter_config.json` +
  `adapter_model.safetensors`), an `.npz` with GGUF tensor names, or the hash of an adapter already
  converted into the pack. Hugging Face Qwen3.5 module names map to GGUF tensors, and an adapter
  trained in Hugging Face's V-head order is permuted for the 4B/9B. `kindi/lora.py` folds the A rows
  into the activation rotation once, at conversion (`A R^T`), so no rotation runs per token, and
  merges `ssm_alpha`/`ssm_beta` adapters into the fp32 weights (zero cost per token). Each adapted
  projection adds one `k_lora` launch after its GEMV; `k_lora2` splits the reduction across the grid.
  Converted adapters live in `PACK/lora/<hash>/`; the prompt-cache fingerprint includes them, so a
  saved opening is never resumed under another adapter.
  - Range: the plain single-card GPU engine (`KINDI_TP=0`). The plan engine, the tandem engine and
    the CPU engine refuse an adapter with a message instead of ignoring it.
  - **Not verified on this host:** there is no CUDA device here, so the LoRA kernels are compiled but
    never run and no real adapter was applied to a model. The 0.7.2 speed and exactness numbers are
    the publisher's. What is tested here is device-free: the fold and the V-head permutation against
    NumPy, the readers (including an `.npz` whose `lora_alpha` scalar upstream's reader mistakes for
    a tensor - fixed in this port), conversion into a synthetic pack, the cache fingerprint and the
    config/CLI wiring (`tests/test_lora.py`).

### Fixed

- **One OpenAI error body on the router and the stream.** `kindi route`'s nine hand-written error
  bodies now go through `kindi/api/errors.error_body`, so they carry `param` and `code` like
  `kindi serve` does; the server's SSE error events, its non-streaming failure mapping and its
  cancelled answer use the same builder too. A bad request field is no longer labelled
  `unsupported_<field>`: only `tool_choice` and `response_format` keep that code, everything else
  reports `invalid_request_error` (PR #58). The 404/405/422 handlers and the `messages` shape check
  already here are stricter than upstream's version of this change.

## 0.7.1-fork (2026-11-24, on official v0.7.1)

This is the fork's release, not official Kindi 0.7.1. After the 2026-11-24 merge below, the fork is
based on official `v0.7.1` (`1cb59d2`, one squashed commit on `v0.7.0`), 43 upstream files changed
(+15,649/−114), mostly the new Inkling 2 Mini engine. The package, both installers and the Docker tags
say `0.7.1-fork`, so the fork cannot be mistaken for an official `0.7.1` patch. The source archive and
installer hashes are recorded in `docs/INSTALL.md`; publication is still a release-owner step.
(The pre-merge 0.6.1-era snapshot — 669 commits / 827 files at `60a039b` — is kept in
`detailed-fork-changelog.md`.)

### The llama.cpp proxy stays out of `kindi serve`, and now says why (2026-11-25)

The `llamacpp` backend is resolvable by the registry but `kindi serve` never builds it, so the
question was whether to wire it in. A probe (a proxy-shaped object mounted on the real app, canned
upstream replies) showed routing is not the cost: `/health`, `/v1/models`, `/metrics`, chat and SSE
all answer, and structured output is already refused with a 400 by the capability gate. What the
proxy cannot provide is the four numbers the app is built on - `eng.ctx` (it would be invented, and
is advertised as `context_length`), `check_prompt` (byte ids are not upstream tokens, so `dry_run`,
`/tokenize`, Anthropic `input_tokens` and the overflow check would be wrong), streaming (the backend
posts `stream: false` and replays a finished reply, so an SSE first frame waits for the whole reply
and cancelling does not abort upstream) and the prompt-cache gauges (`Kindi.runner`; llama-server's
prefix reuse is invisible). So the proxy stays an engine-gateway backend until the backend grows
incremental upstream SSE, a `/props` context probe and `timings`-based usage - each verifiable only
against a live `llama-server`, the gate T4.4 already has open.

What changed in the tree: a pack whose manifest names the proxy is refused with that reason
(`kindi/engine.py` now separates a contract-only backend from a family nobody ported, pinned by
`tests/test_engine_gateway.py`), and the decision, the probe and its output table are recorded in
`docs/BACKENDS.md` ("Serving boundary") and REQ-0019 T4.6.

### Arc A770: four-row vector loads in the packed ternary GEMV (2026-11-25)

The linked worktree has an Arc A770 16 GB installed, so this round worked the Intel path directly
rather than from a plan. `kindi intel-ternary` now has a second kernel: the pack layout stores a
slot's bytes as consecutive **rows**, so `ternary_gemv_part4` loads four rows of a slot with one
4-byte `vload4`, decodes the trits with `(b * 171) >> 9` instead of integer division, and keeps the
same per-phase int32 accumulation, so its partials stay bit-identical to the scalar kernel.
`--kernel auto` (the default) uses it whenever four rows fit, `--kernel scalar` keeps the 2026-11-24
shape and rows not divisible by four take a guarded path.

Measured on the installed A770 through the display driver's OpenCL C runtime, three runs of 20
launches per arm, medians: **1.357x** at 4096x8192 (0.0936 -> 0.0690 ms per GEMV, 95.75 -> 129.88
GB/s effective), **1.223x** at 2048x4096 (0.0571 -> 0.0467 ms) and 1.065x at 1024x2048, with 0
mismatches in every arm. A 4096x128 shape (128 KB of weights) still costs 0.0410 ms per call, which
pins the two-launch dispatch floor at about 40 microseconds: below ~2 MiB of packed weights the GEMV
is overhead-bound, not bandwidth-bound, and that is the next lever for a backend. Probe only - no
engine path, model run or Intel support claim; M2 stays partial. Evidence:
`docs/benchmarks/2026-11-25-i7-8086k-arc-a770-gemv-rows4/`.

### The constrained mask's walk: an exact character skip, and a memo withdrawn (2026-11-25)

The 2026-11-24 round left the trie walk as the largest remaining part of a constrained mask miss
without saying what was inside it, so this round profiled it first. A 22-token `json_schema`
document makes **8,303** automaton steps per mask miss - one per distinct token-start character,
91% of them rejected - and the trie structure, the id sort and the character sets are rounding
error. The walk was asking the automaton 8,219 questions to keep a handful of ids.

An interim cut, a per-character-class `character -> next state` memo, measured 1.486x and was
**withdrawn the same day**: the cached transitions embed the schema stack, so once warm a state of
one class answered with another state's stack, and document states returned masks that the
full-vocabulary sweep disagreed with. Its page
(`docs/benchmarks/2026-11-25-i7-8086k-cpu-trie-walk/`) is kept with a retraction notice.

The shipping cut is exact and shared nothing. `kindi/jsongrammar.py:_accepted_chars(state)` returns
a **superset** of the characters a state can accept, read off the fields `_step` reads - the schema
stack for a key, the surviving terminals for a keyword, the target node for a value slot, fixed sets
for the structural modes - and `None` where acceptance is not analytic (the string bodies, already
covered by the 2026-11-24 safe-set pruning). `TokenClassTable` takes it as `state_chars` and, at a
trie node, looks up only those characters in the node's children instead of asking the automaton
about every distinct first character. Over-reporting costs one step; under-reporting is what the
tests forbid.

Measured on the 2B CPU path, in-process A/B over the document's 23 states with the same table with
and without the hook: the median mask miss fell from **4.923 ms** to **0.119 ms** (**41.37x**),
automaton steps per miss from **8,303** to **38**, and the document's first-visit masks from
**121.1 ms** to **24.9 ms**. Every mask equals the table without the hook *and* the pre-token-class
sweep over all 248,320 vocabulary entries, on every state. `tests/test_token_classes.py` keeps three
guards: the sets are supersets of what `_step` accepts, the masks do not change with the hook on or
off, and a warm table still matches the sweep - that last test fails on the commit that carried the
memo. Served, three interleaved pairs of fresh CPU servers: structured replies **23.11** ->
**20.82** ms/token (**9.9%**), the constrained gap **3.83** -> **1.65** ms/token (**56.9%**), the
unconstrained control flat (**-0.1%**).

REQ-0030 A6's mask and end-to-end bars are now met by the Python path, so no native variant was
started and none is claimed; what is left of a mask miss is 1.65 ms/token, which a native transition
function would have to remove almost entirely to clear the same 5% bar. Run page:
`docs/benchmarks/2026-11-25-i7-8086k-cpu-key-chars/`, claim `req0030-key-chars-20261125.json`.

### Bonsai 2.5 27B on the CPU engine, and the prompt now matches the published template (2026-11-25)

The 2.5 27B went from "header-verified" to "runs": the 6 GB `PTQ1_0` file was downloaded (SHA-256
matched), converted (`7,242 MB in 370 s`), packchecked (10 files) and loaded by the CPU engine on the
same i7-8086K - **339.9 s** to load, then greedy decoding at **4.02 tok/s** best and **3.08 tok/s** worst
over **2 prompts** of **16 tokens**. Replies are identical within a process and across two processes, and
the manifest repeats the header facts (`851 tensors`, 248320 vocabulary, 262144 context, hadamard block
1024). [docs/BONSAI-2-SUPPORT.md](docs/BONSAI-2-SUPPORT.md) section 1a and the support matrix now say
`load+generate, no reference` for the CPU cell.

Two correctness fixes came with it:

- **The prompt was not the published one.** Kindi writes the Bonsai prompt by hand instead of running
  the template, and it serialized tool schemas and non-string tool arguments with plain
  `json.dumps(...)` - insertion order, no HTML escaping - where the template uses Jinja's `tojson`
  (sorted keys, `<`, `>`, `&`, `'` escaped). Every tool-calling request therefore carried a different
  prompt than the model was tuned on. `kindi/chat.py` now writes both through one `_tojson()` helper;
  a jinja2 comparison of the renderer against the template text in the 2.5 GGUF header and in a
  converted 2B pack matches on **26 of 26** cases (`26 cases`, 0 mismatches,
  `docs/benchmarks/2026-11-24-bonsai-chat-template-parity/`), and `tests/test_chat_template.py`'s
  goldens are rendered by jinja2 rather than typed by hand.
- **`kindi verify` could not tell the releases apart.** `verify.reference_for` derived only the size from
  the pack manifest, so a 2.5 27B pack would have been scored against `reference-27b.json`, Bonsai 2's
  tokens, and reported as a port failure. The reference file name now carries the release
  (`reference-2.5-27b.json`); when it is absent the command says `no llama.cpp reference for this model
  in this version of Kindi` and exits 2. No exactness result exists for the 2.5, and none is claimed.

The Arc A770 path is still probe-only and this host has no CUDA device, so the GPU cells stay `not run`.

### Bonsai 2.5 27B: accepted by the catalog, converter and profiles (2026-11-24)

Prism ML released Ternary Bonsai 2.5 27B the day after v0.7.1, aimed at long agentic work: the same
Qwen3.8-27B base and ternary hybrid-attention geometry as the 27B, the same 5.9 GB footprint, and a
retrained model with multi-turn tool use as the target. Before touching the code, the released GGUF's
header was read on this host with one 12 MB HTTP range request over the 6 GB file
(`docs/benchmarks/2026-11-24-bonsai25-27b-header/`): architecture `qwen35`, the 27B geometry
(64 / 5120 / 17408 / 24 / 4 / 16), `PTQ1_0` tensor type 143, 851 tensors with no unknown names,
248320-token vocabulary, 262144 context, and a 8952-character chat template with `reasoning_effort`.
Nothing in the file needs a new kernel, a new tensor mapping or a new template variant.

So the release is wired in where the file shape allows it, and nowhere else:

- `kindi use bonsai-2.5-27b` downloads the `PTQ1_0` file from `prism-ml/Ternary-Bonsai-2.5-27B-GGUF`
  with its SHA-256 and size pinned, converts it through the existing 27B path and selects it.
- The registry id is `bonsai2.5-27b` (`kindi/backends/bonsai.py`), so a 2.5 pack resolves through the
  Bonsai adapter while staying distinguishable from the 2 release.
- `kindi/profiles.py` gains the 2.5 publisher profile: the card's thinking-mode presence penalty is 0.0
  where Bonsai 2's is 1.0; non-thinking is unchanged.
- `docs/SUPPORT-MATRIX.md` was regenerated; the new row says `not run` / `not converted`, because no
  pack was built and no token was generated from the 2.5 weights here. The `-mtp`, `PQ2_0`, `Q2_0` and
  `mmproj` variants remain unsupported or unverified, and `docs/BONSAI-2-SUPPORT.md` section 1a says so
  explicitly.

### Structured replies get 8.4% cheaper on the CPU engine (2026-11-24)

The constrained path was doing work it did not need, in three places, and all three are now exact
replacements rather than tunings. `TokenClassTable._outside_ids_for` ran
`np.setdiff1d(arange(vocab), safe_ids)` — two whole-vocabulary `np.unique` sorts per safe character
set — where the complement of the already-cached safe mask is the same array in one linear scan.
`_safe_ids_for` scanned 248,320 per-token character masks in a Python generator plus
`np.fromiter`; the shared index now keeps the ids containing each character, so a safe mask clears
the ids of the characters that are *not* safe, and the per-token mask objects are gone.
`Sampler._constrained` built `arange(vocab)`, cast it to int64 and ran `np.isin` over the whole
vocabulary; on an engine whose logits have no `ylog_map` a boolean mask over `ylog` selects the
values *and* is the id list. Mapped heads and requests with a frequency/presence penalty still take
the generic path.

Measured on the 2B CPU engine with the same prompt and reply length for both arms (8 tokens, 10
interleaved runs per shape, each arm on its own fresh server): the 22-token schema filter fell from
245.6 ms to 106.1 ms, its slowest miss from 154.6 ms to 14.9 ms, and a structured reply from 26.35
to 24.13 ms/token (-8.4%) while the unconstrained control stayed flat (20.03 to 19.79). The
constrained overhead itself went from 6.32 to 4.34 ms/token. `tests/test_sampling.py` pins the new
pick against the old selection (ids, logits and token, greedy and sampled); the grammar equality
tests against the pre-token-class sweep are unchanged and passing. Page:
`docs/benchmarks/2026-11-24-i7-8086k-cpu-mask-fast/`. Not measured: the GPU engines (no CUDA device
on this host) and the 9B CPU path.

### The generic fallback: a llama.cpp proxy backend (2026-11-24)

REQ-0019 T4.4's recommended generic fallback is now in the tree: `kindi/backends/llamacpp.py`
(registered `llamacpp`) declares no model ids and no architectures, so the registry resolves it
only for models nothing more specific serves (an arbitrary GGUF), and forwards generation to a
running `llama-server`'s `/completion` endpoint through the backend contract. Configuration is
`base_url`/`api_key`/`model` from the load config or `KINDI_LLAMACPP_BASE_URL` /
`KINDI_LLAMACPP_API_KEY` (default `http://127.0.0.1:8080`); `load()` probes `/health` and records
reachability in `info()` without failing, and `generate()` raises `BackendUnavailable` naming the
server and the `llama-server -m <model.gguf>` remedy when it is missing. Limits are stated, not
hidden: the prompt is ChatML-rendered by Kindi (llama.cpp's own template is bypassed), generation
is one non-streaming request, and constrained decoding is unsupported (`supports_json_grammar` and
`supports_token_filter` are False; a `token_filter` raises instead of being ignored). CPU-tested
with an injectable opener (`tests/backends/test_llamacpp.py`, 12 tests); `kindi backends` lists it
as a wildcard. There is still no arbitrary-GGUF claim: no `llama-server` runs on this host, so the
live proxy gate is open and `verify_prompts()` stays empty.

Also in this round: REQ-0030's native-CPU profile is measured (2B, CPU engine — the mask stays the
only qualifying native candidate at 11.2 ms/token short / 1.1 ms/token long; tokenizer 0.034 ms
and framing 0.0027 ms/chunk stay removed), and `tools/grammar_cost.py`'s `walk()` no longer
crashes on the bounded `StateMaskCache` (hit detection reads the miss counter). Pages:
`docs/BACKENDS.md`, `docs/benchmarks/2026-11-24-i7-8086k-cpu-req0030/`.

### The Arc A770 runs a SYCL vector-add through Level Zero (2026-11-24)

REQ-0020 M2 names the SYCL runtime and SYCL device selection, and every probe so far used OpenCL C.
`kindi intel-sycl` now builds `tools/intel_sycl_vecadd.cpp` with the installed oneAPI 2026.2 `icpx`,
runs it under `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, and validates the child's report: **`result:
pass`**, backend `ext_oneapi_level_zero`, device `Intel(R) Arc(TM) A770 Graphics`, driver
`1.17.39894+6`, 0 mismatches and `max_abs_error = 0` at 1,048,576, 1 and 1,000,003 elements (the last
not a multiple of the local size 64, so the bounds guard runs). The child keeps only a Level Zero
GPU and refuses anything else, and with `--selector opencl:gpu` the run exits 1 naming the missing
`level_zero:gpu` device, so a silent OpenCL or CPU fallback cannot pass as an M2 signal. Page:
`docs/benchmarks/2026-11-24-i7-8086k-arc-a770-sycl/`.

Getting there fixed two host-facing gaps. The Level Zero adapter is loaded from the oneAPI `bin`
directories, so `sycl-ls` run without them on `PATH` lists the Arc through OpenCL and no Level Zero
device at all - which is why the earlier entry below recorded "no L0" on a host that has one.
`kindi/intel.py` now locates the oneAPI install (`KINDI_ONEAPI_ROOT`, then `ONEAPI_ROOT`, then the
default root), runs the installed `sycl-ls` with those directories on `PATH`, counts a root as
present only when it holds a compiler or `sycl-ls` (a leftover `oneAPI/Installer` directory used to
count), and the `intel-info` report keeps the command by name rather than a machine path. And the
current `sycl-ls` prints `[backend:kind][backend:index]`, not `[backend:kind:index]`; the parser
accepted only the old layout and silently dropped every device line, so `intel-info` said "none"
while `sycl-ls` listed the Arc. Both layouts parse now, with a test pinning each. M2 is still not
passed: Windows is not the Linux target, this kernel is not Kindi's, and there is no backend,
engine, installer or support-matrix change.

### Arc staging re-checked, and the host's compilers recorded (2026-11-24)

The Arc A770's `kindi intel-gemm` already had the locally staged variant beside the global-load
shape, and its page claimed a consistent 1.02-1.05x. A paired re-check (arms alternated per run,
three runs per arm, ten launches per run) says something narrower and truer: **1.05x at 128 x 128 x
512, 1.06x at 256 x 256 x 2048, and 0.97x at 512 x 512 x 4096**, with the host reference bit-exact
in all 18 runs. The staged group reserves 34 KiB of the device's 64 KiB local memory, so occupancy
is the plausible cost, but the probe measures no counters and the note says so instead of guessing.
Both tables now sit on the same page (`docs/benchmarks/2026-11-24-i7-8086k-arc-a770-gemm-tiled/`),
and the large-shape 1.02x is marked as that session's number rather than a property of the shape.

The Intel oneAPI toolkit installed on this host is now recorded: at the time `sycl-ls` enumerated
the Arc A770 through OpenCL and found no Level Zero device, which the entry above later traced to
the adapter only loading from oneAPI's own `bin` directories (with them on `PATH` the card is listed
through Level Zero and the SYCL smoke passes), so the SYCL runtime half of REQ-0020 M2 is in place. `docs/development/KNOWLEDGE.md` gained the path-free how-to for
clang (CPU build and `--analyze`), Cppcheck and the oneAPI driver, with the exact machine paths in
the gitignored host-tools sheet; `clang --analyze` on `kindi/kcpu.c` reports two warnings, both
reviewed (one false positive at the attention rotation, one cosmetic dead store left alone so the
engine's source hash and its 4/4 CPU evidence stay valid).

### The 27B on the CPU, and a compiler that was there (2026-11-24)

The 27B pack (6.8 GB, 64 layers) now has a measured CPU row without any CUDA device: `kindi verify`
with `KINDI_DEVICE=cpu` is **4/4 token-exact** (40/40 tokens per prompt), 3.7 tok/s over the run
after a 340 s load with a library built from the current `kcpu.c`; the older cached library gave the
same 4/4 at 2.0 tok/s. `kcpu.c` already implemented the block the 27B needs (gated attention, `qn[2*HD]`), so nothing in
the engine changed to make this work; what was missing was the toolchain and the record.

- `kindi/cpu.py` resolves the CPU build compiler through `_find_cc`: `KINDI_CC`, then the name on
  `PATH`, then a standard Windows LLVM install (`Program Files`/`clang+llvm-*` or `LLVM*`, newest
  first). This host had clang+LLVM 18.1.8 installed with `clang` not on `PATH`, so the engine said
  "could not run clang", `kindi info` reported every AVX feature as `unknown`, and the CPU engine
  looked unavailable although everything was installed. The cache key now names the compiler that
  will run, so two installs never share one library.
- `kindi caps` asks the same resolver, so the toolchain line names the compiler instead of claiming
  the CPU engine cannot be built here.
- `kindi plan` names the pack it planned for: the header printed `Hf (None, 64 layers)` for the 27B,
  whose GGUF names itself `Hf` and whose manifest has no `backend`; it now resolves the registry id
  through the adapters (`bonsai2-27b (dense ternary, 64 layers)`).
- `kindi models` no longer calls the 2B/4B/9B "too large" on a host with no card: the human listing
  names the CPU engine for the models it runs, and `GPU only` for the rest (`fits()` is unchanged).
- `docs/SUPPORT-MATRIX.md` gains the 27B CPU verify row; `docs/CPU-SUPPORT.md` records the run and
  keeps the scope statement (an exactness row, not a speed recommendation).
- The `tests/test_cpu_build.py` build-and-load checks now resolve the compiler the way the engine
  does, so they compile and load real libraries on this host instead of skipping.
- With the CPU engine buildable again, REQ-0028 A4's live half is measured: `kindi selftest` on the
  2B pack passed with `dialects agree (2 token(s), 5 chars)`, i.e. `/v1/chat/completions` and
  `/v1/messages` decoded the same greedy tokens through the real engine.

### Upstream merge (2026-11-24)

0.7.1 keeps everything in 0.7.0 (Qwen3.8-Flash-Next, AMD RDNA2, native Windows, several requests at once and card
placement) and adds three changes and one model. Flash-Next now drafts for sampled requests. The CPU engines use one
thread for each physical core. A new check finds the warp-shuffle mistakes that hung kernels on RTX 20, 30 and 40
cards. Kindi also runs Inkling 2 Mini.

Official v0.7.1 is merged onto the fork's `e4e7eea`; 17 conflicted files were resolved by hand. Three things the
merge itself had to settle: the release changelog names `tools/check_warp_masks.py` but the tag does not ship it, so
the fork wrote it (7 reviewed sub-warp masks, wired into CI — it found two real hazards in the merged kernels, a
`chunk.cu` shuffle and the `kindi.cu` `wsum_d`/`wsum_f` half-warp reduction on pre-Volta cards, both fixed); the
release's root `test_inkling2_engine.py` moved to `tools/verify_inkling2.py`, since it needs a model pack and is not
a pytest module; and the merged Inkling modules had to satisfy Ruff and mypy like the rest of the tree (they do,
after the fixes above), while Pyright and Pylint skip that unrun engine by name (`UPSTREAM_UNVERIFIED`, next to the
existing qwen4exp/qx scope) — `kindi/server.py`, the file that serves it, stays fully in scope. **Unrun here:** Inkling 2 Mini, Flash-Next sampled
drafting, the physical-core detection and the three kernel edits need a GPU, a model pack and a Linux host; the
plain CPU suite is the only evidence for the merge.

### New

- **Inkling 2 Mini** (Thinking Machines Lab), experimental. `kindi convert` accepts the IQ2_M GGUF (about 7.5 GiB),
  and `kindi serve` runs it with its chat template, streaming, thinking levels, tool calls, the sampling values from
  the GGUF and prompt reuse. All 3,072 experts fit in VRAM on an 8 GB card. The
  server answers one request at a time for this model: a second request waits for the first. On a GTX 1080 (8 GB,
  full VRAM) it reads a 512-token prompt at 120.9 tokens/s and writes at 19.8 tokens/s with a 2,048-token context.
  On an RTX 3080 10 GB, a context of 32,768 tokens fits.
- **Drafting for sampled requests in Flash-Next.** A request with temperature above 0, top-k, top-p, min-p or a
  repeat penalty now drafts with the MTP head. Each draft is kept with the probability that plain sampling gives
  that token, so the distribution of the output does not change. Greedy requests keep the 0.7.0 stream exactly.
  `KINDI_FN_DRAFT_SAMPLED=0` turns it off.
- **One CPU thread for each physical core.** Where Kindi picks CPU threads itself (the dense CPU engines, the
  Flash-Next expert pool, the DeepSeek expert pools), it takes one logical CPU for each physical core. Environment
  settings still override it. On a Ryzen 9 7940HS, 8 threads decode at 11.5 tokens/s and 16 threads at 10.0 to 10.8.
- **Faster Intel Arc A770 probe: the packed ternary GEMV runs parallel over the 128-weight phases.**
  `kindi intel-ternary` used one work-item per row, which used a fraction of the card; the phases are now split
  over work-items (`--parts`, default one per phase) and a second kernel adds them. Still bit-exact against the
  host reference, and 5.75x to 14.14x faster at layer-sized matrices on the A770 (512x4096 8.88 -> 1.11 ms,
  4096x4096 9.06 -> 1.58 ms, three-run medians). The probe now refuses a shape whose split is not exact rather
  than reporting a device mismatch. Arc is still unsupported: no backend, model or engine path, and M2 stays partial.
- **Structured-output masks are 2.2x to 8.5x cheaper.** Profiling the 22-token GBNF walk on the real 2B vocabulary
  found two dominant costs, both removed. `Grammar._token_safe_sets` proved a state's universal character loop by
  calling `_step` once for every character in the vocabulary's alphabet (8,228 of them, each a full epsilon closure,
  per NFA state); the same set is now read off the NFA's character edges plus a reverse-epsilon closure built once
  at grammar load, with no change to the accepted language. Each mask union then sorted a whole-vocabulary array
  (`np.union1d`, `np.unique`); safe ids and accepted outside ids are disjoint and sorted, so a cached boolean mask
  plus `np.flatnonzero` assembles the same ascending ids linearly. On the 2B, three-run medians: `json_schema` 15 ->
  10.0 ms/token and the unrolled `json.gbnf` 124 -> 57.1 ms/token at 22 tokens (depth 2/3 83/84 -> 9.5 ms), the
  ordered-record GBNF 70 -> 8.4, and at 200 tokens 3 -> 1.7 and 20 -> 7.5 ms/token. The pre-token-class sweep was
  re-run as an oracle on the real 248,320-entry vocabulary: 14 states across the `json.gbnf`, `json_schema` and
  record walks, 0 mismatches. CPU filter measurement only — no GPU was available on this host, so REQ-0027 A4/A5
  stay open. Evidence: `docs/benchmarks/2026-11-24-i7-8086k-grammar-mask-edges/RESULTS.md`, regenerated by
  `tools/grammar_mask_record.py`.

### Fixed

- Inkling 2 Mini hung on RTX 20, 30 and 40 cards (compute capability 7.0 and later) because two warp shuffles named
  lanes that never reach them. Both shuffles now name the lanes that run.
- `tools/check_warp_masks.py` lists every warp shuffle and barrier under a condition that changes inside a warp and
  fails on any that is not reviewed. The release checks run it. The expert kernels now stop with a clear error when
  a model has a shape they cannot handle, in place of a hang.
- The native Windows package stays at 0.7.0: 0.7.1 has no new Windows build.
- `install.sh` and `kindi cards` stop with a clear message when the NVIDIA driver and library versions differ, and
  `install.sh` names the package when `apt-get` cannot install Docker or the NVIDIA container toolkit. It no longer
  takes an `nvidia` runtime entry in `daemon.json` for an installed toolkit.

### A liveness gate for the documented speculation knobs (2026-11-24, REQ-0018)

- `tests/test_knob_liveness.py` classifies every documented `KINDI_SPEC*`/`KINDI_DSPINE*`/
  `KINDI_MTP_HEAD*` knob: six are proven live (setting the knob changes what the code decides —
  the mode, k, whether sampling drafts, whether the block drafter is wanted, the head's verdict),
  fifteen are parked with a reason, and the test fails on a documented speculation knob with no row,
  a stale row, or a knob with no reader left in `kindi/`. This is the general form of the mode-A bug
  (a documented knob that reached nothing while every gate stayed green). It found one real gap:
  `KINDI_MTP_HEAD` is read by the loader but absent from the generated configuration reference.
- The gate now covers four families, not just speculation: `KINDI_GBNF_MAX_DEPTH` (a recursive
  grammar compiles only when opted in), `KINDI_GBNF_MAX_STATES` (a 90-character literal fits a
  20,000-state default and not an 8-state limit), `KINDI_MASK_CACHE_MB` (the cache budget in
  bytes) and `KINDI_RESPONSE_FORMAT=ignore` (a `json_object` request is refused by default and
  accepted when ignored) are proven live too; `KINDI_SAMPLER_ORDER` is parked because it is a
  module constant read at import, so a request cannot set it. 10 live / 16 parked, and deleting
  one env read fails the gate, so it is not vacuous.
- That gap is closed: `kindi/config_reference.py` dropped every environment name that matched a
  config key, so `KINDI_MTP_HEAD` — the Bonsai MTP head path read directly in `kindi/spec.py` and
  `kindi/mtp_heads.py` — was hidden behind the unrelated Nanbeige `mtp_head` key. The generator now
  lists every direct literal read, so the environment table grows 331 -> 341 rows (ten names, and
  the direct read of a shared name is not the same setting as the config key). `KINDI_MTP_HEAD` now
  shows its loader sites, the knob-liveness test's `UNDOCUMENTED` exemption is gone, and
  `tests/test_config_reference.py` pins the shadowed-key row so the generator cannot hide one again.

### `kindi selftest` compares the two API dialects (2026-11-24, REQ-0028 A4)

- `kindi selftest` already posted the same greedy prompt through `/v1/chat/completions` and
  `/v1/messages`; it now compares the answers, which is REQ-0028 A4's no-decoding-change gate. A
  different text or completion-token count between the two dialects fails the run ("the Anthropic
  route changed decoding"), in process and against a running server (`--url`, which the installer
  calls), and the JSON records `dialects{same_text, same_tokens, chat_text, anthropic_text}`. Both
  requests are greedy and greedy repeats are gated separately, so a mismatch is the endpoint rather
  than sampling noise. No live pack run yet: this host has no C compiler for the CPU engine and no
  CUDA device (`docs/MANUAL-HELP-NEEDED.md` §3b).

### `kindi bench --spec` (2026-11-24, REQ-0026)

- `kindi bench` can now benchmark a speculation mode: `kindi bench --spec mtp` (also `ngram`,
  `draft2b`, `both`, `dspine`) prints the effective mode first — the same line `/health` and
  `kindi verify` print — and the generation curve runs inside `spec_begin`/`spec_end` exactly as a
  server request, so the JSON carries the mode, the loaded head, drafted/accepted counts, the
  acceptance ratio and the k used. A mode whose drafter did not attach is a named refusal (exit 2,
  no JSON row), never a plain-engine number with a spec label. The first registered measurement
  (2B mode A, off vs on) still needs a CUDA card.

### Anthropic Messages endpoint (2026-11-24, REQ-0028)

- `POST /v1/messages` answers the same engine path in Anthropic's dialect, so Claude Code and the
  harnesses that expect that shape need no proxy. Content blocks map both ways (`text`, `tool_use` →
  `tool_calls`, `tool_result` → a `tool` message), `system` may be a string or text blocks, `tools`
  and `tool_choice` map field for field, and `stop_sequences` becomes `stop`. The answer is the
  Anthropic envelope (`message`, `content` blocks, `stop_reason`, `usage`); `stream: true` emits
  `message_start` → `content_block_start` → `content_block_delta`(s) → `content_block_stop` →
  `message_delta` → `message_stop`, with a parsed call as a `tool_use` block rather than leaked
  `<tool_call>` text.
- Auth accepts `x-api-key` (what Claude Code sends) beside the existing bearer token, errors use the
  Anthropic envelope (400/401/413/429/500, never a 500 for a bad field), and an `image` or `document`
  block is refused by name because this backend has no vision path.
- `kindi/api/anthropic.py` holds the whole dialect as pure transforms; `server.py` now has one SSE
  relay shared by both streaming routes instead of two copies. **Not claimed:** compatibility — the
  app-level gate passes (`tests/test_anthropic.py`), but no real Claude Code run has happened yet.
- `kindi selftest` now also posts its prompt through `/v1/messages` (in process, and with `--url`
  against a running server), so an installed image proves the Anthropic answer with the real engine
  and not just that the route is registered; a 200 whose body is not a message fails the check. One
  extra short generation per selftest.

### Audit fixes (2026-11-24, after the merge)

- **The Windows installer fetched the previous release.** `install.ps1` still pinned `$KindiWinVersion =
  '0.7.0-fork'`, and that literal is the bundle URL and the version Settings > Apps shows, while the package and
  `install.sh` said `0.7.1-fork`. Both installers and both release URLs now name `0.7.1-fork`, and
  `tests/test_version_consistency.py` pins the three literals and each installer's release tag so a release cannot
  ship the skew again. The `v0.7.1-fork` release itself is still unpublished (the fork's releases page carries only
  `v0.5.0-installer`), so the one-liners 404 until the release owner uploads it; README says so and
  `docs/RELEASE-CHECKLIST.md` lists the exact steps.
- **The constrained-decoding mask cache is now bounded.** `vocab_allowed` cached every automaton state's allowed-id
  array for the life of a request, and a broad state on the 2B vocabulary is 248,320 int64 ids (~2 MB), so a long
  grammar walk grew with the generation (~86 MB at 200 tokens). `kindi/token_classes.py::StateMaskCache` is a 64 MiB
  LRU by default (`KINDI_MASK_CACHE_MB`, `0` disables caching) that evicts the least recently used mask and never
  stores one mask bigger than the whole budget; both `Grammar` and `JsonGrammar` use it, and the tests prove
  eviction leaves every mask equal to the pre-token-class sweep.
- Document drift corrected: `docs/ROADMAP.md`'s status date, `CURRENT-VS-KINDI-0.7.md`'s PR status (31 of 33 branches
  rebased onto v0.7.1; #47 joined #48/#49/#72/#76 as included by content), `docs/WINDOWS.md`'s installer-version
  history and `docs/RELEASE-CHECKLIST.md`'s status.


### Intel Arc A770: pinned work-group (+1.71x) and staged tiles (2026-11-24, REQ-0020 M2)

- The q8 GEMM probe now launches one 256-wide work-group per 128-token x 128-row tile instead of
  letting the runtime choose the shape. Same session, same kernel, 512 x 512 x 4096: 555.1 -> 323.4 ms
  (38.69 -> 66.3 GFLOP/s, **1.71x**) with bit-identical output. It is a correctness fix as well: with
  tiles staged in local memory an unpinned launch let one work-group span two tiles, whose staging
  and barriers then mixed, giving wrong results at more than one tile (0 mismatches at one tile).
- `--staging local` (the default) stages each group's activation words and decoded weight words in
  local memory as the CUDA kernels do: exact, and 1.02-1.05x over the pinned global load
  (67.77 vs 66.17 GFLOP/s at 512 x 512 x 4096). Small because the byte-wise dot and the per-word
  4-trit decode dominate, not global traffic.
- **Device selection is pinned by name, not index.** The driver has returned the integrated UHD 630
  first and the Arc A770 second, and the reverse on other runs, so a bare `--device 0` could publish
  iGPU numbers as Arc. `intel-cl`, `intel-ternary` and `intel-gemm` take `--device-name SUBSTR` and
  record a warning in the report and printed output when the device is not an Arc.
- `kindi intel-gemm` also holds the quantizer's activations in registers between the maximum and
  packing passes (three barriers per band instead of one pair per token); that change measured inside
  the noise of a single launch, so no speedup is claimed for it. No backend, engine path or support
  claim: M2 stays partial. Evidence in
  `docs/benchmarks/2026-11-24-i7-8086k-arc-a770-gemm-tiled/`.

### Intel Arc A770: the engine's q8 GEMM runs on the card (2026-11-24, REQ-0020 M2)

- `kindi intel-gemm` runs the arithmetic the engine's prompt path does, on the A770 in this rig,
  and holds it to the same bit-exact discipline as the earlier probes: `q8x` ports `k_q8x_T2` (one
  256-wide group per 128-column group and 64-token band, one threshold pair per token, `rint`
  round-half-even) and feeds `gemm_tile`, which mirrors `k_mm_g5q` (128 tokens x 128 rows per
  256-work-item block, int32 dots per group, one `f * (xs * dj)` per group, fp32 accumulate in
  group order). The weight bytes are the pack's on-disk 4-trit layout, there is no local-memory
  staging, and the int32 dot is byte by byte because OpenCL C 1.2 has no dot-product instruction.
  The quantizer is bit-exact with the host model at every shape and the GEMM has 0 mismatches at
  4.2-36.9 GFLOP/s (128x128x512 through 512x512x4096); the host reference is checked against an
  independent float64 dot before the device is launched. The local-memory-staged variant and the
  engine's 5-trit lane unpack are named follow-ups, not measurements.
- No backend, engine path or support claim: Windows/OpenCL is precursor work, M2 stays partial, and
  the exactness construction (amax forced to 127) proves the pipeline but not the general
  threshold path. Recorded under `docs/benchmarks/2026-11-24-i7-8086k-arc-a770-ternary-gemm/`.

### Launch-shape contract: block size (2026-11-24, REQ-0023 A2)

- `tests/test_kernel_arity.py` now holds each graph launch's block tuple against the kernel's
  `__launch_bounds__` ceiling: 96 kernels declare one, 28 (site, kernel) pairs are measured and 135
  launches are pinned as dynamic (96 kernels declare no bound; the rest name the kernel or the block
  through a variable or a runtime width). Both counts are fixed, so a new launch is reviewed once.
  A block that exceeds the declaration (`k_big asks 1024 > 256`) fails the CPU suite rather than the
  launch on a card this host does not have; a synthetic mutation proves the check is not vacuous.
- Only the first `__launch_bounds__` argument is a contract (the second is an occupancy hint), and a
  name declared more than once must fit the strictest declaration, because the caller cannot say
  which variant it got. The grid half of A2 stays open: a zero grid launches nothing silently, but
  grid elements are runtime expressions, so that needs a planner-time assertion on a GPU host.

### Upstream merge (2026-11-23)

Official **Kindi v0.7.0** (`8298a0f`, released 2026-11-23) is merged in as `35e4697`, so the branch is
0 commits behind and 702 ahead (931 changed files, `+133,113/−1,946`, 1,066 tracked files vs 216). 39
conflicted files were resolved by hand. What came across: the Flash-Next engine (`qwen4exp_*`, `qx_cpu.c`,
`qx_gpu.cu`), the multi-slot serving stack (`slots.py`, `--slots`, `KINDI_SLOTS_*`), `kindi place` /
`kindi route` / `kindi cards`, the packcheck size+sha256 gate, the request validator (`oai.py`), DeepSeek
V4.1 expert-card calibration and the Windows launcher/tray monitor. **The multi-slot path and the Flash-Next
engines are not wired into this fork's server** - neither could be run here; that is stated in
`CURRENT-VS-KINDI-0.7.md` and in `AGENTS.md`. #48, #49, #72 and #76 are part of 0.7.0 by
content; the other 29 fork PRs are still open.

### Added

- **The container smoke runs `kindi selftest`** — `bash tools/container_smoke.sh` proves a release
  image end to end: `kindi selftest` one-shot (pack, tokenizer, engine, `/health` and the chat route
  in process) and `kindi selftest --url http://127.0.0.1:8093` in the serving container (the real
  Uvicorn route, authenticated with the container's key). It replaces the manual `/health` + one-curl
  step in the release checklist. The selftest's in-process client no longer needs `asyncio.run`
  (3.7+), because the release image is Python 3.6.9, and it clears the app's key for the in-process
  run (no port is bound). No image was built or run here: no GPU on this host.

- **The installer proves the server it starts** — after each server answers `/health`, the installer runs
  `kindi selftest --url http://127.0.0.1:8093` inside that container and stops if it fails; `--no-selftest`
  skips the check. `--url` is standard-library only and loads no pack, so it costs one short request and does
  not compete with the server for the card. The in-process mode now calls the ASGI app directly instead of
  fastapi's test client, which needed `httpx`/`requests` that the images do not install. No GPU run.

- **One-command install check** — `kindi selftest [--json FILE]` loads the selected pack once, calls
  `/health`, runs one short greedy `/v1/chat/completions` request through the real in-process app,
  and prints the effective runtime state, pack hash and timings. A missing pack, tokenizer or engine
  failure, a non-200 health check or an empty generation exits non-zero. Fake-engine tests cover
  those paths; live CPU runs pass on the 2B (24.5 s) and 9B (112.1 s). It binds no port and adds no
  dependency; the installer does not call it yet, and no GPU run was made.
- **Diagnostics read one effective runtime state** — `kindi/runtime.py` resolves the engine,
  TC/HP state, K/V format, context, effective speculation mode and head digest, chunk size,
  threads and VRAM budget/reserve once. The startup log, `/health` (new `runtime` block),
  `kindi info` and `bench --json` all publish that object, so they cannot disagree about what is
  running. `head` is the head that runs (a requested fast head without a fast-head engine reports
  `exact`) and `spec` is the same effective view `/health.drafting` exposes. Before a load the
  block says `loaded: false` and reports the requests; `kindi plan` keeps its pre-load planner.
- **The A770 runs exact compute kernels without oneAPI** — the new host-only
  `kindi intel-cl [--n 1048576] [--dim 128] [--repeats 5] [--json]` compiles a float32 vector-add
  and square GEMM with the driver's own OpenCL C compiler and checks them against a host reference
  that must match bit for bit. On the Windows A770 both kernels pass with zero mismatches:
  1,048,576 adds in 6.5 ms and a 128x128 GEMM at 31.5 GFLOP/s (5 launches, wall clock), with 512
  compute units and driver 32.0.101.9175. This is M2's kernel/GEMM half measured through OpenCL,
  not a SYCL backend: there is no `libksycl.so`, Kindi device selection, model, engine or support
  claim (REQ-0020).
- **The A770 decodes Kindi's packed ternary weights in a kernel** — the new host-only
  `kindi intel-ternary [--rows 64] [--cols 1024] [--repeats 5] [--json]` builds a matrix in the
  pack file's layout (4 trits per byte, `byte = t0 + 3 t1 + 9 t2 + 27 t3`, stored `[column/4][row]`;
  one fp16 scale per 128 weights), uploads those buffers unchanged and runs an OpenCL C GEMV on
  the A770. A host reference that decodes the same bytes matches bit for bit: 0 mismatches at
  64x1024, 128x4096 and 512x4096, with 5 launches in 0.944/2.470/2.437 ms and 0.694/2.123/8.604
  GFLOP/s wall. The probe covers the on-disk pack layout, not the engine's 5-trit lane repack;
  there is no activation quantization, SYCL path, backend, model, engine or support claim
  (REQ-0020).
- **Intel Arc A770 has an executable Level Zero memory path** — the new host-only
  `kindi intel-smoke [--mib 16] [--device 0] [--json]` creates a Level Zero context on the
  installed A770, allocates device and shared USM, copies a known pattern through a command
  list/queue and verifies the round trip. On the Windows rig a 16 MiB run passes with zero
  mismatches in 11.6 ms. The copy group is selected by its advertised COPY capability after the
  measured driver ignored copy commands on group 0; `zeMemCopy` is absent from loader 1.33.1, so
  copies use a command list. There is no kernel, GEMM, model, engine or support claim (REQ-0020).
- **Looped GGUFs now have a reusable execution plan and coverage check** — `kindi/looped.py`
  returns the full logical layer order with the stored block, phase and loop pass for each step,
  and reports stored blocks that are missing from or unexpected in the GGUF tensor index. The new
  host-only `kindi loop-info MODEL.gguf [--json]` prints the plan and coverage without loading
  tensor data, and a changed `block_count` is refused with its measured family and counts instead
  of the generic architecture error. This is map validation only; there is still no looped loader,
  pack, catalog entry or support claim (REQ-0031).
- **The first structured request no longer pays the vocabulary build** — `kindi serve` starts
  `token_classes.prewarm` in a daemon thread after the engine loads, building the process-shared
  index, trie, character sets and id sets for every served structured-output path: dense, Gemma 4,
  and (2026-11-23) Lightning and Nanbeige through the shared `kindi/grammar_prewarm.py` mixin
  (`KINDI_PREWARM_GRAMMAR=0` opts out). On the 2B the build is 2,475 ms and the first mask falls
  from 2,350 ms to 20 ms; repeat masks are 2.5 ms. The same change keeps the REQ-0027 gate at
  15/124 ms/token for `json_schema`/unrolled GBNF at 22 tokens and 3/20 at 200. A4/A5 GPU exactness remains unmeasured; the CPU server's first structured request measured 0.99 s
  with the prewarm ready against 3.60 s with it off (2026-11-23), with no GPU server TTFT. Evidence:
  `docs/benchmarks/2026-11-22-i7-8086k-grammar-prewarm/RESULTS.md`.
- **The dense tandem portfolio is comparable before a model is loaded** — `kindi plan --portfolio`
  prices full GPU, `cpu-first` and `gpu-first` in one dry run with the planner's own byte estimate,
  labelled against free VRAM and with a capacity-first recommendation (full GPU wins whenever it
  fits; no throughput is predicted). `kindi autotune` now also measures `tandem-gpu-first`, because
  the two directions own different output heads and one cannot stand in for the other.
- **DSV4.1 request shapes are validated before token counting or generation** — the served
  `kindi/dsv41_server.py` path now rejects a malformed message/tool/stream/stop shape with an
  OpenAI-style 400 before `count()` or any model work, and the error envelope is pinned by
  app-level tests.
- **Windows secret-baseline scanning no longer trips the command-line limit** — detect-secrets is
  invoked in bounded argv batches and the results are merged, so a large tracked tree no longer
  fails with `WinError 206`; a regression test pins the batching and merge.
- **Mode A's default now follows its measured winning draft length** — explicit `KINDI_SPEC=mtp`
  defaulted to `KINDI_SPEC_K=4`, the setting that cost 0.66-0.89x on the 2B / RTX 4060 Ti; the
  measured k=1 row is 1.12-1.13x at 512 and 24K and a 1.01x wash at 8K. `spec_begin` now defaults
  mode A to k=1 while the model drafter modes keep k=4, and an explicit `KINDI_SPEC_K` still wins.
  Pinned by `tests/test_verify_spec.py`; no output token changes because verify decides every token.
- **The Lightning Q2_0 CPU expert lane has an exactness gate** — `tools/moe_expert_lane.py` (an
  experiment, moved out of `kindi/` so no unused production module ships) computes one routed expert
  with the CUDA kernel's own int32 run sums, fused multiply-add placement and XOR tree; against the
  real `lt_mv_q2`/`lt_mv_q2_down` at hidden 2688 / width 1856 it has **0 differing elements** on
  both matrices with device-resident or host-mapped weights. It is a gate, not a backend: the CPU
  expert costs 2.395 ms, against 0.783 ms for the copy lane (3.06x) and 0.662 ms for the engine's
  direct host-mapped GPU read (3.62x faster than the CPU lane; 1.18x faster than the copy lane), the
  lane is not wired into `lightning.Engine`, and the model row is blocked by the absent Lightning
  pack. `tests/test_moe_expert_lane.py` pins the host arithmetic.
- **The mixed card/CPU Q2_0 expert layer has a bit-exact combine contract** — `Q2Matrix.partials`
  returns the host experts' pre-tree lane partials, `lt_mv_q2_down_parts` and `lt_q2_shared_parts`
  return the card side's, and `lt_q2_combine` performs `lt_mv_q2_down`'s per-lane gateway sum and
  five-step XOR tree. `tools/q2_combine_lane.py` mixes two host and two card experts plus the
  shared expert on synthetic Q2_0 weights and gets **0 differing elements** against one
  `lt_mv_q2_down` launch, with zero and non-zero residuals and the shared-only branch; host
  partials match the parts kernel. It is a gate, not Engine wiring: no Lightning pack exists here,
  so no model verify or tok/s is claimed.
- **The DSV4.1 CLI rejects unreadable paths, invalid counts and malformed prompt/reference content
  before loading a model** — `python -m kindi.dsv41` accepted `--prompts`, `--ids`, `--out`, `--ref`,
  `--dspark` and `--dump` values unchecked until after the model was mapped, and parsed `--free1`
  inside the `numa` branch. Argparse types now require readable input files, parse every `--prompts`
  JSONL line as an object with an integer `id` and a string `text`, parse every `--ref`
  `GEN tok:margin ...` line with token ids in `0..129279` and finite margins, require a writable
  output parent, a complete DSpark directory, token ids in `0..129279`, a finite non-negative
  `--free1`, a non-negative `-n`, positive `--tmax`/`--gen`/`--spec` values and a finite `--conf`;
  commands that need `--prompts` or `--ref` fail with their required set before any model load, and
  `verify`'s default gate files are parsed by the same validators. `tests/test_dsv41_cli.py` pins the
  helpers and the pre-load exits, including malformed JSONL and continuation content.
- **The other standalone CLIs validate late user values before loading** — `python -m
  kindi.dsv41_server` now parses `--tmax`, `--port`, `--cache-gb`, `--ckpt-min`, `--slots`,
  `--temperature`, `--repeat-stop`, `--turn-min`, `--max-tokens` and `--dspark` with bounded argparse
  types instead of bare `int`/`float` or a path opened after the model. `python -m
  kindi.dsv41_dspark` now uses argparse (`convert`/`check`) and requires the three source shards
  before `Shards` maps anything, instead of indexing raw `sys.argv`. `tools/mtp_context_bench.py`
  parses `--ctx`, `--tokens`, `--runs` and comma-separated `--positions` before building the engine,
  where positions used to be split after the engine existed. `tools/spec_ab.py` validates `--modes`
  and `--ctx` at argparse time, and `tools/measure_split_ratio.py` validates its internal probe
  counts before building a split. The gate and harness scripts (`check_long_context`,
  `check_perf_baseline`, `dsv41_hotplan`, `grammar_cost_serve`, `verify_matrix`, `verify_p106`) now
  reject non-finite, negative or zero values for their numeric options at parse time; values already
  validated before loading keep the same valid sets. Tests pin the parsers and the pre-load exits.
- **Python lint and type checking now ship as a measured ratchet** — `tools/check_python_quality.py`
  runs Ruff 0.16.0 with correctness rules `F`, `E9`, `B` and `I` over the Python tree, mypy 2.4.0,
  Pyright 1.1.414 (`basic`, all platforms) and Pylint 4.1.3 (`E`, `F`, import-error disabled) over
  first-party production code (`kindi`, `tools`, `examples` and `bridge/kindi_bridge`). The legacy
  baseline is **332 Ruff findings / 59 mypy errors / 1,035 Pyright errors / 255 Pylint findings**,
  stored per file and rule/error code in `tools/python_quality_baseline.json`; the gate fails only
  on a new or increased finding, so existing debt does not block work. CI runs it,
  `tests/test_python_quality.py` pins the comparison and parser shape, and `tests/test_ci_config.py`
  pins the step and exact tool versions. This closes only the lint/type half of audit M6;
  dependency scanning remains open.
- **The full CPU gate was rerun after the Usage/plan edits** — the first pass found the generated
  configuration reference stale: **1 failed, 1,230 passed, 28 skipped** in 246.53 s. After
  regenerating `docs/CONFIGURATION-REFERENCE.md`, the rerun is **1,231 passed, 28 skipped, 0 failed**
  in 243.91 s at **46.2186%** line+branch coverage (23,467 statements, 8,386 branches). This
  supersedes the older 1,196-pass / 45.4% figure, which predates these edits and also had a stale
  reference in its first pass.
- **The cold-expert decoder keeps its pinned slots across steps** — the decode hit test asked the
  host device-slot table (`lightning.downer`) whether an expert was "already warm", but that table
  is written by `lt_cache_dec`, which skips every cold expert (`hidx[e] < 0`). The hit set could
  never contain a cold expert, so every routed cold expert was read and re-pointed on every token.
  A per-layer slot table now follows `cold_experts.cold_slot_plan`: a repeated expert is served
  from its unchanged slot, a miss takes a free slot or the least-recently-used slot whose owner is
  not routed in that step, and a routed expert's slot is never stolen (so its pointer cannot go
  stale). The prompt path already uses its own `tcp`/`tsp` staging tables, so it does not touch the
  decode table. `cold_stats()` reports `slot_hits` beside `experts_read`. No GPU run, larger-than-RAM
  pack or tok/s claim; the per-layer stream sync and blocking current-layer read stay open (C9).
- **The standalone DSV4.1 server caps request bodies** — `kindi/dsv41_server.py` read
  `Content-Length` with no limit on every endpoint, unlike `kindi/server.py`'s 16 MiB default. It
  now shares the `KINDI_MAX_BODY_MB` setting (16 MiB default), refuses an over-cap body with HTTP
  413 `request_too_large` before reading it, and returns 400 for a malformed length or JSON (the
  warm endpoint used to answer a parse failure with 500). `tests/test_dsv41_server.py` pins the
  helper, the exact-cap boundary and both handler responses. A body error also closes the HTTP/1.1
  connection: the refused body was not read, so reusing the socket would parse its bytes as the
  next request. The close now half-closes the response direction and drains a bounded prefix of the
  refused bytes before the final close, so the 413/400 cannot be lost to a TCP reset; a socket-level
  test sends a 1 MiB body against a small test cap followed by a valid request and pins one 413,
  `Connection: close`, and no second response.
- **The optional Bridge component has a runnable smoke suite and CI job** — `bridge/tests` now
  covers duration/time parsing, the SQLite settings/agent store, agent status and the documented
  jobs parser with only the standard library. `bridge/README.md` no longer claims the full
  UI/PTY/RPC suite is in this source distribution, and `ci.yml` runs the smoke suite.
- **Machine roots are gone from current-facing docs and benchmark evidence** — the forward-slash
  `D:/`/`L:/` roots that the backslash-only secret rule missed are redacted to portable placeholders,
  including the tracked 27B median logs. A new current-tree guard catches forward-slash drive roots
  while still allowing the repository's `T:/kindi` form; the historical secret baseline is unchanged.
- **Every workflow's actions are SHA-pinned** — the existing CI guard now checks every
  `.github/workflows/*.yml`, not only `ci.yml`, so a GPU or Blackwell workflow cannot add a mutable
  action unnoticed.
- **A direct 27B speculation A/B compares outputs byte for byte** — `tools/spec_ab.py` builds one
  27B tensor-core engine with the 2B drafter and its head, then generates the same prompts plain
  and under `ngram`, `draft2b` and `both`, comparing token ids and decoded bytes without consulting
  a reference verdict. The 2026-11-18 run is **12/12 prompt/mode pairs identical** (4 prompts x
  3 modes, 40 greedy tokens each); `docs/MTP.md` links the command and hashes.
- **The server warns at startup when `KINDI_SPEC` has no tensor-core path** — `KINDI_SPEC` without
  `KINDI_TC=1` used to start silently until `/health` was queried. `create_app` now logs
  `warning: speculation off: ...` with the same tensor-core-path reason `/health` reports, built by
  one helper. A parameterized test captures the startup line and compares it with `/health` for
  `mtp` and `draft2b`.
- **`drafting.last` names the speculation mode that produced its drafts** — the last-request block
  now includes `method`, recorded from the engine's per-request effective `spec_mode` when
  generation ends. A client can attribute `passes`/`drafted`/`accepted` without borrowing the
  sibling `drafting.method`, which describes the current effective state rather than necessarily
  the mode of the last request. `/health` docs are updated, and a focused server test pins a last
  block whose method differs from the sibling field.
- **An opt-in GPU smoke test covers the 2B MTP server path** — `tools/mtp_smoke.py` boots the
  real `kindi serve` process with the registered `kindi-mtp-2b-v1.npz` head, sends one greedy
  completion, and fails unless `/health` says `drafting.method=mtp`, both `drafting.last.drafted`
  and `kindi_spec_drafted_tokens_total` are above zero, and the accepted counters are consistent.
  It uses a temporary state directory, an empty config and a free loopback port, and is not
  collected by the CPU-only pytest suite. `docs/MTP.md` documents the command and defaults.
- **The cold-expert reader has a real lookahead prefetch queue** — the old `read_ahead` knob only
  merged adjacent ranges. `ColdExpertReader` now names that behavior `coalesce`, and `prefetch`
  reads exact ranges into a bounded queue before demand; a later `read_many` consumes those bytes
  without a second file read. The Lightning decode path queues the next cold MoE layer's most-used
  cold experts in `_cold_prefetch`. A test prefetches two ranges, then proves the demand read
  returns the same bytes with the reader's read count unchanged; another pins the next-layer
  queue selection. Checks are in `tests/test_cold_experts.py`.
- **`kindi plan` and `kindi autotune` reuse a saved `kindi io` report** — `kindi io --json FILE`
  writes the probe report, and `--io-report FILE` (or `KINDI_IO_REPORT`) feeds it to either command
  through the same normalizer as a fresh probe, so the 64 MiB disk probe is not repeated. A missing,
  unreadable or rate-less report is named in a log line and falls back to one fresh probe; with no
  report selected, the command says how to save one. Tests cover save/reload, the fallback and the
  plan/autotune wiring. This reuses measurements, not another host's numbers: the file is the one
  `kindi io` wrote on this machine.
- **`kindi autotune` now tests an explicit cold-tier host cache size** — Lightning packs with
  measured rates get `expert-cache-<MiB>` beside the decode-slot candidates. It sets both engine
  settings the cache needs (`KINDI_LT_COLD=1` and `KINDI_LT_HOST_MB`) and sizes the budget from the
  same rate-sized working set as the slots (`n_used` plus the RAM/disk/PCIe lookahead, times the
  MoE layer count and expert bytes), capped by the host tier's experts and by the host's currently
  available RAM after the same 10% / 1 GiB headroom as the default host budget. Tests cover the env
  pair, both caps and the missing-rate/no-host-tier cases. No Lightning pack is available here, so
  this is an arithmetic candidate, not a measured cache optimum.
- **`kindi bench` carries the Lightning plan's slot count as a portable flag** — a rate-sized
  `dcache` was shown in the report but the printed follow-up prefixed `KINDI_LT_DCACHE=<planned>`,
  which cmd.exe and PowerShell do not parse as an environment assignment and the child engine
  therefore missed. `kindi bench` now accepts `--dcache N`, applies it as the engine's slot budget
  for that run and restores the caller's own `KINDI_LT_DCACHE` afterwards. `kindi plan` emits
  `... bench --dcache <planned>` (also for an explicit `--dcache`), and tests pin the flag, the
  engine's view of it and the absence of the POSIX prefix.
- **The Lightning plan's follow-up bench command carries its planned context** — the fit and slot
  count are computed at `ctx`, but the command omitted it, so a configured default context could
  produce a different run. The command now passes the global option as
  `... --ctx <planned> bench --dcache <planned>`; a plan test pins the value and that it precedes
  the `bench` subcommand, where the CLI accepts it.
- **The Lightning plan reports the cold host-tier split** — when `KINDI_LT_COLD` is enabled,
  `kindi plan` now runs the engine's own `cold_experts.split_host_tiers` over the planned host set
  and budget. The `host ram` line names the pinned host experts and the file-backed overflow with
  their bytes and budget, the `experts` line no longer calls the whole remainder pinned, and the
  report's `host_tier` object carries the same counts for JSON consumers. A test enables a 1 MiB
  budget on the fake pack and checks the pinned/file-backed counts, the displayed line and the
  byte totals.
- **The kernel-arity check reaches the DSpine graph** — the checker now resolves the 31
  `kf["..."]` launches from the literal iterable that builds `self.kf`, and `self.k` /
  `self.k8 if n <= 8 else self.k16` from their `get_function(...)` assignments. The resolver now
  follows only table assignments rooted at `self`, and every branch of a conditional kernel must
  accept the call's argument count instead of the union of the branches' declared counts; two tests
  pin a foreign table and branches with different arities. DSpine moves from 4/39 measured to 37/39;
  across the four graph builders the coverage is 105 measured / 37 pinned
  (spec 13, dspine 2, mtp 2, tp 20 unresolvable), with every measured site still mutation-proven.
  Conditional **argument** tuples get the same treatment: an `if`/`else` tuple contributes every
  branch's count, so a 6-or-7 tuple against a 7-parameter kernel fails on the 6 branch instead of
  passing on the larger one; `test_every_argument_branch_must_match_the_kernel` pins that case.
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
- **The 2B verify-matrix baseline includes the four speculation cells** — a 2B run with
  `--modes plain,tc,tc-hp --spec-modes ngram,mtp,draft2b,both` saved seven passing cells
  (`tc-ngram`, `tc-mtp`, `tc-draft2b`, `tc-both` each 4/4; 45/120, 94/204, 120/120 and 100/189
  drafts accepted) at `bench-logs/verify-matrix/2b-spec-baseline.json`, which is gitignored by
  design. The repeat `--fail-on-regression` run passed with no status, exactness, coverage or
  throughput regression; the exact commands are in `docs/CARD-SUPPORT.md` and `docs/CI.md`.
- **The 27B drafter speculation cells are a tracked verify-matrix baseline** —
  `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-27b-drafter/verify-baseline.json`
  freezes `tc-ngram`, `tc-draft2b` and `tc-both` (all pass, 3 exact + 1 tie;
  three-pass medians 45.7/71.0/62.0 tok/s) and the exact
  `--baseline ... --fail-on-regression --max-tps-drop-pct 10` command is in that
  page and `docs/CARD-SUPPORT.md`. The 2026-11-18 gate re-run after switching the
  baseline from the first pass to those medians passed with exactness unchanged
  and throughput +9.0/+6.6/+7.9% (49.8/75.7/66.9 tok/s); a missing cell, a lost
  exact prompt or a drop over 10% exits 1. The repeat passes ranged
  44.5-46.7, 70.7-71.1 and 61.3-62.3; the first pass's 49.8/75.7/67.1 values are
  superseded.
- **`kindi io` measures this host's memory tiers** — a RAM copy, an unbuffered disk read and pinned
  PCIe copies both ways, as text or `--json` (`--mib`, `--disk-dir`, `--skip`). The disk probe writes
  a temporary file and reads it with `O_DIRECT` (Linux) or `FILE_FLAG_NO_BUFFERING` (Windows) so it
  measures the device instead of the page cache, and says `buffered` rather than pretending when the
  OS refuses; the PCIe probe uses pinned memory like the engine's staging buffers and reports H2D and
  D2H apart. It loads no model. Measured here on 2026-11-20 with `--mib 64`: 13,266.4 MiB/s RAM,
  2,547.5 MiB/s disk, 6.0/6.1 GiB/s H2D/D2H on the 4060 Ti. The rates feed slot sizing; the
  NVMe cold-expert tier now has its first engine path, whose gate is in `docs/ROADMAP.md`
  "Memory-tier follow-ups".
- **The Lightning cold-expert tier is opt-in and reads only the host overflow, not a page cache** —
  with `KINDI_LT_COLD=1`, the hottest host experts stay pinned under `KINDI_LT_HOST_MB` (default:
  available host RAM less 10% or 1 GiB, whichever is larger) and only the overflow is file-backed.
  The split uses measured expert use when `kindi/lightning-experts.json` has counts, so the hot
  tier is not lost. A prompt stages a layer's host block through a pinned buffer; a decode step
  reads only routed cold experts that are not already in a warm device slot. `kindi/cold_experts.py`
  issues aligned `O_DIRECT`/`FILE_FLAG_NO_BUFFERING` reads where available, coalesces neighbouring
  ranges into one aligned read, and separates requested expert bytes from bytes actually
  transferred; `Engine.cold_stats()` exposes requested/transferred bytes plus exact pinned and
  disk tier bytes. The decode graph is disabled only on this opt-in path because the read follows
  the router, and warm slots take no file read. CPU tests cover the flag, the host/disk split and
  byte accounting, exact bytes, coalescing, prefetch, order, warm-slot skips and short reads. **No larger-than-RAM pack, GPU run, tok/s or verify result is claimed**; those are the
  open gate in `docs/ROADMAP.md`.
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
  pack at ctx 8192: 97.4-97.6 tok/s decode, 3,050 MiB sampled peak; the two tandem candidates were
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

### Fixed

- **Gemma 4 gives back the host RAM its uploads and prompt cache pin ([upstream #44](https://github.com/ahtx/kindi/issues/44))** — the weight upload (`TensorFile.gpu`/`load_resident`) copied with `cupy.asarray`, which stages each tensor in a pinned host buffer and leaves it in CuPy's pool, and `G4Cache` parked conversations in `cupyx.empty_pinned` without returning the free blocks. The upload now uses `empty` + `set` like `engine._upload` and frees the pool after the resident load; parking and restoring a conversation calls `free_all_blocks`, which releases only free blocks. `tests/test_gemma4_tensorfile.py` and `tests/test_gemma4_cache.py` pin both paths without a GPU; the upstream GTX 1070 measurement is not re-run here.
- **Cold-tier readers now have a real teardown path** — `lightning.Engine` created a
  `ColdExpertReader` and a daemon lookahead thread but never closed either, so a process exit could
  kill the thread mid-read and leave the file handle to the OS. `Engine.close()` now joins the
  lookahead and closes the reader once; the registry adapter, `lightning_cli` verify/bench and the
  HTTP server's `kindi.close()`/`serve()` teardown call it. The reader also serializes prefetch and
  demand reads with an internal lock, so a future consumer that forgets an external join cannot
  corrupt the queue or share a Windows file pointer. Tests pin the idempotent close, the lock path
  and the adapter/server teardown; no GPU, model or throughput claim.
- **The Python quality baseline cannot be lowered to hide a new finding** — `--update` now requires
  `--reason`, refuses any new or increased file/rule count, and records the reason, measured commit
  and per-tool delta in `tools/python_quality_baseline.json`. The CI gate's message names the new
  form, and the reviewed debt burn-down still lowers the baseline one family at a time.
- **`--save-baseline` refuses provenance for a different measurement** — when it carried a
  baseline's `logs` list forward, it did not require the new row's `log` to be one of those files,
  so a fresh single pass could inherit a three-pass list it was not part of. It now refuses
  (`SystemExit`, no write) unless the row's `log` is in the carried list; an explicit new `logs`
  list must also contain that `log`, and a missing `log` refuses. Tests cover both carried sources,
  explicit acceptance and refusal, and an unchanged target on refusal.
- **The cold-expert lookahead read now overlaps the current layer's GPU work** — Lightning's
  `_cold_prefetch` read the next cold layer's spans synchronously before the current layer's
  kernels, so the lookahead added file latency instead of hiding it. It now starts the read on a
  daemon thread; `/health`/`cold_stats`, prompt staging and the next cold layer's demand read join
  it before touching the queue, and a prefetch error is re-raised on the caller's thread. The reader
  queue and its byte accounting are unchanged. A new engine-level test queues a span and calls
  `_cold_prepare` without a manual join, proving the demand read consumes both queued spans with no
  second file read. No tok/s, GPU or larger-than-RAM run is claimed; C9/R1 stay open.
- **`kindi verify`/`bench` now pass the configured Nanbeige K/V cache through the engine gateway** —
  `make_engine` did not put `kv_cache` in the adapter config, so a pack served by `kindi serve`
  used the configured cache while verify/bench's registry path fell back to the engine default. The
  gateway now forwards `kv_cache` from verify, bench and calibrate, and
  `tests/test_served_engines.py` records the engine constructor from both paths for the Nanbeige,
  Lightning and Gemma 4 serve families, so a wrapper cannot drift to another class or argument again.
- **The Nanbeige gateway path is explicitly non-drafting** — the registry adapter and
  `nanbeige_cli` (the verify/bench builder) load the base engine only. `mtp_head` and `lookup`
  are the serve wrapper's draft sources, so a head that the adapter's one-token-per-step
  `generate()` never drafts with is no longer loaded there, and `make_engine` no longer takes it.
  `supports_speculation` stays False and now matches the code and tests: the adapter's done event
  reports `drafted=0`/`accepted=0`, verify/bench do not pass `mtp_head`, and
  `tests/test_served_engines.py` pins that only serve attaches the head. C2's claim is narrowed to
  the base engine the paths share; the missing-head warning for serve is unchanged.
- **`kindi plan`'s follow-up commands survive a spaced path** — the printed `bench` and `verify`
  lines formatted the interpreter and pack with `%s`, so `C:\Program Files\Python 313\python.exe`
  or `/mnt/my packs/bonsai` split into several arguments when pasted. The `bench` builders and the
  `verify` builder now quote the argv for the host shell (`shlex.join` on POSIX, cmd.exe's
  `subprocess.list2cmdline` rules on Windows), leaving the flags unchanged. A focused test parses the
  POSIX forms back with `shlex`, pins the Windows double quotes, and runs the printed `verify` lines
  through the host shells; PowerShell still needs its `&` call operator when the quoted executable
  comes first.
- **`kindi serve` refuses backends it cannot drive instead of building a dense engine** — a
  DeepSeek V4.1 pack now exits with the documented `python -m kindi.dsv41_server` command, Cascade
  keeps its converter-only message, and an external `KINDI_BACKEND_MODULES` family is refused
  because the HTTP generation loop has no contract for it. The server's built-in family list is one
  table, and a gateway test pins a registry adapter for every served family.
- **Expert usage counts are validated before they steer an expert tier** — `Usage` now refuses a
  count row that is not a full `n_exp` list of finite, non-negative numbers, and a bad order id
  (out of range, duplicate, non-integer). A partial map stays valid, but `split_host_tiers` reports
  `counts_used`/`counted_layers` and the Lightning engine logs when it had to fall back to the
  usage order. An explicit plan file is strict; the bundled file measured on another expert count
  is ignored rather than indexing a short list.
- **The autotune cold-cache candidate uses the engine's own fit** — `expert-cache-<MiB>` now sizes
  slots from the full expert count and the plan's room (including staging buffers and decode
  slots), then caps the cache by the host tier. A tight room lowers the proposal instead of naming
  a slot count the planner would refuse.
- **`kindi caps` no longer reports FlashAttention-2 as a hardware fact** — the feature table is
  silicon-only again; FA2 is a library algorithm and does not belong beside tensor cores and FP8.
- **The documentation/support gates run in CI** — `check_claims.py`, `project_memory --check`,
  `config_reference --check` and `support_matrix --check` are now one CPU-job step, pinned by
  `tests/test_ci_config.py`.
- **Per-module coverage floors now cover 46 modules** — the historical 2026-11-18 full-suite run
  (1,196 passed, 28 skipped, 45.4% total; its first pass had the then-stale generated reference)
  added floors for the planner, MoE policy, grammar and
  chat parsers, CLIs, the new cold/I/O/autotune modules, and low CPU-reachable floors for the
  graph builders (`spec.py`, `dspine.py`, `mtp.py`, `tp.py`); `engine.py` moves from 19% to 20%.
  The floors sit 10 points below the measured values, and `moe_cpu.py` stays out until its C
  library is built in CI.
- **The installer docs lead with the checksum-verified file** — the stale
  `v0.5.0-installer` one-liners are gone; Linux downloads, verifies, inspects and runs the 0.6.0
  script, and Windows downloads `install.ps1` for review instead of piping it into `iex`.
- **The standalone DSV4.1 CLIs reject malformed comma values at argparse time** —
  `python -m kindi.dsv41 --ctx` requires one or more positive integers (e.g. `2048,8192`), its
  `--auto` and the server's `--cal` require exactly two finite `a,b` numbers, and the server's
  `--ctx` requires a positive integer. `--ctx` used to be converted after parsing (a bare
  `ValueError`), and the calibration pairs were split into floats when the engine started; all four
  now exit 2 before a model or native library is touched, naming the valid set. `tests/test_dsv41_cli.py`
  pins the parsers and both CLI rejection paths.
- **`python -m kindi.dsv41 --range` validates the prompt-id range at argparse time** —
  `--range` now requires `a-b` with two non-negative integers and `a <= b` (e.g. `0-9`); a
  malformed, reversed or single-id value exits 2 before the model or the prompt file is touched,
  where `generate` and `specbench` used to split the string bare and raise `ValueError` after the
  model loaded. `tests/test_dsv41_cli.py` pins the parser and the CLI rejection path.
- **The 27B median evidence no longer carries machine paths** — the three tracked median JSONs
  kept the local Python path and the `~/.kindi` pack/head paths from their run commands. They now
  record `<python>/python.exe` and `~/.kindi/...`, and the history baseline records the seven
  `local drive path` hits per file from the commit that added them (reviewed with
  `check_secret_history.py --all-hits`; no credential was involved).
- **`/health` explains MTP requested without the tensor-core path** — `KINDI_SPEC=mtp` with
  `KINDI_TC=0` reported `requested: "mtp", enabled: false, reason: null`, because `spec.install()`
  never patches `spec_begin` onto the engine without the tensor-core path. `drafting()` now names
  the missing switch when the engine has no speculation path, and
  `tests/test_server.py::test_health_explains_mtp_needs_the_tensor_core_path` pins it with an engine
  that has no `spec_begin`.
- **Comma-separated CLI values now reject unknown tokens with their valid set** — `kindi bench`
  `--split-at` used to drop zero/negative lengths and end a run with a bare `ValueError` for text;
  it now rejects anything that is not a positive integer (or `none`/`off`) before the model loads.
  `--tandem-cpu-layers` now rejects malformed forms (`auto:lots`, `gdn:many`, a bad list entry) at
  the parser with the accepted count/list/`gdn:N`/`auto[:MiB]`/`rate[:FLOOR]` forms, while its
  existing count, subset and GDN-range errors keep naming their valid range. Unit and CLI tests
  cover both; `kindi io --skip` remains the reference for the error style.

### Changed

- **The live comparison is re-based at `53198e5`** — official Kindi `v0.6.1` (`7fececb`) remains
  the latest release and upstream `main` has no post-release commit. The fork is now 664 commits /
  827 files ahead (`+106,882/−1,320`), with 929 tracked files, 28 numbered requirements and zero
  file deletions; `CURRENT-VS-KINDI-0.7.md` and `docs/claims/upstream-delta.json` carry the figures.

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
- **`CURRENT-VS-KINDI-0.7.md` §41's delta figures were 7 commits stale, and the harness now gates
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
  0.66x recorded on 2026-11-17 was at the then-default `KINDI_SPEC_K=4`. Re-measured at depth on the 2B /
  RTX 4060 Ti with one harness (`tools/mtp_context_bench.py`: same engine, plain and mode A
  interleaved, prompt reading excluded): k=1 gives 1.12x at 512 tokens and 1.13x at 24K (4 runs a
  side, 72% acceptance), a wash at 8K (1.01x), while the then-default k=4 still loses (0.89x / 0.66x /
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

## 0.7.0 (2026-11-22)

0.7.0 adds Qwen3.8-Flash-Next, a 125B mixture-of-experts model that runs on one 8 GB card. It also adds AMD RDNA2
support, native Windows with a tray monitor, several requests at once and card placement. Every number is from the
default install. Each comparison uses the same card, model file and context.

### New

- **Qwen3.8-Flash-Next** (about 125B parameters, about 6B used for each token;
  [docs/FLASHNEXT.md](docs/FLASHNEXT.md)). `install.sh --model flashnext` installs ISTA-DASLab's GSQ-RCO IQ3_S file.
  The dense layers, the output head and the most used experts run on the card. The other experts run on the CPU from
  host RAM, and the card's and the CPU's expert kernels give the same bits. Most experts stay in host RAM, so about
  96 GB of RAM or more is recommended. On hosts with less than about 128 GB, set `KINDI_QXG_PLE_LOCK=0`. The server is
  Kindi's OpenAI server with streaming, thinking, tool calls and prompt reuse. `/health` reports the experts on the
  card and the card hit rate. The installer fetches the MTP head from Qwen's release by byte range and converts it on
  your computer (Qwen Community License 1.0; `--no-mtp` leaves it out). `kindi verify` checks the exact path against
  a reference run of the same file (174 of 174 tokens).
  - Speed on one RTX 3060 12 GB with two Xeon E5-2673 v4: it writes 39.4 tokens/s (three runs, new chats). It reads
    489 tokens/s at 4K and 465 at 16K.
  - Three speed features are on by default. Fast mode skips the lowest-ranked CPU expert picks; `KINDI_FAST=0` gives
    exact output, and kbench gives 62 of 65 tasks in both modes. Drafting with the MTP head can differ from plain
    decoding at near-ties below compute capability 8.0; `KINDI_FN_DRAFT=0` gives exact output. The dense layers use
    8-bit integer inputs; `KINDI_QXG_IDENSE=0` gives the float path (kbench: 62 of 65 against 61 of 65).
  - Maxwell and Pascal cards work from compute capability 5.0. Where a card has no `dp4a`, the kernels use exact byte
    products. Kindi loads a cubin for the card, with a PTX fallback. Below compute capability 7.5 the installer picks
    a CUDA 12.x PyTorch build.
  - A Quadro M4000 8 GB (Maxwell) as the only card, with the same CPUs, writes 13.1 tokens/s and reads 4K at 54.7.
    llama.cpp on the same card and file writes 5.4 and reads 4K at 34.2.
- **AMD RDNA2** (gfx1030 to gfx1036; #28). The int8 dot product uses gfx103x's signed dot4, which is exact. The HIP
  kernels compile for gfx1030, and the plan engine runs there. On a Radeon PRO W6800 32 GB, `kindi verify` passes on
  the Ternary Bonsai 2B, 9B and 27B. At 8,192 tokens of context the 27B reads 506 tokens/s and writes 38.8.
  llama.cpp with the same file and q8_0 K/V reads 263 and writes 33.4.
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md) has all the rows.
- **Kindi for Windows.** `kindi.exe` runs natively with its own Python, CuPy and NVRTC. It needs no WSL and no
  `CUDA_PATH`. Kindi Monitor is a tray app with a dashboard window. It shows write and read speed, prompt progress,
  cache hits, context, slots, card memory, utilisation, power, temperature, clocks, host RAM and CPU. It also starts,
  stops and restarts the servers on the PC and selects the model. We tested it on Windows 11 with an RTX 3080, and
  `kindi verify` passed on the 2B and the 27B from the package. `/health` has `slots`, `last` and the card's id.
  Native Kindi loads `nvcuda.dll` itself (#4).
- **More than one request at a time** (`kindi serve --slots N`, config `slots`; #22). The plan engine with the
  tensor-core plans keeps N conversations on one card. Each has its own K/V cache, recurrent state and prompt cache.
  One pass writes one token for every writing conversation, so they share the weight reads. Prompts are read one
  chunk at a time between the passes. Greedy output is the same as when a conversation runs alone (checked token for
  token on the 2B and the 27B). The default stays one request at a time.
  - On an RTX 3080 with `--slots 8`, one pass writes 396 tokens/s in all with the 27B (one slot: 70). It writes 2,852
    with the 2B (one slot: 481).
  - Through the server with `--slots 8`, the 27B finishes 16 requests of 256 tokens at 134 tokens/s in all. The
    default gives 52. One request alone runs as fast as without slots.
  - It works on one card with `KINDI_TC=1` for now ([docs/CONFIGURATION.md](docs/CONFIGURATION.md)).
- **VRAM auto-split on two cards** (`KINDI_TP_SPLIT`, default `auto`). Sometimes the halves of the 27B leave less room
  for the context on one card. Then the other card takes a larger part of the FFN and the output head, so both cards
  keep the same room. On two RTX 3080s with 5 GB taken on one, the context grows from 75,008 to 166,656 tokens
  (tensor cores: 66,816 to 147,968). Cards that fit the context with the halves keep them.
- **Card inventory, placement and one front door** ([docs/INSTALL.md](docs/INSTALL.md)). `kindi cards` lists every
  card with its compute capability, memory, free memory, PCIe link, bandwidth and driver. It says whether Kindi can
  use the card, or why not. `kindi plan` places models on the cards and writes `~/.kindi/plan.json`, so no card is in
  two servers. It uses the fastest card as DeepSeek's main card. It splits a model over two cards by free memory when
  one card is too small. It puts DeepSeek's hot experts on all spare VRAM. `install.sh --place "dsv41:0+" "27b:3"`
  and `--plan FILE` install a placement. With two or more servers, `kindi route` is one OpenAI endpoint. It sends
  each request to the server of its `model`.
- **DeepSeek V4.1 Flash: streamed experts** (`KINDI_DSV_EXPERTS=stream`, the installer's nvme profile). Kindi reads
  only the routed experts of each call from the model file. It uses direct I/O with many reads in flight and a RAM
  expert cache of fixed size. The cache pins the most used experts and prefetches the predicted experts of the next
  two layers. The results are the same bits as the mmap mode. We have not measured speeds on NVMe drives yet
  ([docs/DEEPSEEK.md](docs/DEEPSEEK.md#streamed-experts)).
- **DeepSeek V4.1 Flash: edits far back in a conversation.** The engine copies the window ring to host RAM every
  1,024 positions (about 11 MB each). A request that changes the conversation more than one prompt chunk back
  resumes at the nearest copy. Before, it read the whole prompt again
  ([docs/DEEPSEEK.md](docs/DEEPSEEK.md#edits-far-back-in-a-conversation)). On a 24,386-token conversation with
  message 6 edited, the read takes 228.2 s instead of 256.1 s, and the answers are the same. The first build crashed
  in the ring save, and we fixed it (`tools/dsv41_ring_test.cu`).
- **`tool_choice` and `response_format`** (#38). Kindi now honours them or refuses them with a clear 400 error.
  `required` and a named function force a tool call on Ternary Bonsai. The other engines, `json_object` and
  `json_schema` return the 400. `KINDI_RESPONSE_FORMAT=ignore` gives the old behaviour.

### Fixed

- `install.sh` checks for `tmux` and prints the install commands. Before, a missing `tmux` ended it silently (#41).
- `install.sh --model` is not case sensitive. The CPU engine offers the 4B and the 9B with RAM checks, not only the
  2B (#45).
- Docker no longer runs out of memory with a long context (#40). Kindi counts the container's cgroup RAM limit and the
  parent limits. It does not count clean page cache as used. It skips a saved turn whose K/V copy does not fit in
  host RAM. (PR #48 by atimelessrage, plus our changes.)
- Gemma-4 gives pinned host memory back after the weight upload and when the prompt cache drops copies (#44).
- A tokenizer pack with a missing piece fails at load with a clear error (PR #49, atimelessrage).
- MTP mode A drafts one token by default (PR #72, atimelessrage). Hadamard-4096 packs use the exact tensor-core
  activations by default (the `tc_hp_for` default of PR #76, atimelessrage).
- A container started without `--gpus` ran the CPU engine without a word. It now prints a warning, and `/health` has
  `device`. DOCKER.md has recipes for LAN access (B11), Podman and NAS.
- Kindi Bridge says "For Kindi 0.7.0" and lists DeepSeek, Lightning and DSpark as supported. It takes a server for
  Kindi only when `/health` or `/v1/models` says so (B9).
- The DeepSeek server stops cleanly on SIGTERM and SIGINT. The request in progress finishes, then the exit handlers
  run (B17).
- A new tokenizer test covers Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, CJK, Korean and emoji. It found no
  fault (B13).
- Not fixed: B1 (the web UI sends the locale `en_US`). We found no locale code in Kindi or Bridge, and we wait for the
  reporter's request. B12 (pi did not connect) does not reproduce with the `models.json` provider. pi's built-in
  OpenAI provider sends to api.openai.com, as DOCKER.md says.
- A weight pack whose files change after `kindi convert` is now caught (B16). `kindi convert` records each file's size,
  a fast hash and its sha256 in `kindi_pack.json`; every start checks the sizes, `kindi verify` and every build of the
  5-trit cache check the sha256, and a mismatch names the file. A cache made from other weights is made again. On a
  test host a pack's `bcodes.bin` had changed in place, and every new cache made from it gave wrong tokens with no
  message. Packs made before 0.7.0 have no record and are used as before (`kindi convert --force` adds it).
- The 2B on tensor-core cards with its MTP head (the installer's profile for RTX 30 cards and later) stopped at start
  with `cuGraphAddKernelNode failed (1)` since 0.5.0: the head's fold launch missed one argument (B18). It starts and
  passes `kindi verify` again.
- Two cards (27B, tensor parallel) on cards with many SMs (an RTX 3080): with the 2-bit weights (`KINDI_G5=0`), the
  output head was split into column parts that were never added, so every token was 0 and `kindi verify` failed.
- The installer asks each NVIDIA card on its own. A card whose `nvidia-smi` query fails (exit 14, "infoROM is corrupted")
  gets a warning and its compute capability from the card name; the other cards stay. Before, one such card made the
  installer report no NVIDIA GPU and install for the CPU.
- The DeepSeek installer (`tools/install_dsv41.sh`) picks the ram profile from 496 GiB of RAM, the measured need (the
  466 GiB model in the page cache, the server's 11 GiB and 18 GiB for the system). Before, it asked for 560 GiB, so a
  512 GB build got the nvme profile. It now prints the RAM it found and the RAM the profile needs.
- Blocking sync is off by default with the 580 driver (CUDA 13.0) and later, where it cost 5 to 15% of the write
  speed (#30). On an RTX 3080 with the 580 driver, the 2B writes about 11% faster. `KINDI_BLOCKING_SYNC=1` or `0`
  still decides. The installer's tensor-core profile keeps blocking sync on, as before.
- On native Windows, Kindi loads the CUDA driver from `nvcuda.dll` (#4). The prompt buffer of the 5-trit models is
  sized from the card's own weights, which gives back context that 0.4.0 took (#4). Not tested on native Windows yet.
- `kindi bench` logs the positions it measures and the ones that do not fit the context, and it measures the deepest
  position that fits. Before, positions past the context were dropped without a word (#33).
- `KINDI_P100_FP16_PROMPT=1` on a card whose cuBLAS cannot run the fp16 product (an RTX 3080 with CUDA 10.2) keeps
  the normal prompt path, with one warning. Before, a long prompt gave HTTP 500.
- Nemotron 3.5 Lightning reads `<think>`, `</think>`, `<tool_call>` and `<tool_response>` (and its closing tag) as
  one token each, as the model was trained (B19). Before, they were split into text pieces, as llama.cpp does with this
  GGUF. Prompts are 3 to 21 tokens shorter, and thinking replies follow the model's usual form. `KINDI_LT_ADDED_TAGS=0`
  restores the old split.
- A new test checks the tokenizer and the streamed text on Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, CJK,
  Korean and emoji. No fault was found.

- **AH-Nanbeige4.3-33B on Volta (Tesla V100, sm_70)**: a prompt path of its own (#25). The prompt products run on the
  fp16 tensor cores (mma m8n8k4) and give the int32 dots of the int8 tensor-core kernels, so the products are the same
  bits as on sm_80 and later cards; the prompt attention also runs on the tensor cores. On one V100 32 GB the 33B reads
  566 tokens/s at 512 tokens and 474 at 16K (before: 144 and 106), and on two cards 1,015 at 8K. `kindi verify` of the
  33B now passes on Volta; before, one prompt differed from llama.cpp at a close margin. `KINDI_NB_VOLTA=0` gives the
  old path ([docs/CARDS.md](docs/CARDS.md)).

### Thanks

- atimelessrage contributed four pull requests to 0.7.0. They add the container RAM limit (#48), the tokenizer pack
  check (#49), one MTP draft by default (#72) and exact tensor-core activations for Hadamard-4096 packs (#76).
- The reports and tests that shaped this release came from epscheel, JugglerMaster, vntgeai, rojecas, kidkal,
  scuderiapk, ggeorgovassilis, Immac, exolvo, Rares14324, maximhausleiter and PlagaMedicum.
- Thank you to the Qwen team at Alibaba Cloud for Qwen3.8-Flash-Next and its MTP head. Thank you to ISTA-DASLab for
  the GSQ-RCO IQ3_S GGUF that Kindi runs. The README Acknowledgements name the other model sources.
### Docs

- CARDS.md: why `KINDI_TC=1` can end `kindi verify` prompt 1 early (#31). Its 8-bit activations move the logits by up
  to 0.15, and llama.cpp's own top two at those tokens differ by 0.0064 and 0.0105, so it is a near-tie.
  `KINDI_TC_HP=1` passes all four prompts (on an RTX 3080 it writes about 4% slower).
- CARDS.md: the GeForce GTX 1650 SUPER, 4 GB (Turing, sm_75), a user report (PR #24).
- CARDS.md: the Tesla V100 32 GB (Volta, sm_70), measured on one and two cards, with llama.cpp on the same cards.

## 0.6.1 (2026-11-16)

### Fixed

- The 2B with its MTP head on tensor-core cards stopped at start with `cuGraphAddKernelNode failed (1)`, and
  `kindi verify` failed. A kernel that the MTP head uses got one more argument in 0.5.0, and the MTP head did not pass
  it. Affected: 0.5.0 and 0.6.0, the installer's setup "Ternary-Bonsai-2 2B" on an NVIDIA card of compute capability
  8.0 or later with 9.5 GB or more (RTX 3060 12 GB, 3080, 3090, 40 series, A100), chosen with `--model 2b`,
  `--config` or the setup list (the recommended setup on these cards is the 27B, which does not use the MTP head), and
  any setup that sets `KINDI_MTP_HEAD` or `KINDI_SPEC_DRAFT_HEAD` by hand. The 2B, 4B, 9B and 27B without an MTP head
  were not affected.
- When a CUDA graph cannot be built, the error names the node, its grid and block, and the number of arguments.

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
