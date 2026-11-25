# Engineering knowledge

Durable operational facts that do not belong in a single decision, requirement or user
guide. Keep commands copy-pasteable and avoid local machine paths and credentials.

## Model and card facts worth not re-deriving (2026-11-25)

- **Bonsai 2.5 27B is `qwen35` with `full_attention_interval = 4`**: 64 blocks, 16 full-attention and
  48 Gated-DeltaNet, width 5120, ffn 17408, 24 Q / 4 KV heads at 256 dims, partial RoPE 64, mRoPE
  sections `[11,11,10,0]` (text-degenerate), base 1e7, eps 1e-6, vocabulary 248320, context 262144.
  The per-layer tensor lists are 11 (full) and 14 (GDN); `docs/BONSAI-2-SUPPORT.md` section 1a has
  the table, the file sizes and the CPU run.
- **The `-mtp` file's `blk.64` is a full-attention block with 15 tensors, eight of them Q4_0 (about
  239 MB)**, borrowing `token_embd` and the folded `output.weight`; it needs a K/V ring and no GDN
  state. `docs/MTP.md` has the details, and the note that a CPU speedup or 83-88% acceptance figure
  circulating for it is not ours.
- **A vision projector is a separate `mmproj-*.gguf` of about 600-930 MB**; `kindi mmproj` prices one
  and holds it in host RAM (`mmproj_place = ram`, the default) so it never takes VRAM from the
  language model. No build implements the encoder yet, so image requests are still refused with a
  400; `docs/ROADMAP.md` has the V0-V4 ladder.
- **The Arc A770 reads at 385 GB/s in practice (69% of the 560 GB/s spec) and its launches cost
  about 8.8 us**; the shipped per-matrix GEMV pattern loses 48% of a token to ramp and drain between
  401 small matrices, and one persistent launch at `parts=1` is 1.76x. `docs/INTEL-ARC.md` has the
  table; zero-filled weights are up to 2.4x faster than random ones (memory compression), so never
  measure bandwidth on zeros.
- **`kindi verify` has no reference for 2.5** (it exits 2 rather than compare against Bonsai 2's
  tokens); the 2.5 CPU evidence is load and generate, not token-exactness against a reference.
- **The pack is 7,242 MB from a 6 GB GGUF** (codes 6.717 GB, scales 0.42 GB, f32 0.105 GB), and the
  CPU engine loads it in about 340 s here at 4.02 tok/s.

## Verification layers

| layer | command | what it proves |
|---|---|---|
| Parse and unit gate | `python -m compileall -q kindi && python -m pytest tests/ -q` | Python modules parse; CPU-only behavior and guards hold. |
| CPU build/smoke | `KINDI_CPU_SMOKE=1 KINDI_CPU_CACHE=<cache> python -m pytest tests/test_cpu_build.py tests/test_cpu_threads.py tests/test_cpu_smoke.py -q` | The native CPU library builds and loads (a compiler is required). |
| CPU exactness | `KINDI_CPU_PACK=<pack> python -m pytest tests/test_cpu_layers.py -q` | Per-layer CPU execution matches the reference. |
| GPU exactness | `python -m kindi --pack <pack> --ctx 1024 verify` | Greedy output matches the reference for the selected engine. |
| Generated docs | `python -m kindi.config_reference --check`, `python -m kindi.support_matrix --check`, `python docs/benchmarks/build_cross_mode.py --check` | Generated references match source. |
| Project memory | `python tools/project_memory.py --check` | Requirement history is current. |
| Docs links | `python -m pytest tests/test_docs_links.py -q` | Local Markdown links resolve. |

## Configuration and generated files

- Precedence is built-in defaults < config file < environment < command-line flags.
- `docs/CONFIGURATION-REFERENCE.md` is generated from `kindi/config.py` and the
  `KINDI_*` environment reads. Regenerate it after adding or moving a key.
- `docs/SUPPORT-MATRIX.md` is generated from the model catalog, conversion formats, K/V
  modes and CPU ISA checks. Regenerate it after a support change.
- `docs/benchmarks/CROSS-MODE.md`/`.csv` are generated from measured benchmark files.
- Generated files are checked by tests; a green unit suite does not excuse a stale
  generated reference if you have not run the generator.

## Support boundaries worth remembering

- The exact head and `kindi verify` are the exactness reference; benchmark throughput does
  not replace correctness.
- The 9B's Hadamard block is 4096. A **single card** folds it in 512-point chunks with the
  high-precision activations (`kindi/tp.py`, `KINDI_TC_HP=1` selected automatically; measured
  4/4 exact on the 4060 Ti, 2026-11-18). A **two-card tensor-parallel split** still refuses
  anything but Hadamard 1024 (`tp.py::build_plans`), so 4096 stays single-card (or plain-engine)
  only. The older "the 9B plan is refused" note was the pre-`KINDI_TC_HP` behaviour.
- The 4B tensor-core path uses the standard decode attention because split flash-decode
  diverges on its 2560-wide plan. This is a conditional pass, not a claim about split
  decode on that geometry.
- The 27B plain engine refuses grouped head dimensions it does not support. A refusal is a
  known boundary, not a benchmark failure.
- `k8v4` is experimental, opt-in and quarantined; it is not token-exact (2B verify 2/4 against
  the fp16 reference). See the requirement and decision records before touching it. It keys on
  the head size, not the model name (`spec.py::kv_plan`; the value kernels cover head_dim 256),
  and the plain engine refuses it outright (`KINDI_KV=k8v4` needs the plan engine).
- The 27B GGUF published as `Ternary-Bonsai-2-27B-PTQ1_0.gguf` carries `general.name = "Hf"` and
  `general.basename = "folded"`, so `kindi info`/`kindi plan` print **Hf** as the model name - it is
  faithful metadata, not a conversion defect. `verify` still finds its reference, because
  `reference_for` reads the size from `source_file` (`-27B-` -> `reference-27b.json`). The pack is
  64 layers, width 5120, head_dim 256, Hadamard block **1024** (the tensor-core-friendly case),
  ~7,242 MB, and a full convert takes ~6 minutes on an i7-8086K.
- CPU support requires AVX and chooses one of five explicit ISA tiers (AVX2/FMA/F16C,
  AVX2/F16C, AVX2/FMA, AVX2, AVX-only). Forced-tier exactness on an AVX2 host is not a
  native Sandy Bridge/no-FMA hardware claim.
- The dense Bonsai CPU+GPU tandem runtime is implemented for the 2B/9B in `cpu-first`
  and `gpu-first` placements, with 27B also gated after generic GQA support; the
  Nanbeige host-layer path and MoE/Gemma tandem remain separate.

## Compilers and static analysis on a Windows host

Exact install locations are machine data, so they are not written here (see the note
at the top of this file); the local record with the paths and the versions this host
was verified against is the gitignored host-tools sheet under `bench-logs/`.

- **clang builds the CPU engine.** `kindi/cpu.py:_find_cc` resolves `KINDI_CC`, then
  `PATH`, then a standard LLVM install, so a clang under `Program Files` is used without
  any setup; `kindi caps` prints the compiler it found. The build needs `-mavx2 -mfma
  -mf16c` (the source refuses to compile without AVX).
- **`clang --analyze` is the cheap bug finder for `kindi/kcpu.c`.** Run it with the same
  ISA flags, or the file's own `#error` and `__m256` errors hide the real findings. It
  currently reports two warnings and both are understood: an "assigned value is garbage"
  at the attention rotation that is a false positive (`rms_rows(..., 1, HD)` writes exactly
  `HD` elements; the analyzer does not follow the store through the callee's loop-bound
  parameter), and one dead store of `R` that is left in place on purpose so the engine's
  source hash, and therefore its 4/4 CPU verify evidence, stays valid. Re-run it after any
  change to the C engine and treat a third finding with the same care.
- **Cppcheck 2.22.0 covers the C engine too**, in C mode:
  `--enable=warning,performance,portability --std=c11 --language=c
  --suppress=missingIncludeSystem`. It is clean on `kindi/kcpu.c` today. It cannot see the
  OpenCL kernels, which live in Python string literals (`kindi/intel_gemm.py`, `intel_ternary.py`,
  `intel_cl.py`); those are checked by running the probes against the host reference instead.
- **Intel oneAPI (SYCL/DPC++) is the REQ-0020 M2 toolchain.** `sycl-ls` enumerates the
  backends. It finds the Arc A770 **through Level Zero** only when the oneAPI `bin` directories
  are on `PATH`: the Unified Runtime adapter is loaded from there, so running the installed
  `sycl-ls.exe` by absolute path with a bare `PATH` lists OpenCL devices and no Level Zero
  device. `kindi/intel.py` therefore locates the install (`KINDI_ONEAPI_ROOT`, then
  `ONEAPI_ROOT`, then the default root) and runs the installed `sycl-ls` with those directories
  prepended; `KINDI_ONEAPI_ROOT=<root> python -m kindi intel-info` then prints the L0 device.
  Current `sycl-ls` prints `[backend:kind][backend:index]` per device while older builds printed
  `[backend:kind:index]`; the parser accepts both. `icx-cl` drives MSVC and needs its environment
  (`vcvars64.bat` or a Developer PowerShell); `icpx`/`dpcpp` do not, and `icpx -fsycl` builds the
  smoke after `vcvars64.bat`. `python -m kindi intel-sycl` runs the exact vector-add smoke on
  Level Zero; the engine still has no SYCL backend, so no backend or support claim exists.
- **A CUDA toolkit installer for the NVIDIA route is kept on the host but not installed**, and
  no CUDA device is visible (see `docs/MANUAL-HELP-NEEDED.md` §4), so nothing NVIDIA-side can
  be measured from here.

- **The generic fallback backend is a llama.cpp proxy (REQ-0019 T4.4).**
  `kindi/backends/llamacpp.py` (registered `llamacpp`) is the registry wildcard:
  it declares no model ids or architectures, so it is chosen only for models
  nothing more specific serves. It forwards generation to a running
  `llama-server`'s `/completion`; point it with `KINDI_LLAMACPP_BASE_URL`
  (default `http://127.0.0.1:8080`) and `KINDI_LLAMACPP_API_KEY`. Reachability
  is probed at `load()` and recorded in `info()`; the missing-server error is
  raised by `generate()` with the `llama-server -m <model.gguf>` remedy. It
  renders ChatML itself (llama.cpp's own chat template is bypassed), makes one
  non-streaming request, and cannot do constrained decoding (a `token_filter`
  raises). `tests/backends/test_llamacpp.py` covers it with an injectable
  opener; a live server is the one unrun gate.
- **A registered backend is not a servable one, and the gap is four numbers.**
  `kindi serve` builds `server.Kindi`, which is an engine face (ctx, tokenizer,
  `Runner` prompt cache, sampler), not the Backend contract. A probe
  (2026-11-25) mounted a proxy-shaped object on `create_app` and got 200 from
  `/health`, `/v1/models`, `/metrics`, chat and SSE with the existing 400 for
  structured output - so routing is not what blocks a gateway. What blocks it
  is that a proxy would have to invent `eng.ctx` (advertised as
  `context_length`), `check_prompt` (used by `dry_run`, `/tokenize`,
  Anthropic `input_tokens`, overflow; the proxy's ids are bytes), streaming
  (the backend posts `stream: false` and replays a finished reply, so SSE's
  first frame waits for the whole reply and cancel does not abort upstream)
  and prompt-cache gauges (`Kindi.runner`). Fix the backend first - upstream
  SSE, a `/props` n_ctx probe, `timings` into usage - then a gateway can be
  honest; do not fake a `Runner`.
- **`Backend.capabilities` is declared, not `None`.** The base contract carries
  `capabilities: "BackendCapabilities"` as an annotation only, so a subclass
  assigns its own object without a mypy override error and
  `registry._capabilities_for` still gives an unannotated backend a default
  (it uses `getattr`). Initializing it to `None` was what produced the same
  `assignment` finding in six adapters.
- **`tools/grammar_cost.py` reads the mask cache's miss counter.** The grammar
  caches moved from dicts to the bounded `StateMaskCache` LRU, so `state in
  cache` no longer works; `walk()` marks a step as a hit when
  `cache.misses` did not change. `tests/test_grammar_cost.py` pins it with a
  tiny vocabulary and the real cache class.

- **A constrained mask miss is three costs, not one.** `TokenClassTable.allowed` first asks the
  grammar for safe character sets (`_token_safe_sets`), then either takes the broad-safe slice or
  walks the token trie (`_trie_allowed`). Until 2026-11-24 the broad path also paid two
  whole-vocabulary `np.unique` sorts (`_outside_ids_for` via `np.setdiff1d`) and a 248,320-element
  Python scan (`_safe_ids_for`); both were waste, and both are now linear or slice-based.
- **Inside the trie walk the cost is the automaton, not the trie.** Profiled 2026-11-25: a schema
  mask miss makes ~8,300 `jsongrammar._step` calls, ~8,200 of them one per distinct token-start
  character (91% rejected), with only two distinct states per miss and 761 nodes for the
  end-of-document `trail` state; the id sort and `_chars_under` are rounding error. So the cost
  model is "one Python transition per distinct first character", and the useful lever is to skip the
  characters that state cannot accept: `JsonGrammar._accepted_chars` returns a *superset* of them
  from the fields `_step` reads (schema stack for a key, surviving terminals for a keyword, target
  node for a value slot, fixed sets for the structural modes), and the walk then looks up only those
  characters in a node's children — 8,303 -> 38 steps, median miss 4.923 -> 0.119 ms (41.37x), with
  every mask equal to the full-vocabulary sweep (2026-11-25). Sharing one `character -> next state`
  memo across states is what the first attempt did, and it was withdrawn: the cached states embed the
  schema stack, so a warm table answered with another state's stack and returned wrong masks, and a
  first-visit-only oracle cannot see it — compare *after* warming. A superset hook is safe because an
  extra character only costs a step; a shared cache of states is not, because a wrong state changes
  the answer.
- **The server's constrained overhead is mask + pick, and they are comparable.** An instrumented
  CPU server (2026-11-24) attributed a 24-token structured reply to `allowed` at 8.8 ms/call and
  `Sampler._logits` at 4.4 ms/call before the refactor; after it, 6.3 and 3.0. `_logits` is only
  wasteful when `ylog_map` is None: it then builds `arange(vocab)` for ids that are the array
  positions, which `_masked_logits` avoids. A request with a frequency/presence penalty needs the
  ids back, so it still takes `_logits`.
- **Measure the arms at the same reply length.** Comparing a 24-token structured reply with an
  8-token unconstrained one overstates the constrained overhead (the per-token cost is not linear
  at the start of a reply). `docs/benchmarks/2026-11-24-i7-8086k-cpu-mask-fast/measure.py` posts
  both shapes with `max_tokens=8` and interleaves the runs; the unconstrained arm doubles as the
  host-noise control.

## Git archaeology

```sh
git log --follow --oneline -- docs/development/requirements/REQ-0001.md
git show <commit>:docs/development/requirements/REQ-0001.md
git log -S'REQ-0001' --all -- docs/development/requirements
git show <deletion-commit>^:docs/development/requirements/REQ-0001.md
```

The generated history records 12-character abbreviations. Use the full hash from
`git log` when a command needs it. Never rewrite a requirement ID to hide a reversal; add
a new decision or mark the old one superseded.

## External documentation

- User-facing behavior: `README.md`, `docs/*.md`, `examples/`.
- Agent-facing API usage: `docs/AGENTS.md`.
- Installation and release: `docs/INSTALL.md`, `docs/RELEASE-CHECKLIST.md`.
- Support claims: `docs/SUPPORT-MATRIX.md` and the per-card pages.
- Internal process: this directory.

A user-visible behavior change is incomplete until the external page and examples are
updated. Run the docs-link test after moving or renaming a page.
