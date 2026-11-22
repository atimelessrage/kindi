# Kindi architecture

This is the maintainer's map of the current system. It is deliberately shorter than the
algorithm explanation in [`../HOW-IT-WORKS.md`](../HOW-IT-WORKS.md); that page explains
the kernels, while this page records boundaries and invariants. The generated
[`../SUPPORT-MATRIX.md`](../SUPPORT-MATRIX.md) is the authority for what is currently
supported.

## System map

| component | location | responsibility |
|---|---|---|
| CLI and configuration | `kindi/__main__.py`, `kindi/config.py` | Parse commands and options, apply precedence, select the pack and device. |
| Conversion and packs | `kindi/convert.py`, `kindi/layout.py` | Convert GGUF tensors into the pack format; preserve tokenizer, scales and manifests. |
| Plain GPU engine | `kindi/engine.py`, `kindi/chunk.py`, `kindi/spec.py` | Stream weights, run chunked prompt reads, decode and sampling. |
| Tensor-parallel / plan engine | `kindi/tp.py`, `kindi/spec.py`, `kindi/tc*.cu`, `kindi/tcf.cu` | Split a model over cards and run the tensor-core plans and guards. |
| CPU engine | `kindi/cpu.py`, `kindi/kcpu.c` | Five-tier x86-64 dispatch (AVX2/FMA/F16C, AVX2/F16C, AVX2/FMA, AVX2, AVX-only) with OpenMP and a native Windows build path. |
| CPU+GPU tandem | `kindi/tandem.py` | Splits dense Bonsai layers across CPU and GPU in `cpu-first`/`gpu-first` placements, with a joint prompt cache. |
| Cache and prefix reuse | `kindi/cache.py`, `kindi/chunk.py` | Recurrent-state checkpoints, parked conversations and saved openings. |
| HTTP boundary | `kindi/server.py` | OpenAI/Ollama-compatible endpoints, validation, API key, bounded queue, metrics. |
| HTTP auth | `kindi/api/auth.py` | One constant-time bearer check shared by the OpenAI and DeepSeek servers (DEC-0010). |
| Looped-model gate | `kindi/looped.py` | Recognizes `sensenova_loop` and `inkling2`, maps their logical layers to the stored blocks, and refuses them with REQ-0031 (no loader yet). |
| Installer | `install.sh` | Hardware classification, safe host edits, model download, service setup. |
| Generated references | `kindi/config_reference.py`, `kindi/support_matrix.py`, `docs/benchmarks/build_cross_mode.py` | Render docs from code or measured tables and fail on drift. |
| Project memory | `tools/project_memory.py`, `docs/development/` | Requirements, architecture, decisions, lessons, knowledge and git history. |

## Runtime flow

```text
GGUF file
  -> convert/layout: pack (bcodes, dpm, f32, tokenizer, manifest)
  -> engine selection: plain GPU / plan tensor-core / CPU
  -> prompt read: chunked products, attention and recurrent state
  -> decode: token loop, sampling, optional draft/verify
  -> server: OpenAI / Ollama response, queue accounting, metrics
```

The server loads one engine and serves one request at a time. Additional requests wait in
a bounded queue; beyond `KINDI_QUEUE_MAX` the server returns HTTP 429 with `Retry-After`.
Configuration precedence is built-in default < config file < environment < CLI flag.

## Invariants

- **Greedy exactness is a product promise.** With the exact head, greedy output must match
  llama.cpp for the same GGUF. A change that breaks a verified path needs a recorded
  decision, not just a test edit.
- **The pack is a lossless re-layout.** Conversion may change layout and metadata, not
  model values or tokenization.
- **Cache keys include the engine and pack state.** A change to a kernel, layout,
  quantization mode or supported tensor set must invalidate incompatible caches.
- **Tensor-core guards are explicit.** The plan engine supports Hadamard blocks 512 and
  1024. The 9B's 4096-point block is refused before graph construction; the 4B's
  2560-wide split-decode path falls back to the standard decode attention. Do not turn a
  guard into a silent approximation.
- **The CPU engine is opt-in per device and ISA-checked.** It selects an explicit AVX2
  tier or the AVX-only fallback, and the default thread count is one per physical
  core, not every logical CPU.
- **The server boundary fails closed.** Body limits, request validation, API key checks
  and queue caps are part of the API contract.
- **Generated documents are code.** A source change that alters a generated row requires
  regenerating the file; `--check` must pass before handoff.
- **No credentials or local home paths in tracked files.** `tests/test_no_secrets.py`
  enforces the common cases.

## Requirement map

| requirement | main components |
|---|---|
| [REQ-0001](requirements/REQ-0001.md) | `kindi/verify.py`, exact head, tests |
| [REQ-0002](requirements/REQ-0002.md) | `kindi/convert.py`, `kindi/catalog.py` |
| [REQ-0003](requirements/REQ-0003.md) | `kindi/tp.py`, `kindi/spec.py`, tensor-core kernels |
| [REQ-0004](requirements/REQ-0004.md) | `kindi/cpu.py`, `kindi/kcpu.c` |
| [REQ-0005](requirements/REQ-0005.md) | `kindi/server.py` |
| [REQ-0006](requirements/REQ-0006.md) | `kindi/cache.py`, `kindi/chunk.py` |
| [REQ-0007](requirements/REQ-0007.md) | `kindi/sampling.py`, `kindi/profiles.py` |
| [REQ-0008](requirements/REQ-0008.md) | `install.sh` |
| [REQ-0009](requirements/REQ-0009.md) | `kindi/config.py`, `kindi/config_reference.py` |
| [REQ-0010](requirements/REQ-0010.md) | `docs/`, generated references, link tests |
| [REQ-0011](requirements/REQ-0011.md) | tracked-file secret scan and server auth |
| [REQ-0012](requirements/REQ-0012.md) | `KINDI_KV=k8v4` opt-in path |

## Changing the architecture

When a component boundary, data flow or invariant changes:

1. Update this page and the affected requirement's **Acceptance** and **Evidence**.
2. Record the choice and alternatives in [`DECISIONS.md`](DECISIONS.md).
3. Update the user-facing explanation in `docs/` and examples where behavior is visible.
4. Regenerate generated references and run the project-memory and docs checks.
5. Record any hardware-dependent gap explicitly; do not turn an unverified path into a
   support claim.
