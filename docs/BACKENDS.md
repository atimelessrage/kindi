# Backends

Kindi's inference backends sit behind one plugin contract, so adding a model is
a contained change: one backend module plus docs, with no edits to the server
or the CLI. This page covers the contract (T4.1). The adapter ports of the
existing engines (T4.2) are in progress — the dense Bonsai adapter
(`kindi/backends/bonsai.py`) is done and registered; the other families
(Nanbeige, Nemotron Lightning, DeepSeek V4.1) still take the legacy path.
The authoring checklist and `kindi backends` listing (T4.3) and the generic
fallback (T4.4) are tracked in [REQ-0019](development/requirements/REQ-0019.md).
Today the registry resolves the fake backend, the bonsai adapter and
externally registered backends, and `resolve` says so when asked for
anything else. There is no generic-GGUF claim yet: an arbitrary GGUF cannot
be served today.

## The contract

`kindi/backends/base.py` defines:

- `BackendCapabilities`: name, model ids, GGUF architectures, devices,
  max context, thinking/tools support, the tool-call family key
  (`tool_format`, used by the tool-call parsers), speculation and
  token-filter support, cache kind, max concurrent sequences, greedy
  determinism.
- `SamplingParams`: max tokens, temperature, top-p/top-k, min-p, stop
  strings, seed, and an optional `token_filter` (constrained decoding; the
  field is in the contract now so it does not change twice).
- `GenerationEvent`: one yielded item — a token, a reasoning piece, or the
  final `done` event with finish reason, timing and speculation stats.
- `CancelToken`: a flag polled during prompt processing and decode; a plain
  `threading.Event` satisfies it.
- `Backend`: lifecycle (`load`/`close`/`info`), text primitives (`tokenize`,
  `detokenize`, `apply_chat_template`), `generate`, `cache_stats`, and the
  optional hooks `park`, `reference_gguf`, `verify_prompts`.

The contract is dependency-free and importable on a CPU-only host, including
the Python 3.6 CUDA image. No CuPy import happens at module import time.

## Registry

`kindi/backends/registry.py`:

- `register_backend(factory, ...)` registers a backend class or zero-arg
  factory. Duplicate names are refused.
- `resolve(model_id_or_gguf_path, device=None, config=None)` returns a
  backend instance. A `.gguf` path is matched by its `general.architecture`
  metadata (read with the existing `kindi/gguf.py` reader); a model id is
  matched against declared model ids; `device` filters by declared device
  names. When nothing matches it raises `BackendNotFound` naming what is
  registered.
- External backends load through `KINDI_BACKEND_MODULES=pkg.mod,pkg2.mod`.
  Importing the module registers the backend. Python entry points are
  deferred because `importlib.metadata` does not exist on Python 3.6.

## FakeBackend

`kindi/backends/fake.py` is a deterministic backend for tests: byte-level
tokenization, seeded pseudo-random generation or scripted output, controllable
per-step latency, and cancel support. It implements the whole contract with
no GPU, no weights and no tokenizer files, so the API surface can be tested
on a CPU-only host. The contract tests (`tests/backends/test_contract.py`)
run over the fake backend and auto-extend to every registered backend.

## Inspecting registered backends

`kindi backends` prints one line per registered backend, including external
backends loaded from `KINDI_BACKEND_MODULES`; it is host-only and does not
load a pack or touch the GPU. `kindi backends --json` prints the same registry
rows as JSON for scripting.

## Authoring a backend

1. Subclass `kindi.backends.Backend` and set `capabilities`.
2. Implement lifecycle, text primitives, `generate` and `cache_stats`.
3. Call `register_backend` when the module is imported.
4. Add a contract test and verify with `kindi backends --json`.

This page is the authoring guide for the contract. The Bonsai engine ships as
an adapter (T4.2, first family); the other engine families are not ported yet,
so a third-party backend serving one of those should declare the model ids or
GGUF architectures it serves instead of relying on the built-in engine paths.

## Status and next steps

- T4.1 (this page): contract, registry, FakeBackend. Done.
- T4.2: port the existing engines (Bonsai, Nanbeige, Nemotron Lightning,
  DeepSeek V4.1) behind the contract as thin adapters, with
  `KINDI_LEGACY_ENGINE_PATH=1` as the safety valve for one release. The Bonsai
  adapter (`kindi/backends/bonsai.py`) is done: dense Bonsai packs resolve
  through the registry in `kindi.engine.make_engine` (valve-gated), and
  `kindi verify` 2B on the RTX 4060 Ti is 4/4 on both the adapter path and
  the valve path. The gate was extended the same day to the other packs on
  the host: 4B PASS 3+tie on both paths (the known prompt-3 llama.cpp
  tie), 9B ALL PASS 4/4 on both paths, and adapter bench rows at ctx
  24576 agree with the valve within ~1% — evidence in
  `docs/benchmarks/2026-11-15-i7-8086k-rtx4060ti-t42-bonsai-4b-9b/RESULTS.md`.
  The gate was closed for the 27B the next day (2026-11-16): the pack
  (at `T:/kindi-packs/bonsai-27b`) names itself "Hf" in the manifest, so
  resolution needed a `source_file` fallback — 27B verify is 4/4 on both
  paths and bench agrees within ~1% (see
  `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-t42-bonsai-27b/RESULTS.md`).
  The Nanbeige adapter (`kindi/backends/nanbeige.py`, registered `nanbeige-33b`)
  is ported 2026-11-16, but its GPU gate is blocked: the catalog's ahtx GGUF is
  corrupt as served by Hugging Face (SHA mismatch, garbled tensor infos), so
  no verify/bench parity is claimed; the owner action is in
  `docs/MANUAL-HELP-NEEDED.md`. Nemotron Lightning and DeepSeek V4.1 are not
  ported yet (no packs on the verification host), so their packs still take
  the legacy path.
- T4.3: `kindi backends` listing and this authoring guide. Done 2026-11-15.
- T4.4: generic fallback as a llama.cpp proxy backend. Needs the maintainer
  decision on positioning (REQ-0019). Not done.
- T4.5: native generic backend. Design document only. Not done.

Tracked in [REQ-0019](development/requirements/REQ-0019.md) (experimental).
