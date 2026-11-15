# Backends

Kindi's inference backends sit behind one plugin contract, so adding a model is
a contained change: one backend module plus docs, with no edits to the server
or the CLI. This page covers the contract (T4.1). The adapter ports of the
existing engines (T4.2), the authoring checklist (T4.3) and the generic
fallback (T4.4) are roadmap items tracked in [REQ-0019](development/requirements/REQ-0019.md);
until they land, only the fake backend and externally registered backends
resolve through the registry, and `resolve` says so when asked for anything
else. There is no generic-GGUF claim yet: an arbitrary GGUF cannot be served
today.

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

## Status and next steps

- T4.1 (this page): contract, registry, FakeBackend. Done.
- T4.2: port the existing engines (Bonsai, Nanbeige, Nemotron Lightning,
  DeepSeek V4.1) behind the contract as thin adapters, with
  `KINDI_LEGACY_ENGINE_PATH=1` as the safety valve for one release. Not done.
- T4.3: this page becomes the full authoring guide; `kindi backends` lists
  registered backends. Not done.
- T4.4: generic fallback as a llama.cpp proxy backend. Needs the maintainer
  decision on positioning (REQ-0019). Not done.
- T4.5: native generic backend. Design document only. Not done.

Tracked in [REQ-0019](development/requirements/REQ-0019.md) (experimental).
