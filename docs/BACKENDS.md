# Backends

Kindi's inference backends sit behind one plugin contract, so adding a model is
a contained change: one backend module plus docs, with no edits to the server
or the CLI. This page covers the contract (T4.1). The adapter ports of the
existing engines (T4.2) are ported — the dense Bonsai, Nanbeige, Nemotron
Lightning, Gemma 4 and DeepSeek V4.1 adapters (`kindi/backends/bonsai.py`,
`nanbeige.py`, `lightning.py`, `gemma4.py`, `dsv41.py`) are registered. Cascade 2
is converter/validator only.
The authoring checklist and `kindi backends` listing (T4.3) and the generic
fallback (T4.4) are tracked in [REQ-0019](development/requirements/REQ-0019.md).
Today the registry resolves the fake backend, those five adapters and
externally registered backends, and `resolve` says so when asked for
anything else. There is no generic-GGUF claim yet: an arbitrary GGUF cannot
be served today.

## The contract

`kindi/backends/base.py` defines:

- `BackendCapabilities`: name, model ids, GGUF architectures, devices,
  max context, thinking/tools support, the tool-call family key
  (`tool_format`, used by the tool-call parsers), speculation and
  token-filter and JSON-grammar support, cache kind, max concurrent sequences,
  greedy determinism. A backend that sets `supports_json_grammar` must treat
  `SamplingParams.token_filter` as the full constrained-decoding set: the
  allowed ids are sampled/argmaxed in full, not masked from a top-K or
  temperature-window candidate set that could lose a grammar-legal low-rank
  token.
- `SamplingParams`: max tokens, temperature, top-p/top-k, min-p, stop
  strings, seed, and an optional `token_filter` (constrained decoding; the
  field is in the contract now so it does not change twice).
- `token_filter(state)` receives `{"prompt_ids": [...], "generated": [...]}`
  with the **real** prompt ids: a filter may depend on what the prompt already
  contains (a grammar seeded from it, a "do not stop inside a fenced block"
  rule). All five adapters pass the prompt; three used to pass an empty list.
- Stop strings are matched with `base.StopScanner`, which scans only the text
  appended since the last check plus the longest stop's overlap. A stop that
  straddles two decoded pieces is still found, and a 24K-token reply no longer
  costs O(n^2) character comparisons. An over-context prompt raises
  `BackendUnavailable` naming prompt and context; it does not return an empty
  reply finished for length.
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
  names. When nothing matches — or when two backends match equally well — it
  raises `BackendNotFound` naming the candidates; registration order never
  decides between them.
- `kindi.engine.make_engine` (the engine gateway) asks each adapter's
  `model_id_for_manifest` hook, resolves the pack through the registry and
  logs the backend and model id it chose. It refuses instead of guessing when
  a manifest is claimed by two adapters, when a pack names a family no adapter
  serves (the dense legacy builder would build the wrong engine for it), or
  when a manifest has neither a `backend` nor the dense weight fields. A dense
  pack that no adapter claims — a placeholder `model`/`source_file`, the way
  the 27B once hid from its own adapter — is the one fallback that stays valid,
  and it is logged with the manifest field to check.
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

This page is the authoring guide for the contract. Every shipped engine family
(Bonsai, Nanbeige, Lightning, Gemma 4, DeepSeek V4.1) ships as an adapter
(T4.2); a third-party backend serving something else should declare the model
ids or GGUF architectures it serves instead of relying on an engine path.

## Status and next steps

- T4.1 (this page): contract, registry, FakeBackend. Done.
- T4.2: port the existing engines (Bonsai, Nanbeige, Nemotron Lightning,
  Gemma 4, DeepSeek V4.1) behind the contract as thin adapters, with
  `KINDI_LEGACY_ENGINE_PATH=1` as the safety valve for one release (it forces
  the dense builder, so it applies to the dense family; a pack that names a
  family with no legacy builder is refused rather than mis-built). The Bonsai
  adapter (`kindi/backends/bonsai.py`) is done: dense Bonsai packs resolve
  through the registry in `kindi.engine.make_engine` (valve-gated), and
  `kindi verify` 2B on the RTX 4060 Ti is 4/4 on both the adapter path and
  the valve path. The gate was extended the same day to the other packs on
  the host: 4B PASS 3+tie on both paths (the known prompt-3 llama.cpp
  tie), 9B ALL PASS 4/4 on both paths, and adapter bench rows at ctx
  24576 agree with the valve within ~1% on decode (the read row differs by 1.4%) — evidence in
  `docs/benchmarks/2026-11-15-i7-8086k-rtx4060ti-t42-bonsai-4b-9b/RESULTS.md`.
  The gate was closed for the 27B the next day (2026-11-16): the pack
  (at `~/.kindi/packs/bonsai-27b`) names itself "Hf" in the manifest, so
  resolution needed a `source_file` fallback — 27B verify is 4/4 on both
  paths and bench agrees within ~1% (see
  `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-t42-bonsai-27b/RESULTS.md`).
  The Nanbeige adapter (`kindi/backends/nanbeige.py`, registered `nanbeige-33b`)
  is ported 2026-11-16, but its GPU gate is blocked: the catalog's ahtx GGUF is
  corrupt as served by Hugging Face (SHA mismatch, garbled tensor infos), so
  no verify/bench parity is claimed; the owner action is in
  `docs/MANUAL-HELP-NEEDED.md`. The Lightning adapter
  (`kindi/backends/lightning.py`, registered `nemotron-lightning-30b`) is
  ported 2026-11-16 and now GPU-gated on the RTX 4060 Ti with a clean
  catalog download (size/SHA-256/GGUF all match, fresh pack): `kindi verify`
  is 5 token-exact + 7 llama.cpp ties, 0 fail, and `kindi bench` at ctx 8192
  measures read 788.3–821.3 tok/s and write 165.96–167.41 tok/s. Evidence:
  `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-lightning-30b/RESULTS.md`.
  The Gemma 4 adapter
  (`kindi/backends/gemma4.py`, registered `gemma4-e4b`) and the DeepSeek V4.1
  adapter (`kindi/backends/dsv41.py`, registered `deepseek-v41-flash`) were
  ported 2026-11-16 and are CPU-tested; both GPU gates are blocked because
  neither model is on this host (no verify/bench parity claimed). A DeepSeek
  V4.1 pack is written by hand — there is no converter: `kindi_pack.json` with
  `backend: kindi_dsv41`, the first GGUF shard as `model_path`, and optionally
  `encoding` pointing at the model release's `encoding.py` (the chat
  template). Cascade 2 stays converter/validator only.
- T4.3: `kindi backends` listing and this authoring guide. Done 2026-11-15.
- T4.4: generic fallback as a llama.cpp proxy backend. Needs the maintainer
  decision on positioning (REQ-0019). Not done.
- T4.5: native generic backend. Design document only. Not done.

Tracked in [REQ-0019](development/requirements/REQ-0019.md) (experimental).
