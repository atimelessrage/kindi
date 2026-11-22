# Repo map (T0)

Resolved 2026-11-15 for the backend-interface (Feature 4) and agent-grade API
(Feature 5) delegation plans. Every `[LOCATE]` in those plans is resolved in the
table at the end. This map is the contract for the remaining tasks; if it and
the code disagree, the code wins and this file must be updated.

## Runtime and interpreter floors

| surface | python | notes |
|---|---|---|
| CUDA image `docker/Dockerfile` | 3.6 (Ubuntu 18.04 `python3`) | CUDA 10.2, CuPy 9.6, FastAPI 0.79.1, pydantic 1.9.2. Shared code must import on 3.6. |
| ROCm image `docker/Dockerfile.rocm` | 3.12 (Ubuntu 24.04 `python3`) | ROCm 6.3, CuPy 14.2, FastAPI 0.115.6, pydantic v2. |
| CI `.github/workflows/ci.yml` | 3.11 | CPU suite, coverage floor, installer, static analysis, secret scan. |
| Windows native | system python (3.11 in CI) | CPU path only; see `docs/WINDOWS.md`. |

Known incompatibility with the 3.6 floor (recorded for decision D3, not fixed
here): `kindi/cascade.py`, `kindi/draft_policy.py`, `kindi/moe_compat.py` and
`kindi/sessions.py` use `from __future__ import annotations` (3.7+). They are
imported lazily, so the CUDA image boots, but a 3.6 interpreter cannot import
them. New code in `kindi/backends/` and `kindi/api/` avoids 3.7+ syntax
entirely.

CI exists and is not stubbed: `ci.yml` has cpu-tests (pytest + 27% coverage
floor + per-module floors + perf baseline), cpu-engine-smoke, windows-cpu,
installer-shell, static-analysis and secret-scan jobs, plus `gpu-smoke.yml` and
`blackwell-fp8.yml` as manual lanes. T4.1 therefore adds no new workflow; the
contract tests run in the existing cpu-tests job.

## Modules under `kindi/` by role

FROZEN means rule 5 of the delegation plan: GPU kernels, weight conversion,
model numerics and the sampler core. Adapters may call into these files but
must not change them; anything that seems to need a change stops for the
maintainer.

| module | role | frozen |
|---|---|---|
| `server.py` | FastAPI OpenAI-compatible server; `Kindi` engine facade; queue/scheduler; metrics | no |
| `runtime.py` | one effective runtime-configuration view for `/health`, the startup log, `kindi info` and `bench --json` | no |
| `__main__.py` | CLI (`convert verify bench serve chat info use ...`) | no |
| `dsv41_server.py` | DeepSeek V4.1 stdlib-HTTP server (separate from `server.py`) | no |
| `container.py` | container entrypoint (model fetch, pack build, serve) | no |
| `config.py`, `config_reference.py` | config defaults/layering; generated config reference | no |
| `catalog.py` | model catalog, `kindi use` selection | no |
| `profiles.py` | per-model sampling profiles | no |
| `support_matrix.py` | generated support matrix + card registry | no |
| `sessions.py` | chat session store (host-only) | no |
| `hostram.py`, `vram.py`, `gpudrv.py` | host-RAM preflight, sampled-peak VRAM, driver selection | no |
| `sched.py` | priority scheduler (chat vs background) | no |
| `cache.py` | prompt cache, checkpoints, parked conversations (`Runner`) | no |
| `gemma4_cache.py` | Gemma 4 prompt cache | no |
| `engine.py` | Bonsai engine (plain + plan), decode loop | **FROZEN (numerics)** |
| `tp.py` | tensor-parallel / plan engine | **FROZEN (numerics)** |
| `spec.py` | speculative decoding, kernel compilation | **FROZEN (numerics)** |
| `spec.cu`, `chunk*.cu`, `tc*.cu`, `tca*.cu`, `tcf.cu`, `tcl.cu`, `tcp.cu` | CUDA kernels | **FROZEN** |
| `kindi.cu`, `f16p.cu` | CUDA kernels | **FROZEN** |
| `chunk.py`, `chunk1.py`, `tc.py`, `tc_turing.py` | kernel wrappers / Turing kernels | **FROZEN** |
| `sampling.py` | sampler core (greedy path) | **FROZEN (sampler)** |
| `tandem.py` | CPU+GPU layer-split runtime | no |
| `cpu.py`, `kcpu.c` | native CPU engine + C kernels | **FROZEN (C kernels)** |
| `convert.py` | GGUF -> weight pack conversion | **FROZEN (conversion)** |
| `gguf.py` | minimal GGUF metadata/tensor reader | no (read-only use) |
| `layout.py`, `dtypes.py`, `moe.py`, `expert_cache.py` | weight layouts, dtypes, MoE placement/cache | no |
| `chat.py` | Bonsai chat template + tool-call parser | no |
| `tokenizer.py`, `spm.py` | Bonsai BPE tokenizer, SentencePiece tokenizer | no |
| `verify.py`, `reference*.json` | `kindi verify` + llama.cpp references | **FROZEN (references)** |
| `bench.py`, `kld.py`, `calib.py`, `calibrate.py`, `plan.py`, `cluster.py` | bench, KLD gate, calibration, planner, fast-head clusters | no |
| `draft_policy.py`, `lookup.py`, `mtp.py`, `dspine.py` | speculative drafting helpers | no |
| `nanbeige.py`, `nanbeige_gen.py`, `nanbeige_chat.py`, `nanbeige_cli.py`, `nanbeige_cpu.py` | Nanbeige engine + generation + template + verify + CPU fallback | kernels **FROZEN**; `_gen/_chat/_cli` no |
| `nbq*.cu`, `nbq_cpu.c` | Nanbeige CUDA/C kernels | **FROZEN** |
| `gemma4.py`, `gemma4_tp.py`, `gemma4_chat.py`, `gemma4_tools.py`, `gemma4_tokenizer.py` | Gemma 4 engine, TP engine, template, tool calls, tokenizer | kernels **FROZEN**; `_chat/_tools` no |
| `gemma4*.cu` | Gemma 4 CUDA kernels | **FROZEN** |
| `lightning.py`, `lightning_gen.py`, `lightning_chat.py`, `lightning_cli.py`, `lightning_tokenizer.py` | Nemotron Lightning engine + generation + template + verify + tokenizer | kernels **FROZEN**; `_gen/_chat/_cli` no |
| `lightning*.cu`, `lightning_k35.cu`, `lightning_tc.cu`, `moe_cpu.py`, `moe_cpu.c` | Lightning CUDA/C kernels | **FROZEN** |
| `dsv41.py`, `dsv41_dspark.py`, `dsv41_tokenizer.py` | DeepSeek engine, DSpark drafter, tokenizer | kernels **FROZEN**; tokenizer no |
| `dsv41_*.cu`, `dsv41_cpu.c` | DeepSeek CUDA/C kernels | **FROZEN** |
| `cascade.py`, `moe_compat.py` | Cascade 2 converter/validator; small-MoE recognition table | no |
| `backends/` | **new**: plugin contract, registry, fake backend (T4.1) | no |
| `api/` | **new**: OpenAI error envelope (T5.1) | no |

## Request trace: POST /v1/chat/completions

In `kindi/server.py`:

1. `create_app(cfg)` builds the FastAPI app and one `Kindi` instance via
   `_make_engine` on `GPU_THREAD`.
2. Middleware `_request_meta` assigns an `X-Request-ID` and the access-log meta.
3. The route handler reads the body with `_read_json_body` (16 MiB cap), then
   `_validate_request` turns a malformed request into a 400 `RequestError`
   before anything touches the engine or the prompt cache.
4. `logprobs` is rejected (400) unless the backend supports it (Gemma 4).
5. `K.check_prompt(req)` renders and tokenizes the prompt with no GPU and no
   lock, raising `ContextOverflow` (400 `context_length_exceeded`) before the
   request waits for the engine. `dry_run` returns the exact prompt count here.
6. Admission: `RateLimiter`, then the priority scheduler (`sched.SCHED`) and
   `Gate` (`KINDI_QUEUE_MAX`); a full queue is a 429 with `Retry-After`.
7. The engine runs on `GPU_THREAD` (`K.generate`), relayed as SSE when
   `stream=true` with a `: kindi progress` comment every `KINDI_KEEPALIVE_S`
   (10 s). Client disconnect sets the cancel event; generation stops at the
   next token.
8. `Kindi.generate` dispatches on the pack: `nanbeige`/`lightning` backends
   (`nanbeige_gen.py` / `lightning_gen.py`), Gemma 4 (`_generate_gemma`), or
   the Bonsai path (`chat.render` + `Runner.prepare` + decode loop +
   `chat.parse_tool_calls`).
9. The done info is assembled into the OpenAI completion (or Ollama) response;
   the end-of-turn state is saved to disk in the background.

## How an engine is chosen today

- At **convert** time, `convert.py` reads the GGUF `general.architecture`
  metadata and writes the pack manifest `backend` key (`kindi_gemma4`,
  `kindi_nanbeige`, `kindi_lightning`, `kindi_cascade`, or the Bonsai default).
- At **serve** time, `Kindi.__init__` (`server.py`) branches on that manifest
  `backend` key. There is no model-id or architecture routing at serve time.
- T4.2 adapters therefore key off the manifest `backend` first; registry
  architecture matching (GGUF metadata) serves external backends and a future
  generic backend.

## Chat templates and tool-call formats

| family | template | parser | tool-call format |
|---|---|---|---|
| Bonsai (qwen35 / bonsai27) | `chat.py::render` | `chat.py::parse_tool_calls` | `<tool_call><function=NAME><parameter=K>...</parameter></function></tool_call>` |
| Gemma 4 | `gemma4_chat.py::render` | `gemma4_tools.py::split_reply` (+ `CallGrammar` constrained sampling) | `<\|tool_call>call:NAME{...}<tool_call\|>` |
| Nanbeige | `nanbeige_chat.py::render` | `nanbeige_gen.py::parse_calls` | `<tool_call><function=NAME>...` XML, or a JSON body (`tool_call_format`) |
| Nemotron Lightning | `lightning_chat.py::render` | `lightning_gen.py::calls_of` (normalizes, then `nanbeige_gen.parse_calls`) | same XML as Nanbeige |
| DeepSeek V4.1 | model encoding file (DSML) | `dsv41_server.py` (`truncated_tool_calls`) | DSML blocks; truncated calls never leak markup into content |

## Verify prompts and references

- Bonsai/Gemma: `verify.py::run` / `run_gemma4` against `kindi/reference*.json`
  (4 prompts, first 40 tokens, greedy, llama.cpp-generated).
- Nanbeige / Lightning / Cascade: `nanbeige_cli.verify`, `lightning_cli.verify`,
  `cascade.verify` with their own references.
- DeepSeek: `python -m kindi.dsv41 verify` (98.91% top-1 gate on the
  1,280-token check).

## [LOCATE] resolution

| plan reference | resolved to |
|---|---|
| the server module(s) | `kindi/server.py` (`create_app`, `Kindi`); DeepSeek: `kindi/dsv41_server.py` |
| engine-construction site | `kindi/server.py::Kindi.__init__` via `_make_engine`; CLI via `kindi/__main__.py` |
| CLI commands to route (T4.2) | `kindi/__main__.py` (`serve`, `chat`, `info`, `verify`, `use`, `bench`) |
| GGUF architecture reader | `kindi/gguf.py::GGUF` (`kv["general.architecture"]`) |
| verify | `kindi/verify.py::run` |
| chat templates / tool parsing | table above |
| Dockerfiles | `docker/Dockerfile`, `docker/Dockerfile.rocm` |
| CI | `.github/workflows/ci.yml` (exists; no `tests.yml` added) |
| requirements-dev | `tests/requirements-test.txt` (test-only deps; `openai` and `uvicorn` added for T5.1) |

## Maintainer gate for T0

Review this map before T4.2 (engine adapters) starts. In particular confirm
the FROZEN list and the serve-time selection story (manifest `backend` key),
because both constrain how the adapters may be written.
