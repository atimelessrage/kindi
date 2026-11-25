# Serving the llama.cpp proxy over the HTTP API: a probe, and the decision (2026-11-25)

Question: the `llamacpp` backend is resolvable by the registry but `kindi serve` never builds it.
What would serving it take, and is the smallest honest version worth shipping?

This page is a probe, not a benchmark: no llama-server ran (there is none on this host), and no
model was loaded. The proxy's opener is canned, so every reply here is a stub.

## What the probe does

`probe.py` builds the real `LlamaCppBackend` (canned opener), wraps it in a ~90-line object with the
surface `server.create_app` touches (`eng.ctx`, `stats`, `busy`, `lock`, `generate`, `check_prompt`,
`encode_text`/`decode_ids`, `after_turn`, `head_active`, `drafting`, `progress`), and mounts the real
app with `server._make_engine` monkeypatched. It then drives five requests.

## What the app answered

| request | result |
|---|---|
| `GET /health` | 200, `ctx` 8192 **supplied by the probe**, `cache_kind` none |
| `GET /v1/models` | 200, `context_length`/`max_model_len`/`max_context_length` 8192 (same supplied value) |
| `GET /metrics` | 200, the usual gauges, and **no** `kindi_cache_*` samples (no `Runner`) |
| `POST /v1/chat/completions` | 200, content replayed from the canned reply; `usage.prompt_tokens` 53 is Kindi's byte ids, not upstream tokens |
| `POST /v1/chat/completions` `stream: true` | 200, SSE frames emitted **after** the single non-streaming upstream reply returns |
| `POST /v1/chat/completions` with `response_format: json_object` | 400, `param="response_format"`, from the existing capability gate |

## What that establishes

- Routing is not the work: the HTTP surface (auth, queue, rate limit, error envelope, SSE framing,
  `/v1/messages`, `/api/chat`, sessions) is reachable through the existing app with an adapter,
  and the capability gates already refuse what the proxy cannot do (structured output, tools,
  `logprobs`).
- The work is in the four numbers the app is built on, and a stub only makes them look right:
  - **context**: the probe passes `eng.ctx = 8192`; nothing asked llama-server. `capabilities
    .max_context` is 0, so a real gateway would advertise a made-up window to `/v1/models` unless
    the backend probes llama-server (`/props` → `default_generation_settings.n_ctx`).
  - **prompt tokens**: `Kindi.check_prompt` is the engine's tokenizer. The proxy's ids are bytes
    (its own docstring says they are an in-process transport), so `dry_run`, `/tokenize` and
    Anthropic's `input_tokens` cannot be answered honestly. Upstream `/completion` does return
    `timings.prompt_n`/`predicted_n`/`cache_n`, so a gateway can report real usage *after* the
    fact, but not before it.
  - **streaming**: `kindi/backends/llamacpp.py` posts `stream: False` and replays the reply as
    character events (`tests/backends/test_llamacpp.py` pins that). Mounted on the app, the SSE
    stream therefore holds every frame until generation is finished: time to first token equals the
    whole reply, and a client that cancels mid-stream does not abort the upstream request. That is
    the opposite of what an agent framework reads SSE for.
  - **prompt cache**: the app's cache gauges, session parking and checkpoint reuse live in
    `Kindi.runner`, which a proxy does not have. The upstream server does prefix reuse
    (`cache_prompt: true`), but it is invisible to `/metrics` and to `usage.prompt_tokens_details`.

## Decision

`kindi serve` stays engine-only for this backend; the proxy remains a registry/engine-gateway
backend. Serving it honestly needs upstream streaming, a context probe and token accounting first,
and those need a live `llama-server` to verify - the same gate T4.4 already has open. Recorded in
`docs/development/requirements/REQ-0019.md` (T4.6) and in `docs/BACKENDS.md` ("Serving boundary").

## Reproduce

```sh
python docs/benchmarks/2026-11-25-llamacpp-proxy-serving/probe.py
```

It prints the five status lines above and exits 0; it needs no pack, GPU, network or llama-server.
