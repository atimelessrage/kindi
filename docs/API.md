# API

The OpenAI-compatible HTTP API. This page is the field-by-field contract
(agent frameworks depend on it), the error envelope, and the conformance
suite that pins the wire shape. It records what the server does **today**;
proposed changes are marked as such and are not implemented until the
maintainer approves them (REQ-0018).

## Error envelope

Every error response has the OpenAI shape:

```json
{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
```

- `type` keeps the values existing clients see: `invalid_request_error`,
  `authentication_error`, `request_too_large`, `rate_limit_exceeded`,
  `server_busy`, `server_error`, `cancelled`, `not_found`,
  `method_not_allowed`.
- `code` defaults to the type, except one promise: a prompt that does not fit
  is a 400 with `code: "context_length_exceeded"` and `param: "messages"`, so
  client libraries recognise a context-window error and shorten the
  conversation instead of retrying the same request.
- `param` names the request field at fault when the server knows it
  (for example `messages` on a context overflow, `n` or `response_format`
  on an unsupported value, `id`/`status`/`tag` on the sessions routes),
  else `null`.
- Unknown routes (404) and wrong methods (405) and unhandled failures (500)
  return the same envelope. Internal errors do not leak stack traces.

Implemented in `kindi/api/errors.py`, wired into `kindi/server.py` (T5.1).

## Request fields

`POST /v1/chat/completions`. Unknown fields are ignored. Anything marked
**rejected** is a 400 with the envelope above.

| field | status | notes |
|---|---|---|
| `model` | ignored | the served model id comes from the server configuration |
| `messages` | supported | required, non-empty; roles system/user/assistant/tool |
| `tools`, `tool_choice` | supported | per-family tool-call formats; `tool_choice: "none"` disables tools |
| `stream`, `stream_options.include_usage` | supported | SSE; usage chunk at the end when asked |
| `max_tokens`, `max_completion_tokens` | supported | reply length cap |
| `temperature`, `top_p`, `top_k`, `min_p` | supported | sampling |
| `repeat_penalty`, `presence_penalty`, `repeat_last_n` | supported | Kindi extensions |
| `stop` | supported | string or list of strings |
| `seed` | supported | sampled requests |
| `parallel_tool_calls` | supported | honored on Gemma 4; parsed per family elsewhere |
| `logprobs`, `top_logprobs` | supported on Gemma 4 only | **rejected** (400) on other backends |
| `enable_thinking`, `chat_template_kwargs`, `reasoning_effort`, `thinking_budget_tokens` | supported | Kindi thinking controls |
| `n` | supported only as 1 | **rejected** (400) when any other value is given; multiple completions are not generated |
| `response_format` | `{"type": "text"}`, `{"type": "json_object"}`, `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`, `{"type": "grammar", "grammar": "root ::= ..."}` | decode-side constraint (REQ-0018): every token is restricted to the grammar automaton, so the content parses against the schema or matches the GBNF language. Supported on the dense Bonsai, Gemma 4, Lightning and Nanbeige backends; unknown types, an invalid schema or an invalid/recursive grammar are **rejected** (400, `param: response_format`), and `tools` is rejected together with structured output. Validation compiles the automaton to answer an invalid schema or grammar with a 400 before the engine runs, and generation reuses that compiled object, so a request compiles it once |
| image content parts (`{"type": "image_url", ...}`) and Ollama `images` | **rejected** (400) | Kindi is text-only: no build implements a vision projector, so an image part is refused before the engine runs, **even when the pack manifest declares `vision` or `mmproj`** — the 400 names which of the two cases applies instead of answering about an image the engine never saw |
| `frequency_penalty`, `logit_bias`, `user`, `modalities` and other OpenAI-only fields | ignored | accepted and not read |

The live-HTTP conformance suite (`tests/api/`, real app over uvicorn driven by
the official `openai` client, `KINDI_API_URL` for the real-server gate) covers
the structured-output accept + record round trip, the loud 400 on a backend
without the grammar, and the `/sessions` HTTP round trip (list/show/export/
status/tags/read, malformed-id 400) — all GPU-free against the stub engine.

Implemented 2026-11-15 (decision D2, REQ-0018): `n` greater than 1 and a
`response_format` naming structured output are rejected with a clear 400
naming the parameter instead of being silently ignored; constrained decoding
landed as the JSON-schema/json_object slice (T5.4, REQ-0018): the dense path
restricts the sampler to the ids the schema automaton accepts
(`kindi/jsongrammar.py`), so `json_object`/`json_schema` replies parse by
construction. The Gemma 4 native path gained the same constraint later the
same day (`Kindi._generate_gemma` + `_g4_json_token`): structured output takes
over the reply — a thinking channel is not entered, the rest of the reply is
the JSON document, and stop tokens are admitted only once the automaton is
done. The `supports_json_grammar` flag gates the supported backends at
the HTTP boundary; Lightning and Nanbeige gained the same constraint
2026-11-16 (their `{x}_gen` loops advance the same automaton; CPU-tested,
no GPU structured-output gate claimed — the Lightning GGUF does pass plain
`kindi verify`/`kindi bench` on the 4060 Ti, but not through this path).
`{"type": "grammar"}` takes llama.cpp GBNF
(`kindi/gbnf.py`: rules, alternation, groups, literals with C escapes,
character classes, `.`, `* + ? {m} {m,} {m,n}`, rule references, comments).
Bodies may start on the line after `::=` and continue across newlines inside
parentheses or after `|`, which is llama.cpp's documented layout (its own
`grammars/json.gbnf` is written that way). It is compiled before generation, so a
syntax error or an unknown rule is an enveloped 400 and never a
half-constrained reply. A recursive rule is refused by default, naming the cycle,
because this slice is a finite automaton; `KINDI_GBNF_MAX_DEPTH=N` opts into a
bounded unroll that compiles N levels with a fresh copy per reference, so nesting
up to N is in the language and one level deeper is refused rather than truncated.
That language is exact at the depths it allows (checked against an independent
matcher, `tests/test_gbnf_recursion.py`) but bounded, and the copies cost states:
llama.cpp's own `json.gbnf` reaches array nesting 2 inside the default
20,000-state limit, so `json_schema` remains the path for unbounded recursive JSON. Compilation is bounded: `{m,n}` refuses a
bound above 10,000 copies and the automaton above 20,000 states
(`KINDI_GBNF_MAX_STATES`), both as a 400 rather than a memory request from the
caller.

**What exact reuse costs.** A rule reference is expanded once per use. Sharing a rule's entry and
exit states - what the first slice did - is not a valid NFA: the closure after one use reaches a
later use's continuation and the language widens (the pre-2026-11-17 build accepted 0-5 pairs for
a grammar that said exactly 3), which is why the cache was dropped rather than fixed. Exactness
is paid in states, in proportion to reuse: a flat 40-pair record that references `ws` 82 times
is **5,915 states / ~6 ms** here, against 302 states for the incorrect shared build, while the
loop form of the same practical constraint (`pair (ws "," ws pair)*`) stays **~330 states** at
any document length (zero pairs needs the group wrapped in `?`). The default 20,000 states
therefore holds an enumerated record of roughly 130 copies - a few KB of grammar text - which
`tests/test_gbnf.py` pins at 40 copies; past it the 400 names `*`/`+` (loop instead of
enumeration), a shorter literal and `json_schema` as the alternatives, and `KINDI_GBNF_MAX_STATES`
remains the operator escape hatch (it buys ~1.4 ms per 1,000 states on this host: 12k states in
13 ms, measured 2026-11-17). The same
`supports_json_grammar` capability flag gates it. The
adapter contract carries the same flag on
`BackendCapabilities`; a backend that sets it treats `token_filter` as the
full allowed set for constrained decoding. `n = 1` and
`response_format: {"type": "text"}` are accepted along with all three
structured types above, and purely cosmetic fields keep being ignored. The grammar is schema-ordered
(`kindi/jsongrammar.py`): properties are emitted in schema order, `required`
must be present, `additionalProperties` behaves like OpenAI strict mode
(only schema properties), `integer` is canonical decimal, and `enum`/`const`
accept primitives only.

## Sessions (REQ-0016)

The server also exposes the chat session history that `kindi chat` records
(host-only; these routes never touch the engine, the GPU or the prompt
cache). All of them require the API key when one is set (401 otherwise) and
use the same error envelope; session ids are the store's 12-hex ids, and any
other id (including path-shaped input) is a 400, never a file read.

| route | returns |
|---|---|
| `GET /sessions?status=&tag=` | `{"sessions": [...]}` — summaries (id, title, status, tags, created, updated, unread, protected, message_count), newest first; `status` must be `open`, `needs-review` or `resolved` (400 otherwise) |
| `GET /sessions/<id>` | the full record, messages included; 404 (`not_found`) when there is no such session |
| `GET /sessions/<id>/export` | the raw markdown export (`text/markdown`, `Content-Disposition` filename), exact message text verbatim |
| `POST /sessions/<id>/status` | `{"status": ...}` moves the session through triage; returns the summary |
| `POST /sessions/<id>/tags` | `{"tag": "..."}` adds a (slugified, deduped) tag; returns the summary |
| `POST /sessions/<id>/read` | clears the unread flag; no body fields needed; returns the summary |

Concurrent mutations are serialized: each request gets its own store
instance, so the routes share one lock around the read-modify-write, and a
simultaneous tag post and status post cannot overwrite each other.

## Streaming and queueing

- `stream=true` yields `chat.completion.chunk` objects terminated by
  `data: [DONE]`. A `: kindi progress {...}` comment is sent every 10 s while
  a long prompt is read.
- One request runs at a time on the single engine. Up to `KINDI_QUEUE_MAX`
  (default 8) wait behind it; beyond that the server returns 429 with
  `Retry-After` instead of waiting without a bound. Chat requests are ordered
  before background requests (priorities); a background prompt read pauses
  between chunks for a chat request. A client that disconnects cancels its
  request; generation stops at the next token.
- The planned scheduler work (T5.3: cancel-on-disconnect everywhere,
  configurable overflow behavior, per-request timeouts, queue position in the
  progress comment) is tracked in REQ-0018 and is not implemented yet.

## Observability

- `GET /health`: liveness plus (loopback or authenticated) queue, stats,
  drafting and GPU memory.
- `GET /metrics`: Prometheus text (requests, tokens, speculative stats,
  queue, uptime). The richer per-request metrics in T5.5 are not implemented
  yet.
- `log_json = 1`: one JSON access-log line per inference request with the
  request id, status, duration, queue wait and token counts.

## Conformance suite

`tests/api/` starts the real FastAPI application in a uvicorn thread on a
random local port with a scripted stub engine and drives it with the official
`openai` Python client:

```sh
python -m pytest tests/api -q
```

It pins the non-streaming and streaming completion shapes, usage (including
`stream_options.include_usage` and `prompt_tokens_details.cached_tokens`),
the `finish_reason` vocabulary, the models listing (context limit under every
name agent harnesses read), and the error envelope. The GPU-free run uses the
stub; per-family tool-call fixtures (T5.2) are captured from real models by
the maintainer with `tools/capture_toolcall_fixtures.py`.

### Maintainer gate (GPU)

Against a real pack:

```sh
kindi serve --host 127.0.0.1 --port 8093
python -m pytest tests/api -q   # with KINDI_API_URL pointed at the real server
```

Then smoke test Open WebUI, pi and opencode against the server. No GPU result
is claimed by the CPU suite.

Tracked in [REQ-0018](development/requirements/REQ-0018.md) (proposed).
