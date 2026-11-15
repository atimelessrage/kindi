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
  (for example `messages` on a context overflow), else `null`.
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
| `n` | ignored | today; multiple completions are not generated |
| `response_format` | ignored | today; structured output is T5.4 (REQ-0018) |
| `frequency_penalty`, `logit_bias`, `user`, `modalities` and other OpenAI-only fields | ignored | accepted and not read |

Proposed (decision D2, **not implemented**): reject `n` greater than 1 and a
`response_format` naming `json_schema` with a clear 400 naming the parameter,
instead of silently ignoring them, until they are supported. Purely cosmetic
fields keep being ignored.

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
