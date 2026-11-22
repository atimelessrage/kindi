# Using Kindi with agents and chat clients

Kindi is an OpenAI-compatible endpoint. Use these values:

- Base URL: `http://<host>:8093/v1` (on the same computer: `http://127.0.0.1:8093/v1`).
- Model: `bonsai2-2b` (or your `model_name`).
- API key: your `api_key`. If you did not set an `api_key`, any string works.
- Log probabilities: only the Gemma 4 backend returns them. On Bonsai and Nanbeige, `logprobs: true` is rejected with HTTP 400 rather than silently omitting the field.

For tool calls, Kindi uses the standard `tools` / `tool_calls` fields.

## Structured output

Send `response_format: {"type": "json_object"}` or
`{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`
and the reply is constrained at decode time so it parses against the
schema — on the dense Bonsai, Gemma 4, Nanbeige and Lightning backends.
`{"type": "grammar", "grammar": "root ::= ..."}` takes llama.cpp GBNF
instead. Recursion is a pushdown language, so a recursive rule (llama.cpp's own
`json.gbnf`) is refused by default with the cycle named, and the refusal is a 400
`param: response_format` before generation. Set `KINDI_GBNF_MAX_DEPTH=N` on the server to
unroll it: nesting up to N levels is in the language, one level deeper is simply not an
allowed token (the reply stays legal and gets shallower, never a 500), and
`KINDI_GBNF_MAX_STATES` still bounds the copies. The unrolled language is a bounded
subset; `json_schema` is the exact, unbounded recursive-JSON path. A grammar or schema
that does not compile is a 400 with
`param: response_format`; unknown types are 400 too. See [API.md](API.md)
for the contract.

## Native endpoints

Besides `/v1`, Kindi serves what an agent host needs to observe and to connect a non-OpenAI client:

- `GET /metrics` - Prometheus text. `kindi_requests_total`, `kindi_prompt_tokens_total`,
  `kindi_cached_prompt_tokens_total`, `kindi_completion_tokens_total`, `kindi_spec_drafted_tokens_total`,
  `kindi_spec_accepted_tokens_total`, `kindi_spec_mode_effective{mode}` (the mode `drafting()` resolves;
  `none` when off), `kindi_queue_depth`, `kindi_queue_max`, `kindi_rate_limit_per_min`,
  `kindi_rate_limited_total`, `kindi_uptime_seconds` and `kindi_model_info{model,ctx,head}`. The two
  spec token counters are global; pair their ratio with the effective-mode gauge. It is protected by
  the API key when one is set.
- `log_json = 1` (`KINDI_LOG_JSON=1`) writes one JSON line per inference request on stdout: `request_id`
  (also returned as `X-Request-ID`), `status`, `duration_ms`, `queue_wait_ms`, `stream`, and on success
  `prompt_tokens`, `cached_tokens`, `completion_tokens`, `gen_tok_s` and `finish`. Rejected, cancelled and
  failed requests are logged too; `/health`, `/metrics` and `/v1/models` are not. Headers and the API key
  are never logged.
- `GET /api/tags`, `POST /api/chat`, `POST /api/generate` - Ollama's API, so Ollama clients and chat UIs work
  without an OpenAI shim. `stream` defaults to true and the body is NDJSON; `options.num_predict` maps to
  `max_tokens`; `options.keep_alive`, `num_ctx`, `num_thread` and `format` are ignored. The API key applies here
  too.
- `GET /sessions`, `GET /sessions/<id>`, `GET /sessions/<id>/export`, `POST /sessions/<id>/status`,
  `POST /sessions/<id>/tags` - the chat session history `kindi chat` records (list with status/tag filters,
  full record, raw-markdown export, triage status, tags). Host-only; the API key applies here too. See
  [API.md](API.md) for the shapes.
- `GET /health` is a liveness probe: it always returns `ok`, `kindi`, `commit`, `model` and `ctx`. When
  `api_key` is set, the detailed view - `queue.waiting`, `queue.max`, `queue.rejected`, `busy`, `stats`,
  `uptime_s` and GPU free/total MiB - requires `Authorization: Bearer <api_key>`. With no key, the detailed
  view is loopback-only; a non-loopback bind gets only liveness. Keep `/health` on loopback unless a proxy
  protects the detailed view. The detailed view's `drafting.last` block carries its own `method`:
  the effective speculation mode that produced those drafts (`mtp`, `ngram`, `draft2b`, `both` or
  `dspine`), alongside `passes`, `drafted`, `accepted` and `acceptance`.
- `POST /v1/completions` and `/v1/embeddings` are not implemented and return 404; use
  `POST /v1/chat/completions` for text generation.
- Image input is rejected with HTTP 400 when the selected pack has no vision weights. The current packs are
  text-only; remove `image_url`/`input_image` parts and Ollama `images`, or use a text-only prompt.

## What to set in any client

1. **Context window: 24,576 tokens** (or your `ctx`). Clients that summarise old turns ("compaction") should do it
   before that.
2. **A long request timeout - 30 to 60 minutes.** Kindi reads new prompt text at about 25-33 tokens/s. The first request
   from an agent that writes code often has 5,000+ tokens of system prompt and tool definitions. The first read of
   that text takes about 3 minutes. The cache then saves it (see the README), so later turns and later sessions are
   fast. Some clients stop after 60-120 s and send the request again. This makes the problem worse: the retry waits
   behind the request that Kindi still reads.
3. **One retry at most**, and no parallel requests: Kindi serves one request at a time. Up to 8 requests may
   wait behind the active one; beyond that the server answers **HTTP 429** with `Retry-After: 1`. Wait a second
   and retry once instead of opening more connections. For a shared server, `KINDI_RATE_LIMIT_PER_MIN` adds a
   per-source generation cap; health, metrics and model listing stay exempt. That limit is a best-effort
   in-process operator guard, not an anti-abuse boundary: behind a reverse proxy all clients share the proxy
   address, and direct LAN clients can use many addresses.
4. **A short system prompt and few tools.** Every tool schema is prompt text. Keep the tool list fixed between calls.
   A different set of tools per call changes the start of the prompt, and then the cache cannot help.
5. **Temperature 0.1-0.7.** Greedy (`0`) is reproducible; the model publisher recommends 0.7 / top_p 0.8 / top_k 20 with `presence_penalty` 1.5. Set `sampling_profile = publisher` in `kindi.conf` for the whole publisher profile, including the thinking profile (1.0 / 0.95 / 20, presence 1.0, up to 8K output). A request's own `presence_penalty` (-2..2) overrides the config value.

## Queue load check (2026-11-10)

A native CPU run of `bonsai2-9b` (6 threads, 1024-token context, `KINDI_QUEUE_MAX=8`) with 12
simultaneous non-streaming clients produced **9 HTTP 200** and **3 HTTP 429**. The nine 200s were
the active request plus the eight waiters; the queue never exceeded eight waiters. Every 429 carried
`Retry-After: 1` and returned in 8-10 ms. `/health` reported at most eight waiting requests, and the
final metrics were `kindi_requests_rejected_total 3` with 0 errors. The eight queued requests ran
serially behind the active request. This confirms the queue bounds waiting clients instead of
letting them accumulate.

Watch a long first read with `curl -s localhost:8093/health` - the `busy` field shows `prompt_read` progress (send `Authorization: Bearer <api_key>` for that field when a key is configured).

## Browser clients and CORS

Kindi sends no CORS headers. A browser app served from another origin cannot call it directly; put the app and
Kindi behind the same reverse proxy, or proxy `/v1` through the app's server. Do not expose the API key in browser
JavaScript. This is deliberate: the server is designed for server-side agents and desktop clients, and a permissive
CORS policy would turn a LAN key into a browser-readable credential.


## Examples

**Any OpenAI SDK**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8093/v1", api_key="none", timeout=3600, max_retries=1)
```

**Agents for code that have an OpenAI-compatible provider setting** (for example pi, or similar terminal agents):

1. Add a provider with these values (for pi, the exact `models.json` entry is in
   [DOCKER.md](DOCKER.md#pi-and-other-agents-beside-the-container); do not use pi's built-in OpenAI provider, which
   talks to OpenAI's servers):
   - Base URL: `http://127.0.0.1:8093/v1`.
   - API style: "openai-completions" / "chat completions".
   - Model: `bonsai2-2b`.
   - Context window: `24576`.
   - Max output: `2048`.
2. Increase the request timeout of the agent, as above.

**Pi no-thinking request.** Thinking is off by default, but a harness can make the no-thinking arm explicit by sending `enable_thinking: false` as a top-level chat-completions field:

```sh
curl -s http://localhost:8093/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-2b","messages":[{"role":"user","content":"Say hello."}],"enable_thinking":false}'
```

This is Kindi's request control only; it does not validate or install an external Pi adapter.

A shorter custom system prompt helps the first turn.

**Chat UIs** (Open WebUI, LibreChat and similar): add an OpenAI connection with the base URL above.

## What to expect from a 2B model

For its size, the model does short shell and Python tasks, file edits, factual questions, summaries and tool calls
well. It also makes the usual mistakes of a small model:

- Wrong tool arguments. Agents usually recover when they send the error back to the model.
- Repetition in long answers.
- Weak multi-step reasoning.

Give it small, concrete steps.
