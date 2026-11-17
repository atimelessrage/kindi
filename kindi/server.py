"""Kindi OpenAI-compatible server.

  GET  /health                 status, live progress of the request being served, GPU memory, queue depth
  GET  /v1/models              the model and its real context (context_length, max_model_len, meta.n_ctx)
  POST /v1/chat/completions    messages (+ tools) -> answer; stream=true for SSE; tool calls as OpenAI tool_calls;
                               dry_run=true -> the exact prompt_tokens, no generation
  POST /v1/tokenize            {"content": text} or {"messages": [...], "tools": [...]} -> count, tokens
  POST /v1/detokenize          {"tokens": [...]} -> text
  POST /api/chat               the same, in Ollama's NDJSON form (stream defaults to true)
  POST /api/generate           a raw prompt, Ollama style
  GET  /api/tags               the served model, Ollama style
  GET  /metrics                Prometheus text: requests, tokens, speculative stats, queue, uptime
  GET  /sessions               chat sessions from state_dir/sessions (?status=&tag= filters); host-only, no engine
  GET  /sessions/<id>          one session with its messages; /sessions/<id>/export is the raw markdown (text/markdown)
  POST /sessions/<id>/status   move a session through triage ({"status": "open"|"needs-review"|"resolved"})
  POST /sessions/<id>/tags     add a tag ({"tag": "..."})
  POST /sessions/<id>/read     clear the unread flag (no body fields needed)

A prompt that does not fit is an HTTP 400 (code context_length_exceeded, OpenAI's wording) before the request waits
for the engine. A stream gets an SSE comment ": kindi progress {...}" every KINDI_KEEPALIVE_S (10) seconds while
nothing else goes out; kindi_keepalive=true gives a non-streaming request spaces before its JSON. After each answer
the end-of-turn state is saved to disk in the background (turn_save, see cache.Turns).

One request at a time (one engine on one GPU); up to KINDI_QUEUE_MAX requests wait behind it, and the next one gets
HTTP 429 with Retry-After instead of waiting without a bound (the Ollama endpoints; /v1/chat/completions queues on
the priority scheduler). KINDI_RATE_LIMIT_PER_MIN can also cap generation requests per source address. Thinking is
off unless the request sets chat_template_kwargs.enable_thinking (or top-level enable_thinking); a stream then
carries the thinking as delta.reasoning_content. thinking_budget_tokens (or thinking_budget) caps it: at the budget
the server closes the thinking itself and the model goes on to answer. temperature 0 = greedy (token-exact with
llama.cpp in the exact head mode). Request bodies are capped at 16 MiB (KINDI_MAX_BODY_MB); when api_key is set, it
is compared in constant time."""
import asyncio
import codecs
import concurrent.futures
import contextvars
import hmac
import json
import math
import os
import re
import threading
import time
import uuid

import numpy as np

from . import COMMIT, __version__, chat
from .api import errors as api_errors
from .cache import Cancelled, Runner
from .constrained import STRUCTURED_TYPES, is_structured
from .engine import EOS
from .sampling import Sampler


THINK_CLOSE = "\n\nI have thought enough; time to act.\n</think>\n\n"

# while a long prompt is read nothing else is sent: an SSE comment this often keeps the stream alive and tells the
# client how far the read is (Node's fetch, which pi uses, drops a body that is silent for 300 s)
KEEPALIVE_S = float(os.environ.get("KINDI_KEEPALIVE_S", "10"))


class ContextOverflow(OverflowError):
    """The prompt does not fit the context. The message is OpenAI's wording: client libraries (litellm, which the
    terminus-2 agent harness uses, and the OpenAI SDKs) recognise it as a context-window error and shorten the
    conversation in place of sending the same request again."""

    def __init__(self, n, ctx, reserve=1):
        self.n, self.ctx = int(n), int(ctx)
        OverflowError.__init__(self, "This model's maximum context length is %d tokens. However, your messages "
                               "resulted in %d tokens (%d are needed for the reply to start). Please reduce the length "
                               "of the messages." % (self.ctx, self.n, self.n + reserve))


class RequestError(ValueError):
    """A malformed client request. It is a 400 and must never touch the prompt cache."""

    def __init__(self, msg, param=None):
        super().__init__(msg)
        self.param = param          # the request field at fault, named in the error envelope


def _request_error(msg, param=None):
    raise RequestError(msg, param=param)


def _response_format_schema(response_format):
    """The JSON schema inside a json_schema response_format, or None (REQ-0018).

    Accepts the OpenAI shape {"type": "json_schema", "json_schema": {"name": ...,
    "schema": {...}}} and the bare {"schema": {...}} shape; a json_schema payload
    that IS the schema (its only keys are not "schema"/"name") is taken as the
    schema itself when it does not look like the wrapper.
    """
    js = response_format.get("json_schema")
    if not isinstance(js, dict):
        return None
    if isinstance(js.get("schema"), dict):
        return js["schema"]
    if set(js) <= {"schema", "name", "strict"}:
        return None
    return js


def _response_format_grammar(response_format, req=None):
    """Build the decode grammar for a response_format (assumes `_validate_request` passed).

    Validation already compiled the automaton to answer an invalid grammar with a 400 before the
    engine runs, so it parks the result on the request (`_grammar`) and generation reuses that
    object instead of building the same automaton a second time.
    """
    parked = req.get("_grammar") if isinstance(req, dict) else None
    if parked is not None:
        return parked
    from .jsongrammar import grammar_from_schema, json_object_grammar

    if response_format.get("type") == "json_object":
        return json_object_grammar()
    if response_format.get("type") == "grammar":
        from .gbnf import Grammar

        return Grammar.from_string(response_format.get("grammar") or "")
    return grammar_from_schema(_response_format_schema(response_format))


def _finite_number(value):
    """JSON numbers only; bool is an int in Python, and NaN/Infinity are not valid JSON numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _env_int(name, raw, minimum=0):
    """Parse a server limit at import time with an actionable startup error."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise SystemExit("kindi: %s must be an integer (got %r)" % (name, raw))
    if value < minimum:
        raise SystemExit("kindi: %s must be at least %d (got %d)" % (name, minimum, value))
    return value


MAX_BODY_MB = _env_int("KINDI_MAX_BODY_MB", os.environ.get("KINDI_MAX_BODY_MB", "16"), 1)
MAX_BODY_BYTES = MAX_BODY_MB * 1024 * 1024

# One engine runs one request; this bounds how many clients may wait behind it (KINDI_QUEUE_MAX=0: no waiting).
MAX_QUEUE = _env_int("KINDI_QUEUE_MAX", os.environ.get("KINDI_QUEUE_MAX", "8"), 0)

# The access-log record of the request on this task. The middleware sets it; the generator wrapper
# receives the same dict because the engine runs on the GPU thread, where a context variable does not
# follow.
_CURRENT = contextvars.ContextVar("kindi_request", default=None)


class QueueFull(Exception):
    """The bounded wait behind the active request is full: an HTTP 429 with Retry-After."""


class Gate(object):
    """Admission for the single engine.

    K.lock serialises generation, but nothing bounded the number of clients waiting for it: the executor queue
    is unbounded, so on a loaded server every extra request waited (for minutes) instead of being told to retry.
    The count is the number admitted (one running + up to max_queue waiting); rejections are counted for /metrics.
    """

    def __init__(self, max_queue):
        self.max_queue = max(0, int(max_queue))
        self._lock = threading.Lock()
        self.admitted = 0
        self.rejected = 0

    def enter(self):
        with self._lock:
            if self.admitted >= self.max_queue + 1:
                self.rejected += 1
                raise QueueFull("the server is busy: %d request(s) queued (KINDI_QUEUE_MAX=%d)"
                                % (self.admitted - 1, self.max_queue))
            self.admitted += 1

    def release(self):
        with self._lock:
            if self.admitted:
                self.admitted -= 1

    @property
    def waiting(self):
        with self._lock:
            return max(0, self.admitted - 1)


class RateLimiter(object):
    """A fixed-window per-source limit for generation requests.

    The queue bounds how many requests wait for the engine, but it does not stop one client from refilling the
    queue forever. This limiter is deliberately in-process and per source address: it is a cheap overload guard,
    not a distributed quota. ``per_minute=0`` disables it. Health, metrics and model-listing requests are not
    counted, so a monitor cannot lock itself out.
    """

    def __init__(self, per_minute, clock=None):
        self.per_minute = max(0, int(per_minute))
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._clients = {}
        self.rejected = 0

    def retry_after(self, client):
        """Return 0 if allowed, else the whole seconds until the window resets."""
        if not self.per_minute:
            return 0
        now = self._clock()
        window = int(now // 60)
        key = client or "-"
        with self._lock:
            current = self._clients.get(key)
            if current is None or current[0] != window:
                if len(self._clients) >= 4096:
                    self._clients = {k: v for k, v in self._clients.items() if v[0] == window}
                    if len(self._clients) >= 4096:          # a burst of distinct sources: drop the oldest key
                        self._clients.pop(next(iter(self._clients)))
                self._clients[key] = [window, 1]
                return 0
            if current[1] < self.per_minute:
                current[1] += 1
                return 0
            self.rejected += 1
            return max(1, int(math.ceil(60.0 - (now % 60.0))))


class RequestTooLarge(ValueError):
    """The request body exceeds MAX_BODY_BYTES: an HTTP 413, not an engine error."""


async def _read_json_body(request):
    """Read and parse a JSON body with a hard size cap.

    `await request.json()` reads an unbounded body into memory. This server can be bound to 0.0.0.0, so a
    LAN client must not be able to make the process allocate an arbitrarily large string before the API key
    or the request shape is checked. The declared Content-Length is checked first; `stream()` then enforces
    the same cap even when the header is missing or lies.
    """
    limit = getattr(request.app.state, "max_body_bytes", MAX_BODY_BYTES)
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            too_large = int(declared) > limit
        except ValueError:
            too_large = False
        if too_large:
            raise RequestTooLarge("the request body is larger than %d bytes" % limit)
    body = bytearray()
    async for piece in request.stream():
        body.extend(piece)
        if len(body) > limit:
            raise RequestTooLarge("the request body is larger than %d bytes" % limit)
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise RequestError("the body is not JSON")


_IMAGE_UNSUPPORTED = "this pack has no vision weights; image input is not supported"
_IMAGE_NO_PROJECTOR = ("this pack declares vision weights (mmproj), but no Kindi build implements a "
                       "vision projector yet; image input is not supported")


def _image_error(pack):
    """The 400 text for an image request, which no Kindi build can serve.

    A manifest that declares vision weights does not unlock acceptance: there is no projector in
    any build, so accepting the request would hand a text-only engine an image request and answer
    about an image it never saw. The message says which of the two situations the caller is in.
    """
    return _IMAGE_NO_PROJECTOR if _pack_has_vision(pack) else _IMAGE_UNSUPPORTED


def _pack_has_vision(pack):
    """True only when the selected pack names vision weights in its manifest."""
    if not pack:
        return False
    try:
        with open(os.path.join(pack, "kindi_pack.json"), encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        return False
    return bool(man.get("vision") or man.get("mmproj"))


def _is_image_part(part):
    """An OpenAI-compatible content part that carries an image."""
    return (part.get("type") in ("image_url", "image", "input_image")
            or "image_url" in part or "image" in part)


def _validate_request(req, image_error=_IMAGE_UNSUPPORTED):
    """Check the OpenAI request shape before it reaches the engine or the prompt cache.

    The server takes a raw Request, so this is the boundary that turns a typo or a wrong JSON type into a
    400 RequestError instead of a KeyError/TypeError from chat.render() that would be reported as an engine
    failure (and, before this check, would clear the prompt cache).
    """
    if not isinstance(req, dict):
        _request_error("the body must be a JSON object")

    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        _request_error("messages must be a non-empty list")
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            _request_error("messages[%d] must be an object" % i)
        role = m.get("role")
        if not isinstance(role, str) or role not in ("system", "user", "assistant", "tool"):
            _request_error("messages[%d].role must be system, user, assistant or tool" % i)
        content = m.get("content")
        if content is not None and not isinstance(content, (str, list)):
            _request_error("messages[%d].content must be a string or a list" % i)
        if m.get("images"):
            _request_error("%s (messages[%d].images)" % (image_error, i))
        if isinstance(content, list):
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    _request_error("messages[%d].content[%d] must be an object" % (i, j))
                if _is_image_part(part):
                    _request_error("%s (messages[%d].content[%d])" % (image_error, i, j))
                if "text" in part and not isinstance(part["text"], str):
                    _request_error("messages[%d].content[%d].text must be a string" % (i, j))
        if role == "assistant":
            calls = m.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, list):
                    _request_error("messages[%d].tool_calls must be a list" % i)
                for j, call in enumerate(calls):
                    fn = call.get("function", call) if isinstance(call, dict) else None
                    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                        _request_error("messages[%d].tool_calls[%d] needs a function name" % (i, j))
                    args = fn.get("arguments")
                    if args is not None and not isinstance(args, (str, dict)):
                        _request_error("messages[%d].tool_calls[%d].arguments must be a string or object" % (i, j))

    tools = req.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            _request_error("tools must be a list")
        for i, tool in enumerate(tools):
            fn = tool.get("function", tool) if isinstance(tool, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                _request_error("tools[%d] needs a function name" % i)
            if fn.get("parameters") is not None and not isinstance(fn["parameters"], dict):
                _request_error("tools[%d].function.parameters must be an object" % i)

    if "tool_choice" in req and req["tool_choice"] is not None and not isinstance(req["tool_choice"], (str, dict)):
        _request_error("tool_choice must be a string or an object")
    n = req.get("n")
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int):
            _request_error("n must be an integer", param="n")
        if n < 1:
            _request_error("n must be at least 1", param="n")
        if n > 1:
            _request_error("multiple completions (n > 1) are not supported", param="n")
    response_format = req.get("response_format")
    if response_format is not None:
        if not isinstance(response_format, dict):
            _request_error("response_format must be an object", param="response_format")
        rtype = response_format.get("type")
        if not isinstance(rtype, str):
            _request_error("response_format.type must be a string", param="response_format")
        if rtype != "text" and rtype not in STRUCTURED_TYPES:
            _request_error(
                "structured output (response_format type %r) is not supported; use text, "
                "json_object, json_schema or grammar" % rtype, param="response_format")
        if rtype in STRUCTURED_TYPES:
            if req.get("tools"):
                _request_error("tools are not supported together with structured output",
                               param="response_format")
            if rtype == "grammar":
                grammar_text = response_format.get("grammar")
                if not isinstance(grammar_text, str) or not grammar_text.strip():
                    _request_error(
                        "a grammar response_format needs a non-empty \"grammar\" string (GBNF)",
                        param="response_format")
                from .gbnf import Grammar

                try:
                    req["_grammar"] = Grammar.from_string(grammar_text)
                except ValueError as e:
                    _request_error("invalid grammar: %s" % e, param="response_format")
            if rtype == "json_schema":
                schema = _response_format_schema(response_format)
                if schema is None:
                    _request_error(
                        "json_schema must be an object with a \"schema\" object",
                        param="response_format")
                from .jsongrammar import grammar_from_schema

                try:
                    req["_grammar"] = grammar_from_schema(schema)
                except ValueError as e:
                    _request_error("invalid JSON schema: %s" % e, param="response_format")
    if req.get("stream") is not None and not isinstance(req["stream"], bool):
        _request_error("stream must be true or false")
    if req.get("logprobs") is not None and not isinstance(req["logprobs"], bool):
        _request_error("logprobs must be true or false")
    if req.get("parallel_tool_calls") is not None and not isinstance(req["parallel_tool_calls"], bool):
        _request_error("parallel_tool_calls must be true or false")
    if req.get("enable_thinking") is not None and not isinstance(req["enable_thinking"], bool):
        _request_error("enable_thinking must be true or false")

    stop = req.get("stop")
    if stop is not None and not isinstance(stop, str):
        if not isinstance(stop, list) or any(not isinstance(s, str) for s in stop):
            _request_error("stop must be a string or a list of strings")

    so = req.get("stream_options")
    if so is not None:
        if not isinstance(so, dict):
            _request_error("stream_options must be an object")
        if so.get("include_usage") is not None and not isinstance(so["include_usage"], bool):
            _request_error("stream_options.include_usage must be true or false")
    kw = req.get("chat_template_kwargs")
    if kw is not None:
        if not isinstance(kw, dict):
            _request_error("chat_template_kwargs must be an object")
        if kw.get("enable_thinking") is not None and not isinstance(kw["enable_thinking"], bool):
            _request_error("chat_template_kwargs.enable_thinking must be true or false")
        if kw.get("preserve_thinking") is not None and not isinstance(kw["preserve_thinking"], bool):
            _request_error("chat_template_kwargs.preserve_thinking must be true or false")
        if kw.get("reasoning_effort") is not None and not isinstance(kw["reasoning_effort"], str):
            _request_error("chat_template_kwargs.reasoning_effort must be a string")
    if req.get("reasoning_effort") is not None and not isinstance(req["reasoning_effort"], str):
        _request_error("reasoning_effort must be a string")

    for key, lo in (("max_tokens", 1), ("max_completion_tokens", 1), ("top_k", 0),
                    ("top_logprobs", 0), ("repeat_last_n", -1), ("thinking_budget_tokens", 0),
                    ("thinking_budget", 0), ("seed", None)):
        value = req.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            _request_error("%s must be an integer" % key)
        if lo is not None and value < lo:
            _request_error("%s must be at least %d" % (key, lo))

    for key, lo, hi in (("temperature", 0.0, None), ("top_p", 0.0, 1.0), ("min_p", 0.0, 1.0),
                        ("repeat_penalty", 0.0, None), ("presence_penalty", -2.0, 2.0)):
        value = req.get(key)
        if value is None:
            continue
        if not _finite_number(value):
            _request_error("%s must be a finite number" % key)
        if lo is not None and value < lo:
            _request_error("%s must be at least %s" % (key, lo))
        if hi is not None and value > hi:
            _request_error("%s must be at most %s" % (key, hi))
    if req.get("top_p") == 0:
        _request_error("top_p must be greater than 0")
    if req.get("repeat_penalty") == 0:
        _request_error("repeat_penalty must be greater than 0")
    if req.get("top_logprobs") is not None and req["top_logprobs"] > 20:
        _request_error("top_logprobs must be at most 20")
    return req


def _looping(gen, min_span=240):
    """True when the last tokens are one block of 4..256 tokens repeated (at least 4 times and min_span tokens):\n    a model stuck in a loop. Short legitimate repeats (table rules, a few identical lines) stay below it."""
    n = len(gen)
    for P in range(4, 257):
        reps = max(4, -(-min_span // P))
        if P * reps > n:
            break
        tail = gen[n - P:]
        if all(gen[n - (k + 1) * P:n - k * P] == tail for k in range(1, reps)):
            return True
    return False



def _is_loopback(host):
    """True for the default local bind addresses; a network bind needs an explicit security warning."""
    h = str(host or "").strip().lower()
    return h in ("127.0.0.1", "localhost", "::1") or h.startswith("127.")


def _two_cards():
    from .backends.gemma4 import two_cards
    return two_cards()


class Kindi(object):
    """Engine + tokenizer + cache + sampler, shared by the HTTP layer and `kindi chat`."""

    def __init__(self, cfg, log=print):
        try:
            import cupy as cp
        except ImportError:              # a CPU-only host (KINDI_DEVICE=cpu)
            cp = None
        from .engine import make_engine
        from .tokenizer import Tok
        self.cp, self.cfg, self.log = cp, cfg, log
        pack = cfg["pack"]
        if not os.path.exists(os.path.join(pack, "kindi_pack.json")):
            raise SystemExit("kindi: no model pack at %s - run: kindi convert <model.gguf>" % pack)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        self.pack_id = (man.get("source_sha256") or "pack")[:12]
        self.native = man.get("backend") == "kindi_gemma4"
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0, "read": 0, "generated": 0,
                     "started": 0.0}
        self.nanbeige = None
        if man.get("backend") == "kindi_nanbeige":        # llama-arch K/I-quant GGUF (Nanbeige 4.3 33B)
            from .nanbeige_gen import NanbeigeBackend
            self.variant = "nanbeige"
            self.nanbeige = NanbeigeBackend(man, cfg, log=log)
            self.eng, self.tok, self.runner, self.sample = self.nanbeige.eng, self.nanbeige.tok, None, None
        elif man.get("backend") == "kindi_cascade":        # Nemotron Cascade 2: converter/validator only so far
            raise SystemExit("kindi: Nemotron Cascade 2 is converter/validator only; GPU K-quant Nemotron-H kernels are pending, so it cannot be served yet")
        elif man.get("backend") == "kindi_lightning":     # Nemotron-H MoE GGUF (Nemotron 3.5 Lightning 30B-A3B)
            from .lightning_gen import LightningBackend
            self.variant = "lightning"
            self.nanbeige = LightningBackend(man, cfg, log=log)   # (the same backend interface as Nanbeige's)
            self.eng, self.tok, self.runner, self.sample = self.nanbeige.eng, self.nanbeige.tok, None, None
        elif self.native:
            from .gemma4 import Engine as NativeEngine
            from .gemma4_tokenizer import Tok as GemmaTok
            self.variant = "gemma4"
            self.model_path = man["model_path"]
            man["pack_path"] = pack
            self.tok = GemmaTok(self.model_path)
            if _two_cards():
                from .gemma4_tp import TPEngine as NativeEngine
            self.eng = NativeEngine(man, ctx=cfg["ctx"], log=log)
            from .gemma4_cache import G4Cache
            self.g4cache = G4Cache(self.eng, log=log)
            self.runner = None
            self.sample = None
        else:
            self.variant = chat.template_variant(pack)      # qwen35 (2B/4B/9B) or bonsai27 (the 27B template)
            tandem = str(cfg.get("tandem", "off") or "off").strip().lower().replace("_", "-")
            clusters = None
            if tandem == "off":
                cl_path = os.path.join(pack, "head_clusters.npz")
                if os.path.exists(cl_path):
                    clusters = dict(np.load(cl_path))
            self.eng = make_engine(pack, ctx=cfg["ctx"], clusters=clusters, log=log, tandem=tandem,
                                   cpu_layers=cfg.get("tandem_cpu_layers"),
                                   gpu=cfg.get("tandem_gpu"))
            self.tok = Tok(self.eng.vocab, os.path.join(pack, "merges.txt"))
            if tandem != "off":
                from .tandem import TandemRunner
                self.runner = TandemRunner(self.eng, cfg, self.pack_id, log=log, progress=self.busy)
            else:
                self.runner = Runner(self.eng, cfg, self.pack_id, log, progress=self.busy)
            self.sample = Sampler(self.eng)
        self.supports_logprobs = self.variant == "gemma4"
        self.supports_json_grammar = self.variant in ("qwen35", "bonsai27", "gemma4", "nanbeige", "lightning")   # these paths carry the JSON automaton
        self.lock = threading.RLock()        # reentrant: a background request runs waiting chat requests inside its own turn
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
                      "spec_passes": 0, "spec_drafted": 0, "spec_accepted": 0,
                      "errors": 0, "gen_seconds": 0.0, "last_tok_s": 0.0}
        self.t0 = time.time()

    def head_mode(self):
        """exact | fast: the state file (written by `kindi head`) wins over the config."""
        try:
            v = open(os.path.join(self.cfg["state_dir"], "head_mode")).read().strip().lower()
            if v in ("exact", "fast"):
                return v
        except OSError:
            pass
        return self.cfg["head"]

    def head_active(self):
        """the head that runs: "fast" only when the engine has a fast head (the 27B, Nanbeige and the CPU engine have none)"""
        m = self.head_mode()
        if m == "fast" and getattr(self.eng, "hsl", None) is None and getattr(self.eng, "fast_centroids", None) is None:
            return "exact"
        return m

    def _pick(self, nxt, temperature, top_p, top_k, rng, min_p=0.0, penalty=None, presence=None,
              allowed=None):
        """the next token: already drawn by speculative sampling, or sampled now"""
        ps = getattr(self.eng, "spec_presampled", None)
        if ps is not None:
            self.eng.spec_presampled = None
            return ps
        return self.sample(nxt, temperature, top_p, top_k, rng, min_p, penalty, presence,
                           allowed=allowed)

    def _g4_text(self,token):
        return self.tok.decode_bytes(token).decode("utf-8","replace")

    def _g4_call_token(self,logits,grammar,gst,temperature,top_p,top_k,rng):
        """the next token inside a tool call: the best candidates that keep the call syntax valid. The search widens
        (64, 1,024, 8,192 candidates) until one fits; if none fits, the free choice stands and the parser decides."""
        for width in (64,1024,8192):
            width=min(width,logits.size)
            idx=self.cp.argpartition(logits,-width)[-width:]
            cand=self.cp.asnumpy(idx).astype(np.int64,copy=False); sc=self.cp.asnumpy(logits[idx]).astype(np.float64)
            order=np.argsort(-sc); cand,sc=cand[order],sc[order]
            keep=[i for i,c in enumerate(cand) if int(c) not in self.tok.stop_ids
                  and grammar.advance(gst,self._g4_text(int(c))) is not None]
            if keep:
                cand,sc=cand[keep],sc[keep]
                if temperature<=0: return int(cand[0])
                return Sampler._pick(cand,sc,temperature,top_p,top_k,rng)
        return int(self.cp.argmax(logits).item())

    def _g4_json_token(self,logits,grammar,gstate,temperature,top_p,top_k,rng):
        """the next token inside a structured-output reply: the best grammar-legal one, from the full vocabulary,
        so the finished reply parses against the schema by construction (same allowed-set rule as the dense path)."""
        if not hasattr(self,"_grammar_texts"):
            self._grammar_texts=[self.tok.decode_bytes(i).decode("utf-8","replace") for i in range(len(self.tok.tokens))]
        allowed=grammar.vocab_allowed(gstate,self._grammar_texts)
        if grammar.done(gstate):
            allowed=np.concatenate([allowed,np.array(sorted(self.tok.stop_ids),dtype=np.int64)])
        if allowed.size==0:
            raise RuntimeError("structured-output grammar allowed no token")
        scores=self.cp.asnumpy(logits)[allowed]
        if temperature<=0: return int(allowed[int(np.argmax(scores))])
        return Sampler._pick(allowed,scores.astype(np.float64),temperature,top_p,top_k,rng)

    def _generate_gemma(self,req,cancel=None):
        from .gemma4_chat import render as render_gemma
        from . import gemma4_tools
        cfg=self.cfg; messages=gemma4_tools.prepare_messages(req.get("messages") or [])
        tools=req.get("tools") or None
        if req.get("tool_choice")=="none": tools=None
        kw=dict(req.get("chat_template_kwargs") or {})
        kw["enable_thinking"]=bool(kw.get("enable_thinking",req.get("enable_thinking",False)))
        kw["conversational_mode"]=bool(kw.get("conversational_mode",False))
        prompt=render_gemma(self.model_path,messages,tools,kw,True)
        jgrammar=jstate=None
        rf=req.get("response_format") or {}
        if is_structured(rf):
            jgrammar=_response_format_grammar(rf, req); jstate=jgrammar.start()
            thinking_open=False     # a structured reply is the whole reply: it never enters a thought channel
        else:
            thinking_open=prompt.endswith(gemma4_tools.THINK_OPEN+"thought\n")   # after a tool response, thinking on
        sorter=gemma4_tools.Stream(thinking_open)
        ids=self.tok.encode(prompt)
        if not ids: raise ValueError("Gemma 4 chat template produced an empty prompt")
        if len(ids)>=self.eng.ctx:
            raise ContextOverflow(len(ids),self.eng.ctx)
        if req.get("_kindi_check"): return ids            # check_prompt(): the prompt is valid, nothing to run
        self.eng.shortlist=cfg["fast_clusters"] if self.head_mode()=="fast" and self.eng.fast_centroids is not None else 0
        max_new=int(req.get("max_tokens") or req.get("max_completion_tokens") or cfg["max_tokens"])
        max_new=max(1,min(max_new,self.eng.ctx-len(ids)))
        temperature=float(req.get("temperature",cfg["temperature"]))
        top_p=float(req.get("top_p",cfg["top_p"])); top_k=int(req.get("top_k",cfg["top_k"]))
        # OpenAI logprobs: the model's own distribution (the logits at temperature 1), as llama.cpp reports it
        want_lp=bool(req.get("logprobs")); n_lp=max(0,min(20,int(req.get("top_logprobs") or 0)))
        lp_out=[] if want_lp else None
        seed=req.get("seed")
        rng=np.random.RandomState(seed if seed is not None else int(time.time()*1000)%2**31)
        stop=req.get("stop") or []
        if isinstance(stop,str): stop=[stop]
        stop_seq=[x for x in stop if x]; stop_win=max((len(x) for x in stop_seq), default=0); stop_from=0
        self.eng.reset()
        started=time.time()
        self.busy.update(active=True,phase="prompt",prompt_tokens=len(ids),cached=0,read=0,generated=0,started=started)
        self.stats["requests"]+=1
        self.log("Gemma 4 request: %d prompt tokens (max %d new)"%(len(ids),max_new))
        try:
            batch_size=max(1,min(512,int(os.environ.get("KINDI_G4_PREFILL_BATCH","512"))))   # the KV ring takes 512 at once
            end_prompt=len(ids)-1
            pos=self.g4cache.prepare(ids)                  # positions already on the card (the prompt cache)
            self.busy["cached"]=pos
            while pos<end_prompt:
                if cancel is not None and cancel.is_set():
                    self.busy["active"]=False; return
                end=min(end_prompt,pos+batch_size)
                if end-pos>1:
                    self.eng.forward_batch(ids[pos:end],pos)
                else:
                    self.eng.forward(ids[pos],pos,logits=False)
                self.g4cache.advanced(ids[:end],checkpoint=True)
                pos=end;self.busy["read"]=pos
            if cancel is not None and cancel.is_set():
                self.busy["active"]=False;return
            logits=self.eng.forward(ids[-1],len(ids)-1,logits=True)
            self.g4cache.advanced(ids,checkpoint=True)
            self.busy["read"]=len(ids)
            prompt_done=time.time()
            self.busy["phase"]="generating"
            decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
            content=[]; generated=[]; finish="length"; text=""
            # inside a tool call, only tokens that keep the call syntax valid are sampled (as llama.cpp's grammar does)
            grammar=gemma4_tools.CallGrammar([(t.get("function") or t).get("name") for t in tools]) if tools else None
            call_open=self.tok.ids.get(gemma4_tools.CALL_OPEN)
            parallel=bool(req.get("parallel_tool_calls",False))
            gst=None
            for n in range(max_new):
                if cancel is not None and cancel.is_set():
                    finish="stop"; break
                if jgrammar is not None:
                    token=self._g4_json_token(logits,jgrammar,jstate,temperature,top_p,top_k,rng)
                elif gst is not None:
                    token=self._g4_call_token(logits,grammar,gst,temperature,top_p,top_k,rng)
                elif temperature<=0:
                    token=int(self.cp.argmax(logits).item())
                else:
                    if top_k>0 and top_k<logits.size:
                        shortlist=self.cp.argpartition(logits,-top_k)[-top_k:]
                        scores=self.cp.asnumpy(logits[shortlist])
                        candidates=self.cp.asnumpy(shortlist).astype(np.int64,copy=False)
                    else:
                        scores=self.cp.asnumpy(logits)
                        candidates=np.arange(scores.size,dtype=np.int64)
                    token=Sampler._pick(candidates,scores,temperature,top_p,top_k,rng)
                if token in self.tok.stop_ids:
                    finish="stop"; break
                if want_lp:
                    lz=logits.astype(self.cp.float32); m=lz.max()
                    lse=float((m+self.cp.log(self.cp.exp(lz-m).sum())).item())
                    ent={"token":self._g4_text(token),"id":int(token),"logprob":float(lz[token].item())-lse}
                    if n_lp:
                        ti=self.cp.argpartition(-lz,n_lp)[:n_lp]; tv=self.cp.asnumpy(lz[ti]); ti=self.cp.asnumpy(ti)
                        ent["top_logprobs"]=[{"token":self._g4_text(int(ti[j])),"id":int(ti[j]),"logprob":float(tv[j])-lse}
                                             for j in np.argsort(-tv)]
                    lp_out.append(ent)
                generated.append(token); self.busy["generated"]=len(generated)
                if n+1<max_new:      # queue the next step first: the text work below runs while the GPU computes it
                    logits=self.eng.forward(token,len(ids)+n); self.g4cache.append(token)
                piece=decoder.decode(self.tok.decode_bytes(token))
                text+=piece
                if jgrammar is not None:
                    advanced=jgrammar.advance(jstate,self._g4_text(token))
                    if advanced is None:
                        raise RuntimeError("structured-output grammar rejected a token it had allowed")
                    jstate=advanced
                hit=None
                if stop_seq:
                    _base=max(0,stop_from-(stop_win-1)); _tail=text[_base:]
                    for x in stop_seq:
                        _i=_tail.find(x)
                        if _i>=0:
                            hit=x; hit_at=_base+_i; break
                    if hit is None:
                        stop_from=len(text)
                if hit:
                    text=text[:hit_at]; finish="stop"; break
                if piece:
                    content.append(piece)
                    for kind,part in sorter.feed(piece):
                        # with tools, the text goes out once, parsed, at the end: a client never sees call markup
                        if kind=="text" and tools: continue
                        yield kind,part
                if grammar is not None:
                    if gst is None:
                        if token==call_open: gst=grammar.start()
                    else:
                        gst=grammar.advance(gst,self._g4_text(token))
                        if grammar.done(gst):
                            gst=None
                            if not parallel:               # one call per reply, as llama.cpp without parallel_tool_calls
                                finish="stop"; break
            self.busy["active"]=False
        except BaseException:
            self.busy["active"]=False
            raise
        elapsed=max(0.001,time.time()-prompt_done)
        self.stats["gen_seconds"]+=elapsed
        self.stats["last_tok_s"]=round(len(generated)/elapsed,2)
        self.stats["prompt_tokens"]+=len(ids)
        self.stats["cached_tokens"]+=self.busy.get("cached",0)
        self.stats["completion_tokens"]+=len(generated)
        self.log("Gemma 4 done: %d tokens in %.1fs (%.1f tok/s, %s)"%
                 (len(generated),elapsed,len(generated)/elapsed,finish))
        raw=(gemma4_tools.THINK_OPEN+"thought\n" if thinking_open else "")+"".join(content)
        reply,reasoning,calls=gemma4_tools.split_reply(raw)
        if not tools and calls: reply,calls=raw.strip(),[]
        if tools and not calls and gemma4_tools.CALL_CLOSE in raw:
            self.log("Gemma 4: call markup but no call parsed; the reply starts %r"%raw[:160])
        if calls: finish="tool_calls"
        if tools and reply: yield "text",reply
        yield "done",{"content":reply,"tool_calls":calls,"reasoning":reasoning,"finish":finish,"logprobs":lp_out,
          "usage":{"prompt_tokens":len(ids),"completion_tokens":len(generated),"total_tokens":len(ids)+len(generated),
                   "prompt_tokens_details":{"cached_tokens":self.busy.get("cached",0)}},
          "timing":{"prompt_s":round(prompt_done-started,2),"prompt_new_tokens":len(ids),
                    "gen_s":round(elapsed,2),"gen_tok_s":round(len(generated)/elapsed,2)}}

    def check_prompt(self, req):
        """Renders and counts the prompt of a request without the GPU or the lock, and returns its token ids. Raises
        ContextOverflow (the prompt does not fit the context) or ValueError (the prompt is not valid), so that a
        request gets an HTTP 400 at once: before the first byte of a stream, and before it waits for the engine."""
        g = self.generate(dict(req, _kindi_check=True))
        try:
            while True:
                next(g)
        except StopIteration as e:
            return list(e.value or [])

    def encode_text(self, text, special=True):
        """the token ids of plain text (no chat template)"""
        try:
            return list(self.tok.encode(text, add_bos=special))     # SentencePiece tokenizers (Nanbeige)
        except TypeError:
            return list(self.tok.encode(text))

    def decode_ids(self, ids):
        dec = getattr(self.tok, "decode", None)
        if dec is not None:
            return dec([int(i) for i in ids])
        return b"".join(self.tok.decode_bytes(int(i)) for i in ids).decode("utf-8", "replace")

    def after_turn(self, locked=False):
        """after a request has been answered, on the engine thread: queue the end-of-turn state for disk.
        locked: the caller holds self.lock already (the stream generator)."""
        r = getattr(self, "runner", None)
        if r is None or not hasattr(r, "turn_end"):
            return
        if locked:
            r.turn_end()
        else:
            with self.lock:
                r.turn_end()

    def drafting(self):
        """speculative drafting, for /health (Kindi Bridge shows it): what is actually attached, not
        what KINDI_SPEC asked for - the effective mode, why it differs from the request, the loaded
        head and its digest - and the last request's drafts and how many were accepted (REQ-0024)."""
        last = getattr(self, "last_spec", None)
        requested = os.environ.get("KINDI_SPEC", "0") or "0"
        out = {"enabled": False, "method": None, "requested": requested, "reason": None,
               "head": None, "head_sha256": None, "drafter": None, "last": None}
        if self.nanbeige is not None:
            m = [n for n, on in (("lookup", int(self.cfg.get("lookup") or 0) > 0),
                                 ("mtp", bool(self.cfg.get("mtp_head")))) if on]
            out.update(enabled=bool(m), method="+".join(m) or None)
        elif not self.native and hasattr(self.eng, "spec_begin"):
            from . import spec as _sp
            st = _sp.effective_state(self.eng, requested)
            head = st["head"] or {}
            dr = getattr(self.eng, "drafter", None)
            out.update(enabled=st["mode"] is not None, method=st["mode"], reason=st["reason"],
                       head=head.get("file") if head.get("state") == "loaded" else None,
                       head_sha256=(head.get("sha256") or "")[:12] or None,
                       drafter=getattr(getattr(dr, "eng", None), "pack", None) or (type(dr).__name__ if dr else None))
        if last and last.get("drafted"):
            out["last"] = {"passes": last.get("passes", 0), "drafted": last["drafted"], "accepted": last.get("accepted", 0),
                           "acceptance": round(last.get("accepted", 0) / float(last["drafted"]), 3)}
        return out

    def progress(self):
        """the request being served: phase, tokens reused, read and to read, generated"""
        b = self.busy
        if not b["active"]:
            return None
        return {"phase": b["phase"], "elapsed_s": round(time.time() - b["started"]),
                "prompt_tokens": b["prompt_tokens"], "reused": b["cached"], "read": b["read"],
                "to_read": max(0, b["prompt_tokens"] - b["cached"]), "generated": b["generated"]}

    def generate(self, req, cancel=None):
        """Yields ("text", piece) while generating, then ("done", info). cancel: a threading.Event set when the
        client disconnects (prompt reading and generation stop at the next token)."""
        if self.nanbeige is not None:
            return (yield from self.nanbeige.generate(self, req, cancel))
        if self.native:
            return (yield from self._generate_gemma(req, cancel))
        grammar = None
        rf = req.get("response_format")
        if is_structured(rf):
            grammar = _response_format_grammar(rf, req)
            gstate = grammar.start()
            if not hasattr(self, "_grammar_texts"):
                self._grammar_texts = [self.tok.decode_bytes(i).decode("utf-8", "replace")
                                        for i in range(len(self.tok.vocab))]
        cfg = self.cfg
        messages = req.get("messages") or []
        tools = req.get("tools") or None
        if req.get("tool_choice") == "none":
            tools = None
        think = bool((req.get("chat_template_kwargs") or {}).get("enable_thinking", req.get("enable_thinking", False)))
        budget = req.get("thinking_budget_tokens") or req.get("thinking_budget") or cfg.get("thinking_budget") or 0
        budget = int(budget) if think else 0
        from .profiles import sampling_defaults
        sd = sampling_defaults(cfg, think)          # the publisher profile and the explicit config values
        g = lambda k, d: d if req.get(k) is None else req.get(k)  # noqa: E731

        def dflt(k, fallback):
            v = sd.get(k)
            return fallback if v is None else v
        temperature = float(g("temperature", dflt("temperature", cfg["temperature"])))
        top_p = float(g("top_p", dflt("top_p", cfg["top_p"])))
        top_k = int(g("top_k", dflt("top_k", cfg["top_k"])))
        min_p = float(g("min_p", dflt("min_p", 0.0)))
        rp = float(g("repeat_penalty", dflt("repeat_penalty", 1.0)))
        rlast = int(g("repeat_last_n", dflt("repeat_last_n", 64)))
        pp = float(g("presence_penalty", dflt("presence_penalty", 0.0)))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens")
                      or dflt("max_tokens", cfg["max_tokens"]))
        repeat_stop = bool(cfg.get("repeat_stop"))
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        stop_seq = [s for s in stops if s]
        stop_win = max((len(s) for s in stop_seq), default=0)
        stop_from = 0       # text[:stop_from] is known stop-free (the check runs every token unless thinking)
        seed = req.get("seed")
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000) % 2 ** 31)
        kw = req.get("chat_template_kwargs") or {}
        effort = req.get("reasoning_effort") or kw.get("reasoning_effort") or cfg.get("reasoning_effort") or None
        body, gp = chat.render(messages, tools, think, variant=self.variant, reasoning_effort=effort,
                               preserve_thinking=kw.get("preserve_thinking"))
        Tb = self.tok.encode(body)
        T = Tb + self.tok.encode(gp)
        ctx = self.eng.ctx
        if len(T) + 1 >= ctx:
            raise ContextOverflow(len(T), ctx)
        if req.get("_kindi_check"):                       # check_prompt(): the prompt is valid, nothing to run
            return T
        fast = self.head_mode() == "fast"
        if fast and self.eng.hsl is None:
            fast = False
        self.eng.shortlist = cfg["fast_clusters"] if fast else 0
        max_new = max(1, min(max_new, ctx - len(T) - 1))
        t0 = time.time()
        self.busy.update(active=True, phase="matching cache", prompt_tokens=len(T), cached=0, read=0, generated=0,
                         started=t0)
        self.log("request: %d prompt tokens (max %d new)" % (len(T), max_new))
        try:
            im_end = 248046                                   # <|im_end|> closes the first message
            opening_end = T.index(im_end) + 1 if im_end in T else len(T)
            session = req.get("_kindi_session") or req.get("kindi_session") or req.get("user")
            nxt, cached = self.runner.prepare(T, len(Tb), opening_end, cancel, session=session)
        except Cancelled as e:
            self.busy["active"] = False
            self.log("stopped: %s" % e)
            return
        except BaseException:
            self.busy["active"] = False
            raise
        t1 = time.time()
        self.log("prompt: %d reused, %d read in %.1fs" % (cached, len(T) - cached, t1 - t0))
        self.busy["phase"] = "generating"
        pen = rp != 1.0 and rlast != 0
        if hasattr(self.eng, "spec_begin"):
            if pen or pp or grammar is not None or (min_p > 0 and temperature > 0):  # the speculative paths do not apply these: no drafts
                self.eng.spec_begin(False)
            else:                                              # speculative decoding (KINDI_SPEC), min-p included
                self.eng.spec_begin(temperature <= 0, temperature, top_p, top_k, rng, min_p)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text, gen, finish = "", [], "length"
        # while thinking, pieces go out as ("reasoning", ...) - the last 8 characters are held back in case they are
        # the start of "</think>"; after it, as ("text", ...)
        st = {"think": think, "sent": 0, "forced": False, "lead": think}

        def route(piece):
            if not st["think"]:
                if st["lead"]:                                # the newlines that follow </think>
                    piece = piece.lstrip("\n")
                    st["lead"] = not piece
                return [("text", piece)] if piece else []
            out = []
            if "</think>" in text:
                end = text.index("</think>")
                if end > st["sent"]:
                    out.append(("reasoning", text[st["sent"]:end]))
                after = text[end + len("</think>"):].lstrip("\n")
                st["think"] = False
                if after:
                    st["lead"] = False
                    out.append(("text", after))
            elif len(text) - 8 > st["sent"]:
                out.append(("reasoning", text[st["sent"]:len(text) - 8]))
                st["sent"] = len(text) - 8
            return out

        # repeat_penalty: over the last repeat_last_n tokens of the prompt and the reply (-1: all of them)
        recent = lambda: (rp, set((T + gen)[-rlast:] if rlast > 0 else T + gen)) if pen else None  # noqa: E731

        def presence():
            """presence_penalty: once for every token already in the prompt or the reply (OpenAI's additive penalty)"""
            if not pp:
                return None
            mask = np.zeros(len(self.eng.vocab), dtype=bool)
            mask[np.asarray(T + gen, dtype=np.int64)] = True
            return pp, mask

        allowed = None
        if grammar is not None:
            allowed = grammar.vocab_allowed(gstate, self._grammar_texts)
            if grammar.done(gstate):
                allowed = np.concatenate([allowed, np.array(EOS, dtype=np.int64)])
        tok = self.sample(nxt, temperature, top_p, top_k, rng, min_p, recent(), presence(), allowed=allowed)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    self.log("stopped: the client went away after %d tokens" % len(gen))
                    return
                if tok in EOS:
                    finish = "stop"
                    break
                gen.append(tok)
                self.busy["generated"] = len(gen)
                piece = dec.decode(self.tok.decode_bytes(tok))
                text += piece
                if grammar is not None:
                    advanced = grammar.advance(gstate, piece)
                    if advanced is None:
                        raise RuntimeError(
                            "constrained grammar rejected token %r it had allowed" % piece)
                    gstate = advanced
                # the stop check scans only the not-yet-checked tail (a match verified earlier cannot
                # reappear), instead of the whole reply every token, which made generation O(n^2).
                hit = None
                if stop_seq and not st["think"]:
                    base = max(0, stop_from - (stop_win - 1))
                    tail = text[base:]
                    for s in stop_seq:
                        i = tail.find(s)
                        if i >= 0:
                            hit, hit_at = s, base + i
                            break
                    if hit is None:
                        stop_from = len(text)
                if hit is not None:
                    text = text[:hit_at]
                    finish = "stop"
                    break
                for item in route(piece):
                    yield item
                if len(gen) >= max_new:
                    break
                loop = repeat_stop and len(gen) % 16 == 0 and _looping(gen)
                if loop and not st["think"]:
                    self.log("stopped a repeating loop after %d tokens" % len(gen))
                    finish = "length"
                    break
                nxt = self.runner.step(tok)
                if loop and st["think"] and not st["forced"]:
                    budget = len(gen)                         # a loop in the thinking: close it now, then answer
                if st["think"] and budget and len(gen) >= budget and not st["forced"]:
                    # thinking budget spent: close the thinking for the model, then let it answer
                    st["forced"] = True
                    self.log("thinking budget of %d tokens reached; closing the thinking" % budget)
                    for ft in self.tok.encode(THINK_CLOSE):
                        gen.append(ft)
                        piece = dec.decode(self.tok.decode_bytes(ft))
                        text += piece
                        for item in route(piece):
                            yield item
                        nxt = self.runner.step(ft)
                allowed = None
                if grammar is not None:
                    allowed = grammar.vocab_allowed(gstate, self._grammar_texts)
                    if grammar.done(gstate):
                        allowed = np.concatenate([allowed, np.array(EOS, dtype=np.int64)])
                tok = self._pick(nxt, temperature, top_p, top_k, rng, min_p, recent(), presence(),
                                 allowed=allowed)
        finally:
            sp = self.eng.spec_end() if hasattr(self.eng, "spec_end") else None
            self.busy["active"] = False
            if hasattr(self.runner, "idle"):
                self.runner.idle()                            # (KINDI_CKPT_VRAM: checkpoint host copies now)
        if sp:
            self.last_spec = dict(sp)
        if sp and sp["passes"]:
            self.log("speculative: %d tokens in %d passes, %d of %d drafted tokens accepted" % (
                len(gen), sp["passes"], sp["accepted"], sp["drafted"]))
            self.stats["spec_passes"] += sp["passes"]
            self.stats["spec_drafted"] += sp["drafted"]
            self.stats["spec_accepted"] += sp["accepted"]
        if st["think"] and len(text) > st["sent"]:
            yield "reasoning", text[st["sent"]:]
        t2 = time.time()
        self.log("done: %d tokens in %.1fs (%s)" % (len(gen), t2 - t1, finish))
        self.stats["gen_seconds"] += t2 - t1
        self.stats["last_tok_s"] = round(len(gen) / (t2 - t1), 2) if t2 > t1 else 0.0
        for k, v in (("requests", 1), ("prompt_tokens", len(T)), ("cached_tokens", cached),
                     ("completion_tokens", len(gen))):
            self.stats[k] += v
        reasoning = ""
        if "</think>" in text:
            reasoning, text = text.split("</think>", 1)
            reasoning = reasoning.replace("<think>", "").strip()
            text = text.lstrip("\n")
        content, calls = chat.parse_tool_calls(text, tools) if tools else (text.strip(), [])
        if calls:
            finish = "tool_calls"
        yield "done", {"content": content, "tool_calls": calls, "reasoning": reasoning, "finish": finish,
                       "usage": {"prompt_tokens": len(T), "completion_tokens": len(gen),
                                 "total_tokens": len(T) + len(gen), "prompt_tokens_details": {"cached_tokens": cached}},
                       "timing": {"prompt_s": round(t1 - t0, 2), "prompt_new_tokens": len(T) - cached,
                                  "gen_s": round(t2 - t1, 2),
                                  "gen_tok_s": round(len(gen) / (t2 - t1), 2) if t2 > t1 else 0.0}}


# Every engine call runs on this one thread. CuPy keeps a cuBLAS/hipBLAS handle per thread: on the default executor,
# each new worker thread made a new handle, and on a full card that fails (CUBLAS_STATUS_ALLOC_FAILED) once the K/V and
# the parked conversations hold the memory. One thread makes one handle, at its first use, as one request did before.
# (A handle made before the engine would load the cuBLAS code ahead of the context fit and cost small cards context.)
GPU_THREAD = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="kindi-gpu")

# priority classes (0.6.0): every request names a class with "kindi_priority" (or the header X-Kindi-Priority).
# chat (the default) goes before background (cron jobs, batch work, agents' side work). A background request that
# is reading its prompt stops between two chunks when a chat request arrives; the chat request runs, and the
# background request then goes on from the prompt cache (its tokens read so far stay in the cache, or parked).
PRIO_TURN, PRIO_CHAT, PRIO_BACKGROUND = -1, 0, 1
_PRIO_NAMES = {"chat": PRIO_CHAT, "interactive": PRIO_CHAT, "high": PRIO_CHAT,
               "background": PRIO_BACKGROUND, "cron": PRIO_BACKGROUND, "batch": PRIO_BACKGROUND, "low": PRIO_BACKGROUND}


def priority_of(req, headers=None):
    v = req.get("kindi_priority") or (headers.get("x-kindi-priority") if headers is not None else None) or "chat"
    return _PRIO_NAMES.get(str(v).strip().lower(), PRIO_CHAT)


class Scheduler(object):
    """The engine jobs in priority order on the one engine thread: each task on GPU_THREAD runs the best job waiting
    at that moment (the lowest priority number, then the oldest)."""

    def __init__(self):
        import heapq
        self.heapq, self.heap, self.lk, self.seq = heapq, [], threading.Lock(), 0

    def submit(self, fn, prio=PRIO_CHAT):
        f = concurrent.futures.Future()
        with self.lk:
            self.heapq.heappush(self.heap, (prio, self.seq, fn, f))
            self.seq += 1
        GPU_THREAD.submit(self._run_one)
        return f

    def _pop(self, max_prio=None):
        with self.lk:
            if not self.heap or (max_prio is not None and self.heap[0][0] > max_prio):
                return None
            return self.heapq.heappop(self.heap)

    def _run(self, job):
        _, _, fn, f = job
        if not f.set_running_or_notify_cancel():
            return
        try:
            f.set_result(fn())
        except BaseException as e:  # noqa: BLE001 - the waiting request reports it
            f.set_exception(e)

    def _run_one(self):
        job = self._pop()
        if job is not None:
            self._run(job)

    def waiting(self, max_prio=PRIO_CHAT):
        """jobs waiting with this priority or a better one"""
        with self.lk:
            return sum(1 for j in self.heap if j[0] <= max_prio)

    def counts(self):
        with self.lk:
            return {"chat": sum(1 for j in self.heap if j[0] == PRIO_CHAT),
                    "background": sum(1 for j in self.heap if j[0] == PRIO_BACKGROUND)}

    def run_waiting(self, max_prio=PRIO_CHAT):
        """run the waiting jobs of this priority or better now, on this thread (a background request yields)"""
        n = 0
        while True:
            job = self._pop(max_prio)
            if job is None:
                return n
            self._run(job)
            n += 1


SCHED = Scheduler()


class _Yield(object):
    """the cancel event of a background request: also set while it reads its prompt and a chat request waits"""

    def __init__(self, cancel, K):
        self.cancel, self.K, self.fired = cancel, K, False

    def is_set(self):
        if self.cancel.is_set():
            return True
        if self.K.busy.get("phase") in ("matching cache", "reading prompt", "prompt") and SCHED.waiting(PRIO_CHAT):
            self.fired = True
            return True
        return False

    def set(self):
        self.cancel.set()


def run_generate(K, req, cancel, prio):
    """K.generate with the priority classes: a background request that yields to chat requests runs them here, then
    starts again (from the prompt cache). Yields what K.generate yields."""
    while True:
        c = _Yield(cancel, K) if prio >= PRIO_BACKGROUND else cancel
        done = False
        for item in K.generate(req, c):
            if item[0] == "done":
                done = True
            yield item
        if done or cancel.is_set() or not getattr(c, "fired", False):
            return
        K.yields = getattr(K, "yields", 0) + 1
        K.log("a background request yields to %d chat request(s) while it reads its prompt" % SCHED.waiting(PRIO_CHAT))
        SCHED.run_waiting(PRIO_CHAT)


def _after_engine_error(K, e):
    """An engine error other than a bad request. Drop the cache state (it may be half updated), and if the GPU
    context is broken (an illegal memory access is sticky in CUDA: every later call fails too), exit so that the
    container's restart policy brings the server back, in place of answering every request with an error."""
    stats = getattr(K, "stats", None)
    if isinstance(stats, dict):
        stats["errors"] = stats.get("errors", 0) + 1
    r = getattr(K, "runner", None)
    if r is not None:
        try:
            r.eng.reset(); r.tokens = []; r.ckpt.clear()
        except Exception:  # noqa: BLE001 - reset is best-effort; the engine is being discarded anyway
            pass
    if getattr(K, "cp", None) is None or os.environ.get("KINDI_DEVICE", "gpu") == "cpu":
        return                                   # the CPU engine has no GPU context to lose
    broken = "IllegalAddress" in str(e) or "illegal memory access" in str(e)
    try:
        K.cp.cuda.runtime.deviceSynchronize()
        K.cp.cuda.runtime.memGetInfo()
    except Exception:  # noqa: BLE001 - the probe itself failing proves the context is broken
        broken = True
    if broken:
        print("[kindi] the GPU context is broken (%s): exiting so that the server restarts" % e, flush=True)
        threading.Timer(1.0, os._exit, (70,)).start()


def _make_engine(cfg, log):
    return Kindi(cfg, log)


def create_app(cfg, log=print):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    log("[kindi] building engine for %s ..." % cfg.get("model_name", "model"))
    try:
        K = GPU_THREAD.submit(_make_engine, cfg, log).result()
    except Exception:  # noqa: BLE001 - logged and re-raised; startup must fail, not serve
        log("[kindi] engine construction failed")
        raise
    log("[kindi] engine ready for %s" % cfg.get("model_name", "model"))
    model_id = cfg["model_name"]
    image_error = _image_error(cfg.get("pack"))
    if not cfg.get("api_key") and not _is_loopback(cfg.get("host", "127.0.0.1")):
        log("warning: serving on %s without api_key; /health details are loopback-only - use loopback or set api_key"
            % cfg.get("host"))
    app = FastAPI(title="Kindi", version=__version__)

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):
        status, body, headers = api_errors.response(
            api_errors.OpenAIError("internal server error", status=500,
                                   error_type="server_error", code="server_error"))
        return JSONResponse(body, status_code=status, headers=headers)

    @app.exception_handler(404)
    async def _not_found(request, exc):
        status, body, headers = api_errors.response(
            api_errors.OpenAIError("not found", status=404,
                                   error_type="not_found", code="not_found"))
        return JSONResponse(body, status_code=status, headers=headers)

    @app.exception_handler(405)
    async def _method_not_allowed(request, exc):
        status, body, headers = api_errors.response(
            api_errors.OpenAIError("method not allowed", status=405,
                                   error_type="method_not_allowed", code="method_not_allowed"))
        return JSONResponse(body, status_code=status, headers=headers)

    @app.exception_handler(422)
    async def _validation_failed(request, exc):
        detail = getattr(exc, "detail", None)
        param = None
        if isinstance(detail, list) and detail:
            loc = detail[0].get("loc") if isinstance(detail[0], dict) else None
            if isinstance(loc, list) and loc:
                param = str(loc[-1])
        status, body, headers = api_errors.response(
            api_errors.OpenAIError("request validation failed", status=422,
                                   error_type="invalid_request_error", code="invalid_request_error",
                                   param=param))
        return JSONResponse(body, status_code=status, headers=headers)

    max_body_mb = cfg.get("max_body_mb")
    app.state.max_body_bytes = (MAX_BODY_BYTES if max_body_mb is None
                                else _env_int("max_body_mb", max_body_mb, 1) * 1024 * 1024)
    gate = K.gate = Gate(cfg.get("queue_max", MAX_QUEUE))
    limiter = RateLimiter(cfg.get("rate_limit_per_min", 0))
    access_log = bool(cfg.get("log_json"))

    def _access(meta, status=200, info=None, error=None):
        """One JSON line per inference request when log_json is on. Headers and the key are never logged."""
        if not access_log or not meta or not meta.get("id") or meta.get("logged"):
            return
        meta["logged"] = True
        line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "level": "info" if status < 400 else "warning", "msg": "request",
                "request_id": meta["id"], "method": meta["method"], "path": meta["path"],
                "status": status,
                "duration_ms": round((time.time() - meta["t0"]) * 1000, 1),
                "queue_wait_ms": round(meta["wait"] * 1000, 1),
                "stream": bool(meta["stream"]), "model": model_id}
        if error:
            line["error"] = str(error)[:200]
        if info:
            usage, timing = info.get("usage") or {}, info.get("timing") or {}
            line.update(prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        gen_tok_s=timing.get("gen_tok_s"), finish=info.get("finish"))
        log(json.dumps(line, ensure_ascii=False))

    def _logged(gen, meta, stream):
        """Relay the engine generator unchanged and write its access line when it finishes or fails."""
        meta["stream"] = stream
        info, status, error = None, 200, None
        try:
            for kind, val in gen:
                if kind == "done":
                    info = val
                yield kind, val
        except OverflowError as e:
            status, error = 400, str(e)
            raise
        except ValueError as e:
            status, error = 400, str(e)
            raise
        except Exception as e:  # noqa: BLE001 - the endpoint turns this into its own error response
            status, error = 500, str(e)
            raise
        finally:
            _access(meta, status, info, error)

    @app.middleware("http")
    async def _request_meta(request, call_next):
        rid = request.headers.get("x-request-id", "")
        if not (8 <= len(rid) <= 64 and all(c.isalnum() or c in "-._" for c in rid)):
            rid = "req-" + uuid.uuid4().hex[:16]
        meta = {"id": rid, "method": request.method, "path": request.url.path,
                "t0": time.time(), "wait": 0.0, "stream": False, "logged": False}
        _CURRENT.set(meta)
        request.state.kindi_meta = meta
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    def _err(code, msg, typ="invalid_request_error", headers=None, param=None, error_code=None):
        _access(_CURRENT.get(), code, error=msg)
        status, body, resp_headers = api_errors.response(
            api_errors.OpenAIError(msg, status=code, error_type=typ,
                                   code=error_code if error_code is not None else typ,
                                   param=param),
            status=code, headers=headers)
        return JSONResponse(body, status_code=status, headers=resp_headers)

    def _busy(e):
        return _err(429, str(e), "server_busy", {"Retry-After": "1"})

    def _auth(request):
        key = cfg["api_key"]
        if not key:
            return True
        supplied = request.headers.get("authorization", "")
        return hmac.compare_digest(supplied.encode("utf-8"), ("Bearer " + key).encode("utf-8"))

    def _rate_limited(request):
        """A 429 response when this source address exhausted its generation budget, else None."""
        wait = limiter.retry_after(request.client.host if request.client else "-")
        if wait:
            return _err(429, "rate limit exceeded: retry in %d s (KINDI_RATE_LIMIT_PER_MIN=%d)"
                        % (wait, limiter.per_minute), "rate_limit_exceeded", {"Retry-After": str(wait)})
        return None

    OLLAMA_OPTIONS = (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                      ("repeat_penalty", "repeat_penalty"), ("repeat_last_n", "repeat_last_n"),
                      ("presence_penalty", "presence_penalty"),
                      ("seed", "seed"), ("num_predict", "max_tokens"), ("stop", "stop"))

    def _ollama_req(body, prompt_mode, image_error):
        """Map an Ollama request to the internal one; validate it before it reaches the engine or the cache."""
        if not isinstance(body, dict):
            raise RequestError("the body must be a JSON object")
        stream = body.get("stream", True)
        if not isinstance(stream, bool):
            raise RequestError("stream must be true or false")
        opts = body.get("options") or {}
        if not isinstance(opts, dict):
            raise RequestError("options must be an object")
        if body.get("images"):
            raise RequestError(image_error)
        if prompt_mode:
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise RequestError("prompt must be a non-empty string")
            messages = []
            if body.get("system") is not None:
                if not isinstance(body["system"], str):
                    raise RequestError("system must be a string")
                messages.append({"role": "system", "content": body["system"]})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = body.get("messages")
        req = {"messages": messages}
        for src, dst in OLLAMA_OPTIONS:
            if opts.get(src) is not None:
                req[dst] = opts[src]
        if req.get("max_tokens") in (-1, 0):
            req.pop("max_tokens")                # Ollama's -1 means "until the context is full"
        if body.get("tools") is not None:
            req["tools"] = body["tools"]
        _validate_request(req, image_error)
        return req, stream

    def _ollama_line(info, created, prompt_mode, final=False):
        """The final Ollama object of a request (done=true), from the internal done info."""
        t, u = info["timing"], info["usage"]
        line = {"model": model_id, "created_at": created, "done": True,
                "done_reason": info["finish"], "load_duration": 0,
                "prompt_eval_count": u["prompt_tokens"], "prompt_eval_duration": int(t["prompt_s"] * 1e9),
                "eval_count": u["completion_tokens"], "eval_duration": int(t["gen_s"] * 1e9),
                "total_duration": int((t["prompt_s"] + t["gen_s"]) * 1e9)}
        if prompt_mode:
            line["response"] = "" if final else info["content"]
        else:
            msg = {"role": "assistant", "content": "" if final else info["content"]}
            if info.get("reasoning"):
                msg["thinking"] = info["reasoning"]
            if info.get("tool_calls"):
                msg["tool_calls"] = info["tool_calls"]
            line["message"] = msg
        return line

    async def _relay_ndjson(gen, cancel):
        """Run a blocking generator on the GPU thread and relay its lines, so the event loop stays free."""
        loop = asyncio.get_event_loop()
        q = asyncio.Queue()
        END = object()

        def pump():
            try:
                for item in gen:
                    loop.call_soon_threadsafe(q.put_nowait, item)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, END)

        fut = loop.run_in_executor(GPU_THREAD, pump)
        try:
            while True:
                item = await q.get()
                if item is END:
                    break
                yield item
        finally:
            cancel.set()
            await asyncio.wait([fut], timeout=30)

    async def _ollama(request, prompt_mode):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        limited = _rate_limited(request)
        if limited is not None:
            return limited
        try:
            body = await _read_json_body(request)
        except RequestTooLarge as e:
            return _err(413, str(e), "request_too_large")
        except RequestError as e:
            return _err(400, str(e), param=getattr(e, "param", None))
        try:
            req, stream = _ollama_req(body, prompt_mode, image_error)
        except RequestError as e:
            return _err(400, str(e), param=getattr(e, "param", None))
        try:
            gate.enter()
        except QueueFull as e:
            return _busy(e)
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cancel = threading.Event()
        meta = _CURRENT.get() or {}
        meta["stream"] = bool(stream)
        if stream:
            def events():
                try:
                    t_wait = time.time()
                    with K.lock:
                        meta["wait"] = time.time() - t_wait
                        for kind, val in _logged(K.generate(req, cancel), meta, True):
                            if kind == "text":
                                line = {"model": model_id, "created_at": created, "done": False}
                                if prompt_mode:
                                    line["response"] = val
                                else:
                                    line["message"] = {"role": "assistant", "content": val}
                                yield json.dumps(line, ensure_ascii=False) + "\n"
                            elif kind == "reasoning" and not prompt_mode:
                                yield json.dumps({"model": model_id, "created_at": created, "done": False,
                                                  "message": {"role": "assistant", "content": "",
                                                              "thinking": val}}, ensure_ascii=False) + "\n"
                            else:
                                yield json.dumps(_ollama_line(val, created, prompt_mode, final=True),
                                                 ensure_ascii=False) + "\n"
                except (OverflowError, ValueError) as e:
                    yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
                except Exception as e:  # noqa: BLE001 - report it on the stream, as the SSE path does
                    yield json.dumps({"error": "kindi: %s" % e}, ensure_ascii=False) + "\n"
                    _after_engine_error(K, e)
                finally:
                    gate.release()
            return StreamingResponse(_relay_ndjson(events(), cancel), media_type="application/x-ndjson")

        def run():
            t_wait = time.time()
            with K.lock:
                meta["wait"] = time.time() - t_wait
                info = None
                for kind, val in _logged(K.generate(req, cancel), meta, False):
                    if kind == "done":
                        info = val
                return info
        try:
            fut = asyncio.get_event_loop().run_in_executor(GPU_THREAD, run)
            while not fut.done():
                done, _ = await asyncio.wait([fut], timeout=1.0)
                if not done and await request.is_disconnected():
                    cancel.set()
            info = fut.result()
            if info is None:
                return _err(499, "request cancelled", "cancelled")
            return _ollama_line(info, created, prompt_mode)
        except OverflowError as e:
            return _err(400, str(e), "context_length_exceeded")
        except ValueError as e:
            return _err(400, str(e))
        except Exception as e:  # noqa: BLE001 - an engine failure, not a bad request
            _after_engine_error(K, e)
            return _err(500, "kindi: %s" % e, "server_error")
        finally:
            gate.release()

    @app.get("/health")
    def health(request: Request):
        # A probe needs only liveness, version, model and context. The detailed view (GPU memory, queue, token
        # counters and live progress) is available without a key only on a loopback bind, but is protected when
        # api_key is set: /health is often exposed so a container or LAN monitor can poll it, and that probe must
        # not become an unauthenticated inventory/usage endpoint for a network bind.
        body = {"ok": True, "kindi": __version__, "commit": COMMIT, "model": model_id, "ctx": K.eng.ctx}
        if cfg["api_key"]:
            if not _auth(request):
                return body
        elif not _is_loopback(cfg.get("host", "127.0.0.1")):
            return body
        free = total = 0
        if K.cp is not None and os.environ.get("KINDI_DEVICE", "gpu") != "cpu":
            try:
                free, total = K.cp.cuda.runtime.memGetInfo()
            except Exception:  # noqa: BLE001 - health must stay available after a GPU-context failure
                pass
        b = K.busy
        busy = None
        if b["active"]:
            busy = {"phase": b["phase"], "elapsed_s": round(time.time() - b["started"]),
                    "prompt_tokens": b["prompt_tokens"], "reused": b["cached"],
                    "prompt_read": "%d/%d" % (b["read"], b["prompt_tokens"] - b["cached"]), "generated": b["generated"]}
        body.update({"head": K.head_active(), "uptime_s": round(time.time() - K.t0), "busy": busy, "stats": K.stats,
                     "drafting": K.drafting(),
                     "queue": {"waiting": gate.waiting, "max": gate.max_queue, "rejected": gate.rejected,
                               "scheduler": dict(SCHED.counts(), yields=getattr(K, "yields", 0))},
                     "rate_limit_per_min": limiter.per_minute, "rate_limited": limiter.rejected,
                     "gpu_free_mib": round(free / 1048576.0), "gpu_total_mib": round(total / 1048576.0)})
        return body

    def _overflow(e):
        return JSONResponse(api_errors.error_body(str(e), code="context_length_exceeded",
                                                  param="messages"),
                            status_code=400)

    @app.get("/v1/models")
    def models(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        ctx = K.eng.ctx
        # the real window under every name that agent harnesses read (vLLM, llama.cpp, LM Studio, OpenRouter style)
        return {"object": "list", "data": [{"id": model_id, "object": "model", "created": int(K.t0),
                                            "owned_by": "kindi", "context_length": ctx, "max_model_len": ctx,
                                            "max_context_length": ctx, "meta": {"n_ctx": ctx}}]}

    async def _tokenize(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        try:
            req = await _read_json_body(request)      # the same 16 MiB cap as the chat routes
        except RequestTooLarge as e:
            return _err(413, str(e), "request_too_large")
        except RequestError as e:
            return _err(400, str(e), param=getattr(e, "param", None))
        if not isinstance(req, dict):
            return _err(400, "send a JSON object")
        ctx = K.eng.ctx
        try:
            if req.get("messages") is not None:   # a chat request: the whole prompt, as the chat endpoint builds it
                ids = await asyncio.get_event_loop().run_in_executor(None, K.check_prompt, dict(req, stream=False))
            else:
                text = req.get("content", req.get("prompt"))
                if not isinstance(text, str):
                    return _err(400, "send content (text) or messages")
                ids = K.encode_text(text, req.get("add_special_tokens", True) is not False)
        except ContextOverflow as e:              # counting still works: report the count and that it does not fit
            return {"count": e.n, "max_model_len": ctx, "fits": False, "tokens": []}
        except ValueError as e:
            return _err(400, str(e))
        out = {"count": len(ids), "max_model_len": ctx, "fits": len(ids) + 1 < ctx}
        if req.get("return_tokens", True) is not False:
            out["tokens"] = ids
        return out

    async def _detokenize(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        try:
            req = await _read_json_body(request)      # the same 16 MiB cap as the chat routes
        except RequestTooLarge as e:
            return _err(413, str(e), "request_too_large")
        except RequestError as e:
            return _err(400, str(e), param=getattr(e, "param", None))
        try:
            if not isinstance(req, dict):
                raise ValueError("the body must be a JSON object")
            ids = [int(i) for i in req.get("tokens") or []]
        except (TypeError, ValueError):
            return _err(400, "send tokens: a list of token ids")
        text = K.decode_ids(ids)
        return {"prompt": text, "content": text}

    app.add_api_route("/tokenize", _tokenize, methods=["POST"])
    app.add_api_route("/v1/tokenize", _tokenize, methods=["POST"])
    app.add_api_route("/detokenize", _detokenize, methods=["POST"])
    app.add_api_route("/v1/detokenize", _detokenize, methods=["POST"])

    # Chat sessions (REQ-0016): the same host-only store `kindi chat` records and the `kindi sessions` CLI reads.
    # These routes never touch the engine, the GPU or the prompt cache.
    from .sessions import STATUSES as SESSION_STATUSES
    from .sessions import SessionStore
    from .sessions import slug as session_slug  # filename-safe title fragment, same as the CLI export

    # Each request gets its own store (so list/show see the latest files), so the lock must live here,
    # not on the store: two concurrent status/tag posts would otherwise race their read-modify-write
    # and one update would silently overwrite the other.
    _SESSIONS_LOCK = threading.Lock()

    def _session_store():
        return SessionStore(os.path.join(cfg.get("state_dir") or os.path.join(os.getcwd(), ".kindi"), "sessions"))

    def _session_summary(rec):
        return {"id": rec.get("id"), "title": rec.get("title"), "status": rec.get("status"),
                "tags": rec.get("tags") or [], "created": rec.get("created"), "updated": rec.get("updated"),
                "unread": bool(rec.get("unread")), "protected": bool(rec.get("protected")),
                "message_count": len(rec.get("messages") or [])}

    _SID_RE = re.compile(r"^[0-9a-f]{12}$")

    def _session_or_none(sid):
        """The record for sid, or an error response. Ids are the store's own 12-hex ids, so a path-shaped id is
        refused here rather than becoming a file outside the sessions directory."""
        if not _SID_RE.match(sid or ""):
            return None, _err(400, "invalid session id", "invalid_request_error", param="id")
        try:
            return _session_store().get(sid), None
        except KeyError:
            return None, _err(404, "no session %r" % sid, "not_found", param="id")

    @app.get("/sessions")
    def sessions_list(request: Request, status: str = None, tag: str = None):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        if status is not None and status not in SESSION_STATUSES:
            return _err(400, "status must be one of %s (got %r)" % (list(SESSION_STATUSES), status),
                        "invalid_request_error", param="status")
        with _SESSIONS_LOCK:
            rows = _session_store().list(status=status, tag=tag)
        return {"sessions": [_session_summary(r) for r in rows]}

    @app.get("/sessions/{sid}")
    def sessions_show(request: Request, sid: str):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        with _SESSIONS_LOCK:
            rec, err = _session_or_none(sid)
        if err is not None:
            return err
        return rec

    @app.get("/sessions/{sid}/export")
    def sessions_export(request: Request, sid: str):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        with _SESSIONS_LOCK:
            rec, err = _session_or_none(sid)
            if err is not None:
                return err
            text = _session_store().export_markdown(rec)
            name = "%s-%s.md" % (rec["created"][:10], session_slug(rec["title"]) or rec["id"])
        return Response(text, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="%s"' % name})

    async def _sessions_post(request: Request, sid: str, action):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        if action == "read":
            # clearing the unread flag needs no body fields (an empty request clears it)
            with _SESSIONS_LOCK:
                rec, err = _session_or_none(sid)
                if err is not None:
                    return err
                return _session_summary(_session_store().mark_read(rec["id"]))
        try:
            body = await _read_json_body(request)
        except RequestTooLarge as e:
            return _err(413, str(e), "request_too_large")
        except RequestError as e:
            return _err(400, str(e), param=getattr(e, "param", None))
        if not isinstance(body, dict):
            return _err(400, "the body must be a JSON object")
        if action == "status":
            status = body.get("status")
            if status not in SESSION_STATUSES:
                return _err(400, "status must be one of %s (got %r)" % (list(SESSION_STATUSES), status),
                            "invalid_request_error", param="status")
        else:
            tag = body.get("tag")
            if not isinstance(tag, str) or not tag.strip():
                return _err(400, "tag must be a non-empty string", "invalid_request_error", param="tag")
        with _SESSIONS_LOCK:
            rec, err = _session_or_none(sid)
            if err is not None:
                return err
            store = _session_store()
            if action == "status":
                return _session_summary(store.set_status(rec["id"], body.get("status")))
            return _session_summary(store.add_tag(rec["id"], body.get("tag")))

    async def _sessions_status(request: Request, sid: str):
        return await _sessions_post(request, sid, "status")

    async def _sessions_tags(request: Request, sid: str):
        return await _sessions_post(request, sid, "tags")

    async def _sessions_read(request: Request, sid: str):
        return await _sessions_post(request, sid, "read")

    app.add_api_route("/sessions/{sid}/status", _sessions_status, methods=["POST"])
    app.add_api_route("/sessions/{sid}/tags", _sessions_tags, methods=["POST"])
    app.add_api_route("/sessions/{sid}/read", _sessions_read, methods=["POST"])

    @app.get("/metrics")
    def metrics(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        s = K.stats
        # `/health` already resolves the mode the next request will take; reuse that answer here
        # instead of reading KINDI_SPEC back as if it were the mode that drafted (REQ-0024 A2).
        effective_mode = (K.drafting() or {}).get("method")

        def g(k):
            return s.get(k, 0)

        def label(v):
            return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        lines = []

        def metric(name, typ, help_text, *samples):
            lines.extend(("# HELP %s %s" % (name, help_text), "# TYPE %s %s" % (name, typ)))
            lines.extend(samples)

        metric("kindi_up", "gauge", "1 while the server is up", "kindi_up 1")
        metric("kindi_uptime_seconds", "gauge", "seconds since the engine started",
               "kindi_uptime_seconds %.3f" % (time.time() - K.t0))
        metric("kindi_model_info", "gauge", "the served model",
               'kindi_model_info{model="%s",ctx="%d",head="%s"} 1'
               % (label(model_id), K.eng.ctx, label(K.head_active())))
        metric("kindi_queue_depth", "gauge", "requests admitted and waiting for the engine",
               "kindi_queue_depth %d" % gate.waiting)
        metric("kindi_queue_max", "gauge", "KINDI_QUEUE_MAX: requests that may wait behind the active one",
               "kindi_queue_max %d" % gate.max_queue)
        metric("kindi_requests_total", "counter", "generation requests completed",
               "kindi_requests_total %d" % g("requests"))
        metric("kindi_requests_rejected_total", "counter", "requests rejected because the queue was full",
               "kindi_requests_rejected_total %d" % gate.rejected)
        metric("kindi_rate_limit_per_min", "gauge", "KINDI_RATE_LIMIT_PER_MIN: per-client generation budget",
               "kindi_rate_limit_per_min %d" % limiter.per_minute)
        metric("kindi_rate_limited_total", "counter", "requests rejected by the per-client rate limit",
               "kindi_rate_limited_total %d" % limiter.rejected)
        metric("kindi_request_errors_total", "counter", "requests that failed after admission",
               "kindi_request_errors_total %d" % g("errors"))
        cache_errors = getattr(getattr(K, "runner", None), "cache_errors", 0)
        metric("kindi_cache_errors_total", "counter", "cache fallbacks after a damaged opening or failed unpark",
               "kindi_cache_errors_total %d" % cache_errors)
        opening_save_failures = getattr(getattr(getattr(K, "runner", None), "openings", None), "save_failures", 0)
        metric("kindi_opening_save_failures_total", "counter", "saved-opening writes that failed and were skipped",
               "kindi_opening_save_failures_total %d" % opening_save_failures)
        run_metrics = getattr(getattr(K, "runner", None), "metrics", None)
        if callable(run_metrics):
            m = run_metrics()
            for name, typ, help_text, key in (
                    ("kindi_cache_budget_bytes", "gauge", "host prompt-cache byte budget", "budget_bytes"),
                    ("kindi_cache_used_bytes", "gauge", "host prompt-cache bytes held now", "used_bytes"),
                    ("kindi_cache_peak_bytes", "gauge", "host prompt-cache peak bytes held", "peak_bytes"),
                    ("kindi_cache_checkpoints", "gauge", "host prompt-cache checkpoints held", "checkpoints"),
                    ("kindi_cache_parked", "gauge", "host prompt-cache parked conversations held", "parked"),
                    ("kindi_cache_evictions_total", "counter", "host prompt-cache copies dropped to fit the budget", "evictions"),
                    ("kindi_cache_evicted_bytes_total", "counter", "host prompt-cache bytes evicted", "evicted_bytes"),
                    ("kindi_cache_checkpoints_dropped_total", "counter", "host prompt-cache checkpoints dropped", "checkpoints_dropped"),
                    ("kindi_cache_parked_dropped_total", "counter", "host prompt-cache parked conversations dropped", "parked_dropped")):
                metric(name, typ, help_text, "%s %d" % (name, m[key]))
        tandem_timing = getattr(K.eng, "timing_summary", None)
        if callable(tandem_timing):
            t = tandem_timing()
            if t is not None:
                metric("kindi_tandem_tokens_total", "counter", "tokens timed by the CPU/GPU split probe", "kindi_tandem_tokens_total %d" % t["tokens"])
                for name, help_text, key in (
                        ("kindi_tandem_cpu_ms_per_token", "tandem probe: milliseconds per token on the CPU layers", "cpu_ms_per_token"),
                        ("kindi_tandem_gpu_ms_per_token", "tandem probe: milliseconds per token on the GPU layers", "gpu_ms_per_token"),
                        ("kindi_tandem_transfer_ms_per_token", "tandem probe: milliseconds per token on layer handoffs", "transfer_ms_per_token"),
                        ("kindi_tandem_head_ms_per_token", "tandem probe: milliseconds per token on the output head", "head_ms_per_token"),
                        ("kindi_tandem_total_ms_per_token", "tandem probe: milliseconds per token summed over the split", "total_ms_per_token"),
                        ("kindi_tandem_transfers_per_token", "tandem probe: layer handoffs per token", "transfers_per_token")):
                    metric(name, "gauge", help_text, "%s %.3f" % (name, t[key]))
        g4 = getattr(K, "g4cache", None)
        if g4 is not None:
            m = g4.metrics()
            for name, typ, help_text, key in (
                    ("kindi_g4_cache_budget_bytes", "gauge", "Gemma 4 host prompt-cache byte budget", "budget_bytes"),
                    ("kindi_g4_cache_used_bytes", "gauge", "Gemma 4 host prompt-cache bytes held now", "used_bytes"),
                    ("kindi_g4_cache_peak_bytes", "gauge", "Gemma 4 host prompt-cache peak bytes held", "peak_bytes"),
                    ("kindi_g4_cache_checkpoints", "gauge", "Gemma 4 host ring checkpoints held", "checkpoints"),
                    ("kindi_g4_cache_parked", "gauge", "Gemma 4 parked conversations held in host memory", "parked"),
                    ("kindi_g4_cache_evictions_total", "counter", "Gemma 4 host snapshots and conversations evicted", "evictions"),
                    ("kindi_g4_cache_evicted_bytes_total", "counter", "Gemma 4 host bytes evicted", "evicted_bytes"),
                    ("kindi_g4_cache_checkpoints_dropped_total", "counter", "Gemma 4 checkpoints evicted", "checkpoints_dropped"),
                    ("kindi_g4_cache_parked_dropped_total", "counter", "Gemma 4 parked conversations evicted", "parked_dropped")):
                metric(name, typ, help_text, "%s %d" % (name, m[key]))
        metric("kindi_prompt_tokens_total", "counter", "prompt tokens processed",
               "kindi_prompt_tokens_total %d" % g("prompt_tokens"))
        metric("kindi_cached_prompt_tokens_total", "counter", "prompt tokens served from the cache",
               "kindi_cached_prompt_tokens_total %d" % g("cached_tokens"))
        metric("kindi_completion_tokens_total", "counter", "completion tokens generated",
               "kindi_completion_tokens_total %d" % g("completion_tokens"))
        metric("kindi_spec_mode_effective", "gauge",
               "the effective speculation mode, 1 for the mode now running (none when off)",
               'kindi_spec_mode_effective{mode="%s"} 1' % label(effective_mode or "none"))
        metric("kindi_spec_drafted_tokens_total", "counter",
               "tokens proposed by speculative drafting, over all modes (see kindi_spec_mode_effective)",
               "kindi_spec_drafted_tokens_total %d" % g("spec_drafted"))
        metric("kindi_spec_accepted_tokens_total", "counter",
               "drafted tokens accepted by the verify pass, over all modes (see kindi_spec_mode_effective)",
               "kindi_spec_accepted_tokens_total %d" % g("spec_accepted"))
        metric("kindi_generation_seconds_total", "counter", "seconds spent generating",
               "kindi_generation_seconds_total %.3f" % g("gen_seconds"))
        metric("kindi_last_generation_tokens_per_second", "gauge", "tokens/s of the last generation",
               "kindi_last_generation_tokens_per_second %.2f" % g("last_tok_s"))
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.get("/api/tags")
    def ollama_tags(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        return {"models": [{"name": model_id, "model": model_id,
                            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(K.t0)),
                            "size": 0, "digest": getattr(K, "pack_id", ""),
                            "details": {"format": "kindi", "family": "kindi", "families": ["kindi"],
                                        "parameter_size": "", "quantization_level": "ternary"}}]}

    @app.post("/api/chat")
    async def ollama_chat(request: Request):
        return await _ollama(request, False)

    @app.post("/api/generate")
    async def ollama_generate(request: Request):
        return await _ollama(request, True)

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        limited = _rate_limited(request)
        if limited is not None:
            return limited
        try:
            req = await _read_json_body(request)
        except RequestTooLarge as e:
            return _err(413, str(e), "request_too_large")
        except RequestError as e:
            return _err(400, str(e), param=getattr(e, "param", None))
        try:
            _validate_request(req, image_error)
        except RequestError as e:
            return _err(400, str(e), param=getattr(e, "param", None))
        if req.get("logprobs") and not getattr(K, "supports_logprobs", False):
            return _err(400, "logprobs are not supported by this backend; remove logprobs or use a Gemma 4 pack",
                        "invalid_request_error", param="logprobs")
        rf = req.get("response_format") or {}
        if is_structured(rf) and not getattr(K, "supports_json_grammar", True):
            return _err(400, "structured output (response_format) is not supported by the %s backend" % getattr(K, "variant", ""),
                        "invalid_request_error", param="response_format")
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        # every request is counted before it waits for the engine: a prompt that does not fit gets its HTTP 400 at
        # once (a queued request used to wait for the request ahead of it, then fail), and a dry run ends here
        dry = bool(req.get("dry_run") or req.get("kindi_dry_run"))
        try:
            ids = await asyncio.get_event_loop().run_in_executor(None, K.check_prompt, req)
        except ContextOverflow as e:
            if dry:
                return {"object": "kindi.prompt_count", "model": model_id, "fits": False, "max_model_len": K.eng.ctx,
                        "usage": {"prompt_tokens": e.n, "completion_tokens": 0, "total_tokens": e.n}}
            return _overflow(e)
        except ValueError as e:
            return _err(400, str(e))
        except Exception as e:  # noqa: BLE001 - the prompt could not be counted
            if dry:
                # A preflight whose count failed must not answer "fits": that is the one answer a
                # caller acts on, and a false positive is worse than an error (2026-11-16 audit, F2).
                return _err(500, "the prompt could not be counted (%s: %s), so dry_run cannot say "
                                 "whether it fits" % (type(e).__name__, e), "server_error")
            ids = None                # the request itself reports it on the generation path
        if dry:
            n = len(ids or [])
            return {"object": "kindi.prompt_count", "model": model_id, "fits": n + 1 < K.eng.ctx,
                    "max_model_len": K.eng.ctx, "usage": {"prompt_tokens": n, "completion_tokens": 0,
                                                          "total_tokens": n}}

        def chunk(delta, finish=None, usage=None):
            d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                d["usage"] = usage
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        cancel = threading.Event()
        meta = _CURRENT.get() or {}
        meta["stream"] = bool(req.get("stream"))
        mine = {"started": False}             # this request holds the engine (else it waits behind another one)
        prio = priority_of(req, request.headers)
        if request.headers.get("x-kindi-session"):
            req["_kindi_session"] = request.headers["x-kindi-session"]      # the conversation this request belongs to

        def progress_note():
            if not mine["started"]:
                return {"phase": "queued"}
            return K.progress() or {"phase": "finishing"}

        if req.get("stream"):
            try:
                gate.enter()
            except QueueFull as e:
                return _busy(e)

            def sse():
                with K.lock:
                    mine["started"] = True
                    yield chunk({"role": "assistant", "content": ""})
                    held, in_call = "", False
                    try:
                        for kind, val in _logged(run_generate(K, req, cancel, prio), meta, True):
                            if kind == "reasoning":
                                yield chunk({"reasoning_content": val})
                            elif kind == "text":
                                held += val
                                if in_call:
                                    continue                  # keep the call text until it parses (or not)
                                if req.get("tools") and "<tool_call>" in held:
                                    pre = held[:held.index("<tool_call>")]
                                    if pre:
                                        yield chunk({"content": pre})
                                    held, in_call = held[held.index("<tool_call>"):], True
                                    continue
                                # hold back a possible start of '<tool_call>' / '<think>' / '</think>'
                                cut = max([held.rfind("<")] + [0]) if "<" in held[-12:] else len(held)
                                out, held = held[:cut], held[cut:]
                                if out:
                                    yield chunk({"content": out})
                            else:
                                info = val
                                if info["tool_calls"] and held and not in_call:
                                    yield chunk({"content": held})        # text held back before the calls
                                if info["tool_calls"]:
                                    yield chunk({"tool_calls": [dict(tc, index=i)
                                                                for i, tc in enumerate(info["tool_calls"])]})
                                elif held:
                                    # also an unfinished call (cut off by max_tokens): send its raw text, do not drop it
                                    yield chunk({"content": held})
                                inc = (req.get("stream_options") or {}).get("include_usage")
                                yield chunk({}, info["finish"], info["usage"] if inc else None)
                    except OverflowError as e:
                        yield "data: " + json.dumps({"error": {"message": str(e), "type": "invalid_request_error",
                                                               "code": "context_length_exceeded"}}) + "\n\n"
                    except ValueError as e:
                        yield "data: " + json.dumps({"error": {"message": str(e), "type": "invalid_request_error"}}) + "\n\n"
                    except Exception as e:  # noqa: BLE001 - engine failure becomes an SSE error event, by design
                        yield "data: " + json.dumps({"error": {"message": "kindi: %s" % e, "type": "server_error"}}) + "\n\n"
                        _after_engine_error(K, e)
                    yield "data: [DONE]\n\n"
                    if not cancel.is_set():
                        K.after_turn(locked=True)     # the client has its answer; the next request waits for this
            async def relay():
                # the generator runs in a worker thread; if the client disconnects, Starlette cancels this
                # coroutine, the finally sets `cancel`, and the engine stops at its next token
                loop = asyncio.get_event_loop()
                q = asyncio.Queue()
                END = object()

                def pump():
                    try:
                        for item in sse():
                            loop.call_soon_threadsafe(q.put_nowait, item)
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, END)
                fut = asyncio.wrap_future(SCHED.submit(pump, prio))
                try:
                    while True:
                        # an SSE comment every KEEPALIVE_S while nothing else is sent (waiting in the queue, reading
                        # a long prompt): it keeps the stream alive and says how far the read is. SSE clients
                        # ignore comment lines.
                        try:
                            item = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_S)
                        except asyncio.TimeoutError:
                            yield ": kindi progress " + json.dumps(progress_note()) + "\n\n"
                            continue
                        if item is END:
                            break
                        yield item
                finally:
                    cancel.set()
                    await asyncio.wait([fut], timeout=30)
                    gate.release()
            return StreamingResponse(relay(), media_type="text/event-stream")

        try:
            gate.enter()
        except QueueFull as e:
            return _busy(e)

        def run():
            t_wait = time.time()
            with K.lock:
                meta["wait"] = time.time() - t_wait
                mine["started"] = True
                info = None
                for kind, val in _logged(run_generate(K, req, cancel, prio), meta, False):
                    if kind == "done":
                        info = val
                return info

        def answer(info):
            msg = {"role": "assistant",
                   "content": info["content"] if (info["content"] or not info["tool_calls"]) else None}
            if info["tool_calls"]:
                msg["tool_calls"] = info["tool_calls"]
            if info["reasoning"]:
                msg["reasoning_content"] = info["reasoning"]
            choice = {"index": 0, "message": msg, "finish_reason": info["finish"]}
            if info.get("logprobs") is not None:
                choice["logprobs"] = {"content": info["logprobs"]}
            return {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                    "choices": [choice], "usage": info["usage"], "timing": info["timing"]}

        def failed(e):
            """(status, body) of a request that raised"""
            if isinstance(e, OverflowError):
                return 400, {"error": {"message": str(e), "type": "invalid_request_error", "param": "messages",
                                       "code": "context_length_exceeded"}}
            if isinstance(e, ValueError):
                return 400, {"error": {"message": str(e), "type": "invalid_request_error",
                                       "code": "invalid_request_error"}}
            _after_engine_error(K, e)
            return 500, {"error": {"message": "kindi: %s" % e, "type": "server_error", "code": "server_error"}}

        loop = asyncio.get_event_loop()
        # kindi_keepalive (or the header X-Kindi-Keepalive: 1): a non-streaming client behind a proxy with an idle
        # timeout gets a space every KEEPALIVE_S until the JSON answer. JSON allows whitespace before the value, but
        # the status is then 200 before the answer is known: an error comes as an {"error": ...} body with status
        # 200. So it is off unless the client asks for it.
        if req.get("kindi_keepalive") or request.headers.get("x-kindi-keepalive", "") in ("1", "true"):
            async def body():
                fut = asyncio.wrap_future(SCHED.submit(run, prio))
                try:
                    while True:
                        done, _ = await asyncio.wait([fut], timeout=KEEPALIVE_S)
                        if done:
                            break
                        yield " "
                    try:
                        info = fut.result()
                        out = answer(info) if info is not None else {"error": {"message": "request cancelled",
                                                                               "type": "cancelled"}}
                    except Exception as e:  # noqa: BLE001
                        out = failed(e)[1]
                    yield json.dumps(out, ensure_ascii=False)
                    if not cancel.is_set():
                        SCHED.submit(K.after_turn, PRIO_TURN)
                finally:
                    cancel.set()
                    gate.release()
            return StreamingResponse(body(), media_type="application/json")
        try:
            fut = asyncio.wrap_future(SCHED.submit(run, prio))
            while not fut.done():
                done, _ = await asyncio.wait([fut], timeout=1.0)
                if not done and await request.is_disconnected():
                    cancel.set()
            info = fut.result()
            if info is None:
                return _err(499, "request cancelled", "cancelled")
        except Exception as e:  # noqa: BLE001
            code, out = failed(e)
            return JSONResponse(out, status_code=code)
        finally:
            gate.release()
        # the save of the turn's end state runs on the engine thread after this answer is sent, before the next request
        SCHED.submit(K.after_turn, PRIO_TURN)
        return answer(info)

    return app


def serve(cfg, log=print):
    import uvicorn
    app = create_app(cfg, log)
    log("serving on http://%s:%d/v1 (model %s)" % (cfg["host"], cfg["port"], cfg["model_name"]))
    uvicorn.run(app, host=cfg["host"], port=int(cfg["port"]), log_level="warning")
