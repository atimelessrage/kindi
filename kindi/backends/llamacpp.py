# SPDX-License-Identifier: AGPL-3.0-only
"""llama.cpp proxy backend (REQ-0019 T4.4, the recommended generic fallback).

A wildcard backend: it declares no model ids and no architectures, so the
registry resolves it only for models nothing more specific serves (an
arbitrary GGUF). It does not run a model itself; it forwards requests to a
running ``llama-server`` (llama.cpp's OpenAI-compatible server) over HTTP and
streams the result back through the backend contract, so one Kindi server can
front any GGUF llama.cpp can load.

Configuration (all optional):

- ``base_url`` config key, or ``KINDI_LLAMACPP_BASE_URL``; default
  ``http://127.0.0.1:8080``. No trailing slash needed.
- ``api_key`` config key, or ``KINDI_LLAMACPP_API_KEY``; sent as a bearer
  token when set (llama-server's ``--api-key``).
- ``model`` config key: the model id sent in the request body when set.

Start the upstream server with, for example::

    llama-server -m model.gguf --host 127.0.0.1 --port 8080

Limitations, stated rather than hidden:

- The request goes to llama-server's ``/completion`` endpoint with a prompt
  *text*, rendered by :meth:`apply_chat_template` (ChatML). llama.cpp's own
  chat template for the model is therefore bypassed; a model whose template
  differs may answer worse, and a messages-aware path is future work.
- Generation is one non-streaming request: tokens arrive as one block, not
  as the upstream server produced them, and cancellation is checked before
  and while the reply is replayed, not by aborting the upstream request.
- No constrained decoding: ``supports_json_grammar`` and
  ``supports_token_filter`` are False, and a ``token_filter`` in the
  sampling params raises instead of being silently ignored. Structured
  output must be rejected at the API layer on this backend's
  capabilities.
- The prompt cache is the upstream server's (``cache_prompt`` is sent);
  Kindi's own prefix cache does not apply, so ``cache_kind`` is ``"none"``.

Python 3.6 compatible (the CUDA image), standard library only.
"""
import os

from . import registry
from .base import Backend, BackendCapabilities, CancelToken, GenerationEvent, StopScanner

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
EOS = 0


def _urllib_opener(request):
    """The default opener: one JSON request over urllib, stdlib only."""
    import json
    import urllib.error
    import urllib.request

    data = None
    if request.get("body") is not None:
        data = json.dumps(request["body"]).encode("utf-8")
    req = urllib.request.Request(request["url"], data=data, method=request.get("method", "GET"))
    req.add_header("Content-Type", "application/json")
    api_key = request.get("api_key")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=request.get("timeout", 120)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise registry.BackendUnavailable(
            "llama-server at %s returned HTTP %s%s" % (request["url"], exc.code,
                                                       (": " + detail) if detail else "")) from exc
    except urllib.error.URLError as exc:
        raise registry.BackendUnavailable(
            "cannot reach llama-server at %s (%s); start it with "
            "`llama-server -m <model.gguf>` and set KINDI_LLAMACPP_BASE_URL if it is elsewhere"
            % (request["url"], exc.reason)) from exc


class LlamaCppBackend(Backend):
    """Forward generation to a running llama-server; see the module docstring."""

    capabilities = BackendCapabilities(
        name="llamacpp",
        model_ids=(),
        architectures=(),
        devices=(),
        max_context=0,
        supports_thinking=False,
        supports_tools=False,
        tool_format=None,
        supports_speculation=False,
        supports_token_filter=False,
        supports_json_grammar=False,
        cache_kind="none",
        max_concurrent_sequences=1,
        deterministic_greedy=False,
    )

    def __init__(self, opener=None, timeout=120):
        self._opener = opener if opener is not None else _urllib_opener
        self.timeout = timeout
        self.loaded = False
        self.base_url = ""
        self.api_key = ""
        self.model = ""
        self.reachable = None
        self.health_error = ""

    # lifecycle ------------------------------------------------------------
    def load(self, config):
        config = dict(config or {})
        base = (config.get("base_url") or os.environ.get("KINDI_LLAMACPP_BASE_URL")
                or DEFAULT_BASE_URL)
        self.base_url = str(base).rstrip("/")
        self.api_key = str(config.get("api_key") or os.environ.get("KINDI_LLAMACPP_API_KEY") or "")
        self.model = str(config.get("model") or "")
        self.loaded = True
        # A reachability probe is recorded, not enforced: an unavailable server
        # must not make load() (and every contract test that calls it) fail.
        # generate() is where the missing server is raised, with the hint.
        try:
            self._opener({"method": "GET", "url": self.base_url + "/health",
                          "timeout": 2, "body": None, "api_key": self.api_key})
            self.reachable = True
            self.health_error = ""
        except Exception as exc:  # noqa: BLE001 - recorded, raised again in generate
            self.reachable = False
            self.health_error = str(exc)
        return self

    def close(self):
        self.loaded = False
        self.reachable = None
        self.health_error = ""

    def info(self):
        return {
            "engine": self.capabilities.name,
            "loaded": self.loaded,
            "base_url": self.base_url,
            "model": self.model or None,
            "reachable": self.reachable,
            "health_error": self.health_error or None,
            "cache_kind": self.capabilities.cache_kind,
            "deterministic_greedy": False,
        }

    # text primitives (opaque byte ids; the proxy sends text, not ids) -----
    def tokenize(self, text, add_special_tokens=True):
        """Byte-level ids (id = byte + 1, 0 is EOS), the FakeBackend scheme.

        The ids are an in-process transport only: :meth:`generate` detokenizes
        them back to the prompt text it posts to llama-server, so these are
        not the upstream server's token ids.
        """
        data = (text or "").encode("utf-8")
        ids = [byte + 1 for byte in data]
        if add_special_tokens:
            ids = [EOS] + ids
        return ids

    def detokenize(self, token_ids):
        data = bytearray()
        for token_id in token_ids or []:
            token_id = int(token_id)
            if 1 <= token_id <= 256:
                data.append(token_id - 1)
        return bytes(data).decode("utf-8", "replace")

    def apply_chat_template(self, messages, tools=None, enable_thinking=False):
        """Render ChatML. llama-server's own template is bypassed (see docstring)."""
        lines = []
        for message in messages or []:
            role = message.get("role", "user") if isinstance(message, dict) else "user"
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and "text" in part)
            lines.append("<|im_start|>%s\n%s<|im_end|>" % (role, content))
        lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)

    # generation ------------------------------------------------------------
    def generate(self, prompt_ids, params, session=None, cancel=None):
        if not self.loaded:
            raise registry.BackendUnavailable(
                "llamacpp backend is not loaded; call load() first")
        if params is not None and getattr(params, "token_filter", None) is not None:
            raise registry.BackendUnavailable(
                "the llama.cpp proxy cannot apply a token_filter (constrained "
                "decoding is not supported on this backend); reject structured "
                "output at the API layer instead")
        cancel_token = cancel if cancel is not None else CancelToken()
        if cancel_token.is_set():
            yield GenerationEvent("done", finish_reason="cancelled", timing={})
            return
        body = {
            "prompt": self.detokenize(prompt_ids),
            "n_predict": max(0, int(params.max_tokens)) if params is not None else 0,
            "temperature": float(params.temperature) if params is not None else 0.0,
            "top_p": float(params.top_p) if params is not None else 1.0,
            "stop": list(params.stop or ()) if params is not None else [],
            "cache_prompt": True,
            "stream": False,
        }
        if params is not None and int(params.top_k or 0) > 0:
            body["top_k"] = int(params.top_k)
        if params is not None and int(params.seed or 0) != 0:
            body["seed"] = int(params.seed)
        if self.model:
            body["model"] = self.model
        response = self._opener({"method": "POST", "url": self.base_url + "/completion",
                                 "timeout": self.timeout, "body": body,
                                 "api_key": self.api_key})
        if not isinstance(response, dict) or "content" not in response:
            raise registry.BackendUnavailable(
                "llama-server at %s returned an unexpected reply (keys: %s); "
                "the proxy gate needs llama-server's /completion endpoint"
                % (self.base_url, ", ".join(sorted(response)) if isinstance(response, dict) else type(response).__name__))
        content = response.get("content") or ""
        scanner = StopScanner(params.stop if params is not None else ())
        stopped = bool(scanner.stops and scanner.hit(content))
        if stopped:
            cut = content.find(next(s for s in scanner.stops if s in content))
            content = content[:cut]
        import time
        started = time.time()
        produced = 0
        for ch in content:
            if cancel_token.is_set():
                yield GenerationEvent("done", finish_reason="cancelled", timing={})
                return
            produced += 1
            token_id = ord(ch) + 1 if ord(ch) < 256 else 63
            yield GenerationEvent("token", token_id=token_id, text=ch)
        elapsed = max(0.001, time.time() - started)
        finish = "stop" if stopped else ("length" if response.get("truncated") else "stop")
        yield GenerationEvent(
            "done", finish_reason=finish,
            timing={"gen_s": round(elapsed, 4),
                    "gen_tok_s": round(int(response.get("tokens_predicted") or produced) / elapsed, 2)})

    # cache and verification hooks -----------------------------------------
    def cache_stats(self):
        return {"cache_kind": "none", "base_url": self.base_url,
                "reachable": self.reachable}

    def verify_prompts(self):
        """No verify gate yet: the proxy gate is a live llama-server run."""
        return []


registry.register_backend(LlamaCppBackend)
