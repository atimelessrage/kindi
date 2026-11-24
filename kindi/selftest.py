"""kindi selftest: prove an installed pack, tokenizer, engine and HTTP route in one command.

`kindi verify` needs llama.cpp's reference tokens and `kindi serve` needs a second terminal, so a
fresh install had no single check that the pack, tokenizer, engine and server path all work. This
loads the configured pack once, asks `/health` for the effective runtime state, runs one short
greedy completion through `/v1/chat/completions` and the same prompt once through `/v1/messages`
(Anthropic's shape, REQ-0028), compares the two answers (same greedy request, so the same tokens:
REQ-0028 A4's no-decoding-change gate), and reports the pack hash and timings. It never binds a port
and never needs a GPU-specific test path. Two routes rather than one because the Anthropic endpoint
is translation only: a release image must prove it answers with the real engine, not just that the
route exists, and that it decodes what the OpenAI route decodes.

`--url` checks a server that is already running instead: no pack is loaded, so it costs two short
requests and is safe next to the serving container (the installer uses it). Both modes speak HTTP
in process or over loopback with the standard library only; the server package already brings its
own dependencies, and fastapi's test client would add `httpx`/`requests` to the images.
"""
import json
import os
import time

PROMPT = "Reply with the single word: kindi"


def _write(path, result, log):
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    except OSError as e:
        log("selftest: could not write %s (%s)" % (path, e))


def _fail(result, message, log, json_path):
    result["error"] = message
    log("selftest: FAIL: " + message)
    _write(json_path, result, log)
    return 1


def _health(result, status, hb, log):
    """Record `/health`; returns the failure message, or None when it answered 200."""
    hb = hb if isinstance(hb, dict) else {}
    result["health"] = {"status": status, "model": hb.get("model"), "ctx": hb.get("ctx"),
                        "head": hb.get("head"), "drafting": hb.get("drafting")}
    if status is None:
        return "/health did not answer"
    if status != 200:
        return "/health returned HTTP %d" % status
    rt = hb.get("runtime")
    if isinstance(rt, dict):
        result["runtime"] = rt
    log("selftest: /health 200 (ctx %s, head %s, spec %s)"
        % (hb.get("ctx"), (rt or {}).get("head", hb.get("head")),
           "on" if ((rt or {}).get("spec") or {}).get("enabled") else "off"))
    return None


def _generation(result, status, gb, seconds, max_tokens, log):
    """Record the one chat completion; returns the failure message, or None when it produced tokens."""
    gb = gb if isinstance(gb, dict) else {}
    if status != 200:
        err = gb.get("error")
        detail = err.get("message") if isinstance(err, dict) else err
        if status is None:
            return "generation did not answer"
        return "generation returned HTTP %d: %s" % (status, detail or gb or "no body")
    choices = gb.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    usage = gb.get("usage") or {}
    tokens = usage.get("completion_tokens")
    text = message.get("content") or ""
    gen = {"prompt": PROMPT, "max_tokens": max_tokens, "text": text,
           "tool_calls": message.get("tool_calls") or [],
           "finish": choices[0].get("finish_reason") if choices else None,
           "completion_tokens": tokens, "seconds": round(seconds, 3),
           "request_tok_s": round(tokens / seconds, 2)
           if isinstance(tokens, int) and seconds > 0 else None}
    result["generation"] = gen
    if not choices or not isinstance(tokens, int) or tokens < 1:
        return "generation returned no completion tokens"
    log("selftest: generation %d token(s) in %.0f ms total (%.2f tok/s whole request, finish %s)"
        % (tokens, seconds * 1000.0, tokens / seconds if seconds > 0 else 0.0, gen["finish"]))
    return None


def _chat_body(model, max_tokens):
    return {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}


def _anthropic_body(model, max_tokens):
    return {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]}


def _anthropic_answer(result, status, body, seconds, max_tokens, log):
    """Record the `/v1/messages` answer; returns the failure message, or None when it is a real message.

    A 200 with the wrong shape is a failure: the point of this check is that a release image answers
    Anthropic's dialect with the engine behind it, not that the route is registered.
    """
    body = body if isinstance(body, dict) else {}
    if status != 200:
        err = body.get("error")
        detail = err.get("message") if isinstance(err, dict) else err
        if status is None:
            return "the Anthropic route did not answer"
        return "/v1/messages returned HTTP %d: %s" % (status, detail or body or "no body")
    blocks = [b for b in (body.get("content") or []) if isinstance(b, dict)]
    usage = body.get("usage") or {}
    output_tokens = usage.get("output_tokens")
    text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
    answer = {"prompt": PROMPT, "max_tokens": max_tokens, "text": text,
              "blocks": [b.get("type") for b in blocks], "stop_reason": body.get("stop_reason"),
              "output_tokens": output_tokens, "seconds": round(seconds, 3)}
    result["anthropic"] = answer
    if body.get("type") != "message":
        return "/v1/messages answered type %r, not a message" % body.get("type")
    if not blocks:
        return "/v1/messages returned no content blocks"
    if not isinstance(output_tokens, int) or output_tokens < 1:
        return "/v1/messages returned no output tokens"
    log("selftest: /v1/messages %s in %.0f ms (stop %s, %s output token(s))"
        % (", ".join(answer["blocks"]), seconds * 1000.0, answer["stop_reason"], output_tokens))
    return None


def _dialects_agree(result, log):
    """The two dialects must decode the same text (REQ-0028 A4: the endpoint is translation only).

    Both requests are greedy (temperature 0) with the same prompt and ``max_tokens``, so the same
    engine must produce the same tokens: anything else means the Anthropic route changed decoding,
    which the requirement forbids. Greedy repeats are gated elsewhere (ten identical completions per
    server), so a mismatch here is the endpoint, not sampling noise.
    """
    gen = result.get("generation") or {}
    ant = result.get("anthropic") or {}
    chat_text, ant_text = gen.get("text") or "", ant.get("text") or ""
    chat_tok, ant_tok = gen.get("completion_tokens"), ant.get("output_tokens")
    same_text = chat_text == ant_text
    same_tokens = (chat_tok == ant_tok) if isinstance(chat_tok, int) and isinstance(ant_tok, int) else None
    result["dialects"] = {"same_text": same_text, "same_tokens": same_tokens,
                          "chat_text": chat_text, "anthropic_text": ant_text}
    if not same_text:
        return ("the two dialects decoded different text for the same greedy request "
                "(chat %r, /v1/messages %r), so the Anthropic route changed decoding"
                % (chat_text[:120], ant_text[:120]))
    if same_tokens is False:
        return ("the two dialects report %r and %r completion tokens for the same text"
                % (chat_tok, ant_tok))
    log("selftest: dialects agree (%d token(s), %d chars)" % (chat_tok or 0, len(chat_text)))
    return None


def _asgi(app, method, path, payload=b""):
    """One in-process ASGI request: the app is already an async callable, so no test client is needed."""
    import asyncio

    sent = {"status": None, "body": b""}
    seen = {"body": False}

    async def receive():
        if seen["body"]:
            return {"type": "http.disconnect"}
        seen["body"] = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body":
            sent["body"] += message.get("body") or b""

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": method, "scheme": "http", "path": path, "raw_path": path.encode("ascii"),
             "query_string": b"", "root_path": "",
             "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"application/json"),
                         (b"content-length", str(len(payload)).encode("ascii"))],
             "client": ("127.0.0.1", 0), "server": ("127.0.0.1", 8093), "state": {}}

    async def call():
        await app(scope, receive, send)

    loop = asyncio.new_event_loop()                 # asyncio.run is 3.7+; the release image is Python 3.6
    try:
        loop.run_until_complete(call())
    finally:
        loop.close()
    return sent["status"], sent["body"]


def _asgi_json(app, method, path, body=None):
    status, raw = _asgi(app, method, path, b"" if body is None else json.dumps(body).encode("utf-8"))
    try:
        return status, json.loads(raw.decode("utf-8") or "{}")
    except ValueError:
        return status, {}


def _url_json(url, payload=None, timeout=600):
    """(status, body) over loopback with the standard library; status None when nothing answered."""
    import urllib.error
    import urllib.request
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("KINDI_API_KEY")            # a server started with --lan requires it; docker exec carries it
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8") or "{}"
            return r.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8") or "{}")
        except ValueError:
            body = {}
        return e.code, body
    except (urllib.error.URLError, OSError, ValueError) as e:
        return None, {"error": str(e)}


def run(cfg, log=print, max_tokens=8, json_path=None):
    """Load the pack, prove `/health` and one chat completion, and return 0 only on success."""
    started = time.perf_counter()
    result = {"ok": False, "mode": "in-process", "model": cfg.get("model_name"),
              "pack": {"path": cfg.get("pack"), "source_sha256": None, "pack_id": None},
              "runtime": None, "health": None, "generation": None, "anthropic": None,
              "seconds": None, "error": None}
    max_tokens = max(1, int(max_tokens))
    pack = cfg.get("pack")
    manifest_path = os.path.join(pack, "kindi_pack.json") if pack else ""
    if not pack or not os.path.isdir(pack):
        return _fail(result, "no model pack at %s (run the installer or `kindi convert`)"
                     % (pack or "(unset)"), log, json_path)
    try:
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        return _fail(result, "could not read %s (%s)" % (manifest_path, e), log, json_path)
    sha = man.get("source_sha256")
    result["pack"]["source_sha256"] = sha
    result["pack"]["pack_id"] = (sha or "pack")[:12]
    log("selftest: pack %s (sha256 %s)" % (pack, sha or "not recorded"))

    from . import runtime as runtime_view
    from .server import create_app

    # The detailed /health view is loopback-only; this app is built for loopback and talks to it in
    # process, so nothing is exposed on the network.
    # No port is bound and the app is never exposed, so a keyed container still proves itself here;
    # leave the key in and the in-process chat would answer 401.
    run_cfg = dict(cfg, host="127.0.0.1", api_key="")
    try:
        app = create_app(run_cfg, log=log)
    except (Exception, SystemExit) as e:
        return _fail(result, "engine/tokenizer load failed: %s: %s" % (type(e).__name__, e),
                     log, json_path)

    K = app.state.kindi
    try:
        t0 = time.perf_counter()
        status, hb = _asgi_json(app, "GET", "/health")
        err = _health(result, status, hb, log)
        if err:
            return _fail(result, err, log, json_path)
        rt = runtime_view.describe(cfg, K.eng, K)
        result["runtime"] = rt
        log("selftest: " + runtime_view.render(rt))
        log("selftest: health in %.1f ms" % ((time.perf_counter() - t0) * 1000.0))

        t1 = time.perf_counter()
        try:
            status, gb = _asgi_json(app, "POST", "/v1/chat/completions",
                                    _chat_body(cfg.get("model_name") or hb.get("model") or "model", max_tokens))
        except Exception as e:                     # noqa: BLE001 - a route exception is a failure, not a crash
            return _fail(result, "generation raised %s: %s" % (type(e).__name__, e), log, json_path)
        err = _generation(result, status, gb, time.perf_counter() - t1, max_tokens, log)
        if err:
            return _fail(result, err, log, json_path)

        t2 = time.perf_counter()
        try:
            status, ab = _asgi_json(app, "POST", "/v1/messages",
                                    _anthropic_body(cfg.get("model_name") or hb.get("model") or "model",
                                                    max_tokens))
        except Exception as e:                     # noqa: BLE001 - a route exception is a failure, not a crash
            return _fail(result, "/v1/messages raised %s: %s" % (type(e).__name__, e), log, json_path)
        err = _anthropic_answer(result, status, ab, time.perf_counter() - t2, max_tokens, log)
        if err:
            return _fail(result, err, log, json_path)
        err = _dialects_agree(result, log)
        if err:
            return _fail(result, err, log, json_path)
        result["ok"] = True
    finally:
        close = getattr(K, "close", None)
        if close is not None:
            close()
    result["seconds"] = round(time.perf_counter() - started, 2)
    log("selftest: PASS in %.1f s" % result["seconds"])
    _write(json_path, result, log)
    return 0


def run_url(url, log=print, max_tokens=8, json_path=None, timeout=600):
    """Prove a server that is already running: `/health` and one chat completion over loopback.

    No pack, tokenizer or engine is loaded here, so this is cheap enough for the installer to run
    against the serving container and cannot compete with it for the card. A server that requires
    an API key is authenticated from ``KINDI_API_KEY`` (the installer's container carries it).
    """
    started = time.perf_counter()
    base = (url or "").rstrip("/")
    result = {"ok": False, "mode": "url", "url": base, "model": None,
              "pack": None, "runtime": None, "health": None, "generation": None,
              "anthropic": None, "seconds": None, "error": None}
    max_tokens = max(1, int(max_tokens))
    if not base:
        return _fail(result, "no server URL", log, json_path)
    status, hb = _url_json(base + "/health", timeout=min(timeout, 60))
    err = _health(result, status, hb, log)
    if err:
        return _fail(result, err, log, json_path)
    result["model"] = hb.get("model")
    log("selftest: server %s (%s)" % (base, result["model"] or "model not reported"))
    t1 = time.perf_counter()
    status, gb = _url_json(base + "/v1/chat/completions",
                           _chat_body(result["model"] or "model", max_tokens), timeout=timeout)
    err = _generation(result, status, gb, time.perf_counter() - t1, max_tokens, log)
    if err:
        return _fail(result, err, log, json_path)
    t2 = time.perf_counter()
    status, ab = _url_json(base + "/v1/messages",
                           _anthropic_body(result["model"] or "model", max_tokens), timeout=timeout)
    err = _anthropic_answer(result, status, ab, time.perf_counter() - t2, max_tokens, log)
    if err:
        return _fail(result, err, log, json_path)
    err = _dialects_agree(result, log)
    if err:
        return _fail(result, err, log, json_path)
    result["ok"] = True
    result["seconds"] = round(time.perf_counter() - started, 2)
    log("selftest: PASS in %.1f s" % result["seconds"])
    _write(json_path, result, log)
    return 0
