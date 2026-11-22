"""kindi selftest: prove an installed pack, tokenizer, engine and HTTP route in one command.

`kindi verify` needs llama.cpp's reference tokens and `kindi serve` needs a second terminal, so a
fresh install had no single check that the pack, tokenizer, engine and server path all work. This
loads the configured pack once, asks `/health` for the effective runtime state, runs one short
greedy completion through `/v1/chat/completions`, and reports the pack hash and timings. It never
binds a port and never needs a GPU-specific test path.
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


def run(cfg, log=print, max_tokens=8, json_path=None):
    """Load the pack, prove `/health` and one chat completion, and return 0 only on success."""
    started = time.perf_counter()
    result = {"ok": False, "model": cfg.get("model_name"), "pack": {"path": cfg.get("pack"),
              "source_sha256": None, "pack_id": None}, "runtime": None, "health": None,
              "generation": None, "seconds": None, "error": None}
    max_tokens = max(1, int(max_tokens))
    pack = cfg.get("pack")
    manifest_path = os.path.join(pack, "kindi_pack.json") if pack else ""
    if not pack or not os.path.isfile(manifest_path):
        return _fail(result, "no model pack at %s - run: kindi convert <model.gguf>"
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
    try:
        from fastapi.testclient import TestClient
    except ImportError as e:                       # pragma: no cover - the server extra is installed here
        return _fail(result, "selftest needs fastapi's test client (%s)" % e, log, json_path)

    # The detailed /health view is loopback-only; an in-process TestClient is not loopback, so this
    # asks for it explicitly without exposing anything on the network.
    run_cfg = dict(cfg, host="127.0.0.1")
    run_cfg.setdefault("api_key", "")
    try:
        app = create_app(run_cfg, log=log)
    except (Exception, SystemExit) as e:
        return _fail(result, "engine/tokenizer load failed: %s: %s" % (type(e).__name__, e),
                     log, json_path)

    K = app.state.kindi
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            t0 = time.perf_counter()
            hr = client.get("/health")
            try:
                hb = hr.json()
            except ValueError:
                hb = {}
            result["health"] = {"status": hr.status_code, "model": hb.get("model"), "ctx": hb.get("ctx"),
                                "head": hb.get("head"), "drafting": hb.get("drafting")}
            if hr.status_code != 200:
                return _fail(result, "/health returned HTTP %d" % hr.status_code, log, json_path)
            rt = runtime_view.describe(cfg, K.eng, K)
            result["runtime"] = rt
            log("selftest: " + runtime_view.render(rt))
            log("selftest: /health 200 in %.1f ms (ctx %s, head %s, spec %s)"
                % ((time.perf_counter() - t0) * 1000.0, hb.get("ctx"), rt.get("head"),
                   "on" if (rt.get("spec") or {}).get("enabled") else "off"))

            body = {"model": cfg.get("model_name") or hb.get("model") or "model",
                    "messages": [{"role": "user", "content": PROMPT}],
                    "temperature": 0, "max_tokens": max_tokens, "stream": False}
            t1 = time.perf_counter()
            gr = client.post("/v1/chat/completions", json=body)
            try:
                gb = gr.json()
            except ValueError:
                gb = {}
            gen_seconds = time.perf_counter() - t1
            if gr.status_code != 200:
                err = gb.get("error") if isinstance(gb, dict) else None
                detail = err.get("message") if isinstance(err, dict) else err
                return _fail(result, "generation returned HTTP %d: %s" % (gr.status_code, detail or gb),
                             log, json_path)
            choices = gb.get("choices") or []
            message = (choices[0].get("message") or {}) if choices else {}
            usage = gb.get("usage") or {}
            tokens = usage.get("completion_tokens")
            text = message.get("content") or ""
            gen = {"prompt": PROMPT, "max_tokens": max_tokens, "text": text,
                   "tool_calls": message.get("tool_calls") or [],
                   "finish": choices[0].get("finish_reason") if choices else None,
                   "completion_tokens": tokens, "seconds": round(gen_seconds, 3),
                   "request_tok_s": round(tokens / gen_seconds, 2)
                   if isinstance(tokens, int) and gen_seconds > 0 else None}
            result["generation"] = gen
            if not choices or not isinstance(tokens, int) or tokens < 1:
                return _fail(result, "generation returned no completion tokens", log, json_path)
            log("selftest: generation %d token(s) in %.0f ms total (%.2f tok/s whole request, finish %s)"
                % (tokens, gen_seconds * 1000.0, tokens / gen_seconds if gen_seconds > 0 else 0.0,
                   gen["finish"]))
            result["ok"] = True
    finally:
        close = getattr(K, "close", None)
        if close is not None:
            close()
    result["seconds"] = round(time.perf_counter() - started, 2)
    log("selftest: PASS in %.1f s" % result["seconds"])
    _write(json_path, result, log)
    return 0
