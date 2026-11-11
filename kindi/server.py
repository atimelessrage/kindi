"""Kindi OpenAI-compatible server.

  GET  /health                 status, live progress of the request being served, GPU memory, queue depth
  GET  /v1/models              the model
  POST /v1/chat/completions    messages (+ tools) -> answer; stream=true for SSE; tool calls as OpenAI tool_calls
  POST /api/chat               the same, in Ollama's NDJSON form (stream defaults to true)
  POST /api/generate           a raw prompt, Ollama style
  GET  /api/tags               the served model, Ollama style
  GET  /metrics                Prometheus text: requests, tokens, speculative stats, queue, uptime

One request at a time (one engine on one GPU); up to KINDI_QUEUE_MAX requests wait behind it, and the next one gets
HTTP 429 with Retry-After instead of waiting without a bound. Thinking is off unless the request sets
chat_template_kwargs.enable_thinking (or top-level enable_thinking); a stream then carries the thinking as
delta.reasoning_content. thinking_budget_tokens (or thinking_budget) caps it: at the budget the server closes the
thinking itself and the model goes on to answer. temperature 0 = greedy (token-exact with llama.cpp in the exact head
mode). Request bodies are capped at 16 MiB (KINDI_MAX_BODY_MB); when api_key is set, it is compared in constant time."""
import asyncio
import codecs
import concurrent.futures
import hmac
import json
import math
import os
import threading
import time
import uuid

import numpy as np

from . import COMMIT, __version__, chat
from .cache import Cancelled, Runner
from .engine import EOS
from .sampling import Sampler


THINK_CLOSE = "\n\nI have thought enough; time to act.\n</think>\n\n"


class RequestError(ValueError):
    """A malformed client request. It is a 400 and must never touch the prompt cache."""


def _request_error(msg):
    raise RequestError(msg)


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


class RequestTooLarge(ValueError):
    """The request body exceeds MAX_BODY_BYTES: an HTTP 413, not an engine error."""


async def _read_json_body(request):
    """Read and parse a JSON body with a hard size cap.

    `await request.json()` reads an unbounded body into memory. This server can be bound to 0.0.0.0, so a
    LAN client must not be able to make the process allocate an arbitrarily large string before the API key
    or the request shape is checked. The declared Content-Length is checked first; `stream()` then enforces
    the same cap even when the header is missing or lies.
    """
    limit = MAX_BODY_BYTES
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


def _validate_request(req, has_vision=False):
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
        if not has_vision and m.get("images"):
            _request_error("%s (messages[%d].images)" % (_IMAGE_UNSUPPORTED, i))
        if isinstance(content, list):
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    _request_error("messages[%d].content[%d] must be an object" % (i, j))
                if not has_vision and _is_image_part(part):
                    _request_error("%s (messages[%d].content[%d])" % (_IMAGE_UNSUPPORTED, i, j))
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
    """Gemma 4 in tensor parallel: KINDI_G4_TP=1 always, 0 never; auto (the default) when exactly two GPUs of the
    same kind are visible"""
    mode = os.environ.get("KINDI_G4_TP", "auto")
    if mode in ("0", "1"):
        return mode == "1"
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n != 2:
            return False
        names = [cp.cuda.runtime.getDeviceProperties(i)["name"] for i in range(2)]
        return names[0] == names[1]
    except Exception:
        return False

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
            clusters = None
            cl_path = os.path.join(pack, "head_clusters.npz")
            if os.path.exists(cl_path):
                clusters = dict(np.load(cl_path))
            self.eng = make_engine(pack, ctx=cfg["ctx"], clusters=clusters, log=log)
            self.tok = Tok(self.eng.vocab, os.path.join(pack, "merges.txt"))
            self.runner = Runner(self.eng, cfg, self.pack_id, log, progress=self.busy)
            self.sample = Sampler(self.eng)
        self.supports_logprobs = self.variant == "gemma4"
        self.lock = threading.Lock()
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

    def _pick(self, nxt, temperature, top_p, top_k, rng, min_p=0.0, penalty=None, presence=None):
        """the next token: already drawn by speculative sampling, or sampled now"""
        ps = getattr(self.eng, "spec_presampled", None)
        if ps is not None:
            self.eng.spec_presampled = None
            return ps
        return self.sample(nxt, temperature, top_p, top_k, rng, min_p, penalty, presence)

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
        thinking_open=prompt.endswith(gemma4_tools.THINK_OPEN+"thought\n")   # after a tool response, thinking on
        sorter=gemma4_tools.Stream(thinking_open)
        ids=self.tok.encode(prompt)
        if not ids: raise ValueError("Gemma 4 chat template produced an empty prompt")
        if len(ids)>=self.eng.ctx:
            raise OverflowError("the Gemma 4 prompt has %d tokens; context is %d"%(len(ids),self.eng.ctx))
        if req.get("_kindi_check"): return                # check_prompt(): the prompt is valid, nothing to run
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
                if gst is not None:
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
                hit=next((x for x in stop if x and x in text),None)
                if hit:
                    text=text[:text.index(hit)]; finish="stop"; break
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
        """Renders and counts the prompt of a request without the GPU or the lock. Raises OverflowError (the prompt does
        not fit the context) or ValueError (the prompt is not valid), so that a streaming request gets an HTTP 400
        before its first byte."""
        for _ in self.generate(dict(req, _kindi_check=True)):
            pass

    def generate(self, req, cancel=None):
        """Yields ("text", piece) while generating, then ("done", info). cancel: a threading.Event set when the
        client disconnects (prompt reading and generation stop at the next token)."""
        if self.nanbeige is not None:
            yield from self.nanbeige.generate(self, req, cancel)
            return
        if self.native:
            yield from self._generate_gemma(req,cancel)
            return
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
            raise OverflowError("the prompt has %d tokens; the context is %d" % (len(T), ctx))
        if req.get("_kindi_check"):                       # check_prompt(): the prompt is valid, nothing to run
            return
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
            nxt, cached = self.runner.prepare(T, len(Tb), opening_end, cancel)
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
            if pen or pp or (min_p > 0 and temperature > 0):  # the speculative paths do not apply these: no drafts
                self.eng.spec_begin(False)
            else:
                self.eng.spec_begin(temperature <= 0, temperature, top_p, top_k, rng)   # speculative decoding (KINDI_SPEC)
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

        tok = self.sample(nxt, temperature, top_p, top_k, rng, min_p, recent(), presence())
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
                hit = None if st["think"] else next((s for s in stops if s and s in text), None)
                if hit:
                    text = text[:text.index(hit)]
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
                tok = self._pick(nxt, temperature, top_p, top_k, rng, min_p, recent(), presence())
        finally:
            sp = self.eng.spec_end() if hasattr(self.eng, "spec_end") else None
            self.busy["active"] = False
            if hasattr(self.runner, "idle"):
                self.runner.idle()                            # (KINDI_CKPT_VRAM: checkpoint host copies now)
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
        except Exception:
            pass
    if getattr(K, "cp", None) is None or os.environ.get("KINDI_DEVICE", "gpu") == "cpu":
        return                                   # the CPU engine has no GPU context to lose
    broken = "IllegalAddress" in str(e) or "illegal memory access" in str(e)
    try:
        K.cp.cuda.runtime.deviceSynchronize()
        K.cp.cuda.runtime.memGetInfo()
    except Exception:
        broken = True
    if broken:
        print("[kindi] the GPU context is broken (%s): exiting so that the server restarts" % e, flush=True)
        threading.Timer(1.0, os._exit, (70,)).start()


def _make_engine(cfg, log):
    return Kindi(cfg, log)


def create_app(cfg, log=print):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    K = GPU_THREAD.submit(_make_engine, cfg, log).result()
    model_id = cfg["model_name"]
    has_vision = _pack_has_vision(cfg.get("pack"))
    if not cfg.get("api_key") and not _is_loopback(cfg.get("host", "127.0.0.1")):
        log("warning: serving on %s without api_key; /health is unauthenticated - use loopback or set api_key"
            % cfg.get("host"))
    app = FastAPI(title="Kindi", version=__version__)
    gate = K.gate = Gate(MAX_QUEUE)

    def _err(code, msg, typ="invalid_request_error", headers=None):
        return JSONResponse({"error": {"message": msg, "type": typ, "code": typ}},
                            status_code=code, headers=headers)

    def _busy(e):
        return _err(429, str(e), "server_busy", {"Retry-After": "1"})

    def _auth(request):
        key = cfg["api_key"]
        if not key:
            return True
        supplied = request.headers.get("authorization", "")
        return hmac.compare_digest(supplied.encode("utf-8"), ("Bearer " + key).encode("utf-8"))

    OLLAMA_OPTIONS = (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                      ("repeat_penalty", "repeat_penalty"), ("repeat_last_n", "repeat_last_n"),
                      ("presence_penalty", "presence_penalty"),
                      ("seed", "seed"), ("num_predict", "max_tokens"), ("stop", "stop"))

    def _ollama_req(body, prompt_mode, has_vision):
        """Map an Ollama request to the internal one; validate it before it reaches the engine or the cache."""
        if not isinstance(body, dict):
            raise RequestError("the body must be a JSON object")
        stream = body.get("stream", True)
        if not isinstance(stream, bool):
            raise RequestError("stream must be true or false")
        opts = body.get("options") or {}
        if not isinstance(opts, dict):
            raise RequestError("options must be an object")
        if not has_vision and body.get("images"):
            raise RequestError(_IMAGE_UNSUPPORTED)
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
        _validate_request(req, has_vision)
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
        try:
            body = await _read_json_body(request)
        except RequestTooLarge as e:
            return _err(413, str(e), "request_too_large")
        except RequestError as e:
            return _err(400, str(e))
        try:
            req, stream = _ollama_req(body, prompt_mode, has_vision)
        except RequestError as e:
            return _err(400, str(e))
        try:
            gate.enter()
        except QueueFull as e:
            return _busy(e)
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cancel = threading.Event()
        if stream:
            def events():
                try:
                    with K.lock:
                        for kind, val in K.generate(req, cancel):
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
            with K.lock:
                info = None
                for kind, val in K.generate(req, cancel):
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
    def health():
        free = total = 0
        if K.cp is not None and os.environ.get("KINDI_DEVICE", "gpu") != "cpu":
            try:
                free, total = K.cp.cuda.runtime.memGetInfo()
            except Exception:  # noqa: BLE001 - health must stay available after a GPU-context failure
                pass
        b = K.busy
        busy = None
        if b["active"] and K.lock.locked():
            busy = {"phase": b["phase"], "elapsed_s": round(time.time() - b["started"]),
                    "prompt_tokens": b["prompt_tokens"], "reused": b["cached"],
                    "prompt_read": "%d/%d" % (b["read"], b["prompt_tokens"] - b["cached"]), "generated": b["generated"]}
        return {"ok": True, "kindi": __version__, "commit": COMMIT, "model": model_id, "ctx": K.eng.ctx, "head": K.head_active(),
                "uptime_s": round(time.time() - K.t0), "busy": busy, "stats": K.stats,
                "queue": {"waiting": gate.waiting, "max": gate.max_queue, "rejected": gate.rejected},
                "gpu_free_mib": round(free / 1048576.0), "gpu_total_mib": round(total / 1048576.0)}

    @app.get("/v1/models")
    def models(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        return {"object": "list", "data": [{"id": model_id, "object": "model", "created": int(K.t0),
                                            "owned_by": "kindi", "context_length": K.eng.ctx}]}

    @app.get("/metrics")
    def metrics(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        s = K.stats

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
        metric("kindi_request_errors_total", "counter", "requests that failed after admission",
               "kindi_request_errors_total %d" % g("errors"))
        cache_errors = getattr(getattr(K, "runner", None), "cache_errors", 0)
        metric("kindi_cache_errors_total", "counter", "cache fallbacks after a damaged opening or failed unpark",
               "kindi_cache_errors_total %d" % cache_errors)
        metric("kindi_prompt_tokens_total", "counter", "prompt tokens processed",
               "kindi_prompt_tokens_total %d" % g("prompt_tokens"))
        metric("kindi_cached_prompt_tokens_total", "counter", "prompt tokens served from the cache",
               "kindi_cached_prompt_tokens_total %d" % g("cached_tokens"))
        metric("kindi_completion_tokens_total", "counter", "completion tokens generated",
               "kindi_completion_tokens_total %d" % g("completion_tokens"))
        metric("kindi_spec_drafted_tokens_total", "counter", "tokens proposed by speculative drafting",
               "kindi_spec_drafted_tokens_total %d" % g("spec_drafted"))
        metric("kindi_spec_accepted_tokens_total", "counter", "drafted tokens accepted by the verify pass",
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
        try:
            req = await _read_json_body(request)
        except RequestTooLarge as e:
            return _err(413, str(e), "request_too_large")
        except RequestError as e:
            return _err(400, str(e))
        try:
            _validate_request(req, has_vision)
        except RequestError as e:
            return _err(400, str(e))
        if req.get("logprobs") and not getattr(K, "supports_logprobs", False):
            return _err(400, "logprobs are not supported by this backend; remove logprobs or use a Gemma 4 pack",
                        "invalid_request_error")
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

        def chunk(delta, finish=None, usage=None):
            d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                d["usage"] = usage
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        cancel = threading.Event()

        if req.get("stream"):
            # an oversized or invalid prompt is an HTTP 400 now: after the first byte the status is always 200
            try:
                await asyncio.get_event_loop().run_in_executor(None, K.check_prompt, req)
            except OverflowError as e:
                return _err(400, str(e), "context_length_exceeded")
            except ValueError as e:
                return _err(400, str(e))
            except Exception:  # noqa: BLE001 - anything else: the stream reports it
                pass

            try:
                gate.enter()
            except QueueFull as e:
                return _busy(e)

            def _sse_locked():
                with K.lock:
                    yield chunk({"role": "assistant", "content": ""})
                    held, in_call = "", False
                    try:
                        for kind, val in K.generate(req, cancel):
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
                        yield "data: " + json.dumps({"error": {"message": str(e), "type": "context_length_exceeded"}}) + "\n\n"
                    except ValueError as e:
                        yield "data: " + json.dumps({"error": {"message": str(e), "type": "invalid_request_error"}}) + "\n\n"
                    except Exception as e:
                        yield "data: " + json.dumps({"error": {"message": "kindi: %s" % e, "type": "server_error"}}) + "\n\n"
                        _after_engine_error(K, e)
                    yield "data: [DONE]\n\n"

            def sse():
                try:
                    for item in _sse_locked():
                        yield item
                finally:
                    gate.release()

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
                fut = loop.run_in_executor(GPU_THREAD, pump)
                try:
                    while True:
                        # while a long prompt is read, nothing else is sent: an SSE comment every 15 s keeps the
                        # stream alive (Node's fetch, which pi uses, drops a body that is silent for 300 s)
                        try:
                            item = await asyncio.wait_for(q.get(), timeout=15)
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"
                            continue
                        if item is END:
                            break
                        yield item
                finally:
                    cancel.set()
                    await asyncio.wait([fut], timeout=30)
            return StreamingResponse(relay(), media_type="text/event-stream")

        try:
            gate.enter()
        except QueueFull as e:
            return _busy(e)

        def run():
            with K.lock:
                info = None
                for kind, val in K.generate(req, cancel):
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
        except OverflowError as e:
            return _err(400, str(e), "context_length_exceeded")
        except ValueError as e:
            return _err(400, str(e))
        except Exception as e:
            _after_engine_error(K, e)
            return _err(500, "kindi: %s" % e, "server_error")
        finally:
            gate.release()
        msg = {"role": "assistant", "content": info["content"] if (info["content"] or not info["tool_calls"]) else None}
        if info["tool_calls"]:
            msg["tool_calls"] = info["tool_calls"]
        if info["reasoning"]:
            msg["reasoning_content"] = info["reasoning"]
        choice = {"index": 0, "message": msg, "finish_reason": info["finish"]}
        if info.get("logprobs") is not None:
            choice["logprobs"] = {"content": info["logprobs"]}
        return {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [choice], "usage": info["usage"], "timing": info["timing"]}

    return app


def serve(cfg, log=print):
    import uvicorn
    app = create_app(cfg, log)
    log("serving on http://%s:%d/v1 (model %s)" % (cfg["host"], cfg["port"], cfg["model_name"]))
    uvicorn.run(app, host=cfg["host"], port=int(cfg["port"]), log_level="warning")
