"""Tool-call conformance suite (REQ-0018): one contract per shipped backend parser.

Contract for every backend (Bonsai XML is in tests/test_tool_calls.py):
valid calls parse to OpenAI tool_calls with JSON-string arguments; a truncated
or malformed call never leaks raw markup as a call; quoted examples are not
calls; calls for tools the request did not offer are dropped.
"""
import json

from kindi import gemma4_tools, nanbeige_gen
from kindi.dsv41_server import truncated_tool_calls
from kindi.lightning_gen import calls_of, close_call, cut_calls, normalize_calls

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city"]}}}]


def _args(call):
    return json.loads(call["function"]["arguments"])


# --- Gemma 4 native format (<|tool_call>call:NAME{...}<tool_call|>) ---

def test_gemma4_valid_call_typed_arguments():
    # string values are <|"|>text<|"|>
    text = ('Check this.<|tool_call>call:get_weather'
            '{city:<|"|>Paris<|"|>, days:3}<tool_call|>')
    content, reasoning, calls = gemma4_tools.split_reply(text)
    assert content == "Check this."
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"city": "Paris", "days": 3}
    assert "<|tool_call>" not in content and "<tool_call|>" not in content


def test_gemma4_nested_and_list_arguments():
    text = ('<|tool_call>call:get_weather{city:<|"|>Oslo<|"|>, opts:{units:<|"|>metric<|"|>},'
            ' tags:[<|"|>a<|"|>, <|"|>b<|"|>], rainy:true, extra:null}<tool_call|>')
    _, _, calls = gemma4_tools.split_reply(text)
    assert _args(calls[0]) == {"city": "Oslo", "opts": {"units": "metric"},
                               "tags": ["a", "b"], "rainy": True, "extra": None}


def test_gemma4_truncated_call_stays_in_content():
    text = '<|tool_call>call:get_weather{city:<|"|>Par'
    content, _, calls = gemma4_tools.split_reply(text)
    assert calls == []
    assert "<|tool_call>" in content  # kept as text per split_reply contract (nothing lost)


def test_gemma4_malformed_call_stays_in_content():
    text = '<|tool_call>call:get_weather{city Paris}<tool_call|>'
    content, _, calls = gemma4_tools.split_reply(text)
    assert calls == []
    assert content  # not dropped entirely


def test_gemma4_thinking_extraction():
    text = '<|channel>thought\nLet me check.<channel|>Done.<|tool_call>call:get_weather{city:<|"|>Rome<|"|>}<tool_call|>'
    content, reasoning, calls = gemma4_tools.split_reply(text)
    assert reasoning == "Let me check."
    assert content == "Done."
    assert len(calls) == 1


# --- Nanbeige / Lightning XML+JSON (shared parse_calls) ---

def test_nanbeige_valid_xml_call():
    text = ('<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n'
            '<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>')
    calls = nanbeige_gen.parse_calls(text, TOOLS)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"city": "Paris", "days": 3}


def test_nanbeige_json_form_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lyon"}}\n</tool_call>'
    calls = nanbeige_gen.parse_calls(text, TOOLS)
    assert len(calls) == 1
    assert _args(calls[0]) == {"city": "Lyon"}


def test_nanbeige_quoted_example_is_not_a_call():
    text = 'The format is `<tool_call><function=write>path=/tmp/x</function></tool_call>` literally.'
    assert nanbeige_gen.parse_calls(text, TOOLS) == []


def test_nanbeige_unknown_tool_is_dropped():
    text = ('<tool_call>\n<function=example_function_name>\n<parameter=x>\n1\n</parameter>\n'
            '</function>\n</tool_call>')
    assert nanbeige_gen.parse_calls(text, TOOLS) == []


def test_nanbeige_missing_required_parameter_is_dropped():
    text = ('<tool_call>\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n'
            '</function>\n</tool_call>')
    assert nanbeige_gen.parse_calls(text, TOOLS) == []


def test_nanbeige_truncated_call_has_no_markup_leak():
    # a cut call at end-of-reply: its (partial) parameters parse if the required
    # ones are present; the raw XML must not surface as a tool call argument.
    text = ('<tool_call>\n<function=get_weather>\n<parameter=city>\nPar')
    calls = nanbeige_gen.parse_calls(text, TOOLS)
    for c in calls:
        assert "<tool_call>" not in c["function"]["arguments"]
        assert "<parameter" not in c["function"]["arguments"]


def test_lightning_normalize_slips():
    tail = ('<parameter=city=Paris>\n</function_call>')
    fixed = normalize_calls(tail)
    assert "<parameter=city>" in fixed and "</function>" in fixed
    assert "=" not in fixed.split("<parameter=city>")[0][-6:]


def test_lightning_close_call_closes_open_tags():
    tail = '<tool_call>\n<function=get_weather>\n<parameter=city>\nParis'
    closed = close_call(tail)
    assert closed.endswith("</tool_call>")
    calls = calls_of(closed, TOOLS)
    assert len(calls) == 1 and _args(calls[0]) == {"city": "Paris"}


def test_lightning_cut_calls_counts_closes():
    tail = ('<tool_call><function=a></function></tool_call> middle '
            '<tool_call><function=b></function></tool_call> end')
    assert cut_calls(tail, 1).endswith("</tool_call>")
    assert cut_calls(tail, 1).count("</tool_call>") == 1
    assert cut_calls(tail, 9) == tail


# --- DeepSeek DSML truncated block ---

class _Enc(object):
    dsml_token = "|DSML|"
    tool_calls_block_name = "tool_calls"
    tool_call_tag_name = " invoke"
    tool_parameter_tag_name = " parameter"


def _dsml(name, params, close=True):
    s = '<|DSML| invoke name="%s">\n' % name  # bare function name, as in real DSML
    for k, (v, is_str) in params.items():
        s += '<|DSML| parameter name="%s" string="%s">%s</|DSML| parameter>' % (
            k, "true" if is_str else "false", v)
    if close:
        s += '</|DSML| invoke>'
    return s


def test_deepseek_complete_call():
    block = _dsml("get_weather", {"city": ("Paris", True)})
    calls = truncated_tool_calls(_Enc(), block)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert "DSML" not in calls[0]["function"]["arguments"]


def test_deepseek_cut_inside_value_returns_unclosed_partial():
    block = _dsml("get_weather", {"city": ("Par", True)}, close=False)
    calls = truncated_tool_calls(_Enc(), block)
    assert len(calls) == 1
    args = calls[0]["function"]["arguments"]
    assert args.startswith('{"city": "Par')
    assert not args.endswith("}")  # never closed: must not parse, must not run
    with __import__("pytest").raises(ValueError):
        json.loads(args)
    assert "DSML" not in args and "<|DSML|" not in args


def test_deepseek_cut_name_returns_no_call():
    block = '<|DSML| invoke name="funct'
    assert truncated_tool_calls(_Enc(), block) == []


JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "x", "schema": {
    "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}}
TOKENS = ["<eos>", "{", '"ok"', ":", "true", "}", "<", "!"]
TYPES = [3, 1, 1, 1, 1, 1, 1, 1]
SCRIPT = [1, 2, 3, 4, 5, 0]         # { "ok" : true } then EOS


def _logits(legal):
    import numpy as np
    z = np.zeros(len(TOKENS))
    z[6] = 100.0                     # '<' is always the model's favorite and never grammar-legal
    z[legal] = 50.0
    return z


class _Cp(object):
    asnumpy = staticmethod(lambda x: __import__("numpy").asarray(x))
    flatnonzero = staticmethod(__import__("numpy").flatnonzero)

    class cuda(object):
        class Device(object):
            def __init__(self, dev):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False


def _fake_k():
    from types import SimpleNamespace
    from kindi.config import DEFAULTS
    return SimpleNamespace(
        cfg=dict(DEFAULTS, max_tokens=32, temperature=0.0, top_p=1.0, top_k=0),
        busy={}, stats={"requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
                       "completion_tokens": 0})


def test_nanbeige_structured_output_skips_the_illegal_top_token(monkeypatch):
    import kindi.nanbeige_chat as chat
    from kindi.nanbeige_gen import NanbeigeBackend
    monkeypatch.setattr(chat, "render", lambda *a, **k: "")
    monkeypatch.setattr(chat, "prompt_ids", lambda tok, text: [1])

    class _Tok(object):
        tokens = TOKENS
        types = TYPES
        eos = 0

        def piece(self, i):
            return TOKENS[i].encode()

        def encode(self, s, add_bos=False):
            return []

    class _Card(object):
        dev = 0
        logits = _logits(SCRIPT[0])

    class _Engine(object):
        ctx = 4096
        cp = _Cp
        cards = [_Card()]
        i = 0

        def prefill_batch(self, ids, pos0=0):
            return 0

        def step(self, tok, pos):
            return 0

        def step_begin(self, tok, pos):
            self.i += 1
            self.cards[0].logits = _logits(SCRIPT[min(self.i, len(SCRIPT) - 1)])

        def step_end(self):
            return 0

    backend = NanbeigeBackend.__new__(NanbeigeBackend)
    backend.log = lambda *a: None
    backend.tok = _Tok()
    backend.stop_ids = {0}
    backend.eng = _Engine()
    backend.cached = []
    backend.lookup = 0
    backend.lk_stats = [0] * 6
    backend.mtp = False
    req = {"messages": [{"role": "user", "content": "hi"}],
           "max_tokens": 16, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
           "response_format": JSON_SCHEMA}
    items = list(backend.generate(_fake_k(), req))
    done = next(v for kind, v in items if kind == "done")
    assert json.loads(done["content"]) == {"ok": True}
    assert done["finish"] == "stop"


def test_lightning_structured_output_skips_the_illegal_top_token(monkeypatch):
    import kindi.lightning_chat as chat
    from kindi.lightning_gen import LightningBackend
    monkeypatch.setattr(chat, "render", lambda *a, **k: "")

    class _Tok(object):
        vocab = TOKENS
        types = TYPES
        eos = 0
        sid = {"<|im_start|>": 99}

        def decode_bytes(self, i):
            return TOKENS[i].encode()

        def encode(self, text, **k):
            return [1]

    class _Engine(object):
        ctx = 4096
        cp = _Cp
        pos = 0
        i = 0

        def __init__(self):
            self.logits = _logits(SCRIPT[0])

        def reset(self):
            self.pos = 0

        def prefill(self, ids, pos=None):
            self.pos = (pos or 0) + len(ids)
            return 0

        def step(self, tok=None):
            self.pos += 1
            self.i += 1
            self.logits = _logits(SCRIPT[min(self.i, len(SCRIPT) - 1)])
            return 0

        def snapshot(self):
            pass

        def restore(self):
            pass

    backend = LightningBackend.__new__(LightningBackend)
    backend.log = lambda *a: None
    backend.path = "x.gguf"
    backend.tok = _Tok()
    backend.stop_ids = {0}
    backend.im_start = 99
    backend.eng = _Engine()
    backend.cached = []
    backend.snap = []
    req = {"messages": [{"role": "user", "content": "hi"}],
           "max_tokens": 16, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
           "response_format": JSON_SCHEMA}
    items = list(backend.generate(_fake_k(), req))
    done = next(v for kind, v in items if kind == "done")
    assert json.loads(done["content"]) == {"ok": True}
    assert done["finish"] == "stop"


def test_deepseek_no_markup_in_any_result():
    block = (_dsml("get_weather", {"city": ("Paris", True)})
             + _dsml("get_weather", {"city": ("Par", True)}, close=False))
    calls = truncated_tool_calls(_Enc(), block)
    assert len(calls) == 2
    for c in calls:
        blob = c["function"]["name"] + c["function"]["arguments"]
        assert "DSML" not in blob and "invoke" not in blob
