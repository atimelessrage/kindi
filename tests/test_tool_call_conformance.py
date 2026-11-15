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


def test_deepseek_no_markup_in_any_result():
    block = (_dsml("get_weather", {"city": ("Paris", True)})
             + _dsml("get_weather", {"city": ("Par", True)}, close=False))
    calls = truncated_tool_calls(_Enc(), block)
    assert len(calls) == 2
    for c in calls:
        blob = c["function"]["name"] + c["function"]["arguments"]
        assert "DSML" not in blob and "invoke" not in blob
