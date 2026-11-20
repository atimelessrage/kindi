"""Tool-call parsing: the model's <tool_call> XML to OpenAI tool_calls."""
import json

from kindi import chat

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]


def test_single_call_with_typed_arguments():
    text = ("Checking the forecast.\n\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n"
            "</function>\n</tool_call>")
    content, calls = chat.parse_tool_calls(text, TOOLS)
    assert content == "Checking the forecast."
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_") and len(call["id"]) == 29
    assert call["function"]["name"] == "get_weather"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3}


def test_plain_text_is_not_a_call():
    content, calls = chat.parse_tool_calls("Just an answer.", TOOLS)
    assert content == "Just an answer."
    assert calls == []


def test_truncated_call_is_returned_verbatim():
    text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar"
    content, calls = chat.parse_tool_calls(text, TOOLS)
    assert calls == []
    assert content == text.strip()


def test_multiple_calls_keep_their_order():
    text = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
            "</function>\n</tool_call>")
    content, calls = chat.parse_tool_calls(text, TOOLS)
    assert content == ""
    assert [json.loads(c["function"]["arguments"])["city"] for c in calls] == ["Paris", "Rome"]


def test_untyped_parameter_keeps_a_json_object():
    text = ('<tool_call>\n<function=extra>\n<parameter=opts>\n{"a": 1}\n</parameter>\n'
            "</function>\n</tool_call>")
    _, calls = chat.parse_tool_calls(text, [])
    assert json.loads(calls[0]["function"]["arguments"]) == {"opts": {"a": 1}}
