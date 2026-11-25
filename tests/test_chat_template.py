"""Chat template golden strings for the Qwen3.5 (2B/4B/9B) and 27B variants.

The 27B goldens are the published Bonsai 2.5 template rendered with jinja2
(`docs/benchmarks/2026-11-24-bonsai-chat-template-parity/`), not this renderer's own output.
"""
import json

import pytest

from kindi import chat

GEN_OFF = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
GEN_ON = '<|im_start|>assistant\n<think>\n'
USER_BODY = '<|im_start|>user\nHi<|im_end|>\n'
SYS_BODY = '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n'
HIST_BODY = '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n'
ON_BODY = '<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n'


def test_user_only_golden():
    body, gen = chat.render([{"role": "user", "content": "Hi"}])
    assert body == USER_BODY
    assert gen == GEN_OFF


def test_system_and_user_golden():
    body, gen = chat.render([{"role": "system", "content": "Be brief."},
                             {"role": "user", "content": "Hi"}])
    assert body == SYS_BODY
    assert gen == GEN_OFF


def test_assistant_history_golden():
    body, _ = chat.render([{"role": "user", "content": "Hi"},
                           {"role": "assistant", "content": "Hello"}])
    assert body == HIST_BODY


def test_thinking_changes_only_the_generation_prompt():
    # with thinking on the template also writes the effort line as a system message
    body, gen = chat.render([{"role": "user", "content": "Hi"}], variant="bonsai27",
                            enable_thinking=True)
    assert body == ON_BODY
    assert gen == GEN_ON


def test_27b_variant_matches_the_published_template():
    body, gen = chat.render([{"role": "user", "content": "Hi"}], variant="bonsai27",
                            enable_thinking=True)
    assert body + gen == '<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n'


def test_tools_block():
    tool = {"type": "function", "function": {"name": "get_weather",
                                             "parameters": {"type": "object",
                                                            "properties": {"city": {"type": "string"}}}}}
    body, _ = chat.render([{"role": "user", "content": "Weather?"}], tools=[tool])
    assert body.startswith("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n"
                           "<tools>\n")
    # the templates render `tool | tojson`: sorted keys, so "function" precedes "type"
    assert "\n<tools>\n" + json.dumps(tool, ensure_ascii=False, sort_keys=True) + "\n</tools>" in body
    assert chat.TOOL_INSTR in body
    assert body.endswith("<|im_start|>user\nWeather?<|im_end|>\n")


def test_tool_schemas_use_the_templates_tojson_in_both_variants():
    """Both published templates render `tool | tojson`; another key order is another prompt.

    docs/benchmarks/2026-11-24-bonsai-chat-template-parity/ holds the byte-for-byte check against
    the template text from the released GGUF files.
    """
    tool = {"type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
    want = json.dumps(tool, ensure_ascii=False, sort_keys=True)
    for variant in ("qwen35", "bonsai27"):
        body, _ = chat.render([{"role": "user", "content": "x"}], tools=[tool], variant=variant)
        assert "\n<tools>\n" + want + "\n</tools>" in body


def test_tool_arguments_are_tojson_in_history():
    """Non-string argument values go through tojson, HTML escapes included; strings stay raw."""
    tool = {"type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
    messages = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "f",
                                  "arguments": {"window": {"to": "a<b & c's"},
                                                "days": ["mon"], "city": "Paris"}}}]},
                {"role": "tool", "content": "ok"}]
    for variant in ("qwen35", "bonsai27"):
        body, _ = chat.render(messages, tools=[tool], variant=variant)
        assert "<parameter=city>\nParis\n</parameter>" in body              # a string stays raw
        assert r'<parameter=window>' "\n" r'{"to": "a\u003cb \u0026 c\u0027s"}' "\n</parameter>" in body
        assert '<parameter=days>\n["mon"]\n</parameter>' in body


def test_tojson_matches_jinja_sorted_keys():
    assert chat._tojson({"b": 1, "a": [2, 3]}) == '{"a": [2, 3], "b": 1}'
    assert chat._tojson({"k": "a<b&c'd"}) == r'{"k": "a\u003cb\u0026c\u0027d"}'


def test_template_variant():
    assert chat.template_variant({"model": "Ternary-Bonsai-2-27B"}) == "bonsai27"
    assert chat.template_variant({"model": "Ternary-Bonsai-2.5-27B"}) == "bonsai27"
    assert chat.template_variant({"model": "Ternary-Bonsai-2-2B"}) == "qwen35"
    assert chat.template_variant({"chat_template": "{{ reasoning_effort }}"}) == "bonsai27"


def test_27b_reasoning_effort_and_error():
    # the 2.5 card: low is not supported and the model behaves close to xhigh, so the xhigh
    # instruction is rendered (the template's own low text is not what the model was trained on)
    low, _ = chat.render([{"role": "user", "content": "Hi"}], variant="bonsai27",
                         enable_thinking=True, reasoning_effort="low")
    xhigh, _ = chat.render([{"role": "user", "content": "Hi"}], variant="bonsai27",
                           enable_thinking=True, reasoning_effort="xhigh")
    assert low == xhigh and "Reasoning effort is set to xhigh" in low
    # round-5 audit 5.8/3a: OpenAI's vocabulary is low|medium|high, so a legal `high` must not be a 400
    # while the vendor's unsupported `low` is accepted; both are aliases of the one measured instruction.
    high, _ = chat.render([{"role": "user", "content": "Hi"}], variant="bonsai27",
                          enable_thinking=True, reasoning_effort="high")
    assert high == xhigh
    assert chat.ACCEPTED_EFFORTS == ("xhigh", "high", "medium", "low")
    chat.check_effort(None)                       # None is the template default
    with pytest.raises(ValueError, match="low and high render the xhigh"):
        chat.check_effort("max")
    with pytest.raises(ValueError, match="Unexpected reasoning effort"):
        chat.render([{"role": "user", "content": "Hi"}], variant="bonsai27",
                    enable_thinking=True, reasoning_effort="max")


def test_rejects_invalid_message_lists():
    with pytest.raises(ValueError, match="No messages"):
        chat.render([])
    with pytest.raises(ValueError, match="No user query"):
        chat.render([{"role": "assistant", "content": "x"}])
    with pytest.raises(ValueError, match="System message must be at the beginning"):
        chat.render([{"role": "user", "content": "x"}, {"role": "system", "content": "y"}])
