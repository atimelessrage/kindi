"""Chat template golden strings for the Qwen3.5 (2B/4B/9B) and 27B variants."""
import pytest

from kindi import chat

GEN_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
GEN_ON = "<|im_start|>assistant\n<think>\n"


def test_user_only_golden():
    body, gen = chat.render([{"role": "user", "content": "Hi"}])
    assert body == "<|im_start|>user\nHi<|im_end|>\n"
    assert gen == GEN_OFF


def test_system_and_user_golden():
    body, gen = chat.render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
    assert body == ("<|im_start|>system\nBe brief.<|im_end|>\n"
                    "<|im_start|>user\nHi<|im_end|>\n")
    assert gen == GEN_OFF


def test_assistant_history_golden():
    body, _ = chat.render([{"role": "user", "content": "Hi"},
                           {"role": "assistant", "content": "Hello"}])
    assert body == ("<|im_start|>user\nHi<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n")


def test_thinking_changes_only_the_generation_prompt():
    body, gen = chat.render([{"role": "user", "content": "Hi"}], enable_thinking=True)
    assert body == "<|im_start|>user\nHi<|im_end|>\n"
    assert gen == GEN_ON


def test_tools_block():
    tool = {"type": "function", "function": {"name": "get_weather",
                                             "parameters": {"type": "object",
                                                            "properties": {"city": {"type": "string"}}}}}
    body, _ = chat.render([{"role": "user", "content": "Weather?"}], tools=[tool])
    assert body.startswith("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n")
    assert '{"type": "function", "function": {"name": "get_weather"' in body
    assert chat.TOOL_INSTR in body
    assert body.endswith("<|im_start|>user\nWeather?<|im_end|>\n")


def test_template_variant():
    assert chat.template_variant({"model": "Ternary-Bonsai-2-27B"}) == "bonsai27"
    assert chat.template_variant({"model": "Ternary-Bonsai-2-2B"}) == "qwen35"
    assert chat.template_variant({"chat_template": "{{ reasoning_effort }}"}) == "bonsai27"


def test_27b_reasoning_effort_and_error():
    body, _ = chat.render([{"role": "user", "content": "Hi"}], variant="bonsai27",
                          enable_thinking=True, reasoning_effort="low")
    assert "Keep your thinking brief" in body
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
