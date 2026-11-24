# SPDX-License-Identifier: AGPL-3.0-only
"""Anthropic Messages translation for ``kindi serve`` (REQ-0028).

Claude Code and several agent harnesses speak Anthropic's ``POST /v1/messages``
rather than OpenAI's chat shape. This module is the whole dialect difference,
in both directions, with no decoding change:

* ``to_chat(body)`` maps an Anthropic request onto the internal (OpenAI-shaped)
  request the engine already validates and runs: ``system`` (string or text
  blocks) becomes a system message, ``content`` block lists become text,
  ``tool_use`` becomes ``tool_calls``, a ``tool_result`` becomes a ``tool``
  message, ``tools``/``tool_choice`` map field for field, and
  ``stop_sequences`` becomes ``stop``.
* ``message(info, ...)`` maps the engine's finished answer to the Anthropic
  envelope (``content`` blocks, ``stop_reason``, ``usage``).
* ``stream(chunks, ...)`` turns the engine's ``(kind, value)`` generator into
  the Anthropic SSE event order: ``message_start`` -> ``content_block_start``
  -> ``content_block_delta``(s) -> ``content_block_stop`` -> ``message_delta``
  -> ``message_stop``, plus ``error`` when the engine fails mid-answer.

No FastAPI, HTTP or engine import: the functions are plain dict/string
transforms, so the whole gate runs on a CPU-only host and on Python 3.6
(the container image).
"""
import json

#: OpenAI ``finish_reason`` -> Anthropic ``stop_reason``.
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                "function_call": "tool_use", "content_filter": "end_turn"}

#: Anthropic has no "length" or "tool_calls" beyond these; keep the four the API defines.
ANTHROPIC_STOP_REASONS = ("end_turn", "max_tokens", "stop_sequence", "tool_use")


class TranslationError(ValueError):
    """A request this endpoint cannot translate; the server answers 400, never 500."""

    def __init__(self, message, param=None):
        ValueError.__init__(self, message)
        self.param = param


def error_payload(message, error_type="invalid_request_error"):
    """The Anthropic error envelope (``{"type": "error", "error": {...}}``)."""
    return {"type": "error", "error": {"type": error_type, "message": message}}


def stop_reason(finish):
    """Map the engine's ``finish_reason``; an unknown one is ``end_turn`` (a reply did finish)."""
    return STOP_REASONS.get(finish, "end_turn")


# --------------------------------------------------------------------------- request


def _text(value, where):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise TranslationError("%s must be a string" % where, where)


def _blocks(content, where):
    """A content field as a list of block dicts (a bare string is one text block)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise TranslationError("%s must be a string or a list of content blocks" % where, where)
    for i, block in enumerate(content):
        if not isinstance(block, dict):
            raise TranslationError("%s[%d] must be an object" % (where, i), where)
        if not isinstance(block.get("type"), str):
            raise TranslationError("%s[%d].type must be a string" % (where, i), where)
    return content


def _system_text(system):
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        raise TranslationError("system must be a string or a list of text blocks", "system")
    parts = []
    for i, block in enumerate(system):
        if not isinstance(block, dict) or block.get("type") != "text":
            raise TranslationError("system[%d] must be a text block" % i, "system")
        parts.append(_text(block.get("text"), "system[%d].text" % i))
    return "".join(parts)


def _tool_result_text(block, where):
    """The text of a ``tool_result`` block (its ``content`` is a string or text blocks)."""
    inner = block.get("content")
    if inner is None:
        return ""
    if isinstance(inner, str):
        return inner
    if not isinstance(inner, list):
        raise TranslationError("%s.content must be a string or a list of blocks" % where, where)
    parts = []
    for i, part in enumerate(inner):
        if not isinstance(part, dict):
            raise TranslationError("%s.content[%d] must be an object" % (where, i), where)
        kind = part.get("type")
        if kind == "text":
            parts.append(_text(part.get("text"), "%s.content[%d].text" % (where, i)))
        elif kind == "image":
            raise TranslationError("images are not supported by this endpoint (%s.content[%d])"
                                   % (where, i), where)
        else:
            raise TranslationError("%s.content[%d] has an unsupported block type %r"
                                   % (where, i, kind), where)
    return "".join(parts)


def _tool_use_call(block, where):
    name = block.get("name")
    if not isinstance(name, str) or not name:
        raise TranslationError("%s.tool_use needs a name" % where, where)
    arguments = block.get("input")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise TranslationError("%s.tool_use.input must be an object" % where, where)
    call_id = block.get("id") or "toolu_%s" % name
    if not isinstance(call_id, str):
        raise TranslationError("%s.tool_use.id must be a string" % where, where)
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def _user_messages(content, where):
    """A user turn -> zero or more OpenAI messages (tool results precede the text)."""
    blocks = _blocks(content, where)
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return [{"role": "user", "content": _text(blocks[0].get("text"), where + "[0].text")}]
    results, texts = [], []
    for i, block in enumerate(blocks):
        kind = block.get("type")
        if kind == "text":
            texts.append(_text(block.get("text"), "%s[%d].text" % (where, i)))
        elif kind == "tool_result":
            tool_id = block.get("tool_use_id")
            if not isinstance(tool_id, str) or not tool_id:
                raise TranslationError("%s[%d].tool_result needs a tool_use_id" % (where, i), where)
            results.append({"role": "tool", "tool_call_id": tool_id,
                            "content": _tool_result_text(block, "%s[%d]" % (where, i))})
        elif kind in ("image", "document"):
            raise TranslationError("images and documents are not supported by this endpoint "
                                   "(%s[%d])" % (where, i), where)
        else:
            raise TranslationError("%s[%d] has an unsupported block type %r" % (where, i, kind), where)
    out = list(results)
    if texts:
        out.append({"role": "user", "content": "".join(texts)})
    return out


def _assistant_message(content, where):
    blocks = _blocks(content, where)
    texts, calls = [], []
    for i, block in enumerate(blocks):
        kind = block.get("type")
        if kind == "text":
            texts.append(_text(block.get("text"), "%s[%d].text" % (where, i)))
        elif kind == "tool_use":
            calls.append(_tool_use_call(block, "%s[%d]" % (where, i)))
        elif kind in ("thinking", "redacted_thinking"):
            continue                      # a previous turn's reasoning is not fed back
        else:
            raise TranslationError("%s[%d] has an unsupported block type %r" % (where, i, kind), where)
    message: dict = {"role": "assistant", "content": "".join(texts)}
    if calls:
        message["tool_calls"] = calls
    return message


def _tool(tool, index):
    if not isinstance(tool, dict):
        raise TranslationError("tools[%d] must be an object" % index, "tools")
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise TranslationError("tools[%d] needs a name" % index, "tools")
    function: dict = {"name": name}
    if isinstance(tool.get("description"), str):
        function["description"] = tool["description"]
    schema = tool.get("input_schema")
    function["parameters"] = schema if isinstance(schema, dict) else {"type": "object", "properties": {}}
    return {"type": "function", "function": function}


def _tool_choice(choice):
    if choice is None:
        return None
    if not isinstance(choice, dict):
        raise TranslationError("tool_choice must be an object", "tool_choice")
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise TranslationError("tool_choice.type=tool needs a name", "tool_choice")
        return {"type": "function", "function": {"name": name}}
    raise TranslationError("tool_choice.type must be auto, any, none or tool", "tool_choice")


def to_chat(body):
    """An Anthropic Messages body -> the internal (OpenAI-shaped) chat request."""
    if not isinstance(body, dict):
        raise TranslationError("the body must be a JSON object")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise TranslationError("stream must be a boolean", "stream")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TranslationError("messages must be a non-empty list", "messages")
    chat = []
    system = _system_text(body.get("system"))
    if system:
        chat.append({"role": "system", "content": system})
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TranslationError("messages[%d] must be an object" % i, "messages")
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise TranslationError("messages[%d].role must be user or assistant" % i, "messages")
        where = "messages[%d].content" % i
        if role == "user":
            chat.extend(_user_messages(message.get("content"), where))
        else:
            described = _assistant_message(message.get("content"), where)
            if not described["content"] and not described.get("tool_calls"):
                raise TranslationError("%s has no text and no tool_use block" % where, "messages")
            chat.append(described)
    request: dict = {"messages": chat, "stream": stream}
    if body.get("max_tokens") is not None:
        max_tokens = body["max_tokens"]
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TranslationError("max_tokens must be an integer", "max_tokens")
        if max_tokens < 1:
            raise TranslationError("max_tokens must be at least 1", "max_tokens")
        request["max_tokens"] = max_tokens
    for name in ("temperature", "top_p", "top_k"):
        if body.get(name) is not None:
            request[name] = body[name]
    stops = body.get("stop_sequences")
    if stops is not None:
        if not isinstance(stops, list) or any(not isinstance(s, str) for s in stops):
            raise TranslationError("stop_sequences must be a list of strings", "stop_sequences")
        if stops:
            request["stop"] = list(stops)
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise TranslationError("tools must be a list", "tools")
        if tools:
            request["tools"] = [_tool(tool, i) for i, tool in enumerate(tools)]
    choice = _tool_choice(body.get("tool_choice"))
    if choice is not None:
        request["tool_choice"] = choice
    return request


# --------------------------------------------------------------------------- response


def _json_input(arguments):
    """A tool call's arguments as the object an Anthropic ``tool_use.input`` holds."""
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {"_raw": arguments}
    return value if isinstance(value, dict) else {"_value": value}


def _items(value):
    return value if isinstance(value, list) else []


def message(info, model_id, message_id):
    """A finished engine answer -> the Anthropic ``message`` envelope."""
    info = info or {}
    blocks = []
    text = info.get("content") or ""
    if text:
        blocks.append({"type": "text", "text": text})
    for i, call in enumerate(_items(info.get("tool_calls"))):
        function = call.get("function") or {}
        blocks.append({"type": "tool_use",
                       "id": call.get("id") or "toolu_%d_%s" % (i, message_id),
                       "name": function.get("name") or "",
                       "input": _json_input(function.get("arguments"))})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    usage = info.get("usage") or {}
    return {"id": message_id, "type": "message", "role": "assistant", "model": model_id,
            "content": blocks, "stop_reason": stop_reason(info.get("finish")),
            "stop_sequence": None,
            "usage": {"input_tokens": int(usage.get("prompt_tokens") or 0),
                      "output_tokens": int(usage.get("completion_tokens") or 0)}}


# --------------------------------------------------------------------------- stream


def event(name, payload):
    """One SSE frame in Anthropic's form (``event:`` line + ``data:`` JSON)."""
    return "event: %s\ndata: %s\n\n" % (name, json.dumps(payload, ensure_ascii=False))


def _message_start(model_id, message_id, input_tokens):
    return event("message_start", {
        "type": "message_start",
        "message": {"id": message_id, "type": "message", "role": "assistant", "model": model_id,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": int(input_tokens or 0), "output_tokens": 1}}})


def _block_start(index, block):
    return event("content_block_start", {"type": "content_block_start", "index": index,
                                         "content_block": block})


def _text_delta(index, text):
    return event("content_block_delta", {"type": "content_block_delta", "index": index,
                                         "delta": {"type": "text_delta", "text": text}})


def _input_delta(index, partial_json):
    return event("content_block_delta", {"type": "content_block_delta", "index": index,
                                         "delta": {"type": "input_json_delta",
                                                   "partial_json": partial_json}})


def _block_stop(index):
    return event("content_block_stop", {"type": "content_block_stop", "index": index})


def error_event(message_text, error_type="api_error"):
    return event("error", error_payload(message_text, error_type))


def stream(chunks, model_id, message_id, input_tokens=0):
    """Engine ``(kind, value)`` items -> the Anthropic SSE sequence.

    Text is held back from a ``<tool_call>`` onward, exactly as the OpenAI route
    does, so a parsed call is emitted as a ``tool_use`` block instead of leaking
    its raw text; text held when the call does not parse is flushed as a text
    delta before the block closes.
    """
    yield _message_start(model_id, message_id, input_tokens)
    index, text_open, held, in_call = 0, False, "", False
    info = None

    def open_text():
        return _block_start(index, {"type": "text", "text": ""})

    for kind, value in chunks:
        if kind != "text":
            if kind == "done":
                info = value
            continue
        held += value
        if not in_call and "<tool_call>" in held:
            head = held[:held.index("<tool_call>")]
            held = held[held.index("<tool_call>"):]
            in_call = True
            if head:
                if not text_open:
                    text_open = True
                    yield open_text()
                yield _text_delta(index, head)
            continue
        if in_call:
            continue
        cut = max(held.rfind("<"), 0) if "<" in held[-12:] else len(held)
        out, held = held[:cut], held[cut:]
        if out:
            if not text_open:
                text_open = True
                yield open_text()
            yield _text_delta(index, out)
    info = info or {}
    calls = _items(info.get("tool_calls"))
    if held and not calls:
        if not text_open:
            text_open = True
            yield open_text()
        yield _text_delta(index, held)
    if text_open:
        yield _block_stop(index)
        index += 1
    for i, call in enumerate(calls):
        function = call.get("function") or {}
        block = {"type": "tool_use", "id": call.get("id") or "toolu_%d_%s" % (i, message_id),
                 "name": function.get("name") or "", "input": {}}
        yield _block_start(index, block)
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if isinstance(arguments, dict) else {}, ensure_ascii=False)
        yield _input_delta(index, arguments or "{}")
        yield _block_stop(index)
        index += 1
    usage = info.get("usage") or {}
    yield event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason(info.get("finish")), "stop_sequence": None},
        "usage": {"output_tokens": int(usage.get("completion_tokens") or 0)}})
    yield event("message_stop", {"type": "message_stop"})
