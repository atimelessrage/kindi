"""Render the published Bonsai chat templates with jinja2 and with Kindi, and diff the two.

Kindi renders both Bonsai templates without a jinja runtime (`kindi/chat.py`: `qwen35` for the
2B/4B/9B and `bonsai27` for the 27B family, including 2.5). Because a prompt that differs from the
template the model was trained on is a different prompt, this compares the two renderers with the
template text taken from the released GGUF metadata:

- `docs/benchmarks/2026-11-24-bonsai25-27b-header/header.json` (Bonsai 2.5 27B, `tokenizer.chat_template`)
- a local Bonsai 2 2B pack's `kindi_pack.json` (`chat_template`), which carries the 2B/4B/9B template

    python compare_template.py --pack ~/.kindi/packs/bonsai-2b [--out parity.json]

`arguments` as a JSON *string* is not a template case: the templates iterate `arguments|items`, so
they only accept a mapping. Kindi parses the string before rendering (OpenAI clients send strings),
and that path is covered by `tests/test_chat_template.py` instead.
"""
import argparse
import difflib
import json
import os
import sys

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from kindi import chat  # noqa: E402

HEADER = os.path.join(HERE, "..", "2026-11-24-bonsai25-27b-header", "header.json")

TOOL = {"type": "function",
        "function": {"name": "get_weather",
                     "description": "Weather for a city",
                     "parameters": {"type": "object",
                                    "properties": {"city": {"type": "string"},
                                                   "days": {"type": "integer"}},
                                    "required": ["city"]}}}


def load_templates(pack):
    templates = {}
    with open(HEADER, encoding="utf-8") as f:
        templates["bonsai25-27b"] = ("bonsai27", json.load(f)["kv"]["tokenizer.chat_template"])
    if pack:
        with open(os.path.join(os.path.expanduser(pack), "kindi_pack.json"), encoding="utf-8") as f:
            templates["bonsai2-2b"] = ("qwen35", json.load(f)["chat_template"])
    return templates


def jinja_render(template, messages, tools=None, enable_thinking=None, reasoning_effort=None,
                 preserve_thinking=None):
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    tpl = env.from_string(template)
    # Undefined, not None: the template's `default('xhigh')` only fires for an undefined variable.
    kwargs = dict(messages=messages, tools=tools, add_generation_prompt=True, add_vision_id=False)
    for key, value in (("enable_thinking", enable_thinking), ("reasoning_effort", reasoning_effort),
                       ("preserve_thinking", preserve_thinking)):
        if value is not None:
            kwargs[key] = value
    return tpl.render(**kwargs)


def kindi_render(variant, messages, tools=None, enable_thinking=False, reasoning_effort=None,
                 preserve_thinking=None):
    body, gen = chat.render(messages, tools=tools, enable_thinking=enable_thinking, variant=variant,
                            reasoning_effort=reasoning_effort, preserve_thinking=preserve_thinking)
    return body + gen


CASES = [
    ("user only, thinking off",
     dict(messages=[{"role": "user", "content": "Hi"}], enable_thinking=False)),
    ("user only, thinking on",
     dict(messages=[{"role": "user", "content": "Hi"}], enable_thinking=True)),
    ("user only, effort medium",
     dict(messages=[{"role": "user", "content": "Hi"}], enable_thinking=True, reasoning_effort="medium")),
    ("user only, effort low",
     dict(messages=[{"role": "user", "content": "Hi"}], enable_thinking=True, reasoning_effort="low")),
    ("system + user",
     dict(messages=[{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"}], enable_thinking=False)),
    ("system + user, thinking on",
     dict(messages=[{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"}], enable_thinking=True)),
    ("tools only, thinking off",
     dict(messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[TOOL],
          enable_thinking=False)),
    ("tools only, thinking on",
     dict(messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[TOOL],
          enable_thinking=True)),
    ("tools + system + user",
     dict(messages=[{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Weather in Paris?"}], tools=[TOOL],
          enable_thinking=False)),
    ("assistant history keeps reasoning",
     dict(messages=[{"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello", "reasoning_content": "thinking"}],
          enable_thinking=True)),
    ("assistant history drops reasoning",
     dict(messages=[{"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello", "reasoning_content": "thinking"}],
          enable_thinking=True, preserve_thinking=False)),
    ("tool call + tool response",
     dict(messages=[{"role": "user", "content": "Weather in Paris?"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                    {"role": "tool", "content": "18 C"}],
          tools=[TOOL], enable_thinking=False)),
    ("tool call with dict/list/scalar arguments",
     dict(messages=[{"role": "user", "content": "Weather in Paris?"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": {"city": "Paris",
                                                    "window": {"from": "09:00", "to": "17:00"},
                                                    "days": ["mon", "tue"],
                                                    "note": "a < b & c's"}}}]},
                    {"role": "tool", "content": "18 C"}],
          tools=[TOOL], enable_thinking=False)),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=os.path.expanduser("~/.kindi/packs/bonsai-2b"),
                    help="a Bonsai 2 pack whose manifest carries the 2B/4B/9B template")
    ap.add_argument("--out", help="write the per-case result as JSON")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    templates = load_templates(a.pack)
    report, failures = [], 0
    for tname, (variant, template) in templates.items():
        for case_name, case in CASES:
            want = jinja_render(template, **case)
            try:
                got = kindi_render(variant, **case)
            except Exception as exc:                       # noqa: BLE001 - the report is the point
                failures += 1
                report.append({"template": tname, "case": case_name, "match": False,
                               "error": "%s: %s" % (type(exc).__name__, exc)})
                print("FAIL %-14s %-45s raised %s" % (tname, case_name, exc))
                continue
            ok = got == want
            failures += 0 if ok else 1
            report.append({"template": tname, "case": case_name, "match": ok,
                           "template_chars": len(want), "kindi_chars": len(got),
                           "template_sha256": __import__("hashlib").sha256(want.encode()).hexdigest()})
            print("%s %-14s %-45s %d chars" % ("ok  " if ok else "FAIL", tname, case_name, len(want)))
            if not ok and a.verbose:
                for line in difflib.unified_diff(want.splitlines(True), got.splitlines(True),
                                                 fromfile="jinja2", tofile="kindi", n=1):
                    print("    " + line.rstrip("\n"))
    total = len(report)
    print("\n%d of %d cases match across %d templates" % (total - failures, total, len(templates)))
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"cases": report, "match": total - failures, "total": total,
                       "templates": sorted(templates)}, f, indent=1)
            f.write("\n")
        print("wrote %s" % a.out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
