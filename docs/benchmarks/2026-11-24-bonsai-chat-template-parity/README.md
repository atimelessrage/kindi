# Bonsai chat templates: Kindi's renderer against the published `tokenizer.chat_template`

Host: i7-8086K, Windows, Python 3.11, jinja2. Date: 2026-11-24.

## Why

Kindi hand-writes the Bonsai prompt instead of running Jinja at request time, so a divergence from the
published template is a different prompt with the same weights: the model degrades and nothing raises.
The 2.5 intake compared them for the first time and found one.

## What was done

`compare_template.py` renders a fixed set of conversations twice - once with jinja2 from the template
string a GGUF ships (`bonsai25-27b` from the header run's `header.json`, `bonsai2-2b` from a local
`kindi_pack.json`) and once with `kindi.chat.render()` - and compares the bytes:

```sh
python docs/benchmarks/2026-11-24-bonsai-chat-template-parity/compare_template.py > parity.json
```

The 26 cases cover the generation prompt with thinking on and off, the reasoning-effort line, a system
message, multi-turn history, a tool schema in `<tools>`, tool calls whose arguments are objects/lists
rather than strings, and the tool result in `<tool_response>`.

## Result

| template | cases | byte-identical |
|---|---|---|
| `Ternary-Bonsai-2.5-27B` (from `header.json`) | 13 | 13 |
| `Ternary-Bonsai-2-2B` (from a converted pack) | 13 | 13 |

`26 of 26` cases match, 0 mismatches (`parity.json`).

## What it fixed

Tool schemas and non-string tool arguments were emitted as `json.dumps(...)` in insertion order, while
the template uses Jinja's `tojson`: sorted keys and HTML-safe escapes (`<`, `>`, `&`, `'`,
U+2028/U+2029). So Kindi put `"type"` before `"function"` and left `<` unescaped inside a tool result.
`kindi/chat.py` now serializes both through one `_tojson()` helper, which `tests/test_chat_template.py`
pins against goldens rendered by jinja2 from the same template text.

## What this does not show

No model ran: this is prompt text only. It does not measure whether the 2.5 weights answer better with
the corrected prompt, and it does not touch the `-mtp` draft head's template.
