#!/usr/bin/env python3
"""Check the canonical ternary-MoE census.

`docs/TERNARY_MOE.md` is the one source for the Qwen3.6-35B-A3B ternary rows, the other
published-build rows, their expert-slot arithmetic and the counted/unread totals. The roadmap links
there instead of keeping a copy.

    python scripts/check_ternary_moe.py [docs/TERNARY_MOE.md]

The checker is deliberately narrow: it reads the exact table headers this page owns, recomputes each
row's product, and pins the row and list counts. A new row or a changed number must update this
script in the same change, so a census edit cannot silently pass as documentation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = ROOT / "docs" / "TERNARY_MOE.md"

EXPECTED_QWEN_ROWS = 2
EXPECTED_OTHER_ROWS = 6
EXPECTED_TOTAL_ROWS = 8
EXPECTED_OTHER_SLOTS = 51_864
EXPECTED_UNREAD = 3
EXPECTED_CHECKED = 3

COUNT_HEADER = ("what", "count")
QWEN_HEADER = ("build", "contents", "format and runtime",
               "why it cannot count as a llama.cpp candidate yet")
OTHER_HEADER = ("build", "base and ternary scope", "format and runtime", "status here")

LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
SLOT = re.compile(
    r"(\d[\d,]*)\s+(?:routed-)?experts?(?:\s+slots)?\b[^()\n]*"
    r"\((\d+)\s*[×x]\s*(\d+)(?=[;)])"
)
NOTA_BLOCKS = re.compile(
    r"(\d+)\s+blocks\s*\((\d+)\s+MoE,\s*(\d+)\s+Mamba-2,\s*(\d+)\s+attention\)"
)
FLINT_LAYERS = re.compile(r"(\d+)\s+layers\s*\((\d+)\s+dense\s*\+\s*(\d+)\s+MoE\)")
FLINT_TENSORS = re.compile(
    r"(\d+)\s+fused expert projection tensors\s*\((\d+)\s*[×x]\s*gate/up/down\)"
)
MICRO_LAYERS = re.compile(
    r"(\d+)\s+layers,\s*4 experts per odd layer\s*\((\d+)\s+MoE layers"
)


class CensusError(Exception):
    pass


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _sections(text):
    sections = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items()}


def _table(sections, section_name, header):
    body = sections.get(section_name)
    if body is None:
        raise CensusError("missing section: ## %s" % section_name)
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        if tuple(_cells(line)) != header:
            continue
        rows = []
        for row_line in lines[index + 2:]:
            if not row_line.lstrip().startswith("|"):
                break
            cells = _cells(row_line)
            if len(cells) != len(header):
                raise CensusError(
                    "%s: table row has %d cells, expected %d: %s"
                    % (section_name, len(cells), len(header), row_line.strip())
                )
            rows.append(dict(zip(header, cells, strict=True)))
        return rows
    raise CensusError("%s: missing table with header %r" % (section_name, header))


def _bullets(section):
    return [line.strip()[2:].strip() for line in section.splitlines()
            if line.strip().startswith("- ")]


def _link(row, table_name, problems):
    matches = LINK.findall(row["build"])
    if len(matches) != 1:
        problems.append("%s: row needs exactly one build link: %s"
                        % (table_name, row["build"]))
        return None
    label, url = matches[0]
    if not label.strip():
        problems.append("%s: empty link label for %s" % (table_name, url))
    return url


def _count(row, problems):
    try:
        return int(row["count"].replace(",", ""))
    except ValueError:
        problems.append("Counts: %r is not an integer" % row["count"])
        return None


def check_text(text):
    """Return every mismatch in one census document (empty means valid)."""
    problems = []
    sections = _sections(text)

    def table(section_name, header):
        try:
            return _table(sections, section_name, header)
        except CensusError as exc:
            problems.append(str(exc))
            return []

    qwen_rows = table("Qwen3.6-35B-A3B builds", QWEN_HEADER)
    other_rows = table("Other published builds", OTHER_HEADER)
    counts = table("Counts", COUNT_HEADER)

    summary = {}
    for row in counts:
        label = row["what"]
        if label in summary:
            problems.append("Counts: duplicate row %r" % label)
        value = _count(row, problems)
        if value is not None:
            summary[label] = value

    expected_counts = (
        ("Qwen3.6-35B-A3B build rows", EXPECTED_QWEN_ROWS),
        ("other published-build rows", EXPECTED_OTHER_ROWS),
        ("total counted build rows", EXPECTED_TOTAL_ROWS),
        ("expert slots across the six other-build rows", EXPECTED_OTHER_SLOTS),
        ("unread leads (not counted)", EXPECTED_UNREAD),
        ("checked and not counted", EXPECTED_CHECKED),
    )
    for label, expected in expected_counts:
        if label not in summary:
            problems.append("Counts: missing %r" % label)
        elif summary[label] != expected:
            problems.append("Counts: %s is %d, expected %d"
                            % (label, summary[label], expected))

    if len(qwen_rows) != EXPECTED_QWEN_ROWS:
        problems.append("Qwen3.6 table: %d rows, expected %d"
                        % (len(qwen_rows), EXPECTED_QWEN_ROWS))
    if len(other_rows) != EXPECTED_OTHER_ROWS:
        problems.append("Other table: %d rows, expected %d"
                        % (len(other_rows), EXPECTED_OTHER_ROWS))
    if len(qwen_rows) + len(other_rows) != EXPECTED_TOTAL_ROWS:
        problems.append("Census: %d total rows, expected %d"
                        % (len(qwen_rows) + len(other_rows), EXPECTED_TOTAL_ROWS))

    links = []
    slot_total = 0
    for row in qwen_rows:
        url = _link(row, "Qwen3.6 table", problems)
        if url:
            links.append(url)
    for row in other_rows:
        url = _link(row, "Other table", problems)
        if not url:
            continue
        links.append(url)
        scope = row["base and ternary scope"]
        matches = SLOT.findall(scope)
        if len(matches) != 1:
            problems.append("Other table: %s needs one 'N expert slots (A × B)' product"
                            % url)
            continue
        slots, layers, per_layer = matches[0]
        slots = int(slots.replace(",", ""))
        if slots != int(layers) * int(per_layer):
            problems.append("Other table: %s says %d slots, but %s × %s = %d"
                            % (url, slots, layers, per_layer,
                               int(layers) * int(per_layer)))
        slot_total += slots

    if slot_total != EXPECTED_OTHER_SLOTS:
        problems.append("Other table: slot total is %d, expected %d"
                        % (slot_total, EXPECTED_OTHER_SLOTS))
    if len(links) != len(set(links)):
        problems.append("Census: duplicate build links")
    if len(links) != EXPECTED_TOTAL_ROWS:
        problems.append("Census: %d linked rows, expected %d"
                        % (len(links), EXPECTED_TOTAL_ROWS))

    for row in other_rows:
        url = (LINK.findall(row["build"]) or [("", "")])[0][1]
        scope = row["base and ternary scope"]
        if "boro64" in url:
            match = NOTA_BLOCKS.search(scope)
            if not match:
                problems.append("Nota row: block arithmetic is missing")
            else:
                total, moe, mamba, attention = (int(part) for part in match.groups())
                if total != moe + mamba + attention:
                    problems.append(
                        "Nota row: %d blocks, but %d + %d + %d = %d"
                        % (total, moe, mamba, attention, moe + mamba + attention)
                    )
        if "kabcpc" in url:
            match = FLINT_LAYERS.search(scope)
            if not match:
                problems.append("Flint row: layer arithmetic is missing")
            else:
                total, dense, moe = (int(part) for part in match.groups())
                if total != dense + moe:
                    problems.append("Flint row: %d layers, but %d + %d = %d"
                                    % (total, dense, moe, dense + moe))
            match = FLINT_TENSORS.search(scope)
            if not match:
                problems.append("Flint row: fused-tensor arithmetic is missing")
            else:
                tensors, layers = (int(part) for part in match.groups())
                if tensors != layers * 3:
                    problems.append("Flint row: %d fused tensors, but %d × 3 = %d"
                                    % (tensors, layers, layers * 3))
        if "micro-terse" in url:
            match = MICRO_LAYERS.search(scope)
            if not match:
                problems.append("micro-terse row: MoE-layer arithmetic is missing")
            else:
                layers, moe_layers = (int(part) for part in match.groups())
                if moe_layers != layers // 2:
                    problems.append("micro-terse row: %d layers, but %d odd MoE layers"
                                    % (layers, moe_layers))

    unread = _bullets(sections.get("Unread leads", ""))
    checked = _bullets(sections.get("Checked and not counted", ""))
    if len(unread) != EXPECTED_UNREAD:
        problems.append("Unread leads: %d bullets, expected %d"
                        % (len(unread), EXPECTED_UNREAD))
    if len(checked) != EXPECTED_CHECKED:
        problems.append("Checked and not counted: %d bullets, expected %d"
                        % (len(checked), EXPECTED_CHECKED))
    if "Granite 4.1 30B" not in sections.get("Not an MoE", ""):
        problems.append("Not an MoE: Granite 4.1 30B is missing")

    return problems


def check_doc(path=None):
    target = Path(path) if path is not None else DEFAULT_DOC
    try:
        return check_text(target.read_text(encoding="utf-8"))
    except OSError as exc:
        return ["cannot read %s: %s" % (target, exc)]


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    target = Path(args[0]) if args else DEFAULT_DOC
    problems = check_doc(target)
    for problem in problems:
        print("FAIL %s" % problem)
    if problems:
        return 1
    print("OK: %d Qwen3.6 rows + %d other rows = %d counted rows; %d expert slots; "
          "%d unread leads; %d checked and not counted"
          % (EXPECTED_QWEN_ROWS, EXPECTED_OTHER_ROWS, EXPECTED_TOTAL_ROWS,
             EXPECTED_OTHER_SLOTS, EXPECTED_UNREAD, EXPECTED_CHECKED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
