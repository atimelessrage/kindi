# SPDX-License-Identifier: AGPL-3.0-only
"""The logical-to-stored block map for the two published looped GGUFs (REQ-0031).

The map is measured, not assumed: SenseNova stores 33 of its 57 logical blocks (0-27 and
52-56; 28-51 are the second pass over 4-27), and Inkling stores 24 of 48 (0-23; 18-41 are
the second and third passes over 6-17). A loader that walks `range(block_count)` would ask
for tensors that are not in the file, so the boundary is pinned here.
"""
import json

import pytest

from kindi import convert
from kindi.looped import (
    SPECS,
    execution_plan,
    format_ranges,
    layout,
    mismatch,
    refusal,
    render,
    report_dict,
    spec_from_gguf,
    stored_index,
    stored_indices,
)


class FakeGGUF(object):
    def __init__(self, kv, tensors=()):
        self.kv = kv
        self.tensors = list(tensors)


def test_the_two_published_families_are_recognized_from_their_arch_key():
    assert set(SPECS) == {"sensenova_loop", "inkling2"}
    spec = spec_from_gguf(FakeGGUF({"general.architecture": "sensenova_loop"}))
    assert (spec.name, spec.logical, spec.stored) == ("SenseNova-Loop-9B", 57, 33)
    spec = spec_from_gguf(FakeGGUF({"general.architecture": "inkling2"}))
    assert (spec.name, spec.logical, spec.stored) == ("Inkling 2 Mini", 48, 24)


def test_an_unknown_arch_or_a_changed_block_count_is_not_described_with_this_map():
    assert spec_from_gguf(FakeGGUF({"general.architecture": "llama"})) is None
    assert spec_from_gguf(FakeGGUF({})) is None
    changed = FakeGGUF({"general.architecture": "inkling2", "inkling2.block_count": 24})
    assert spec_from_gguf(changed) is None
    assert "block_count 24" in mismatch(changed) and "48 logical layers" in mismatch(changed)
    assert "block_count 24" in refusal(changed)


def test_sensenova_replays_4_to_27_and_then_continues_at_52():
    spec = SPECS["sensenova_loop"]
    assert stored_indices(spec) == list(range(0, 28)) + list(range(52, 57))
    assert stored_index(spec, 3) == 3
    assert stored_index(spec, 4) == 4
    assert stored_index(spec, 27) == 27
    assert stored_index(spec, 28) == 4          # second pass
    assert stored_index(spec, 51) == 27
    assert stored_index(spec, 52) == 52         # post-loop, stored under its logical index
    assert stored_index(spec, 56) == 56         # nextn layer
    assert stored_index(spec, 57) is None


def test_inkling_runs_6_to_17_three_times_and_continues_at_18():
    spec = SPECS["inkling2"]
    assert stored_indices(spec) == list(range(24))
    assert stored_index(spec, 5) == 5
    assert stored_index(spec, 6) == 6
    assert stored_index(spec, 17) == 17
    assert stored_index(spec, 18) == 6          # second pass
    assert stored_index(spec, 29) == 17
    assert stored_index(spec, 30) == 6          # third pass
    assert stored_index(spec, 41) == 17
    assert stored_index(spec, 42) == 18         # post-loop, physical index
    assert stored_index(spec, 47) == 23
    assert stored_index(spec, 48) is None


def test_execution_plan_marks_the_replayed_passes():
    plan = execution_plan(SPECS["sensenova_loop"])
    assert [step.logical for step in plan] == list(range(57))
    assert (plan[4].phase, plan[4].pass_index, plan[4].stored) == ("loop", 1, 4)
    assert (plan[28].phase, plan[28].pass_index, plan[28].stored) == ("loop", 2, 4)
    assert (plan[51].phase, plan[51].pass_index, plan[51].stored) == ("loop", 2, 27)
    assert (plan[52].phase, plan[52].pass_index, plan[52].stored) == ("post", 0, 52)

    inkling = execution_plan(SPECS["inkling2"])
    assert (inkling[18].pass_index, inkling[18].stored) == (2, 6)
    assert (inkling[30].pass_index, inkling[30].stored) == (3, 6)
    assert (inkling[42].phase, inkling[42].stored) == ("post", 18)
    assert format_ranges([0, 1, 2, 4, 5, 7]) == "0-2,4-5,7"


def test_layout_reports_missing_and_unexpected_block_tensors():
    spec = SPECS["sensenova_loop"]
    tensors = [("blk.%d.attn_norm.weight" % i, [], 0, 0)
               for i in stored_indices(spec) if i != 5]
    tensors.append(("blk.99.unexpected", [], 0, 0))
    report = layout(FakeGGUF({"general.architecture": "sensenova_loop"}, tensors))
    assert report.missing == (5,)
    assert report.unexpected == (99,)
    assert set(report.present) == (set(stored_indices(spec)) - {5}) | {99}
    payload = report_dict(report)
    assert payload["arch"] == "sensenova_loop" and len(payload["plan"]) == 57
    text = "\n".join(render(report))
    assert "missing stored blocks: 5" in text
    assert "unexpected block tensors: 99" in text
    assert "map check" in text


def test_loop_info_cli_prints_the_plan_and_coverage(monkeypatch, capsys, tmp_path):
    from kindi import __main__ as cli
    from kindi import gguf as gguf_module

    spec = SPECS["inkling2"]
    tensors = [("blk.%d.attn_norm.weight" % i, [], 0, 0) for i in stored_indices(spec)]
    monkeypatch.setattr(gguf_module, "GGUF",
                        lambda _path: FakeGGUF({"general.architecture": "inkling2"}, tensors))
    conf = tmp_path / "kindi.conf"
    conf.write_text("[kindi]\n", encoding="utf-8")
    monkeypatch.setenv("KINDI_CONFIG", str(conf))

    assert cli.main(["loop-info", "model.gguf", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "Inkling 2 Mini"
    assert payload["missing_blocks"] == []
    assert len(payload["plan"]) == 48

    assert cli.main(["loop-info", "model.gguf"]) == 0
    text = capsys.readouterr().out
    assert "Inkling 2 Mini (inkling2)" in text
    assert "map check" in text

    tensors.pop(0)
    assert cli.main(["loop-info", "model.gguf"]) == 1
    assert "missing stored blocks: 0" in capsys.readouterr().out


@pytest.mark.parametrize("arch,name,counts", [
    ("sensenova_loop", "SenseNova-Loop-9B", "33 stored blocks serve 57"),
    ("inkling2", "Inkling 2 Mini", "24 stored blocks serve 48"),
])
def test_the_refusal_names_the_family_the_map_and_the_requirement(arch, name, counts):
    message = refusal(FakeGGUF({"general.architecture": arch}))
    assert name in message and counts in message
    assert "looped transformer" in message
    assert "REQ-0031" in message


@pytest.mark.parametrize("arch", ["sensenova_loop", "inkling2"])
def test_convert_refuses_a_looped_gguf_before_its_generic_arch_error(tmp_path, monkeypatch, arch):
    kv = {"general.architecture": arch, arch + ".block_count": 57 if arch == "sensenova_loop" else 48}
    monkeypatch.setattr(convert, "GGUF", lambda _path: FakeGGUF(kv))
    with pytest.raises(SystemExit) as excinfo:
        convert.convert("model.gguf", str(tmp_path))
    assert "looped transformer" in str(excinfo.value)
    assert "expected a qwen35" not in str(excinfo.value)
