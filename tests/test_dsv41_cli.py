"""The standalone DSV4.1 CLIs reject malformed, missing or unreadable inputs at argparse time,
before a model, a native library or a server socket is touched."""
import argparse

import pytest

from kindi import dsv41, dsv41_server


def test_context_sizes_are_positive_comma_separated_integers():
    assert dsv41._ctx_arg("2048, 8192") == [2048, 8192]
    for bad in ("8192,fast", "8192,", "", "0", "-1", "2048, 0"):
        with pytest.raises(argparse.ArgumentTypeError,
                           match=r"comma-separated positive integers.*2048,8192"):
            dsv41._ctx_arg(bad)


def test_calibration_is_two_finite_comma_separated_numbers():
    assert dsv41._cal_arg("1, 0.5") == (1.0, 0.5)
    for bad in ("1", "1,0,2", "1,fast", "nan,0", "1,inf", "1,"):
        with pytest.raises(argparse.ArgumentTypeError,
                           match=r"two comma-separated finite numbers: a,b"):
            dsv41._cal_arg(bad)


def test_server_context_is_a_positive_integer():
    assert dsv41._positive_int("32768") == 32768
    for bad in ("0", "-1", "fast", "8192,4096"):
        with pytest.raises(argparse.ArgumentTypeError, match=r"positive integer"):
            dsv41._positive_int(bad)


def test_prompt_range_is_two_non_negative_ordered_integers():
    assert dsv41._range_arg("0-3") == (0, 3)
    assert dsv41._range_arg(" 10 - 20 ") == (10, 20)
    assert dsv41._range_arg("7-7") == (7, 7)
    for bad in ("1", "1-", "-1", "1-2-3", "1-fast", "3-1", "", "1,2"):
        with pytest.raises(argparse.ArgumentTypeError,
                           match=r"non-negative integers a-b with a <= b.*0-9"):
            dsv41._range_arg(bad)


@pytest.mark.parametrize("argv, expected", [
    (["bench", "model.gguf", "--ctx", "8192,fast"], "comma-separated positive integers"),
    (["specbench", "model.gguf", "--auto", "1,fast"], "two comma-separated finite numbers"),
    (["generate", "model.gguf", "--prompts", "p.jsonl", "--range", "3-1"], "non-negative integers"),
])
def test_dsv41_rejects_bad_values_before_loading(tmp_path, argv, expected, capsys):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"id": 0, "text": "x"}\n', encoding="utf-8")
    argv = [str(prompts) if value == "p.jsonl" else value for value in argv]
    with pytest.raises(SystemExit) as exc:
        dsv41.main(argv)
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("argv, expected", [
    (["model.gguf", "--ctx", "0"], "positive integer"),
    (["model.gguf", "--cal", "1,fast"], "two comma-separated finite numbers"),
])
def test_dsv41_server_rejects_bad_values_before_loading(argv, expected, capsys):
    with pytest.raises(SystemExit) as exc:
        dsv41_server.main(argv)
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


def test_file_and_number_validators_reject_bad_values(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id": 0, "text": "x"}\n', encoding="utf-8")
    assert dsv41._readable_file(str(prompts)) == str(prompts)
    for bad in (str(tmp_path / "missing.jsonl"), str(tmp_path)):
        with pytest.raises(argparse.ArgumentTypeError, match="existing readable file"):
            dsv41._readable_file(bad)

    ids = tmp_path / "ids.txt"
    ids.write_text("1 2 129279\n", encoding="utf-8")
    assert dsv41._ids_arg(str(ids)) == [1, 2, 129279]
    bad_ids = tmp_path / "bad.ids"
    bad_ids.write_text("1 nope\n", encoding="utf-8")
    empty_ids = tmp_path / "empty.ids"
    empty_ids.write_text("", encoding="utf-8")
    out_of_range = tmp_path / "range.ids"
    out_of_range.write_text(str(dsv41.VOCAB), encoding="utf-8")
    for bad in (str(bad_ids), str(empty_ids), str(out_of_range), str(tmp_path / "missing.ids")):
        with pytest.raises(argparse.ArgumentTypeError, match="token ids"):
            dsv41._ids_arg(bad)

    out = tmp_path / "out.jsonl"
    assert dsv41._out_arg(str(out)) == str(out)
    for bad in (str(tmp_path / "missing" / "out.jsonl"), str(tmp_path), ""):
        with pytest.raises(argparse.ArgumentTypeError, match="existing writable directory"):
            dsv41._out_arg(bad)

    dspark = tmp_path / "dspark"
    dspark.mkdir()
    for name in ("kindi-dspark.bin", "kindi-dspark.bin.json", "config.json"):
        (dspark / name).write_bytes(b"{}")
    assert dsv41._dspark_dir(str(dspark)) == str(dspark)
    (dspark / "config.json").unlink()
    with pytest.raises(argparse.ArgumentTypeError, match="DSpark directory"):
        dsv41._dspark_dir(str(dspark))


def test_number_validators_accept_their_valid_sets():
    assert dsv41._free_gb_arg("12") == 12.0
    assert dsv41._free_gb_arg("0.5") == 0.5
    assert dsv41._free_gb_arg("0") == 0.0
    for bad in ("-1", "nan", "inf", "fast"):
        with pytest.raises(argparse.ArgumentTypeError, match="finite non-negative number of GB"):
            dsv41._free_gb_arg(bad)

    assert dsv41._positive_count_arg("1") == 1
    assert dsv41._positive_count_arg("64") == 64
    for bad in ("0", "-1", "1.5", "fast"):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            dsv41._positive_count_arg(bad)

    assert dsv41._non_negative_int_arg("0") == 0
    assert dsv41._non_negative_int_arg("64") == 64
    for bad in ("-1", "1.5", "fast"):
        with pytest.raises(argparse.ArgumentTypeError, match="non-negative integer"):
            dsv41._non_negative_int_arg(bad)

    assert dsv41._finite_float_arg("0") == 0.0
    assert dsv41._finite_float_arg("-1.5") == -1.5
    for bad in ("nan", "inf", "-inf", "fast"):
        with pytest.raises(argparse.ArgumentTypeError, match="finite number"):
            dsv41._finite_float_arg(bad)


def test_dsv41_rejects_late_inputs_before_loading(tmp_path, capsys):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id": 0, "text": "x"}\n', encoding="utf-8")
    bad_ids = tmp_path / "bad.ids"
    bad_ids.write_text("1 nope\n", encoding="utf-8")
    cases = [
        (["generate", "model.gguf", "--prompts", str(tmp_path / "missing.jsonl")],
         "existing readable file"),
        (["generate", "model.gguf", "--ids", str(bad_ids)], "token ids"),
        (["generate", "model.gguf", "--ids", str(tmp_path / "missing.ids")], "token ids"),
        (["generate", "model.gguf", "--out", str(tmp_path / "missing" / "out.jsonl")],
         "existing writable directory"),
        (["generate", "model.gguf", "--dump", str(tmp_path / "missing" / "dump.bin")],
         "existing writable directory"),
        (["numa", "model.gguf", "--free1", "nan"], "finite non-negative number of GB"),
        (["generate", "model.gguf", "--prompts", str(prompts), "-n", "-1"],
         "non-negative integer"),
        (["bench", "model.gguf", "--tmax", "0"], "positive integer"),
        (["bench", "model.gguf", "--gen", "0"], "positive integer"),
        (["specbench", "model.gguf", "--prompts", str(prompts), "--spec", "0"],
         "positive integer"),
        (["specbench", "model.gguf", "--prompts", str(prompts), "--conf", "nan"],
         "finite number"),
        (["specbench", "model.gguf", "--prompts", str(prompts),
          "--dspark", str(tmp_path / "missing-dspark")], "DSpark directory"),
        (["teacher", "model.gguf", "--prompts", str(prompts),
          "--ref", str(tmp_path / "missing.ref")], "existing readable file"),
        (["tokens", "model.gguf"], "--prompts FILE.jsonl is required"),
        (["cachetest", "model.gguf"], "--prompts FILE.jsonl is required"),
        (["specbench", "model.gguf"], "--prompts FILE.jsonl is required"),
        (["teacher", "model.gguf"], "--prompts FILE.jsonl is required"),
        (["teacher", "model.gguf", "--prompts", str(prompts)],
         "--ref FILE is required"),
    ]
    for argv, expected in cases:
        with pytest.raises(SystemExit) as exc:
            dsv41.main(argv)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert expected in err, (argv, err)


def test_verify_rejects_missing_gate_files_before_loading(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dsv41, "GATE_DIR", str(tmp_path / "missing-gate"))
    with pytest.raises(SystemExit) as exc:
        dsv41.main(["verify", "model.gguf"])
    assert exc.value.code == 2
    assert "gate file not found" in capsys.readouterr().err
