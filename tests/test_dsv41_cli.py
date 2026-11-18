"""The standalone DSV4.1 CLIs reject malformed comma-separated options at argparse time, before a
model, a native library or a server socket is touched."""
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
def test_dsv41_rejects_bad_values_before_loading(argv, expected, capsys):
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
