"""Config precedence: defaults < config file < environment < command-line overrides."""
import os

import pytest

from kindi import config

ENV_KEYS = ["KINDI_" + k.upper() for k in config.DEFAULTS] + ["KINDI_CONFIG"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """No KINDI_* leakage from the host, no ./kindi.conf, no ~/.kindi/kindi.conf, and a pack path that
    does not exist. The last one is a default lookup path: without moving `~`, a real config on the
    machine running the tests silently replaces the built-in defaults."""
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))          # POSIX expanduser
    monkeypatch.setenv("USERPROFILE", str(tmp_path))   # Windows expanduser
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_conf(tmp_path, body):
    p = tmp_path / "kindi.conf"
    p.write_text("[kindi]\n" + body, encoding="utf-8")
    return str(p)


def test_built_in_defaults():
    cfg = config.load()
    assert cfg["ctx"] == 24576
    assert cfg["temperature"] == 0.7
    assert cfg["head"] == "exact"
    assert cfg["source"] == "(built-in defaults)"
    assert isinstance(cfg["ctx"], int) and isinstance(cfg["temperature"], float)
    assert cfg["presence_penalty"] == 0.0 and isinstance(cfg["presence_penalty"], float)
    assert cfg["sampling_profile"] == "default"
    assert cfg["explicit"] == frozenset()


def test_file_beats_defaults(clean_env):
    path = write_conf(clean_env, "ctx = 4096\ntemperature = 0.25\nhead = fast\n")
    cfg = config.load(explicit=path)
    assert cfg["ctx"] == 4096
    assert cfg["temperature"] == 0.25
    assert cfg["head"] == "fast"
    assert cfg["source"] == path


def test_environment_beats_file(clean_env, monkeypatch):
    path = write_conf(clean_env, "ctx = 4096\n")
    monkeypatch.setenv("KINDI_CTX", "2048")
    assert config.load(explicit=path)["ctx"] == 2048


def test_command_line_beats_environment(clean_env, monkeypatch):
    path = write_conf(clean_env, "ctx = 4096\n")
    monkeypatch.setenv("KINDI_CTX", "2048")
    assert config.load(explicit=path, overrides={"ctx": 1024})["ctx"] == 1024


def test_environment_is_coerced_to_the_key_type(monkeypatch):
    monkeypatch.setenv("KINDI_CTX", "8192")
    monkeypatch.setenv("KINDI_TEMPERATURE", "0.0")
    cfg = config.load()
    assert cfg["ctx"] == 8192 and isinstance(cfg["ctx"], int)
    assert cfg["temperature"] == 0.0 and isinstance(cfg["temperature"], float)


def test_path_keys_are_expanded_and_absolute(clean_env):
    path = write_conf(clean_env, "pack = ~/packs/test\n")
    cfg = config.load(explicit=path)
    assert cfg["pack"] == os.path.abspath(os.path.expanduser("~/packs/test"))


def test_unknown_key_is_rejected(clean_env):
    path = write_conf(clean_env, "nonsense = 1\n")
    with pytest.raises(SystemExit, match="unknown config key"):
        config.load(explicit=path)


def test_missing_explicit_file_is_rejected(clean_env):
    with pytest.raises(SystemExit, match="config file not found"):
        config.load(explicit=str(clean_env / "does-not-exist.conf"))


def test_explicit_keys_are_tracked(clean_env, monkeypatch):
    path = write_conf(clean_env, "temperature = 0.25\n")
    monkeypatch.setenv("KINDI_TOP_P", "0.5")
    cfg = config.load(explicit=path, overrides={"ctx": 1024})
    assert cfg["explicit"] == frozenset({"temperature", "top_p", "ctx"})


def test_invalid_sampling_profile_is_rejected(clean_env):
    path = write_conf(clean_env, "sampling_profile = fast\n")
    with pytest.raises(SystemExit, match="sampling_profile must be default or publisher"):
        config.load(explicit=path)
