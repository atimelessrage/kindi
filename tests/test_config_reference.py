# SPDX-License-Identifier: AGPL-3.0-only
"""The generated configuration reference must be current and complete.

`docs/CONFIGURATION-REFERENCE.md` is rendered from `kindi/config.py` and from every `KINDI_*`
read under `kindi/`. These tests fail when a key or switch is added, renamed or removed without
regenerating the file (`python -m kindi.config_reference`), which is what keeps a switch like
`KINDI_TC` from going missing again.
"""
import re
from pathlib import Path

from kindi import config, config_reference


def reference_text():
    return Path(config_reference.REFERENCE).read_text(encoding="utf-8").replace("\r\n", "\n")


def test_reference_is_current():
    assert reference_text() == config_reference.render(), \
        "docs/CONFIGURATION-REFERENCE.md is stale; run `python -m kindi.config_reference`"


def test_every_documented_switch_is_read_by_the_code():
    """The generated reference can only see a *deficit* (a read with no row). The prose guide can
    also drift the other way: a row survives after its read is deleted, and then a documented knob
    silently does nothing. `KINDI_STREAM_LOAD` was exactly that until 2026-11-14."""
    prose = Path(config_reference.ROOT, "docs", "CONFIGURATION.md").read_text(encoding="utf-8")
    named = set(re.findall(r"\b(KINDI_[A-Z0-9_]+)\b", prose))
    read = set(config_reference.env_vars())
    keys = {"KINDI_" + k.upper() for k in config.DEFAULTS}
    surplus = sorted(named - read - keys)
    assert not surplus, ("documented in docs/CONFIGURATION.md but never read by the code (or the "
                         "row is stale - drop it): %s" % surplus)


def test_every_config_key_is_listed():
    text = reference_text()
    missing = [k for k in config.DEFAULTS if "| `%s` |" % k not in text]
    assert not missing, "config keys missing from the reference: %s" % missing


def test_the_config_table_has_one_row_per_default():
    # one row per DEFAULTS key, in the section before the environment table
    config_section = reference_text().split("## Environment-only variables", 1)[0]
    rows = [line for line in config_section.splitlines() if line.startswith("| `")]
    assert len(rows) == len(config.DEFAULTS)


def test_every_environment_variable_is_listed():
    text = reference_text()
    keys = {"KINDI_" + k.upper() for k in config.DEFAULTS}     # covered by the config table
    missing = [name for name in config_reference.env_vars() if name not in keys and
               "| `%s` |" % name not in text]
    assert not missing, "environment variables missing from the reference: %s" % sorted(missing)


def test_the_tensor_core_switch_is_documented():
    # KINDI_TC has no config key (it is read in kindi/spec.py and kindi/tp.py); it must still show up
    assert "| `KINDI_TC` |" in reference_text()
    assert "KINDI_TC" in config_reference.env_vars()


def test_the_scan_reads_environment_gets_and_subscripts(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import os\n"
        "A = os.environ.get('KINDI_FROM_GET', '7')\n"
        "B = os.getenv('KINDI_FROM_GETENV', '9')\n"
        "C = os.environ['KINDI_FROM_SUBSCRIPT']\n"
        "D = os.environ.setdefault('KINDI_FROM_SETDEFAULT', '3')\n"
        "def f():\n"
        "    dynamic = os.environ.get('KINDI_' + 'X')\n",
        encoding="utf-8")
    found = config_reference.env_vars([str(sample)])
    assert found["KINDI_FROM_GET"]["defaults"] == ["7"]
    assert found["KINDI_FROM_GET"]["sites"] == [("sample.py", 2)]
    assert found["KINDI_FROM_GETENV"]["defaults"] == ["9"]
    assert found["KINDI_FROM_GETENV"]["sites"] == [("sample.py", 3)]
    assert "KINDI_FROM_SUBSCRIPT" in found
    assert found["KINDI_FROM_SETDEFAULT"]["defaults"] == ["3"]
    assert found["KINDI_FROM_SETDEFAULT"]["sites"] == [("sample.py", 5)]
    assert all(name != "KINDI_" for name in found)      # a built name is not a literal read


def test_the_scan_covers_subpackages():
    """`KINDI_BACKEND_MODULES` is read in `kindi/backends/`; a top-level-only
    scan silently dropped it from the reference (the N2/N3 class again)."""
    found = config_reference.env_vars()
    assert "KINDI_BACKEND_MODULES" in found, "subpackage reads are not scanned"
    sites = [s for s, _ in found["KINDI_BACKEND_MODULES"]["sites"]]
    assert "backends/registry.py" in sites
    assert "| `KINDI_BACKEND_MODULES` |" in reference_text()


def test_a_config_key_read_directly_is_listed_with_its_direct_sites():
    """`KINDI_MTP_HEAD` is the config key `mtp_head` *and* a direct read: `spec.py` loads that
    file as the Bonsai MTP head. The generator used to drop every name that matched a config
    key, so the Bonsai path was documented by hand in `docs/MTP.md` and missing from the
    generated reference (found by the knob-liveness gate, 2026-11-24)."""
    text = reference_text()
    assert "| `KINDI_MTP_HEAD` |" in text, "a direct read that shadows a config key is missing"
    row = next(line for line in text.splitlines() if line.startswith("| `KINDI_MTP_HEAD` |"))
    assert "kindi/mtp_heads.py:" in row and "more" in row, row
    # the Nanbeige config key is still described in the config table
    assert "| `mtp_head` | string | (empty) |" in text
