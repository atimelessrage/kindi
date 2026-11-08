# SPDX-License-Identifier: AGPL-3.0-only
"""The generated configuration reference must be current and complete.

`docs/CONFIGURATION-REFERENCE.md` is rendered from `kindi/config.py` and from every `KINDI_*`
read under `kindi/`. These tests fail when a key or switch is added, renamed or removed without
regenerating the file (`python -m kindi.config_reference`), which is what keeps a switch like
`KINDI_TC` from going missing again.
"""
from pathlib import Path

from kindi import config, config_reference


def reference_text():
    return Path(config_reference.REFERENCE).read_text(encoding="utf-8").replace("\r\n", "\n")


def test_reference_is_current():
    assert reference_text() == config_reference.render(), \
        "docs/CONFIGURATION-REFERENCE.md is stale; run `python -m kindi.config_reference`"


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
        "B = os.environ.get('KINDI_NO_DEFAULT')\n"
        "C = os.environ['KINDI_FROM_SUBSCRIPT']\n"
        "def f():\n"
        "    dynamic = os.environ.get('KINDI_' + 'X')\n",
        encoding="utf-8")
    found = config_reference.env_vars([str(sample)])
    assert found["KINDI_FROM_GET"]["defaults"] == ["7"]
    assert found["KINDI_FROM_GET"]["sites"] == [("sample.py", 2)]
    assert found["KINDI_NO_DEFAULT"]["defaults"] == []
    assert "KINDI_FROM_SUBSCRIPT" in found
    assert all(name != "KINDI_" for name in found)      # a built name is not a literal read
