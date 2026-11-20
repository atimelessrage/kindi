"""The example config must stay a complete, copyable reference.

A missing key is easy to introduce and hard to notice: the example is what users copy. This test pins the
example to kindi.config.DEFAULTS, so a new default cannot ship without a line in kindi.conf.example.
"""
import re
from pathlib import Path

from kindi import config


def test_example_config_lists_every_default_key():
    root = Path(__file__).resolve().parents[1]
    text = (root / "kindi.conf.example").read_text(encoding="utf-8")
    for key in config.DEFAULTS:
        assert re.search(r"^\s*#?\s*%s\s*=" % re.escape(key), text, re.MULTILINE), \
            "kindi.conf.example is missing the default key %r" % key


def test_configuration_doc_lists_every_default_key():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    missing = [key for key in config.DEFAULTS if "`%s`" % key not in text]
    assert not missing, "docs/CONFIGURATION.md is missing the keys: %s" % missing
