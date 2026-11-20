# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""Dependency-free smoke coverage for the Bridge command and state layer.

The full Bridge UI/PTY/RPC suite lives outside this source distribution. These tests keep the
documented `python -m unittest discover -s tests -v` command honest and cover the pure helpers
and SQLite schema that the installer's `--with-bridge` path relies on.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kindi_bridge import agents, cli  # noqa: E402  (after the source-root setup)
from kindi_bridge.store import Store  # noqa: E402


class DurationTests(unittest.TestCase):
    def test_duration_units_and_fraction(self):
        self.assertEqual(cli.duration("20m"), 1200)
        self.assertEqual(cli.duration("1.5h"), 5400)
        self.assertEqual(cli.duration("2d"), 172800)

    def test_duration_refuses_an_empty_or_nonpositive_value(self):
        for value in ("bad", "0s", "-1m"):
            with self.assertRaises(ValueError):
                cli.duration(value)


class ParseTimeTests(unittest.TestCase):
    def test_now_and_in_duration(self):
        now = 1_700_000_000.0
        self.assertEqual(cli.parse_time("now", now=now), now)
        self.assertEqual(cli.parse_time("in 20m", now=now), now + 1200)

    def test_clock_time_moves_to_the_next_day_when_past(self):
        now = time.mktime((2026, 11, 18, 23, 30, 0, 0, 0, -1))
        self.assertGreater(cli.parse_time("08:00", now=now), now)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "state.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_settings_round_trip(self):
        self.assertIsNone(self.store.setting("endpoint"))
        self.store.setting("endpoint", "http://127.0.0.1:8093/health")
        self.assertEqual(self.store.setting("endpoint"), "http://127.0.0.1:8093/health")

    def test_agent_add_and_roster(self):
        self.store.add_agent("tester", str(self.root), ["pi", "--mode", "rpc"])
        row = self.store.agent("tester")
        self.assertEqual(row["name"], "tester")
        self.assertEqual(row["state"], "offline")
        self.assertEqual(agents.status(row), "Offline")
        roster = agents.roster(self.store)
        self.assertEqual([a["name"] for a in roster], ["tester"])
        self.assertEqual(roster[0]["mission"], "No mission yet")

    def test_agent_add_refuses_a_missing_directory(self):
        with self.assertRaises(ValueError):
            self.store.add_agent("missing", str(self.root / "not-here"), ["pi"])

    def test_parser_exposes_the_documented_jobs_command(self):
        parsed = cli.parser().parse_args(["jobs", "add", "read the release notes"])
        self.assertEqual(parsed.command, "jobs")
        self.assertEqual(parsed.action, "add")
        self.assertEqual(parsed.prompt, "read the release notes")


if __name__ == "__main__":
    unittest.main()
