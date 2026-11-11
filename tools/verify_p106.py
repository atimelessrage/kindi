#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Record the P106-100 single-card, server/cache and two-card 27B checks in one run.

The default run requires exactly two visible NVIDIA P106-100 cards.  It records:

  * `kindi verify` for the 2B, 4B and 9B packs on one P106;
  * a real server on one P106, including a cold chat, a saved prompt opening, an
    in-process cache hit, a restart, a cache hit after the restart, and an invalid
    request that must not clear the cache;
  * `kindi verify` for the 27B pack with KINDI_CARDS=2 and the legacy two-card
    profile, requiring log evidence that the plan engine used two cards.

Every child process gets a clean environment: inherited KINDI_* switches are
removed, CUDA_VISIBLE_DEVICES names the P106 indices, and the exact settings are
written to the result.  Raw logs, results.json and RESULTS.md go to
bench-logs/p106-verify by default.

This script records evidence; it does not edit docs/CARDS.md or make a support
claim.  A full run refuses to start when fewer or more than two P106-100 cards
are visible, so a result is never silently a one-card run.  Use --single-only
when only one P106 is installed.

Examples:
  python tools/verify_p106.py --pack-dir ~/.kindi/packs
  python tools/verify_p106.py --single-only
  python tools/verify_p106.py --visible-devices 1,2 --out /tmp/p106-run
  python tools/verify_p106.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from verify_matrix import parse_verify_output  # noqa: E402

MODELS = ("2b", "4b", "9b")
# The current installer uses ~/.kindi/packs/bonsai-*; older P106 instructions and
# the container image use /data/pack, pack4b, pack9b and pack27b.
PACK_CANDIDATES = {
    "2b": ("bonsai-2b", "pack"),
    "4b": ("bonsai-4b", "pack4b"),
    "9b": ("bonsai-9b", "pack9b"),
    "27b": ("bonsai-27b", "pack27b"),
}
DUAL_MODEL = "27b"
CARD_FIELDS = (
    "index", "name", "memory.total", "memory.free", "compute_cap",
    "pcie.link.gen.current", "pcie.link.width.current", "driver_version",
)
CARD_QUERY = "nvidia-smi --query-gpu=" + ",".join(CARD_FIELDS) + " --format=csv,noheader,nounits"
SAVED_RE = re.compile(r"saved an opening of (\d+) tokens")
LOADED_RE = re.compile(r"loaded a saved opening of (\d+) tokens")
PLAN_RE = re.compile(r"plan engine ready:\s*(\d+)\s+cards?")
CACHE_METRIC_RE = re.compile(r"^kindi_cached_prompt_tokens_total\s+(\d+)\s*$", re.M)


def _display_path(path: Path) -> str:
    try:
        return "~/" + path.expanduser().resolve().relative_to(Path.home().resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def _int(value: str):
    try:
        return int(value.strip())
    except (AttributeError, ValueError):
        try:
            return int(float(value.strip()))
        except (AttributeError, ValueError):
            return None


def parse_card_rows(text: str) -> list[dict]:
    """Parse the nounits CSV from nvidia-smi without serial numbers or UUIDs."""
    cards = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(CARD_FIELDS):
            continue
        raw = dict(zip(CARD_FIELDS, parts))
        index = _int(raw["index"])
        if index is None:
            continue
        cards.append({
            "index": index,
            "name": raw["name"],
            "memory_total_mib": _int(raw["memory.total"]),
            "memory_free_mib": _int(raw["memory.free"]),
            "compute_cap": raw["compute_cap"],
            "pcie_gen": raw["pcie.link.gen.current"],
            "pcie_width": raw["pcie.link.width.current"],
            "driver_version": raw["driver_version"],
        })
    return cards


def query_cards() -> tuple[list[dict], str]:
    try:
        proc = subprocess.run(
            shlex.split(CARD_QUERY), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], "cannot run nvidia-smi: %s" % exc
    if proc.returncode != 0:
        return [], "nvidia-smi exited %d: %s" % (proc.returncode, (proc.stderr or proc.stdout).strip()[:300])
    cards = parse_card_rows(proc.stdout)
    if not cards:
        return [], "nvidia-smi returned no GPU rows"
    return cards, ""


def is_p106(card: dict) -> bool:
    return "P106-100" in card.get("name", "").upper()


def pack_for(pack_dir: Path, model: str) -> Path:
    """Prefer the current bonsai-* name, then the legacy P106/container name."""
    for name in PACK_CANDIDATES[model]:
        candidate = pack_dir / name
        if (candidate / "kindi_pack.json").is_file():
            return candidate
    return pack_dir / PACK_CANDIDATES[model][0]


def select_cards(cards: list[dict], visible: str | None, single_only: bool) -> tuple[list[dict], str]:
    if visible:
        by_index = {card["index"]: card for card in cards}
        wanted = []
        for part in visible.split(","):
            part = part.strip()
            if not part:
                continue
            index = _int(part)
            if index is None or index not in by_index:
                return [], "CUDA_VISIBLE_DEVICES index %r is not in the nvidia-smi table" % part
            wanted.append(by_index[index])
        if not wanted:
            return [], "--visible-devices is empty"
    else:
        wanted = [card for card in cards if is_p106(card)]

    if not wanted:
        return [], "no NVIDIA P106-100 found; this runner is for a P106 host"
    if len({card["index"] for card in wanted}) != len(wanted):
        return [], "CUDA_VISIBLE_DEVICES names a card twice"
    for card in wanted:
        if not is_p106(card):
            return [], "selected card %s is not a P106-100" % card["name"]

    if single_only:
        if len(wanted) > 2:
            return [], "single-only run selects %d cards; use --visible-devices to choose one" % len(wanted)
        return wanted[:1], ""
    if len(wanted) != 2:
        return [], ("the full record needs exactly two P106-100 cards; found %d. "
                    "Use --single-only for one card" % len(wanted))
    return wanted, ""


def _clean_env(visible: list[dict], extra: dict[str, str], base: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if base is None else base
    env = {key: value for key, value in source.items() if not key.startswith("KINDI_")}
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(card["index"]) for card in visible)
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def single_env(cards: list[dict], ctx: int, base: dict[str, str] | None = None) -> dict[str, str]:
    return _clean_env(cards, {
        "KINDI_DEVICE": "gpu",
        "KINDI_HEAD": "exact",
        "KINDI_KV": "q8_0",
        "KINDI_SPEC": "0",
        "KINDI_CARDS": "1",
        "KINDI_TP": "0",
        "KINDI_TC": "0",
        "KINDI_FD": "0",
        "KINDI_ATT_TC": "0",
        "KINDI_CTX": str(ctx),
        "KINDI_BLOCKING_SYNC": "0",
        "KINDI_CHUNK_Q8": "1",
    }, base)


def dual_env(cards: list[dict], ctx: int, base: dict[str, str] | None = None) -> dict[str, str]:
    return _clean_env(cards, {
        "KINDI_DEVICE": "gpu",
        "KINDI_HEAD": "exact",
        "KINDI_KV": "q8_0",
        "KINDI_SPEC": "0",
        "KINDI_CARDS": "2",
        "KINDI_TC": "0",
        "KINDI_FD": "0",
        "KINDI_ATT_TC": "0",
        "KINDI_CTX": str(ctx),
        "KINDI_BLOCKING_SYNC": "0",
        "KINDI_CHUNK_Q8": "1",
        "KINDI_CHECKPOINT_BUDGET_MB": "4096",
        "KINDI_REPEAT_STOP": "1",
    }, base)


def server_env(cards: list[dict], ctx: int) -> dict[str, str]:
    env = single_env(cards, ctx)
    env.update({
        "KINDI_PROMPT_CACHE": "1",
        "KINDI_PREFIX_MIN": "512",
        "KINDI_PREFIX_MAX": "2",
        "KINDI_CHECKPOINT_EVERY": "256",
        "KINDI_DENSE_SPACING": "256",
        "KINDI_REPEAT_STOP": "1",
    })
    return env


def cuda_version() -> str | None:
    """The CUDA version in nvidia-smi's header; no serial numbers or UUIDs are read."""
    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", proc.stdout or "")
    return match.group(1) if match else None


def git_info() -> dict:
    def run(*args: str):
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    head = run("rev-parse", "--short", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "commit": head.stdout.strip() if head and head.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch and branch.returncode == 0 else None,
        "dirty": bool(status and status.returncode == 0 and status.stdout.strip()),
    }


def parse_engine_evidence(text: str) -> dict:
    match = PLAN_RE.search(text)
    return {
        "plan_cards": int(match.group(1)) if match else None,
        "p106_seen": "P106-100" in text,
    }


def verify_status(parsed_status: str, evidence: dict, dual: bool) -> tuple[str, str]:
    """A verify exit code alone is not evidence for a P106 or a two-card run."""
    status = parsed_status
    note = ""
    if status == "pass" and not dual and not evidence.get("p106_seen"):
        status = "fail"
        note = "the log does not name a P106-100"
    if dual and status == "pass" and evidence.get("plan_cards") != 2:
        status = "fail"
        note = "the plan engine did not report two cards"
    return status, note


def _missing_row(phase: str, model: str, pack: Path, reason: str = "pack not found") -> dict:
    return {
        "phase": phase,
        "model": model,
        "status": "missing",
        "prompts": [],
        "prompts_pass": 0,
        "prompts_tie": 0,
        "prompts_fail": 0,
        "tokens_per_second": None,
        "exit_code": None,
        "duration_s": 0.0,
        "first_diff_token": None,
        "first_diff_gap": None,
        "summary": "%s: %s" % (reason, _display_path(pack)),
        "engine_evidence": {"plan_cards": None, "p106_seen": False},
        "log": "",
        "command": [],
    }


def run_verify(model: str, pack: Path, out_dir: Path, env: dict[str, str], timeout: float,
               dual: bool = False, config_path: Path | None = None) -> dict:
    phase = "dual" if dual else "single"
    log_rel = Path("logs") / ("%s-%s.log" % (phase, model))
    log_path = out_dir / log_rel
    command = [sys.executable, "-m", "kindi"]
    if config_path is not None:
        command += ["--config", str(config_path)]
    command += ["--pack", str(pack), "verify"]
    started = time.monotonic()
    exit_code: int | None = None
    output = ""
    try:
        proc = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
        exit_code = proc.returncode
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        output = stdout + stderr + "\nTIMEOUT after %.1f s\n" % timeout
    except OSError as exc:
        exit_code = 127
        output = "script error: %s\n" % exc
    duration = time.monotonic() - started

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", newline="\n")
    parsed = parse_verify_output(output, exit_code)
    evidence = parse_engine_evidence(output)
    status, note = verify_status(parsed["status"], evidence, dual)
    summary = parsed["summary"]
    if note:
        summary = (summary + " - " + note).strip(" -")
    return {
        "phase": phase,
        "model": model,
        "status": status,
        "prompts": parsed["prompts"],
        "prompts_pass": parsed["prompts_pass"],
        "prompts_tie": parsed["prompts_tie"],
        "prompts_fail": parsed["prompts_fail"],
        "tokens_per_second": parsed["tokens_per_second"],
        "exit_code": exit_code,
        "duration_s": duration,
        "first_diff_token": parsed["first_diff_token"],
        "first_diff_gap": parsed["first_diff_gap"],
        "summary": summary,
        "engine_evidence": evidence,
        "log": log_rel.as_posix(),
        "command": (["python", "-m", "kindi"] +
                    (["--config", _display_path(config_path)] if config_path is not None else []) +
                    ["--pack", _display_path(pack), "verify"]),
        "config": {key: value for key, value in env.items()
                   if key.startswith("KINDI_") or key.startswith("CUDA_")},
    }


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class Server:
    """A temporary Kindi server; the state directory survives a stop/start pair."""

    def __init__(self, *, tag: str, pack: Path, state_dir: str, ctx: int, env: dict[str, str],
                 log_path: Path, startup_timeout: float, chat_timeout: float,
                 config_path: Path | None = None):
        self.tag, self.pack, self.state_dir, self.ctx = tag, pack, state_dir, ctx
        self.env, self.log_path, self.startup_timeout, self.chat_timeout = env, log_path, startup_timeout, chat_timeout
        self.config_path = config_path
        self.port = free_port()
        self.log = None
        self.proc = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        try:
            command = [sys.executable, "-u", "-m", "kindi"]
            if self.config_path is not None:
                command += ["--config", str(self.config_path)]
            command += ["--pack", str(self.pack), "--state-dir", self.state_dir, "--ctx", str(self.ctx),
                        "serve", "--host", "127.0.0.1", "--port", str(self.port)]
            self.proc = subprocess.Popen(
                command,
                cwd=ROOT, env=self.env, stdout=self.log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError("server %s died during startup; see %s" % (self.tag, self.log_path))
                try:
                    health = self.get_json("/health")
                    if health.get("ok"):
                        return self
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            raise RuntimeError("server %s did not become healthy; see %s" % (self.tag, self.log_path))
        except BaseException:
            self._stop()
            raise

    def _stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __exit__(self, *exc):
        self._stop()
        return False

    def _url(self, path: str) -> str:
        return "http://127.0.0.1:%d%s" % (self.port, path)

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self._url(path), timeout=10) as response:
            return json.load(response)

    def get_text(self, path: str) -> str:
        with urllib.request.urlopen(self._url(path), timeout=10) as response:
            return response.read().decode("utf-8", "replace")

    def post_json(self, path: str, body: dict) -> tuple[int, object]:
        request = urllib.request.Request(
            self._url(path), json.dumps(body).encode("utf-8"),
            {"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.chat_timeout) as response:
                raw = response.read().decode("utf-8", "replace")
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            status = exc.code
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, raw

    def chat(self, messages: list[dict], n: int = 16) -> tuple[str, dict]:
        status, body = self.post_json("/v1/chat/completions", {
            "model": "x", "messages": messages, "max_tokens": n, "temperature": 0,
        })
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError("chat returned HTTP %s: %s" % (status, body))
        choice = body["choices"][0]["message"]
        return choice["content"], body["usage"]

    def lines(self) -> list[str]:
        return self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()


def _cache_check(checks: list[dict], name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def run_server_cache(pack: Path, out_dir: Path, env: dict[str, str], ctx: int, timeout: float,
                     server_cls=Server, keep_state: bool = False,
                     config_path: Path | None = None) -> dict:
    """Start, save, resume, restart and invalidate the prompt cache on one card."""
    checks: list[dict] = []
    state = tempfile.mkdtemp(prefix="kindi-p106-cache-")
    logs = out_dir / "logs"
    result = {"status": "fail", "checks": checks, "log": "logs/server-save.log",
              "restart_log": "logs/server-restart.log", "state_kept": False,
              "command": (["python", "-u", "-m", "kindi"] +
                          (["--config", _display_path(config_path)] if config_path is not None else []) +
                          ["--pack", _display_path(pack), "--state-dir", "<temporary>",
                           "--ctx", str(ctx), "serve", "--host", "127.0.0.1", "--port", "<free>"]),
              "config": {key: value for key, value in env.items()
                         if key.startswith("KINDI_") or key.startswith("CUDA_")}}
    rules = " ".join("Rule %d: answer briefly, cite the file and the line, never guess." % i for i in range(40))
    system = ("You are a careful assistant for a software team. " + rules + " " +
              "When a rule conflicts with the request, say so and follow the request. " +
              "The working folder is src/, the date is 2026-11-10, and the release is 0.5.0. " * 4)
    messages = lambda q: [{"role": "system", "content": system}, {"role": "user", "content": q}]  # noqa: E731
    try:
        with server_cls(tag="server-save", pack=pack, state_dir=state, ctx=ctx, env=env,
                        log_path=logs / "server-save.log", startup_timeout=timeout,
                        chat_timeout=timeout, config_path=config_path) as server:
            health = server.get_json("/health")
            _cache_check(checks, "health", health.get("ok") is True,
                         "model=%s ctx=%s head=%s" % (health.get("model"), health.get("ctx"), health.get("head")))
            cold, cold_usage = server.chat(messages("Name one use of a hash map."))
            cold_cached = int(cold_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
            _cache_check(checks, "cold chat", bool(cold) and cold_cached == 0,
                         "%d reply chars, cached %d" % (len(cold), cold_cached))
            _cache_check(checks, "opening is above prefix_min",
                         int(cold_usage.get("prompt_tokens", 0)) > 600,
                         "%s prompt tokens" % cold_usage.get("prompt_tokens"))

            _, second_usage = server.chat(messages("Name one drawback of a hash map."))
            saved = [int(match.group(1)) for line in server.lines() for match in [SAVED_RE.search(line)] if match]
            saved_tokens = max(saved) if saved else None
            _cache_check(checks, "parting after the opening saves it",
                         saved_tokens is not None and saved_tokens >= 512,
                         "saved %s tokens" % saved_tokens if saved_tokens is not None else "no save line")

            warm, warm_usage = server.chat(messages("Name one use of a hash map."))
            warm_cached = int(warm_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
            _cache_check(checks, "in-process cache hit", warm_cached >= 512,
                         "cached %d of %s prompt tokens" % (warm_cached, warm_usage.get("prompt_tokens")))
            _cache_check(checks, "in-process reply is token-identical", warm == cold,
                         "same reply" if warm == cold else "reply changed")
            _cache_check(checks, "second parting did not duplicate the opening",
                         len(saved) == 1, "%d save line(s)" % len(saved))

        with server_cls(tag="server-restart", pack=pack, state_dir=state, ctx=ctx, env=env,
                        log_path=logs / "server-restart.log", startup_timeout=timeout,
                        chat_timeout=timeout, config_path=config_path) as server:
            _cache_check(checks, "restart lists the saved opening",
                         any("saved openings: 1 in" in line for line in server.lines()),
                         "saved openings line present" if any("saved openings: 1 in" in line for line in server.lines())
                         else "no saved openings line")
            restored, restored_usage = server.chat(messages("Name one use of a hash map."))
            loaded = [int(match.group(1)) for line in server.lines() for match in [LOADED_RE.search(line)] if match]
            restored_cached = int(restored_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
            _cache_check(checks, "restart loads the opening", bool(loaded) and max(loaded) >= 512,
                         "loaded %s tokens" % (max(loaded) if loaded else None))
            _cache_check(checks, "restart resumes token-identically", restored == cold,
                         "same reply" if restored == cold else "reply changed")
            _cache_check(checks, "restart reports cached prompt tokens", restored_cached >= 512,
                         "cached %d of %s prompt tokens" % (restored_cached, restored_usage.get("prompt_tokens")))

            bad_status, _ = server.post_json("/v1/chat/completions", {"messages": []})
            _cache_check(checks, "invalid request is 400", bad_status == 400, "HTTP %d" % bad_status)
            after_bad, after_usage = server.chat(messages("Name one use of a hash map."))
            after_cached = int(after_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
            _cache_check(checks, "invalid request keeps the cache", after_bad == cold and after_cached >= 512,
                         "cached %d of %s prompt tokens" % (after_cached, after_usage.get("prompt_tokens")))

            metrics = server.get_text("/metrics")
            match = CACHE_METRIC_RE.search(metrics)
            metric_cached = int(match.group(1)) if match else 0
            _cache_check(checks, "metrics report cached prompt tokens", metric_cached > 0,
                         "kindi_cached_prompt_tokens_total %d" % metric_cached)
    except Exception as exc:  # noqa: BLE001 - a failed server phase is a recorded result
        _cache_check(checks, "server/cache run", False, "%s: %s" % (type(exc).__name__, exc))
    finally:
        if keep_state:
            result["state_kept"] = True
            result["state_dir"] = _display_path(Path(state))
        else:
            shutil.rmtree(state, ignore_errors=True)
    result["checks"] = checks
    result["status"] = "pass" if checks and all(check["ok"] for check in checks) else "fail"
    return result


def _md(value) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def write_results(out_dir: Path, payload: dict) -> tuple[Path, Path]:
    json_path = out_dir / "results.json"
    md_path = out_dir / "RESULTS.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    host = payload.get("host", {})
    git = payload.get("git", {})
    preflight = payload.get("preflight", {})
    cards = preflight.get("selected_cards", [])
    lines = [
        "# P106-100 verification record", "",
        "Run started: %s (UTC)" % payload.get("started_utc"),
        "Duration: %.1f s" % payload.get("duration_s", 0.0), "",
        "Commit: `%s` on `%s`%s" % (git.get("commit"), git.get("branch"),
                                     " (dirty worktree)" if git.get("dirty") else ""),
        "Python: %s  " % host.get("python"),
        "Platform: %s  " % host.get("platform"),
        "Driver: %s  " % host.get("driver_version"),
        "CUDA: %s" % host.get("cuda_version"),
        "Visible P106-100 indices: `%s`" % preflight.get("visible_devices"), "",
        "| card | memory MiB | compute | PCIe |", "|---|---:|---|---|",
    ]
    for card in cards:
        lines.append("| %s | %s | %s | gen %s x%s |" % (
            _md(card.get("name")), _md(card.get("memory_total_mib")), _md(card.get("compute_cap")),
            _md(card.get("pcie_gen")), _md(card.get("pcie_width"))))
    lines += ["", "## Single-card verify", "",
              "| model | status | prompts pass/tie/fail | tokens/s | exit | log |",
              "|---|---|---:|---:|---:|---|"]
    for row in payload.get("single", []):
        lines.append("| %s | %s | %d/%d/%d | %s | %s | %s |" % (
            _md(row.get("model")), _md(row.get("status")), row.get("prompts_pass", 0),
            row.get("prompts_tie", 0), row.get("prompts_fail", 0),
            "n/a" if row.get("tokens_per_second") is None else "%.1f" % row["tokens_per_second"],
            _md(row.get("exit_code")), _md(row.get("log"))))
    lines += ["", "## Server and prompt cache", "", "| check | result | detail |", "|---|---|---|"]
    server = payload.get("server_cache", {})
    for check in server.get("checks", []):
        lines.append("| %s | %s | %s |" % (_md(check.get("name")), "PASS" if check.get("ok") else "FAIL",
                                             _md(check.get("detail"))))
    lines += ["", "Server logs: `%s`, `%s`" % (server.get("log") or "", server.get("restart_log") or ""), "",
              "## Two-card 27B", ""]
    dual = payload.get("dual", {})
    if dual.get("status") == "skipped":
        lines.append("Skipped: %s" % _md(dual.get("reason")))
    else:
        lines += ["| model | status | prompts pass/tie/fail | tokens/s | plan cards | log |",
                  "|---|---|---:|---:|---:|---|"]
        lines.append("| %s | %s | %d/%d/%d | %s | %s | %s |" % (
            _md(dual.get("model")), _md(dual.get("status")), dual.get("prompts_pass", 0),
            dual.get("prompts_tie", 0), dual.get("prompts_fail", 0),
            "n/a" if dual.get("tokens_per_second") is None else "%.1f" % dual["tokens_per_second"],
            _md((dual.get("engine_evidence") or {}).get("plan_cards")), _md(dual.get("log"))))
    lines += ["", "## Verdict", "", "`%s`" % payload.get("verdict", "fail"), ""]
    if payload.get("verdict") == "pass":
        lines.append("All selected phases passed. This is a recorded result, not a support claim: "
                     "add the numbers to docs/CARDS.md and docs/P106.md after review.")
    else:
        lines.append("The run did not pass every selected phase; do not record a support claim.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return json_path, md_path


def _split_models(value: str) -> list[str]:
    wanted = {part.strip().lower() for part in value.split(",") if part.strip()}
    unknown = sorted(wanted - set(MODELS))
    if unknown:
        raise SystemExit("unknown model: %s (choose from %s)" % (", ".join(unknown), ", ".join(MODELS)))
    if not wanted:
        raise SystemExit("no model selected")
    return [model for model in MODELS if model in wanted]


def _print_plan(args, cards: list[dict], visible: str) -> None:
    print("P106-100 cards:")
    for card in cards:
        print("  %d  %s  %s MiB  sm_%s  PCIe %s x%s" % (
            card["index"], card["name"], card["memory_total_mib"], card["compute_cap"],
            card["pcie_gen"], card["pcie_width"]))
    print("CUDA_VISIBLE_DEVICES=%s" % visible)
    single = single_env(cards[:1], args.ctx)
    single_config = " ".join("%s=%s" % (key, value) for key, value in sorted(single.items())
                             if key.startswith("KINDI_"))
    config = _display_path(args.out / "kindi.conf")
    for model in args.models:
        print("would run: single %s" % model)
        print("  " + shlex.join(["python", "-m", "kindi", "--config", config, "--pack",
                                 _display_path(pack_for(args.pack_dir, model)), "verify"]))
        print("  " + single_config)
    print("would run: server/cache on %s" % args.server_model)
    print("  " + shlex.join(["python", "-u", "-m", "kindi", "--config", config, "--pack",
                             _display_path(pack_for(args.pack_dir, args.server_model)),
                             "--state-dir", "<temporary>", "--ctx", str(args.ctx), "serve",
                             "--host", "127.0.0.1", "--port", "<free>"]))
    if not args.single_only:
        print("would run: two-card %s verify" % DUAL_MODEL)
        print("  " + shlex.join(["python", "-m", "kindi", "--config", config, "--pack",
                                 _display_path(pack_for(args.pack_dir, DUAL_MODEL)), "verify"]))
        print("  " + " ".join("%s=%s" % (key, value) for key, value in sorted(dual_env(cards, args.ctx).items())
                                  if key.startswith("KINDI_")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack-dir", type=Path, default=Path.home() / ".kindi" / "packs",
                    help="installed pack root (default: ~/.kindi/packs)")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="single-card models (default: 2b,4b,9b)")
    ap.add_argument("--server-model", choices=MODELS, default="2b",
                    help="model served for the cache check (default: 2b)")
    ap.add_argument("--ctx", type=int, default=24576,
                    help="requested context for the server and the two-card verify (default: 24576)")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-command timeout in seconds (default: 1800)")
    ap.add_argument("--out", type=Path, default=ROOT / "bench-logs" / "p106-verify",
                    help="output directory (default: bench-logs/p106-verify)")
    ap.add_argument("--visible-devices", default=None,
                    help="nvidia-smi indices for CUDA_VISIBLE_DEVICES (default: all P106-100s)")
    ap.add_argument("--single-only", action="store_true",
                    help="record one card and skip the two-card 27B phase")
    ap.add_argument("--keep-state", action="store_true",
                    help="keep the server state directory after the run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running or writing")
    args = ap.parse_args(argv)

    if args.ctx < 1024:
        raise SystemExit("--ctx must be at least 1024")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    args.models = _split_models(args.models)
    args.pack_dir = args.pack_dir.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    started = datetime.now(timezone.utc)
    cards, error = query_cards()
    selected, select_error = select_cards(cards, args.visible_devices, args.single_only)
    visible = ",".join(str(card["index"]) for card in selected) if selected else (args.visible_devices or "0,1")
    preflight = {
        "ok": not error and not select_error,
        "error": error or select_error,
        "all_cards": cards,
        "selected_cards": selected,
        "visible_devices": visible,
    }

    if args.dry_run:
        if selected:
            _print_plan(args, selected, visible)
        else:
            print("preflight: %s" % preflight["error"])
            print("would use CUDA_VISIBLE_DEVICES=%s once two P106-100s are visible" % visible)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "kindi.conf"
    config_path.write_text("[kindi]\n", encoding="utf-8", newline="\n")
    result = {
        "schema": 1,
        "script": "tools/verify_p106.py",
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": 0.0,
        "git": git_info(),
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "driver_version": selected[0]["driver_version"] if selected else None,
            "cuda_version": cuda_version(),
        },
        "config": {
            "pack_dir": _display_path(args.pack_dir),
            "models": args.models,
            "server_model": args.server_model,
            "dual_model": DUAL_MODEL,
            "ctx": args.ctx,
            "timeout_s": args.timeout,
            "single_only": args.single_only,
            "config_file": _display_path(config_path),
        },
        "preflight": preflight,
        "single": [],
        "server_cache": {"status": "blocked", "checks": []},
        "dual": {"status": "skipped", "reason": "--single-only"} if args.single_only else {"status": "blocked"},
        "verdict": "blocked",
    }

    if not preflight["ok"]:
        print("preflight failed: %s" % preflight["error"], flush=True)
        result["duration_s"] = (datetime.now(timezone.utc) - started).total_seconds()
        json_path, md_path = write_results(args.out, result)
        print("wrote %s and %s" % (json_path, md_path))
        return 2

    single_cards = selected[:1]
    single = single_env(single_cards, args.ctx)
    total = len(args.models)
    for index, model in enumerate(args.models, 1):
        pack = pack_for(args.pack_dir, model)
        print("[%d/%d] single %s" % (index, total, model), flush=True)
        if not (pack / "kindi_pack.json").is_file():
            row = _missing_row("single", model, pack)
        else:
            row = run_verify(model, pack, args.out, single, args.timeout, config_path=config_path)
        result["single"].append(row)
        print("  %s" % row["status"], flush=True)

    print("[server] %s" % args.server_model, flush=True)
    server_pack = pack_for(args.pack_dir, args.server_model)
    if not (server_pack / "kindi_pack.json").is_file():
        result["server_cache"] = {
            "status": "missing", "checks": [],
            "log": "logs/server-save.log", "restart_log": "logs/server-restart.log",
            "reason": "pack not found: %s" % _display_path(server_pack),
        }
    else:
        result["server_cache"] = run_server_cache(
            server_pack, args.out, server_env(single_cards, args.ctx), args.ctx, args.timeout,
            keep_state=args.keep_state, config_path=config_path,
        )
    print("  %s" % result["server_cache"]["status"], flush=True)

    if not args.single_only:
        print("[dual] %s" % DUAL_MODEL, flush=True)
        dual_pack = pack_for(args.pack_dir, DUAL_MODEL)
        if not (dual_pack / "kindi_pack.json").is_file():
            row = _missing_row("dual", DUAL_MODEL, dual_pack)
        else:
            row = run_verify(DUAL_MODEL, dual_pack, args.out, dual_env(selected, args.ctx),
                             args.timeout, dual=True, config_path=config_path)
        result["dual"] = row
        print("  %s" % row["status"], flush=True)

    phases = [row["status"] for row in result["single"]]
    phases.append(result["server_cache"]["status"])
    if not args.single_only:
        phases.append(result["dual"]["status"])
    result["verdict"] = "pass" if phases and all(status == "pass" for status in phases) else "fail"
    result["duration_s"] = (datetime.now(timezone.utc) - started).total_seconds()
    json_path, md_path = write_results(args.out, result)
    print("\nverdict: %s" % result["verdict"])
    print("wrote %s and %s" % (json_path, md_path))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
