#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in GPU smoke test for the 2B MTP head on a real `kindi serve` process.

The manual check boots the server with mode A attached, sends one greedy completion and then
looks at `/health` and `/metrics`. This script does exactly that, but exits non-zero unless both
views agree that the head drafted tokens:

* `/health` reports `drafting.enabled`, `drafting.method == "mtp"` and a head before the request;
* `/metrics` reports a non-zero `kindi_spec_drafted_tokens_total`;
* `/health`'s `drafting.last.method` is `mtp` and `drafting.last.drafted` is non-zero after the
  request.

It is a tool, not a `tests/test_*.py` file, so the CPU-only suite (`pytest.ini` sets
`testpaths = tests`) never collects it. The child server gets a clean `KINDI_*` environment, an
empty config file, a temporary state directory and a loopback-only port; `--log FILE` keeps the
server log after the run, with a run record at the top: the pack, the head and its digest, the
context, `KINDI_SPEC_K`, the GPU, the exact serve command with its `KINDI_*` environment, and the
request count. The same record is printed, so a log or a transcript is enough evidence on its own.

Example (on a tensor-core GPU host):

    python tools/mtp_smoke.py --pack ~/.kindi/packs/bonsai-2b \\
      --head ~/.kindi/models/kindi-mtp-2b-v1.npz

Exit codes: 0 pass, 2 missing pack/head/GPU or bad arguments, 1 a server or counter failure.
"""
import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                       # `python tools/mtp_smoke.py` from the repository root

DEFAULT_PACK = "~/.kindi/packs/bonsai-2b"
DEFAULT_HEAD = "~/.kindi/models/kindi-mtp-2b-v1.npz"
HEAD_URL = "https://huggingface.co/ahtx/kindi-mtp-heads"
METRIC_DRAFTED = "kindi_spec_drafted_tokens_total"
METRIC_ACCEPTED = "kindi_spec_accepted_tokens_total"

GPU_PROBE = r"""
import json
import cupy as cp

count = cp.cuda.runtime.getDeviceCount()
if count < 1:
    raise SystemExit("no CUDA device")
dev = cp.cuda.Device()
cc = dev.compute_capability
if isinstance(cc, (tuple, list)):
    major, minor = int(cc[0]), int(cc[1])
else:
    major, minor = divmod(int(cc), 10)
props = cp.cuda.runtime.getDeviceProperties(dev.id)
name = props.get("name", "GPU")
if isinstance(name, bytes):
    name = name.decode("utf-8", "replace")
print(json.dumps({"name": str(name), "major": major, "minor": minor}))
"""


class SmokeError(Exception):
    def __init__(self, message, code=1):
        Exception.__init__(self, message)
        self.code = code


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _port(text):
    value = int(text)
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
    return value


def _positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _path(value):
    return os.path.abspath(os.path.expanduser(value))


def _check_pack(path):
    if not os.path.isdir(path):
        raise SmokeError("pack directory not found: %s (expected a converted bonsai-2b pack)" % path, 2)
    manifest_path = os.path.join(path, "kindi_pack.json")
    if not os.path.isfile(manifest_path):
        raise SmokeError("pack has no kindi_pack.json: %s (run `kindi convert` first)" % path, 2)
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SmokeError("cannot read %s: %s" % (manifest_path, exc), 2)
    identity = " ".join((str(manifest.get("model") or ""), str(manifest.get("model_id") or ""),
                         os.path.basename(os.path.normpath(path)))).lower()
    if "2b" not in identity:
        raise SmokeError("pack %s does not look like the 2B pack (manifest model %r); this MTP head is 2B-only"
                         % (path, manifest.get("model")), 2)
    return manifest


def _check_head(path):
    if not os.path.isfile(path):
        raise SmokeError("MTP head not found: %s\n"
                         "download the registered 61 MB head from %s and pass --head" % (path, HEAD_URL), 2)
    try:
        from kindi import mtp_heads
        info = mtp_heads.verify(path)
    except Exception as exc:                       # HeadError, OSError and a local import error
        raise SmokeError("MTP head refused: %s" % exc, 2)
    if not info.get("registered"):
        raise SmokeError("MTP head %s is not in kindi/mtp_heads.json; refusing an unregistered file for this smoke"
                         % path, 2)
    return info


def _check_gpu():
    """Probe CUDA in a short-lived child so this process holds no CUDA context while serving."""
    try:
        proc = subprocess.run([sys.executable, "-c", GPU_PROBE], cwd=ROOT, env=os.environ.copy(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeError("CUDA/CuPy probe failed: %s" % exc, 2)
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise SmokeError("CUDA/CuPy is not usable on this host: %s"
                         % (detail[-1] if detail else "probe exited %d" % proc.returncode), 2)
    try:
        info = json.loads((proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise SmokeError("CUDA/CuPy probe returned no device information", 2)
    if (int(info["major"]), int(info["minor"])) < (8, 0):
        raise SmokeError("MTP needs a tensor-core GPU (sm_80+); found %s (sm_%d%d)"
                         % (info["name"], info["major"], info["minor"]), 2)
    return "%s (sm_%d%d)" % (info["name"], info["major"], info["minor"])


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url, timeout):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _get_json(url, timeout):
    return json.loads(_get(url, timeout))


def _post_json(url, body, timeout):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise SmokeError("completion request failed with HTTP %d: %s" % (exc.code, detail[:1000]))


def _wait_health(base, proc, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SmokeError("server exited before becoming healthy (exit %s)" % proc.returncode)
        try:
            health = _get_json(base + "/health", timeout=5)
        except Exception as exc:                   # connection refused while uvicorn starts
            last_error = exc
        else:
            if health.get("ok") is True:
                return health
            last_error = "health not ok: %r" % (health.get("ok"),)
        time.sleep(0.5)
    raise SmokeError("server did not become healthy within %.0f s%s"
                     % (timeout, " (%s)" % last_error if last_error else ""))


def _wait_for_drafted(base, proc, timeout):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SmokeError("server exited before the drafted-token counter appeared (exit %s)" % proc.returncode)
        try:
            health = _get_json(base + "/health", timeout=5)
        except Exception:
            time.sleep(0.2)
            continue
        last = (health.get("drafting") or {}).get("last")
        if last and int(last.get("drafted") or 0) > 0:
            return health, last
        time.sleep(0.2)
    raise SmokeError("drafting.last did not show drafted tokens after the request (last=%r)" % (last,))


def _check_last_method(last):
    """Reject a post-request block that reports a mode other than MTP.

    The pre-request health check can show `mtp` while the request itself falls back
    to another drafter; the last block is the evidence that mode A actually ran.
    """
    if last.get("method") != "mtp":
        raise SmokeError("drafting.last.method is %r, not mtp (last=%r)"
                         % (last.get("method"), last))


def _metric(text, name):
    match = re.search(r"(?m)^%s(?:\{[^}]*\})?\s+([0-9]+(?:\.[0-9]+)?)\s*$" % re.escape(name), text)
    if not match:
        raise SmokeError("metric %s is missing from /metrics" % name)
    return int(float(match.group(1)))


def _tail(path, lines=60):
    try:
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def _show_log(path, title="server log"):
    lines = _tail(path)
    if not lines:
        return
    print("--- %s (%s) ---" % (title, path), file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print("--- end %s ---" % title, file=sys.stderr)


def _stop(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()


def _env(pack, head, state_dir, ctx, spec_k):
    # A clean set makes the smoke independent of the caller's shell and config. The empty
    # config file passed on the command line also keeps a ./kindi.conf from changing auth,
    # rate limits, or the effective engine.
    env = {key: value for key, value in os.environ.items() if not key.startswith("KINDI_")}
    env.update({
        "PYTHONUNBUFFERED": "1",
        "KINDI_PACK": pack,
        "KINDI_STATE_DIR": state_dir,
        "KINDI_DEVICE": "gpu",
        "KINDI_HEAD": "exact",
        "KINDI_TC": "1",
        "KINDI_TP": "1",
        "KINDI_SPEC": "mtp",
        "KINDI_MTP_HEAD": head,
        "KINDI_SPEC_K": str(spec_k),
        "KINDI_CTX": str(ctx),
        "KINDI_API_KEY": "",
        "KINDI_RATE_LIMIT_PER_MIN": "0",
        "KINDI_QUEUE_MAX": "0",
        "KINDI_TURN_SAVE": "0",
        "KINDI_PREFIX_MAX": "0",
        "KINDI_PARK_MAX": "0",
        "KINDI_LOG_JSON": "0",
    })
    return env


def _run_record(pack, manifest, head, head_info, ctx, spec_k, gpu, command, env, requests):
    """The provenance a future MTP claim needs, independent of the terminal transcript."""
    return {
        "pack": pack,
        "model": manifest.get("model") or manifest.get("model_id") or "unknown",
        "head": head,
        "head_sha256": head_info.get("sha256") or "",
        "head_bytes": int(head_info.get("bytes") or 0),
        "ctx": int(ctx),
        "spec_k": int(spec_k),
        "gpu": gpu,
        "serve_argv": [str(part) for part in command],
        "serve_env": {key: str(value) for key, value in env.items() if key.startswith("KINDI_")},
        "requests": int(requests),
    }


def _serve_command(record):
    """The serve argv with the run's KINDI_* environment, in POSIX shell form."""
    parts = ["%s=%s" % (key, shlex.quote(value))
             for key, value in sorted(record["serve_env"].items())]
    parts.extend(shlex.quote(part) for part in record["serve_argv"])
    return " ".join(parts)


def _record_lines(record):
    return [
        "pack: %s (%s)" % (record["pack"], record["model"]),
        "head: %s sha256 %s (%d bytes, registered)" % (
            record["head"], record["head_sha256"], record["head_bytes"]),
        "ctx: %d" % record["ctx"],
        "KINDI_SPEC_K: %d" % record["spec_k"],
        "gpu: %s" % record["gpu"],
        "requests: %d" % record["requests"],
        "serve command: %s" % _serve_command(record),
    ]


def _record_header(record):
    """The comment block written at the top of the server log."""
    return ("# mtp_smoke run record\n"
            + "".join("# %s\n" % line for line in _record_lines(record))
            + "# end record\n")


def _run(args, temp_dir):
    pack = _path(args.pack)
    head = _path(args.head)
    manifest = _check_pack(pack)
    head_info = _check_head(head)
    gpu = _check_gpu()
    port = args.port or _free_port()
    base = "http://127.0.0.1:%d" % port
    state_dir = os.path.join(temp_dir, "state")
    empty_config = os.path.join(temp_dir, "kindi.conf")
    os.makedirs(state_dir, exist_ok=True)
    open(empty_config, "w", encoding="utf-8").close()
    log_path = _path(args.log) if args.log else os.path.join(temp_dir, "server.log")
    if args.log:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    proc = None
    try:
        env = _env(pack, head, state_dir, args.ctx, args.spec_k)
        command = [sys.executable, "-m", "kindi", "--config", empty_config,
                   "serve", "--host", "127.0.0.1", "--port", str(port)]
        record = _run_record(pack, manifest, head, head_info, args.ctx, args.spec_k, gpu,
                             command, env, requests=1)
        print("mtp_smoke: run record")
        for line in _record_lines(record):
            print("  %s" % line)
        log = open(log_path, "wb")
        try:
            log.write(_record_header(record).encode("utf-8"))
            log.flush()
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()

        health = _wait_health(base, proc, args.timeout)
        draft = health.get("drafting") or {}
        if not draft.get("enabled"):
            raise SmokeError("MTP is not enabled: requested=%r reason=%r"
                             % (draft.get("requested"), draft.get("reason")))
        if draft.get("method") != "mtp":
            raise SmokeError("server selected method %r, not mtp (requested=%r reason=%r)"
                             % (draft.get("method"), draft.get("requested"), draft.get("reason")))
        if not draft.get("head"):
            raise SmokeError("server reports MTP enabled but names no head: %r" % (draft,))
        print("mtp_smoke: server ready at %s (model %s, head %s)"
              % (base, health.get("model"), draft.get("head")))

        request = {
            "model": health.get("model"),
            "messages": [{"role": "user", "content": "List three short colors, one per line."}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": False,
        }
        response = _post_json(base + "/v1/chat/completions", request, args.timeout)
        choices = response.get("choices") or []
        if not choices:
            raise SmokeError("completion response has no choices: %r" % (response,))
        content = ((choices[0].get("message") or {}).get("content") or "")
        usage = response.get("usage") or {}

        health, last = _wait_for_drafted(base, proc, min(args.timeout, 30.0))
        metrics = _get(base + "/metrics", timeout=10)
        drafted = _metric(metrics, METRIC_DRAFTED)
        accepted = _metric(metrics, METRIC_ACCEPTED)
        if drafted <= 0:
            raise SmokeError("%s is %d after one greedy completion" % (METRIC_DRAFTED, drafted))
        if accepted < 0 or accepted > drafted:
            raise SmokeError("inconsistent spec counters: drafted=%d accepted=%d" % (drafted, accepted))
        _check_last_method(last)
        last_drafted = int(last.get("drafted") or 0)
        last_accepted = int(last.get("accepted") or 0)
        if last_drafted <= 0:
            raise SmokeError("drafting.last.drafted is %d after one greedy completion" % last_drafted)
        if last_accepted < 0 or last_accepted > last_drafted:
            raise SmokeError("inconsistent drafting.last: drafted=%d accepted=%d"
                             % (last_drafted, last_accepted))

        preview = " ".join(content.split())
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print("mtp_smoke: PASS")
        print("  drafting.last: method=%s passes=%s drafted=%d accepted=%d"
              % (last.get("method"), last.get("passes"), last_drafted, last_accepted))
        print("  metrics: %s=%d %s=%d" % (METRIC_DRAFTED, drafted, METRIC_ACCEPTED, accepted))
        print("  completion: %s tokens, %r" % (usage.get("completion_tokens", "?"), preview))
        if args.log:
            print("  server log: %s" % log_path)
        return 0
    finally:
        _stop(proc)


def main(argv=None):
    head_default = os.environ.get("KINDI_MTP_HEAD")
    if not head_default and os.path.isfile(_path(DEFAULT_HEAD)):
        head_default = DEFAULT_HEAD
    pack_default = os.environ.get("KINDI_PACK")
    if not pack_default and os.path.isdir(_path(DEFAULT_PACK)):
        pack_default = DEFAULT_PACK

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="This script starts a real server and needs a tensor-core GPU; it is not part of the CPU suite.",
    )
    parser.add_argument("--pack", default=pack_default,
                        help="converted 2B pack (default: KINDI_PACK, else %s)" % DEFAULT_PACK)
    parser.add_argument("--head", default=head_default,
                        help="registered MTP head (default: KINDI_MTP_HEAD, else %s)" % DEFAULT_HEAD)
    parser.add_argument("--port", type=_port, default=0,
                        help="loopback port (default: 0, choose a free port)")
    parser.add_argument("--ctx", type=_positive_int, default=4096,
                        help="server context for the smoke (default: 4096)")
    parser.add_argument("--spec-k", type=_positive_int, default=1,
                        help="tokens the MTP head drafts per pass (default: 1, the measured win)")
    parser.add_argument("--max-tokens", type=_positive_int, default=16,
                        help="greedy completion tokens (default: 16)")
    parser.add_argument("--timeout", type=_positive_float, default=180.0,
                        help="seconds allowed for server startup and the request (default: 180)")
    parser.add_argument("--log", default=None,
                        help="keep the server log at this path, with the run record at the top "
                             "(default: a temporary file)")
    args = parser.parse_args(argv)

    if not args.pack:
        print("mtp_smoke: no pack (--pack or KINDI_PACK)", file=sys.stderr)
        return 2
    if not args.head:
        print("mtp_smoke: no head (--head or KINDI_MTP_HEAD); download %s" % HEAD_URL, file=sys.stderr)
        return 2

    temp_dir = tempfile.mkdtemp(prefix="kindi-mtp-smoke-")
    log_path = _path(args.log) if args.log else os.path.join(temp_dir, "server.log")
    try:
        return _run(args, temp_dir)
    except SmokeError as exc:
        print("mtp_smoke: %s" % exc, file=sys.stderr)
        _show_log(log_path)
        return exc.code
    except KeyboardInterrupt:
        print("mtp_smoke: interrupted", file=sys.stderr)
        return 130
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
