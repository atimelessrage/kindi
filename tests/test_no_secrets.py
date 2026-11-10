"""Fail when a tracked file contains a credential or a local home path.

This is a cheap, dependency-free guard for the common accident: a key, token or local path is
added to a source file or doc. It scans `git ls-files`, skips binary files, and reports only the file, line and detector name,
never the matched text.

It does not replace a full history scan: commit metadata and deleted lines are outside its scope.
Run `detect-secrets` over the unpushed diff before a release.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".gguf", ".bin", ".npz", ".npy", ".dll", ".exe", ".lib", ".exp", ".so", ".o", ".a",
        ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".xz"}

# Placeholders and documented examples are not credentials.
ALLOW = re.compile(r"(?i)(?:example|placeholder|your[-_]|none|redacted|choose-a-long-random-string|random-string|\*\*\*|<[^>]+>)")

PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("local home path", re.compile(r"(?i)(?:[A-Z]:[\\/]Users[\\/][^\\/\s\"'<>]+|/home/(?![\[(])[^/\s\"'<>]+)")),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]

# Assignments with a long literal value, excluding the common placeholders and hashes.
GENERIC = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})")
HEX = re.compile(r"^[0-9a-fA-F]{32,}$")


def tracked_files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True,
                         stdout=subprocess.PIPE).stdout
    for raw in out.split(b"\0"):
        if raw:
            yield raw.decode("utf-8", "replace")


def findings():
    for rel in tracked_files():
        if Path(rel).suffix.lower() in SKIP:
            continue
        try:
            raw = (ROOT / rel).read_bytes()
        except OSError:
            continue
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16", "replace")
        elif b"\x00" in raw:
            continue
        else:
            text = raw.decode("utf-8", "replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS:
                match = pattern.search(line)
                if match and not ALLOW.search(match.group(0)):
                    yield rel, lineno, name
            m = GENERIC.search(line)
            if m and not ALLOW.search(m.group(0)) and not HEX.match(m.group(2)):
                yield rel, lineno, "secret assignment"


def test_no_credentials_or_home_paths_in_tracked_files():
    bad = list(findings())
    assert not bad, "possible secrets/personal data in tracked files:\n" + "\n".join(
        "%s:%d: %s" % item for item in bad[:40])


if __name__ == "__main__":
    test_no_credentials_or_home_paths_in_tracked_files()
    print("ALL PASS")
