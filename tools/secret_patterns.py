"""The one pattern set behind both secret guards.

`tests/test_no_secrets.py` scans the files that exist now (tracked and untracked);
`tools/check_secret_history.py` scans every commit's diff, so a credential that was committed
and deleted afterwards is still found. The patterns live here so the two cannot drift, and
both report only the file, line and detector name - never the matched text.

Categories, on purpose:
  * credentials: key material and provider token shapes;
  * machine data: a Windows home path, a Unix home path, a backslash drive path, a cloud
    storage folder (a synced Dropbox or OneDrive account: the account name and the folder tree
    are personal even when no file inside is sensitive);
  * identifiers: an email address, unless it is a documented placeholder or a no-reply
    address of a hosting forge.
"""
import re

SKIP = {".gguf", ".bin", ".npz", ".npy", ".dll", ".exe", ".lib", ".exp", ".so", ".o", ".a",
        ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".xz"}

# Placeholders and documented examples are not credentials.
# `KindiInstallerTest` is the deliberate -AppName/-DistroDir fixture of the documented
# installer run (docs/WINDOWS.md), not a machine layout that leaked into the tree.
ALLOW = re.compile(r"(?i)(?:example|placeholder|your[-_]|none|redacted|choose-a-long-random-string|random-string|KindiInstallerTest|\*\*\*|<[^>]+>)")

# Reserved and no-reply domains: an address here identifies an account, not a person's mailbox.
ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "kindi.invalid",
                         "users.noreply.github.com", "noreply.github.com", "codebuff.com",
                         "localhost")

PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    # /home/kindi is the fixed non-root container user the Dockerfiles create, not a person's home directory
    ("local home path", re.compile(r"(?i)(?:[A-Z]:[\\/]Users[\\/][^\\/\s\"'<>]+|/home/(?![\[(]|kindi\b)[^/\s\"'<>]+)")),
    # A backslash-form drive path is machine-local data (a pack dir, a model cache): the docs write
    # repository paths as T:/kindi, so this shape is the machine-specific one. The 2026-11-16 audit
    # found exactly that shape (a drive letter, then a backslash-separated pack directory) in
    # tracked benchmark JSONs.
    ("local drive path", re.compile(r"(?i)\b[A-Z]:\\+(?:[A-Za-z0-9_.$~-]+\\+)+")),
    # A synced folder is a person's account and working tree even when the contents are mundane:
    # the account name and the sibling folder names are personal data.
    ("cloud storage path", re.compile(r"(?i)\b(?:dropbox|onedrive|google\s*drive|icloud\s*drive|nextcloud|owncloud|seafile)[\\/]+[^\s\"'<>|)]+")),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]

# Assignments with a long literal value, excluding the common placeholders and hashes.
GENERIC = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})")
HEX = re.compile(r"^[0-9a-fA-F]{32,}$")


def allowed_email(value):
    """True for a placeholder or a no-reply forge address (never a person's mailbox)."""
    domain = value.rsplit("@", 1)[-1].lower().strip(".")
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_EMAIL_DOMAINS)


def hits_in_line(line):
    """Detector names for one line of text, as a tuple (empty for a clean line)."""
    found = []
    for name, pattern in PATTERNS:
        match = pattern.search(line)
        if not match or ALLOW.search(match.group(0)):
            continue
        if name == "email address" and allowed_email(match.group(0)):
            continue
        found.append(name)
    match = GENERIC.search(line)
    if match and not ALLOW.search(match.group(0)) and not HEX.match(match.group(2)):
        found.append("secret assignment")
    return tuple(found)
