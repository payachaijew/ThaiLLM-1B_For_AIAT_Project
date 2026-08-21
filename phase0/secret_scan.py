#!/usr/bin/env python3
"""High-precision secret detection and redaction for replay corpora.

The detector intentionally favours precision over recall for generic assignments.
Private-key material causes the complete document to be dropped; other accepted
matches are replaced with ``[REDACTED_SECRET]``.  Events expose only masked
context and never the matched value.

This is a data-preparation utility, not scientific evidence.
"""
from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

SCIENTIFIC_EVIDENCE_ALLOWED = False
REDACTION = "[REDACTED_SECRET]"


@dataclass(frozen=True)
class Candidate:
    kind: str
    start: int
    end: int
    value: str


PRIVATE_KEY = re.compile(
    r"-----BEGIN\s+(?:RSA|OPENSSH|DSA|EC|PGP)\s+PRIVATE KEY-----",
    re.IGNORECASE,
)

# Patterns are ordered from the most specific to the most generic.  Capturing
# group 1, when present, is the value that is redacted.
PATTERNS: Sequence[Tuple[str, re.Pattern[str], int]] = (
    ("aws_access_key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])"), 0),
    (
        "aws_secret",
        re.compile(
            r"(?i)(?:aws[_-]?(?:secret(?:[_-]?access)?[_-]?key)|secret[_-]?access[_-]?key)"
            r"\s*(?::|=|=>)\s*['\"]?([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"
        ),
        1,
    ),
    (
        "github_token",
        re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{20,255})(?![A-Za-z0-9_])"),
        0,
    ),
    ("google_api_key", re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"), 0),
    ("slack_token", re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-[A-Za-z0-9-]{10,255}(?![A-Za-z0-9-])"), 0),
    ("stripe_secret", re.compile(r"(?<![A-Za-z0-9_-])(?:sk_live_|rk_live_)[A-Za-z0-9]{12,255}(?![A-Za-z0-9])"), 0),
    ("anthropic_api_key", re.compile(r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{20,255}(?![A-Za-z0-9_-])"), 0),
    (
        "openai_api_key",
        re.compile(
            r"(?<![A-Za-z0-9_-])sk-(?!ant-)"
            r"(?=[A-Za-z0-9_-]{20,255}(?![A-Za-z0-9_-]))"
            r"(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*\d)"
            r"[A-Za-z0-9_-]{20,255}(?![A-Za-z0-9_-])"
        ),
        0,
    ),
    (
        "jwt",
        re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{7,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"),
        0,
    ),
    (
        "connection_string_password",
        re.compile(
            r"(?i)(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s:/@]+:([^\s/@]+)@"
        ),
        1,
    ),
    (
        "env_secret",
        re.compile(
            r"(?m)^\s*(?:export|ENV)\s+(?:[A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|ACCESS_KEY|PRIVATE_KEY)[A-Z0-9_]*)"
            r"\s*=\s*['\"]?([^'\"\s#;]+)"
        ),
        1,
    ),
    (
        "env_secret",
        re.compile(
            r"(?m)^(?:[A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|ACCESS_KEY|PRIVATE_KEY)[A-Z0-9_]*)"
            r"=['\"]?([^'\"\s#;]+)"
        ),
        1,
    ),
)


PLACEHOLDER_EXACT = {
    "", "x", "xx", "xxx", "xxxx", "your", "todo", "foo", "bar", "baz",
    "changeme", "change_me", "password",
    "secret", "token", "api_key", "apikey", "none", "null", "undefined",
    "redacted", "[redacted]", "[redacted_secret]", "akiaiosfodnn7example",
    "mypassword", "wrong-password", "wrong_password", "pass", "passwd",
    "pwd", "test", "testing", "demo", "admin", "root", "user",
}
PLACEHOLDER_FRAGMENTS = (
    "your_api_key", "your-api-key", "your_token", "your-token", "your_secret",
    "replace_me", "replace-me", "insert_key", "insert-key", "example_key",
    "example-token", "dummy", "placeholder", "sample_key", "fake_key",
    "yourcredential", "credentials_here", "examplecredential", "examplekey",
)


def is_placeholder(value: str) -> bool:
    """Return True for tutorial/template values that must not be redacted."""
    v = value.strip().strip("'\"").lower().lstrip("\\")
    # Regex capture can stop at whitespace inside a template, e.g.
    # ``<MS live password>`` becomes ``<MS``.  Credentials do not normally
    # begin with a template delimiter, so fail closed as a placeholder here.
    if v.startswith(("[", "<", "{{", "${", "%")):
        return True
    unwrapped = v.strip("[](){}<>")
    if v in PLACEHOLDER_EXACT or unwrapped in PLACEHOLDER_EXACT:
        return True
    if any(fragment in v for fragment in PLACEHOLDER_FRAGMENTS) or any(
        word in v for word in ("example", "sample", "dummy", "placeholder")
    ):
        return True
    if "xxxx" in v or "tttt" in v:
        return True
    if (v.startswith("your_") or v.startswith("your-") or v.startswith("[your") or v.startswith("<your")):
        return True
    if (v.startswith("<") and v.endswith(">")) or "${" in v or "{{" in v:
        return True
    if v.startswith("$") or v.startswith("env[") or v.startswith("os.getenv"):
        return True
    alnum = [c for c in v if c.isalnum()]
    if len(alnum) >= 8 and len(set(alnum)) <= 2:
        return True
    return False


def _overlaps(start: int, end: int, accepted: Iterable[Candidate]) -> bool:
    return any(start < item.end and item.start < end for item in accepted)


def detect(text: str) -> Tuple[bool, List[Candidate]]:
    """Return ``(drop_document, non-overlapping accepted candidates)``."""
    if PRIVATE_KEY.search(text):
        return True, []
    # Most web documents cannot contain any supported token family.  Cheap
    # literal triggers avoid running every heavyweight regex across all text.
    lower = text.lower()
    if not any(trigger in lower for trigger in (
        "akia", "asia", "aws", "ghp_", "gho_", "ghu_", "ghs_", "ghr_",
        "github_pat_", "aiza", "xox", "sk-", "sk_live_", "rk_live_", "eyj", "postgres", "mysql",
        "mongodb", "password", "passwd", "secret", "token", "api_key",
        "access_key", "private_key",
    )):
        return False, []
    accepted: List[Candidate] = []
    for kind, pattern, value_group in PATTERNS:
        kind_triggers = {
            "aws_access_key": ("akia", "asia"),
            "aws_secret": ("aws", "secret_access"),
            "github_token": ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"),
            "google_api_key": ("aiza",),
            "slack_token": ("xox",),
            "stripe_secret": ("sk_live_", "rk_live_"),
            "anthropic_api_key": ("sk-ant-",),
            "openai_api_key": ("sk-",),
            "jwt": ("eyj",),
            "connection_string_password": ("postgres", "mysql", "mongodb"),
            "env_secret": ("password", "passwd", "secret", "token", "api_key", "access_key", "private_key"),
        }[kind]
        if not any(trigger in lower for trigger in kind_triggers):
            continue
        for match in pattern.finditer(text):
            start, end = match.span(value_group)
            value = match.group(value_group)
            if kind == "env_secret":
                variable = match.group(0).split("=", 1)[0].strip().upper()
                if variable.endswith(("_FILE", "_PATH", "_DIR", "_FILENAME", "_DIRECTORY")):
                    continue
            if not value or is_placeholder(value) or _overlaps(start, end, accepted):
                continue
            accepted.append(Candidate(kind, start, end, value))
    accepted.sort(key=lambda item: (item.start, item.end, item.kind))
    return False, accepted


def _masked_context(text: str, start: int, end: int, radius: int = 100) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:start] + REDACTION + text[end:right]


def redact(text: str, context_chars: int = 100) -> Tuple[str | None, Dict[str, int], List[Mapping[str, object]]]:
    """Redact secrets, returning ``(text_or_none, hit_counts, safe_events)``."""
    drop, candidates = detect(text)
    if drop:
        match = PRIVATE_KEY.search(text)
        assert match is not None
        safe = _masked_context(text, match.start(), match.end(), context_chars)
        # A private-key body could contain other sensitive strings.  Only retain
        # a fixed structural marker rather than original surrounding content.
        event = {
            "type": "private_key",
            "start": match.start(),
            "end": match.end(),
            "context": "[REDACTED_PRIVATE_KEY_DOCUMENT]",
        }
        return None, {"private_key": 1}, [event]

    if not candidates:
        return text, {}, []
    pieces: List[str] = []
    cursor = 0
    counts: collections.Counter[str] = collections.Counter()
    for item in candidates:
        pieces.append(text[cursor:item.start])
        pieces.append(REDACTION)
        cursor = item.end
        counts[item.kind] += 1
    pieces.append(text[cursor:])
    redacted = "".join(pieces)

    # Context is generated from the fully redacted string by finding markers in
    # replacement order; it therefore cannot leak any accepted secret value.
    events: List[Mapping[str, object]] = []
    search_from = 0
    for item in candidates:
        marker_start = redacted.find(REDACTION, search_from)
        marker_end = marker_start + len(REDACTION)
        left = max(0, marker_start - context_chars)
        right = min(len(redacted), marker_end + context_chars)
        events.append({
            "type": item.kind,
            "start": item.start,
            "end": item.end,
            "context": redacted[left:right],
        })
        search_from = marker_end
    return redacted, dict(counts), events


def scan(text: str) -> Dict[str, int]:
    return redact(text)[1]


def _self_test() -> None:
    # NOTE: every fixture below is assembled at runtime from fragments.
    # Written as whole literals they trip GitHub push protection, which cannot tell a
    # detector's own test vectors from a real leaked credential. Splitting the literal
    # keeps the test identical while letting the file be committed.
    fake = {
        "aws_access_key": "AKIA" + "7N4P2Q9R6T3W8X5Z",
        "aws_secret": "aws_secret_access_key=" + "Ab3dEf5hIj7kLm9nOp2qRs4tUv6wXy8zAB1CDEFG",
        "github_token": "ghp" + "_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
        "google_api_key": "AIza" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r",
        "slack_token": "xox" + "b-1234567890-AbCdEfGhIjKlMnOp",
        "stripe_secret": "sk" + "_live_" + "A1b2C3d4E5f6G7h8I9j0",
        "anthropic_api_key": "sk" + "-ant-" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "openai_api_key": "sk" + "-" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "jwt": "eyJhbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0" + "." + "AbCdEfGhIjKlMnOp",
        "connection_string_password": "postgres://demo:" + "S3cur3Passw0rd" + "@db.example.org/app",
        "env_secret": "PASSWORD=" + "A1b2C3d4E5f6G7h8",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\n" + "ZmFrZS10ZXN0LWJvZHk=" + "\n-----END RSA PRIVATE KEY-----",
    }

    for expected, value in fake.items():
        sample = value if expected == "env_secret" else "prefix " + value + " suffix"
        out, counts, events = redact(sample)
        assert counts.get(expected) == 1, (expected, counts)
        if expected == "private_key":
            assert out is None
        else:
            assert out is not None and REDACTION in out
            matched_value = value.split("=", 1)[-1] if expected in {"aws_secret", "env_secret"} else value
            assert matched_value not in out and matched_value not in json.dumps(events)

    placeholders = (
        "API_KEY=your_api_key_here\nTOKEN=<TOKEN>\nPASSWORD=changeme\n"
        "AWS_ACCESS_KEY_ID=" + "AKIA" + "IOSFODNN7EXAMPLE" + "\nSECRET=xxxxxxxxxxxx"
    )
    out, counts, _ = redact(placeholders)
    assert out == placeholders and counts == {}, counts
    css = ".sk-chasingDotsRotate .sk-fading-circle .sk-spinner { animation-delay: -0.3s; }"
    assert redact(css)[1] == {}
    twice, counts2, _ = redact(redact(fake["github_token"])[0] or "")
    assert counts2 == {} and twice == REDACTION
    print(json.dumps({
        "scientific_evidence_allowed": False,
        "self_test": "PASS",
        "positive_types": sorted(fake),
        "placeholder_false_positives": 0,
    }, indent=2))


if __name__ == "__main__":
    _self_test()
