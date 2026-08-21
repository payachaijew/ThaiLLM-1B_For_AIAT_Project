#!/usr/bin/env python3
"""High-precision English/international PII redaction for replay data.

Unlike ``pii_filter.py`` this module does not contain Thai-specific identifiers.
Phone, SSN and payment-card candidates use structural/context guards to avoid
code versions, timestamps and arbitrary number sequences.

This is a data-preparation utility, not scientific evidence.
"""
from __future__ import annotations

import collections
import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

SCIENTIFIC_EVIDENCE_ALLOWED = False


@dataclass(frozen=True)
class Candidate:
    kind: str
    start: int
    end: int
    replacement: str


EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
    r"\.[A-Za-z]{2,24}(?![A-Za-z0-9.-])"
)
SSN = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4})(?!\d)")
CARD = re.compile(r"(?<!\d)((?:\d[ -]?){13,19})(?!\d)")
PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])"
    r"\d{2,4}[\s.-]\d{3,4}(?!\w)"
)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!(?:\d|\.\d))")

SSN_CUE = re.compile(r"(?i)(?:\bssn\b|\bsocial\s+security(?:\s+number)?\b|\btaxpayer\s+id\b)")
CARD_CUE = re.compile(
    r"(?i)(?:credit\s*card|debit\s*card|card\s*(?:number|no\.?|#)|visa|mastercard|amex|payment\s*card)"
)
PHONE_CUE = re.compile(r"(?i)\b(?:phone|telephone|tel|mobile|cell|call|contact|fax)\b")
IP_CUE = re.compile(
    r"(?i)(?:\bip(?:v4)?\b|\bip[_-]?address\b|\bremote[_-]?ip\b|"
    r"\b(?:host|server|endpoint|socket|bind|listen|dns)\b|(?:https?|tcp|udp)://)"
)


def _luhn(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _near_cue(text: str, start: int, end: int, cue: re.Pattern[str], radius: int = 48) -> bool:
    return bool(cue.search(text[max(0, start - radius):min(len(text), end + radius)]))


def _valid_phone(text: str, match: re.Match[str]) -> bool:
    value = match.group(0)
    digits = re.sub(r"\D", "", value)
    if not 10 <= len(digits) <= 15 or len(set(digits)) <= 2:
        return False
    # A leading plus, a parenthesised area code or a nearby phone cue is
    # required. Bare grouped numbers in math/code are otherwise far too noisy.
    if value.lstrip().startswith("+") or "(" in value:
        return True
    return _near_cue(text, match.start(), match.end(), PHONE_CUE, radius=24)


def _valid_public_ip_context(text: str, start: int, end: int) -> bool:
    """Require syntax/cue immediately attached to the dotted quad.

    A document-level network word is insufficient: standards and source code
    frequently contain section/version numbers such as 19.2.3.2 near words
    like ``bind`` or ``host``.
    """
    left = text[max(0, start - 64):start]
    right = text[end:min(len(text), end + 16)]
    if re.search(r"(?i)(?:https?|tcp|udp)://(?:[^/@\s]+(?::[^/@\s]*)?@)?$", left):
        return True
    if re.search(
        r"(?i)(?:\bip(?:v4)?(?:[_ -]?address(?:es)?)?|\bremote[_-]?ip|"
        r"\bhost|\bserver|\bendpoint|\blisten|\bdns|\bbind)"
        r"(?:\s+(?:is|of|to))?\s*(?:[:=]|=>)?\s*['\"]?$",
        left,
    ):
        return True
    return bool(re.match(r"(?:/\d{1,2}|:\d{2,5})(?!\d)", right))


def _overlaps(item: Candidate, accepted: Iterable[Candidate]) -> bool:
    return any(item.start < other.end and other.start < item.end for other in accepted)


def detect(text: str) -> List[Candidate]:
    candidates: List[Candidate] = []
    lower = text.lower()

    # Specific contextual identifiers precede the more permissive phone rule.
    if any(cue in lower for cue in ("ssn", "social security", "taxpayer id")):
        for match in SSN.finditer(text):
            if _near_cue(text, match.start(1), match.end(1), SSN_CUE):
                candidates.append(Candidate("us_ssn", match.start(1), match.end(1), "[SSN]"))
    if any(cue in lower for cue in ("credit card", "creditcard", "debit card", "card number", "card no", "visa", "mastercard", "amex", "payment card")):
        for match in CARD.finditer(text):
            if _near_cue(text, match.start(1), match.end(1), CARD_CUE) and _luhn(match.group(1)):
                item = Candidate("credit_card", match.start(1), match.end(1), "[CARD]")
                if not _overlaps(item, candidates):
                    candidates.append(item)
    if "@" in text:
        for match in EMAIL.finditer(text):
            item = Candidate("email", match.start(), match.end(), "[EMAIL]")
            if not _overlaps(item, candidates):
                candidates.append(item)
    network_trigger = any(cue in lower for cue in (
        "http://", "https://", "ip_address", "ip-address", "remote_ip", "remote-ip",
        "ipv4", " ip ", "host", "server", "endpoint", "socket", "bind", "listen", "dns",
    ))
    if "." in text and network_trigger:
        for match in IPV4.finditer(text):
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            immediate_left = text[max(0, match.start()-16):match.start()]
            has_network_shape = _valid_public_ip_context(text, match.start(), match.end())
            looks_like_version = bool(re.search(r"(?i)(?:version|ver\.?|release|section)\s*$", immediate_left))
            if not (address.version == 4 and address.is_global and has_network_shape and not looks_like_version):
                continue
            item = Candidate("public_ip", match.start(), match.end(), "[PUBLIC_IP]")
            if not _overlaps(item, candidates):
                candidates.append(item)
    phone_trigger = any(cue in lower for cue in ("phone", "telephone", "tel:", "tel.", "mobile", "cell", "call", "contact", "fax"))
    if "+" in text or "(" in text or phone_trigger:
        for match in PHONE.finditer(text):
            if _valid_phone(text, match):
                item = Candidate("phone", match.start(), match.end(), "[PHONE]")
                if not _overlaps(item, candidates):
                    candidates.append(item)
    return sorted(candidates, key=lambda item: (item.start, item.end, item.kind))


def redact(text: str, context_chars: int = 100) -> Tuple[str, Dict[str, int], List[Mapping[str, object]]]:
    matches = detect(text)
    if not matches:
        return text, {}, []
    pieces: List[str] = []
    cursor = 0
    counts: collections.Counter[str] = collections.Counter()
    for item in matches:
        pieces.append(text[cursor:item.start])
        pieces.append(item.replacement)
        cursor = item.end
        counts[item.kind] += 1
    pieces.append(text[cursor:])
    redacted = "".join(pieces)

    events: List[Mapping[str, object]] = []
    search_from = 0
    for item in matches:
        marker_start = redacted.find(item.replacement, search_from)
        marker_end = marker_start + len(item.replacement)
        events.append({
            "type": item.kind,
            "start": item.start,
            "end": item.end,
            "context": redacted[max(0, marker_start-context_chars):min(len(redacted), marker_end+context_chars)],
        })
        search_from = marker_end
    return redacted, dict(counts), events


def scan(text: str) -> Dict[str, int]:
    return redact(text)[1]


def _self_test() -> None:
    positives = {
        "email": "Contact Ada.Lovelace+lab@example.co.uk today.",
        "phone": "Telephone: +44 20 7946 0958.",
        "us_ssn": "Social Security Number: 123-45-6789.",
        "credit_card": "Credit card number: 4111 1111 1111 1111.",
        "public_ip": "Production endpoint is 8.8.8.8.",
    }
    for expected, sample in positives.items():
        out, counts, events = redact(sample)
        assert counts.get(expected) == 1, (expected, counts)
        assert sample != out and events

    negatives = (
        "version 1.2.3 and timestamp 2026-08-21 12:30:45",
        "private hosts 10.0.0.1 172.16.4.2 192.168.1.1 localhost 127.0.0.1",
        "build id 123-45-6789 without an identity cue",
        "reference 166617 106017 106617 and 4111111111111111 without a card cue",
        "semver 212-555-121 is not a complete telephone number",
    )
    for sample in negatives:
        out, counts, _ = redact(sample)
        assert out == sample and counts == {}, (sample, counts)
    twice, counts, _ = redact(redact(positives["email"])[0])
    assert counts == {} and "[EMAIL]" in twice
    print(json.dumps({
        "scientific_evidence_allowed": False,
        "self_test": "PASS",
        "positive_types": sorted(positives),
        "negative_false_positives": 0,
    }, indent=2))


if __name__ == "__main__":
    _self_test()
