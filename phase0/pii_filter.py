#!/usr/bin/env python3
"""PII detection and redaction for Thai web text.

Design choice: REDACT, do not drop.
A document containing one phone number is still perfectly good Thai prose. Dropping it
throws away language for the sake of one token. Replacing the number with a placeholder
removes the personal data and keeps the sentence structure the model needs to learn.
This follows what Dolma and the BigScience pipeline do.

Thai national ID numbers are validated with their checksum before redaction, so ordinary
13-digit numbers are not destroyed.

scientific_evidence_allowed = false (data preparation).
"""
from __future__ import annotations
import re, sys, json, unicodedata

# ---------------------------------------------------------------- patterns
# NOTE: \b must NOT be used here. Thai characters are word characters under Python's
# Unicode \w, so a Thai letter directly followed by an ASCII letter is a word-to-word
# transition and \b does not match. An earlier version used \b and detected ZERO emails
# in 40,000 Thai documents that contained 809 '@' characters. Use explicit lookarounds.
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])")

# Thai mobile/landline. Requires a separator or exact length so that ordinary
# numbers (prices, years, ids) are not swept up.
PHONE = re.compile(r"(?<![\d])(?:\+66[\s-]?|0)(?:[689]\d{1}|[2-7])[\s-]?\d{3}[\s-]?\d{3,4}(?![\d])")

# 13 digits, optionally grouped as 1-2345-67890-12-3
NATID = re.compile(r"(?<![\d])(\d)[\s-]?(\d{4})[\s-]?(\d{5})[\s-]?(\d{2})[\s-]?(\d)(?![\d])")

# Thai bank accounts are 10-12 digits and usually appear next to a banking cue word.
BANK = re.compile(
    r"(?:บัญชี|เลขที่บัญชี|โอนเข้า|พร้อมเพย์|กสิกร|ไทยพาณิชย์|กรุงเทพ|กรุงไทย|ธนาคาร)"
    r"[^\d]{0,20}(?<![\d])(\d[\s-]?){10,12}(?![\d])")

# A bare 13-19 digit run is NOT enough: roughly one in ten random digit strings passes
# Luhn, and Thai web text is full of long reference and product numbers. Measured false
# positives included "166617 106017 106617" and "1316423283236106241". Require a payment
# context cue within the preceding 30 characters.
CREDIT = re.compile(
    r"(?:บัตรเครดิต|บัตรเดบิต|เลขบัตร|credit\s*card|visa|mastercard)[^\d]{0,30}"
    r"(?<![\d])((?:\d[ -]?){13,19})(?![\d])", re.I)

LINE_ID = re.compile(r"(?:ไลน์|line)\s*(?:id|ไอดี)?\s*[:：]?\s*@?([A-Za-z0-9._-]{4,20})", re.I)

URL_USER = re.compile(r"(?:facebook\.com|fb\.com|instagram\.com|twitter\.com|x\.com)/([A-Za-z0-9._-]{3,})", re.I)


def _luhn(num: str) -> bool:
    d = [int(c) for c in num][::-1]
    s = 0
    for i, x in enumerate(d):
        if i % 2:
            x *= 2
            if x > 9:
                x -= 9
        s += x
    return s % 10 == 0


def _thai_id_valid(digits: str) -> bool:
    """Thai national ID checksum: sum(d[i] * (13-i)) for i in 0..11, then (11 - sum%11) % 10."""
    if len(digits) != 13 or not digits.isdigit():
        return False
    if digits[0] == "0":
        return False
    total = sum(int(digits[i]) * (13 - i) for i in range(12))
    return (11 - total % 11) % 10 == int(digits[12])


def redact(text: str) -> tuple[str, dict]:
    """Return (redacted_text, counts_by_type)."""
    counts = {}

    def bump(k, n=1):
        counts[k] = counts.get(k, 0) + n

    def sub(rx, repl, key, guard=None):
        nonlocal text
        def f(m):
            if guard and not guard(m):
                return m.group(0)
            bump(key)
            return repl
        text = rx.sub(f, text)

    sub(EMAIL, "[EMAIL]", "email")
    sub(NATID, "[THAI_ID]", "thai_national_id",
        guard=lambda m: _thai_id_valid("".join(m.groups())))
    sub(CREDIT, "[CARD]", "credit_card",
        guard=lambda m: _luhn(re.sub(r"\D", "", m.group(1))) and 13 <= len(re.sub(r"\D", "", m.group(1))) <= 19)
    sub(BANK, "[BANK_ACCOUNT]", "bank_account")
    sub(PHONE, "[PHONE]", "phone")
    sub(LINE_ID, "[LINE_ID]", "line_id")
    sub(URL_USER, "[SOCIAL_HANDLE]", "social_handle")
    return text, counts


def scan(text: str) -> dict:
    """Count without modifying."""
    return redact(text)[1]


if __name__ == "__main__":
    demo = (
        "ติดต่อ สมชาย ได้ที่ somchai.test@example.com หรือโทร 081-234-5678 "
        "เลขบัตรประชาชน 1-1011-12345-52-1 โอนเข้าบัญชี กสิกร 1234567890 "
        "ไลน์ไอดี @somchai99 ราคา 1,250 บาท เมื่อปี 2567 ครับ"
    )
    out, c = redact(demo)
    print("IN :", demo)
    print("OUT:", out)
    print("HIT:", json.dumps(c, ensure_ascii=False))
    print("\nnote: ราคา 1,250 และปี 2567 ต้องไม่ถูกแตะ")
