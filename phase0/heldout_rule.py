#!/usr/bin/env python3
"""THE anti-contamination rule. Import this from BOTH the held-out builder and
the training data pipeline. If the two ever use different logic, every retention
and acquisition number in the project becomes untrustworthy.

Design: a deterministic hash BUCKET, not a list of ids.

  * A list only excludes documents we happened to materialise. Near-duplicates
    and unscanned shards leak straight into training.
  * A bucket rule is stateless, scales to any corpus size, needs no lookup table,
    and excludes the whole bucket even from shards the builder never opened.

Cost: HELDOUT_BUCKETS/TOTAL_BUCKETS of the corpus is withheld from training.
At 1% of a 6.5B-token Thai corpus that is ~65M tokens -- negligible.
"""
from __future__ import annotations
import hashlib, unicodedata

TOTAL_BUCKETS = 100
HELDOUT_BUCKETS = {0}
RULE_ID = "HELDOUT-BUCKET-V1"


def normalise(text: str) -> str:
    """NFC + whitespace collapse, so trivial formatting differences cannot move a
    document between buckets."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def doc_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def bucket(text: str) -> int:
    return int(doc_hash(text)[:8], 16) % TOTAL_BUCKETS


def is_heldout(text: str) -> bool:
    """True  -> evaluation only, MUST NOT be trained on.
       False -> eligible for training."""
    return bucket(text) in HELDOUT_BUCKETS


def is_trainable(text: str) -> bool:
    return not is_heldout(text)


if __name__ == "__main__":
    import json
    print(json.dumps({
        "rule_id": RULE_ID,
        "total_buckets": TOTAL_BUCKETS,
        "heldout_buckets": sorted(HELDOUT_BUCKETS),
        "heldout_fraction": len(HELDOUT_BUCKETS) / TOTAL_BUCKETS,
        "normalisation": "NFC + whitespace collapse",
        "hash": "sha256, first 8 hex chars mod total_buckets",
    }, indent=2))
