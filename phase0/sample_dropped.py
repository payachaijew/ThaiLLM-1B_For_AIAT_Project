#!/usr/bin/env python3
"""Save examples of the documents each cleaning rule threw away, so the filters can be
audited by eye.

clean_corpus.py COUNTED drops but did not keep them -- 11.9M documents were discarded with
no way to inspect them. This re-scans a few shards under the identical rules and writes
N examples per rule.

Especially aimed at spam_gambling, which removed 10.6 percent of the corpus by bytes
against an audit estimate of 3.25 percent. That gap needs a human to look at it.

scientific_evidence_allowed = false.
"""
from __future__ import annotations
import json, re, sys, gzip
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from heldout_rule import doc_hash, is_heldout
from clean_corpus import THAI, SPAM, MIN_CHARS, MAX_CHARS, MIN_THAI_RATIO

PER_RULE = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = HERE.parent / "data" / "dropped_samples"


def classify(t: str, seen: set):
    if is_heldout(t):                              return "heldout_bucket"
    n = len(t)
    if n < MIN_CHARS:                              return "too_short"
    if n > MAX_CHARS:                              return "too_long"
    if len(THAI.findall(t)) / max(n, 1) < MIN_THAI_RATIO: return "low_thai_ratio"
    for k, rx in SPAM.items():
        if rx.search(t):                           return f"spam_{k}"
    h = int(doc_hash(t)[:16], 16)
    if h in seen:                                  return "exact_duplicate"
    seen.add(h)
    return None


def main():
    import pyarrow.parquet as pq
    cache = Path.home() / ".cache/huggingface/hub/datasets--aisingapore--SEA-PILE-v2"
    shards = sorted(cache.rglob("*.parquet"))[:SHARDS]
    OUT.mkdir(parents=True, exist_ok=True)
    seen, buckets = set(), {}
    scanned = 0

    for sh in shards:
        for batch in pq.ParquetFile(sh).iter_batches(batch_size=4000):
            d = batch.to_pydict()
            for i in range(len(d["text"])):
                t = d["text"][i]; scanned += 1
                rule = classify(t, seen)
                if rule is None:
                    continue
                b = buckets.setdefault(rule, [])
                if len(b) < PER_RULE:
                    which = next((k for k, rx in SPAM.items() if rx.search(t)), None)
                    b.append({
                        "rule": rule,
                        "url": d["url"][i],
                        "dump": d["dump"][i],
                        "chars": len(t),
                        "matched_terms": sorted(set(SPAM[which].findall(t)))[:5] if which else None,
                        "text_head": t[:600],
                    })
            if all(len(buckets.get(r, [])) >= PER_RULE
                   for r in ["too_short", "spam_gambling", "spam_lottery", "exact_duplicate"]):
                break
        else:
            continue
        break

    idx = {}
    for rule, rows in sorted(buckets.items()):
        p = OUT / f"{rule}.jsonl"
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        idx[rule] = len(rows)
        print(f"  {rule:18s} {len(rows):4d} ตัวอย่าง -> {p.name}")

    (OUT / "INDEX.json").write_text(json.dumps({
        "scanned_documents": scanned,
        "shards_scanned": [s.name for s in shards],
        "examples_per_rule": idx,
        "purpose": "Eyeball audit of the cleaning filters. Each file holds up to "
                   f"{PER_RULE} documents that rule removed, with the matched spam terms.",
        "scientific_evidence_allowed": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"\n[+] scanned {scanned:,} docs -> {OUT}")


if __name__ == "__main__":
    main()
