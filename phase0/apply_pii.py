#!/usr/bin/env python3
"""Second pass: redact PII across the cleaned Thai corpus.

Runs over data/clean/th/*.jsonl.gz rather than re-reading the raw parquet, so it costs
about 25 minutes instead of the ~6 hours a full re-clean would take.

Redacts rather than drops: a document containing one phone number is still good Thai
prose. Replacing the number keeps the language and removes the personal data.

scientific_evidence_allowed = false (data preparation).
"""
from __future__ import annotations
import gzip, json, sys, time, datetime, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pii_filter import redact

SRC = HERE.parent / "data" / "clean" / "th"
DST = HERE.parent / "data" / "clean_pii" / "th"


def main():
    shards = sorted(SRC.glob("*.jsonl.gz"))
    if not shards:
        sys.exit(f"no input shards in {SRC}")
    DST.mkdir(parents=True, exist_ok=True)
    print(f"[*] {len(shards)} shards  {SRC} -> {DST}", flush=True)

    hits = collections.Counter()
    docs = docs_touched = 0
    bytes_in = bytes_out = 0
    t0 = time.time()

    for si, sp in enumerate(shards):
        out = DST / sp.name
        with gzip.open(sp, "rt", encoding="utf-8") as fin, \
             gzip.open(out, "wt", encoding="utf-8") as fout:
            for line in fin:
                r = json.loads(line)
                t = r["text"]
                bytes_in += len(t.encode())
                new, c = redact(t)
                docs += 1
                if c:
                    docs_touched += 1
                    hits.update(c)
                    r["text"] = new
                    r["pii_redacted"] = c
                    r["utf8_bytes"] = len(new.encode())
                bytes_out += len(r["text"].encode())
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        el = time.time() - t0
        print(f"[{si+1}/{len(shards)}] docs={docs:,} touched={docs_touched:,} "
              f"({100*docs_touched/docs:.2f}%)  {el/60:.1f} min  "
              f"eta {(len(shards)-si-1)*el/(si+1)/60:.1f} min", flush=True)

    man = {
        "manifest_id": "CLEAN-SEA-PILE-TH-PII-V1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "input": {"path": "data/clean/th", "manifest": "CLEAN-SEA-PILE-TH-V1"},
        "output": {"path": "data/clean_pii/th", "shards": len(shards), "documents": docs,
                   "utf8_bytes": bytes_out},
        "policy": "redact_in_place_not_drop",
        "documents_touched": docs_touched,
        "documents_touched_rate": round(docs_touched / docs, 5),
        "redactions_by_type": dict(hits),
        "bytes_removed": bytes_in - bytes_out,
        "detector_fixes_applied": [
            "EMAIL: \\b word boundaries replaced with lookarounds. Thai letters are word "
            "characters under Python Unicode \\w, so Thai-adjacent emails never matched. "
            "The pre-fix detector found ZERO emails in 40,000 documents containing 809 '@'.",
            "CREDIT: now requires a payment context cue. A bare Luhn-valid 13-19 digit run "
            "false-positived on reference numbers such as '166617 106017 106617'.",
            "THAI_ID: validated with the official checksum before redaction.",
        ],
        "limitations": [
            "Regex detectors, not a trained NER model. Names, addresses and dates of birth "
            "are NOT detected and remain in the corpus.",
            "Line IDs are detected by a keyword cue and will miss bare handles.",
            "No manual review of any document was performed.",
        ],
    }
    (DST.parent.parent / "clean_pii_th_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n")

    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")
    print(f"docs {docs:,}  touched {docs_touched:,} ({100*docs_touched/docs:.2f}%)")
    print(f"bytes removed: {bytes_in-bytes_out:,}")
    for k, v in hits.most_common():
        print(f"  {k:20s} {v:9,}")


if __name__ == "__main__":
    main()
