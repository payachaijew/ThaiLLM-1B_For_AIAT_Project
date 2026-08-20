#!/usr/bin/env python3
"""Phase 1 -- clean the SEA-PILE-v2 Thai corpus.

Applies, in order:
  1. HELDOUT-BUCKET-V1 exclusion   (evaluation documents can never enter training)
  2. exact deduplication            (measured 21.0% of documents in the audit)
  3. length floor                   (59.4% of documents are under 500 chars)
  4. Thai script ratio floor        (1.0% of documents fail this)
  5. Thai gambling / lottery filter (~5% of bytes; the Mangosteen paper reports
                                     English-centric pipelines miss exactly this)

Writes gzipped JSONL shards plus a manifest with per-filter accounting, so every
document dropped can be attributed to a named rule.

scientific_evidence_allowed = false (data preparation, not a result).
"""
from __future__ import annotations
import json, gzip, re, sys, time, datetime, hashlib, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from heldout_rule import doc_hash, is_heldout, RULE_ID

OUT = HERE.parent / "data" / "clean" / "th"
MIN_CHARS, MAX_CHARS = 500, 100000
MIN_THAI_RATIO = 0.5
DOCS_PER_OUT_SHARD = 200000

THAI = re.compile(r"[฀-๿]")
SPAM = {
    "gambling": re.compile(r"บาคาร่า|คาสิโน|สล็อต|แทงบอล|เว็บพนัน|ufabet|สมัครสมาชิก.{0,30}เครดิตฟรี|ฝากถอน.{0,20}ออโต้|เครดิตฟรี", re.I),
    "lottery":  re.compile(r"หวย|เลขเด็ด|ตรวจสลากกินแบ่ง|เลขท้าย.{0,10}ตัว"),
    "adult":    re.compile(r"หนังโป๊|คลิปหลุด|หีสวย|xxx\s*ไทย", re.I),
}


def main():
    import pyarrow.parquet as pq
    cache = Path.home() / ".cache/huggingface/hub/datasets--aisingapore--SEA-PILE-v2"
    shards = sorted(cache.rglob("*.parquet"))
    if not shards:
        sys.exit("no shards on disk")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[*] {len(shards)} shards -> {OUT}", flush=True)

    seen = set()
    drop = collections.Counter()
    drop_bytes = collections.Counter()
    kept = kept_bytes = seen_docs = 0
    out_idx, buf = 0, []
    t0 = time.time()

    def flush():
        nonlocal out_idx, buf
        if not buf:
            return
        p = OUT / f"clean-{out_idx:05d}.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            for r in buf:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"    wrote {p.name}  ({len(buf):,} docs)", flush=True)
        out_idx += 1; buf = []

    for si, sh in enumerate(shards):
        for batch in pq.ParquetFile(sh).iter_batches(batch_size=4000):
            d = batch.to_pydict()
            for i in range(len(d["text"])):
                t = d["text"][i]
                seen_docs += 1
                b = len(t.encode())

                if is_heldout(t):
                    drop["heldout_bucket"] += 1; drop_bytes["heldout_bucket"] += b; continue
                n = len(t)
                if n < MIN_CHARS:
                    drop["too_short"] += 1; drop_bytes["too_short"] += b; continue
                if n > MAX_CHARS:
                    drop["too_long"] += 1; drop_bytes["too_long"] += b; continue
                if len(THAI.findall(t)) / n < MIN_THAI_RATIO:
                    drop["low_thai_ratio"] += 1; drop_bytes["low_thai_ratio"] += b; continue
                hit = next((k for k, rx in SPAM.items() if rx.search(t)), None)
                if hit:
                    drop[f"spam_{hit}"] += 1; drop_bytes[f"spam_{hit}"] += b; continue
                h = doc_hash(t)
                key = int(h[:16], 16)
                if key in seen:
                    drop["exact_duplicate"] += 1; drop_bytes["exact_duplicate"] += b; continue
                seen.add(key)

                kept += 1; kept_bytes += b
                buf.append({"text": t, "doc_sha256": h, "url": d["url"][i],
                            "dump": d["dump"][i], "utf8_bytes": b})
                if len(buf) >= DOCS_PER_OUT_SHARD:
                    flush()
        el = time.time() - t0
        print(f"[{si+1}/{len(shards)}] seen={seen_docs:,} kept={kept:,} "
              f"({100*kept/seen_docs:.1f}%) {kept_bytes/1e9:.2f} GB  "
              f"{el/60:.0f} min  eta {(len(shards)-si-1)*el/(si+1)/60:.0f} min", flush=True)
    flush()

    QWEN_BPT = 5.0468
    man = {
        "manifest_id": "CLEAN-SEA-PILE-TH-V1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "source": {"repo": "aisingapore/SEA-PILE-v2", "subset": "th",
                   "revision": "77573cc84631412a781daa8e6f72cf322d4207f0", "shards": len(shards)},
        "exclusion_rule": RULE_ID,
        "filters": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
                    "min_thai_ratio": MIN_THAI_RATIO, "spam_patterns": sorted(SPAM)},
        "input": {"documents": seen_docs},
        "output": {"documents": kept, "utf8_bytes": kept_bytes,
                   "shards": out_idx, "retention_rate_docs": round(kept / seen_docs, 4)},
        "dropped_documents": dict(drop),
        "dropped_bytes": dict(drop_bytes),
        "estimated_qwen_tokens": round(kept_bytes / QWEN_BPT / 1e9, 3),
        "limitations": [
            "Exact dedup only. Near-duplicate (MinHash/SimHash) dedup is NOT applied.",
            "Spam filters are recall-oriented regexes, not classifiers. They over-drop "
            "legitimate news about gambling and under-drop obfuscated spam.",
            "No PII filter. No benchmark decontamination. Both are still required.",
            "Token estimate uses the measured Qwen Thai bytes/token of 5.0468; recount exactly "
            "after tokenisation.",
        ],
    }
    (OUT.parent.parent / "clean_th_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")

    print(f"\n=== DONE in {(time.time()-t0)/60:.0f} min ===")
    print(f"in  {seen_docs:,} docs")
    print(f"out {kept:,} docs ({100*kept/seen_docs:.1f}%)  {kept_bytes/1e9:.2f} GB  "
          f"~{man['estimated_qwen_tokens']}B Qwen tokens")
    print("\ndropped (docs / GB):")
    for k in sorted(drop, key=lambda x: -drop_bytes[x]):
        print(f"  {k:18s} {drop[k]:10,}  {drop_bytes[k]/1e9:6.2f} GB")


if __name__ == "__main__":
    main()
