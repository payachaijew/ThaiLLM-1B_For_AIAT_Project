#!/usr/bin/env python3
"""Phase 1 preview -- quality audit of the SEA-PILE-v2 Thai shards already on disk.

Answers: what is actually in this corpus, and how much cleaning does it still need
beyond what the dataset card says was already done?

scientific_evidence_allowed = false (corpus profiling, not a model result).
"""
from __future__ import annotations
import json, sys, collections, statistics, re, hashlib, datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout_rule import doc_hash

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 200000

# Thai-specific spam categories. The Mangosteen paper (arXiv:2507.14664) reports that
# English-centric pipelines miss exactly these, gambling above all.
PATTERNS = {
    "gambling":   re.compile(r"บาคาร่า|คาสิโน|สล็อต|แทงบอล|เว็บพนัน|ufabet|สมัครสมาชิก.{0,30}เครดิตฟรี|ฝากถอน.{0,20}ออโต้|เครดิตฟรี", re.I),
    "lottery":    re.compile(r"หวย|เลขเด็ด|ตรวจสลากกินแบ่ง|เลขท้าย.{0,10}ตัว"),
    "adult":      re.compile(r"หนังโป๊|คลิปหลุด|หีสวย|เย็ด|xxx\s*ไทย", re.I),
    "seo_spam":   re.compile(r"(รับทำ\s*seo|โปรโมทเว็บ|ซื้อขาย\s*backlink|รับจ้างเขียนบทความ)", re.I),
    "boilerplate":re.compile(r"(ข้ามไปยังเนื้อหา|เมนูหลัก|ลิขสิทธิ์.{0,20}สงวนไว้|คุกกี้.{0,30}ยอมรับ)"),
}
THAI = re.compile(r"[฀-๿]")


def main():
    import pyarrow.parquet as pq
    cache = Path.home() / ".cache/huggingface/hub/datasets--aisingapore--SEA-PILE-v2"
    shards = sorted(cache.rglob("*.parquet"))
    if not shards:
        sys.exit("no local shards found")
    print(f"[*] {len(shards)} shards on disk, sampling up to {SAMPLE:,} docs\n", flush=True)

    n = 0
    dumps, domains = collections.Counter(), collections.Counter()
    lens, thai_ratios = [], []
    flags = collections.Counter()
    flagged_bytes = collections.Counter()
    seen, dup, dup_bytes = set(), 0, 0
    total_bytes = 0

    for sh in shards:
        for batch in pq.ParquetFile(sh).iter_batches(batch_size=4000):
            d = batch.to_pydict()
            for i in range(len(d["text"])):
                t = d["text"][i]
                n += 1
                if n > SAMPLE:
                    break
                b = len(t.encode()); total_bytes += b
                lens.append(len(t))
                dumps[d["dump"][i]] += 1
                try: domains[urlparse(d["url"][i]).netloc.lower().lstrip("www.")] += 1
                except Exception: pass
                thai_ratios.append(len(THAI.findall(t)) / max(len(t), 1))
                for name, rx in PATTERNS.items():
                    if rx.search(t):
                        flags[name] += 1; flagged_bytes[name] += b
                h = doc_hash(t)
                if h in seen: dup += 1; dup_bytes += b
                else: seen.add(h)
            if n > SAMPLE: break
        if n > SAMPLE: break

    n = min(n, SAMPLE)
    lens.sort()
    p = lambda q: lens[int(q * len(lens))]
    rep = {
        "audit_id": "CORPUS-AUDIT-SEA-PILE-TH-2026-08-18",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "shards_on_disk": len(shards),
        "documents_sampled": n,
        "total_utf8_bytes": total_bytes,
        "length_chars": {"mean": round(statistics.mean(lens)), "p10": p(.10), "median": p(.50),
                         "p90": p(.90), "p99": p(.99), "max": lens[-1]},
        "short_doc_rate_under_500_chars": round(sum(1 for x in lens if x < 500) / n, 4),
        "thai_script_ratio": {"mean": round(statistics.mean(thai_ratios), 4),
                              "under_0.5": round(sum(1 for r in thai_ratios if r < .5) / n, 4)},
        "exact_duplicate_rate": round(dup / n, 4),
        "exact_duplicate_bytes_share": round(dup_bytes / total_bytes, 4),
        "crawl_dumps": dict(dumps.most_common(10)),
        "top_domains": dict(domains.most_common(20)),
        "content_flags_doc_rate": {k: round(v / n, 4) for k, v in flags.items()},
        "content_flags_byte_share": {k: round(flagged_bytes[k] / total_bytes, 4) for k in flags},
        "notes": [
            "Dataset card states dedup was applied WITHIN each snapshot only.",
            "Regex flags are recall-oriented screens, not classifiers. They overcount "
            "(a news article about gambling law matches 'gambling') and undercount obfuscated spam. "
            "Treat as a lower bound on the cleaning still needed, not as a filter.",
        ],
    }
    out = Path(__file__).resolve().parent / "corpus_audit_sea_pile_th.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n")

    print(f"docs sampled     : {n:,}   ({total_bytes/1e6:.0f} MB)")
    print(f"length chars     : p10={p(.10)} median={p(.50)} p90={p(.90)} p99={p(.99)} max={lens[-1]:,}")
    print(f"under 500 chars  : {rep['short_doc_rate_under_500_chars']*100:.1f}%")
    print(f"thai ratio <0.5  : {rep['thai_script_ratio']['under_0.5']*100:.1f}%")
    print(f"exact duplicates : {rep['exact_duplicate_rate']*100:.1f}% of docs, "
          f"{rep['exact_duplicate_bytes_share']*100:.1f}% of bytes")
    print("\ncontent flags (doc rate / byte share):")
    for k in PATTERNS:
        print(f"  {k:12s} {rep['content_flags_doc_rate'].get(k,0)*100:6.2f}%  /  "
              f"{rep['content_flags_byte_share'].get(k,0)*100:6.2f}%")
    print("\ntop 12 domains:")
    for dom, c in list(domains.most_common(12)):
        print(f"  {c:7,}  {dom}")
    print("\ncrawl dumps:", ", ".join(sorted(dumps)[:6]), "...")
    print(f"\n[+] wrote {out}")


if __name__ == "__main__":
    main()
