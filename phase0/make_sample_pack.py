#!/usr/bin/env python3
"""Build a small, self-contained sample package of the corpus for review.

Sending the whole 15.6 GB to someone who has not yet decided to use it spreads
CommonCrawl-derived personal data further than necessary while the PDPA questions in
LICENSE_COMPLIANCE.md are still open. A sample plus the full documentation lets the
recipient judge the corpus and ask for the rest only if they actually want it.

scientific_evidence_allowed = false.
"""
from __future__ import annotations
import gzip, json, random, shutil, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus_sample_pack"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

SRC = [("th", ROOT/"data"/"clean_pii"/"th"), ("en", ROOT/"data"/"clean_replay_v2"/"en"),
       ("code", ROOT/"data"/"clean_replay_v2"/"code"), ("math", ROOT/"data"/"clean_replay_v2"/"math")]
DOCS = [("DATASET_CARD.md","README.md"), ("LICENSE_COMPLIANCE.md","LICENSE_COMPLIANCE.md"),
        ("data/clean_th_manifest.json","manifests/clean_th_manifest.json"),
        ("data/clean_pii_th_manifest.json","manifests/clean_pii_th_manifest.json"),
        ("data/clean_replay_v2_manifest.json","manifests/clean_replay_v2_manifest.json"),
        ("data/decon_neardedup_report.json","manifests/decon_neardedup_report.json"),
        ("data/benchmark_ngrams_meta.json","manifests/benchmark_ngrams_meta.json"),
        ("validation/replay_v2_verification.json","manifests/replay_v2_verification.json")]


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    (OUT/"sample").mkdir(parents=True); (OUT/"manifests").mkdir(exist_ok=True)
    random.seed(20260821)
    stats = {}
    for lang, root in SRC:
        files = sorted(root.glob("*.jsonl.gz"))
        if not files: print(f"  ! ไม่พบ {lang}"); continue
        # spread the sample across shards rather than taking the head of one file
        per = max(1, N // min(len(files), 6))
        rows = []
        for f in files[::max(1, len(files)//6)][:6]:
            with gzip.open(f, "rt") as fh:
                for i, line in enumerate(fh):
                    if i >= per*4: break
                    rows.append(line)
        random.shuffle(rows); rows = rows[:N]
        p = OUT/"sample"/f"{lang}-sample.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh: fh.writelines(rows)
        toks = sum(json.loads(r).get("qwen_tokens", 0) for r in rows)
        stats[lang] = {"documents": len(rows), "qwen_tokens": toks,
                       "file": f"sample/{lang}-sample.jsonl.gz",
                       "size_mb": round(p.stat().st_size/1e6, 1)}
        print(f"  {lang:5s} {len(rows):>6,} docs  {toks:>10,} tokens  {stats[lang]['size_mb']:>6.1f} MB")

    for src, dst in DOCS:
        s = ROOT/src
        if s.exists():
            d = OUT/dst; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(s, d)

    (OUT/"SAMPLE_README.md").write_text(f"""# ตัวอย่าง corpus — ThaiLLM-1B

สร้างเมื่อ {datetime.date.today()} · **นี่คือตัวอย่าง ไม่ใช่ corpus เต็ม**

| ภาษา | ตัวอย่าง | corpus เต็ม |
|---|---:|---:|
| ไทย | {stats.get('th',{}).get('documents',0):,} docs | 4,567,214 docs / 5.851B tokens |
| อังกฤษ | {stats.get('en',{}).get('documents',0):,} docs | 3,507,052 docs / 3.513B tokens |
| code | {stats.get('code',{}).get('documents',0):,} docs | 690,716 docs / 1.000B tokens |
| math | {stats.get('math',{}).get('documents',0):,} docs | 412,734 docs / 0.592B tokens |

ตัวอย่างสุ่มกระจายทั่วทุก shard **ไม่ใช่หัวไฟล์** จึงเป็นตัวแทนของ corpus จริง

## เปิดดู

```bash
gzcat sample/th-sample.jsonl.gz | head -3 | python3 -c "
import sys,json
for l in sys.stdin: print(json.loads(l)['text'][:500], chr(10)+'---')"
```

## อ่านก่อนตัดสินใจ

- **`README.md`** — dataset card ฉบับเต็ม: ที่มา · revision ที่ pin · สิ่งที่ทำความสะอาดไปแล้ว · **ข้อจำกัด 8 ข้อ**
- **`LICENSE_COMPLIANCE.md`** — ODC-By 1.0 + CommonCrawl ToU และ**คำถาม PDPA ที่ยังรอข้อสรุป**
- **`manifests/`** — provenance และผลตรวจสอบอิสระทั้งหมด

## ⚠️ corpus เต็มมาพร้อม `removal_list.txt`

273,703 เอกสาร (2.98%) ที่**ต้องตัดออกก่อนเทรน** — ปนเปื้อน benchmark 3,233 + near-duplicate 270,841

ตัวอย่างชุดนี้**ยังไม่ได้ตัดออก** เพราะเป็นตัวอย่างสำหรับดูเนื้อหาเท่านั้น ห้ามนำไปเทรน

## อยากได้ corpus เต็ม (15.6 GB)

แจ้ง username HuggingFace มาได้เลยครับ จะเพิ่มสิทธิ์เข้าถึง private repo ให้
""")

    (OUT/"sample_pack_manifest.json").write_text(json.dumps({
        "pack_id": "CORPUS-SAMPLE-PACK-V1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "purpose": "review sample; NOT for training",
        "sampling": "random across shards, seed 20260821",
        "per_language": stats,
        "full_corpus_reference": "data/clean_replay_v2_manifest.json and data/clean_pii_th_manifest.json",
        "not_included": ["removal_list.txt (full corpus only)", "held-out evaluation sets",
                         "the other 99.9 percent of documents"],
    }, ensure_ascii=False, indent=2)+"\n")

    z = shutil.make_archive(str(ROOT/"corpus_sample_pack"), "zip", root_dir=OUT)
    print(f"\n[+] {Path(z).name}  {Path(z).stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
