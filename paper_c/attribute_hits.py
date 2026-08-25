#!/usr/bin/env python3
"""หาว่าเอกสารที่ถูกจับว่าปนเปื้อน ตรงกับ benchmark ชุดไหน

decon_and_neardedup.py สแกนกับ index ก้อนเดียวที่รวมทุก benchmark เข้าด้วยกัน จึงบอกได้แค่ว่า
"ตรงกี่จุด" ไม่ได้บอกว่า "ตรงกับอะไร" ปัญหาคือ 87% ของ index เป็นข้อสอบภาษาอังกฤษ
(HellaSwag 53% + MMLU 34%) ส่วนข้อสอบไทยทั้งห้าชุดรวมกันได้ 1.3%

เอกสารไทยที่ถูกจับ จึงอาจกำลังตรงกับข้อความอังกฤษที่ฝังอยู่ในหน้าเว็บไทย ไม่ใช่ข้อสอบไทยรั่ว
ข้ออ้างหลักของ paper ขึ้นอยู่กับความต่างนี้ทั้งหมด

  PYTHONHASHSEED=0 python3 attribute_hits.py --lang th
"""
from __future__ import annotations
import argparse, collections, glob, gzip, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase0"))
from build_benchmark_index import hashes, NGRAM, SPECS   # noqa: E402

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("FATAL: ต้องรันด้วย PYTHONHASHSEED=0 ไม่งั้น hash จะไม่ตรงกับ index เดิม")

STRIDE = 1          # เอกสารที่ต้องตรวจมีแค่หลักร้อย ตรวจทุกตำแหน่งได้
MAXCHARS = 40000    # ต้องตรงกับที่ decon_and_neardedup.py ใช้


def per_benchmark_sets():
    """สร้าง n-gram set แยกราย benchmark ไม่รวมเป็นก้อนเดียวเหมือน index เดิม"""
    import warnings; warnings.filterwarnings("ignore")
    from datasets import load_dataset
    out = {}
    for name, repo, kw, fn in SPECS:
        try:
            ds = load_dataset(repo, **kw)
        except Exception as e:
            print(f"  ! {name}: โหลดไม่ได้ ({type(e).__name__})", flush=True)
            continue
        hs = []
        for r in ds:
            try:
                t = fn(r) if fn else " ".join(
                    str(v) for v in r.values() if isinstance(v, str) and v.strip())
            except Exception:
                continue
            h = hashes(t)
            if h.size:
                hs.append(h)
        arr = np.unique(np.concatenate(hs)) if hs else np.empty(0, dtype=np.uint64)
        out[name] = arr
        print(f"  {name:18s} {arr.size:9,} n-grams", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="th")
    ap.add_argument("--hits", default=str(ROOT / "data/decontamination_hits.jsonl"))
    ap.add_argument("--corpus", default=None, help="ค่าเริ่มต้นเดาจาก --lang")
    ap.add_argument("--out", default=str(HERE / "attributed_hits.jsonl"))
    a = ap.parse_args()

    corpus = a.corpus or str(ROOT / f"data/clean_pii/{a.lang}") if a.lang == "th" \
        else str(ROOT / f"data/clean_replay_v2/{a.lang}")

    want = {}
    for line in open(a.hits):
        r = json.loads(line)
        if r["lang"] == a.lang:
            want[r["doc_sha256"]] = r
    print(f"[*] เอกสารที่ต้องตรวจ: {len(want):,}", flush=True)

    print("[*] สร้าง n-gram set แยกราย benchmark", flush=True)
    bench = per_benchmark_sets()

    print("[*] ไล่หาข้อความของเอกสารเหล่านั้นใน corpus", flush=True)
    found, rows = 0, []
    for f in sorted(glob.glob(f"{corpus}/*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                did = r.get("doc_sha256")
                if did not in want:
                    continue
                found += 1
                h = hashes(r["text"][:MAXCHARS], NGRAM, STRIDE)
                per = {}
                for name, arr in bench.items():
                    if not h.size or not arr.size:
                        continue
                    pos = np.searchsorted(arr, h); pos[pos >= arr.size] = 0
                    n = int(np.count_nonzero(arr[pos] == h))
                    if n:
                        per[name] = n
                rec = dict(want[did])
                rec["per_benchmark"] = per
                rec["top_benchmark"] = max(per, key=per.get) if per else None
                rec["total_attributed"] = sum(per.values())
                rows.append(rec)
                if found % 100 == 0:
                    print(f"    {found:,}/{len(want):,}", flush=True)
    print(f"[*] เจอข้อความ {found:,} จาก {len(want):,}", flush=True)

    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    top = collections.Counter(r["top_benchmark"] for r in rows)
    print("\n=== benchmark ที่เอกสารตรงด้วยมากที่สุด ===")
    for k, v in top.most_common():
        print("  %-20s %4d เอกสาร (%.1f%%)" % (str(k), v, 100 * v / max(len(rows), 1)))
    thai = {"m3exam_th", "belebele_th", "thaiexam_onet", "thaiexam_ic",
            "thaiexam_tgat", "thaiexam_tpat1", "thaiexam_a_level"}
    n_thai = sum(v for k, v in top.items() if k in thai)
    print("\n  ตรงกับข้อสอบไทยเป็นหลัก : %d (%.1f%%)" % (n_thai, 100 * n_thai / max(len(rows), 1)))
    print("  ตรงกับข้อสอบอังกฤษเป็นหลัก: %d (%.1f%%)" % (len(rows) - n_thai,
                                                    100 * (len(rows) - n_thai) / max(len(rows), 1)))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
