#!/usr/bin/env python3
"""วัดว่าอัตราการปนเปื้อนไวต่อการตั้งค่าแค่ไหน

งานที่รายงานตัวเลข contamination มักไม่บอกว่าใช้ n-gram ยาวเท่าไหร่ stride เท่าไหร่ เกณฑ์กี่จุด
ถ้าตัวเลขไวต่อค่าพวกนี้มาก ตัวเลขที่ไม่ระบุค่าก็เทียบข้ามงานไม่ได้ นี่คือสิ่งที่สคริปต์นี้วัด

ใช้ตัวอย่างสุ่มแบบกำหนดได้ (hash bucket) แทนการสแกน 4.57 ล้านฉบับซ้ำหกรอบ
ตัวอย่างเดียวกันถูกใช้กับทุก config จึงเทียบกันได้ตรง ๆ

  PYTHONHASHSEED=0 python3 sensitivity.py --sample 200000
"""
from __future__ import annotations
import argparse, glob, gzip, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase0"))
from build_benchmark_index import hashes, norm   # noqa: E402

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("FATAL: ต้องรันด้วย PYTHONHASHSEED=0")

GRAMS = [32, 50, 64]
STRIDES = [1, 8]
THRESHOLDS = [1, 2, 5, 10, 20, 50]
MAXCHARS = 40000


def in_sample(did: str, rate: int) -> bool:
    """ตัวอย่างแบบกำหนดได้จาก doc_sha256 — ไม่ต้องเก็บรายชื่อ และทำซ้ำได้"""
    return int(did[:8], 16) % rate == 0


def build_index(n: int):
    """สร้าง index ที่ความยาว n จาก benchmark ทั้งหมด"""
    import warnings; warnings.filterwarnings("ignore")
    from datasets import load_dataset
    from build_benchmark_index import SPECS
    allh = []
    for name, repo, kw, fn in SPECS:
        try:
            ds = load_dataset(repo, **kw)
        except Exception:
            continue
        hs = []
        for r in ds:
            try:
                t = fn(r) if fn else " ".join(
                    str(v) for v in r.values() if isinstance(v, str) and v.strip())
            except Exception:
                continue
            h = hashes(t, n, 1)
            if h.size:
                hs.append(h)
        if hs:
            allh.append(np.unique(np.concatenate(hs)))
    return np.unique(np.concatenate(allh)) if allh else np.empty(0, dtype=np.uint64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data/clean_pii/th"))
    ap.add_argument("--sample", type=int, default=200000, help="จำนวนเอกสารเป้าหมาย")
    ap.add_argument("--out", default=str(HERE / "sensitivity.json"))
    a = ap.parse_args()

    total_docs = 4567214
    rate = max(1, total_docs // a.sample)
    print(f"[*] สุ่ม 1 ใน {rate} ของ {total_docs:,} -> ~{total_docs//rate:,} เอกสาร", flush=True)

    print("[*] โหลดข้อความตัวอย่าง", flush=True)
    texts = []
    for f in sorted(glob.glob(f"{a.corpus}/*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                did = r.get("doc_sha256") or hashlib.sha256(r["text"].encode()).hexdigest()
                if in_sample(did, rate):
                    texts.append(r["text"][:MAXCHARS])
    print(f"[*] ได้ {len(texts):,} เอกสาร", flush=True)

    results = {}
    for n in GRAMS:
        print(f"\n[*] สร้าง index ที่ {n} ตัวอักษร", flush=True)
        idx = build_index(n); idx.sort()
        print(f"    {idx.size:,} n-grams", flush=True)
        for stride in STRIDES:
            t0 = time.time()
            counts = np.zeros(len(texts), dtype=np.int64)
            for i, t in enumerate(texts):
                h = hashes(t, n, stride)
                if h.size:
                    pos = np.searchsorted(idx, h); pos[pos >= idx.size] = 0
                    counts[i] = int(np.count_nonzero(idx[pos] == h))
                if (i + 1) % 25000 == 0:
                    print(f"    {n}ch stride{stride}: {i+1:,}/{len(texts):,}", flush=True)
            key = f"gram{n}_stride{stride}"
            results[key] = {
                "ngram_chars": n, "stride": stride,
                "index_size": int(idx.size), "docs_scanned": len(texts),
                "flagged_at_threshold": {str(k): int((counts >= k).sum()) for k in THRESHOLDS},
                "rate_at_threshold": {str(k): float((counts >= k).sum()) / len(texts)
                                      for k in THRESHOLDS},
                "seconds": round(time.time() - t0, 1),
            }
            r2 = results[key]["rate_at_threshold"]["2"]
            print(f"    -> ที่เกณฑ์ 2 จุด: {results[key]['flagged_at_threshold']['2']:,} "
                  f"({r2*100:.4f}%)  [{results[key]['seconds']:.0f}s]", flush=True)

    out = {"analysis_id": "CONTAM-SENSITIVITY-V1",
           "scientific_evidence_allowed": False,
           "sample_rule": f"doc_sha256[:8] % {rate} == 0 (deterministic)",
           "eval_suite": "THAILLM-EVAL-FROZEN-V1",
           "configs": results}
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
