#!/usr/bin/env python3
"""Plan C step 2 -- benchmark decontamination and near-duplicate detection in ONE corpus pass.

Both tasks need the same character n-gram hashes per document, so they share the expensive
part and the corpus is read once instead of twice.

  decontamination : any document containing >= MIN_HITS benchmark n-grams is flagged.
  near-duplicates : MinHash signature -> LSH banding -> first occurrence kept, rest flagged.

Nothing is deleted. The output is a REMOVAL LIST that the tokenisation step consumes, so the
decision stays reversible and auditable.

REQUIRES PYTHONHASHSEED=0.
scientific_evidence_allowed = false.
"""
from __future__ import annotations
import gzip, json, glob, sys, os, time, datetime, collections
from pathlib import Path
import numpy as np

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("FATAL: run with PYTHONHASHSEED=0 or hashes will not match the benchmark index")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))
from build_benchmark_index import hashes, NGRAM  # noqa: E402

STRIDE = 8          # corpus stride; index is stride 1 so every alignment is still covered
MIN_HITS = 2        # >=2 distinct benchmark n-grams before flagging, to blunt single-window flukes
K = 32              # MinHash permutations
BANDS, ROWS = 8, 4  # 8 bands x 4 rows = 32
MAXCHARS = 40000    # cap per document; near-dups share their opening, and this bounds cost

SOURCES = [
    ("th",   DATA / "clean_pii" / "th"),
    ("en",   DATA / "clean_replay_v2" / "en"),
    ("code", DATA / "clean_replay_v2" / "code"),
    ("math", DATA / "clean_replay_v2" / "math"),
]

rng = np.random.default_rng(20260821)
PRIME = np.uint64((1 << 61) - 1)
A = rng.integers(1, 1 << 60, size=K, dtype=np.uint64)
B = rng.integers(0, 1 << 60, size=K, dtype=np.uint64)


def signature(h: np.ndarray) -> np.ndarray:
    if h.size == 0:
        return np.zeros(K, dtype=np.uint64)
    return ((h[:, None] * A + B) % PRIME).min(axis=0)


def main():
    idx = np.load(DATA / "benchmark_ngrams.npz")["ngrams"]
    idx.sort()
    print(f"[*] benchmark index: {idx.size:,} n-grams", flush=True)

    buckets = collections.defaultdict(list)   # band key -> first doc id
    decon = collections.Counter()
    decon_rows, dup_rows = [], []
    per_lang = collections.Counter()
    t0 = time.time()
    total = 0

    for lang, root in SOURCES:
        files = sorted(glob.glob(str(root / "*.jsonl.gz")))
        if not files:
            print(f"  ! no shards for {lang} at {root}", flush=True); continue
        n = 0
        for f in files:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    did = r.get("doc_sha256") or ""
                    h = hashes(r["text"][:MAXCHARS], NGRAM, STRIDE)
                    n += 1; total += 1

                    if h.size:
                        pos = np.searchsorted(idx, h)
                        pos[pos >= idx.size] = 0
                        hits = int(np.count_nonzero(idx[pos] == h))
                        if hits >= MIN_HITS:
                            decon[lang] += 1
                            decon_rows.append({"lang": lang, "doc_sha256": did,
                                               "ngram_hits": hits,
                                               "url": r.get("url"), "chars": len(r["text"])})
                        sig = signature(h)
                        for b in range(BANDS):
                            key = (b, sig[b * ROWS:(b + 1) * ROWS].tobytes())
                            if key in buckets:
                                dup_rows.append({"lang": lang, "doc_sha256": did,
                                                 "duplicate_of": buckets[key]})
                                break
                            buckets[key] = did
                    if total % 200000 == 0:
                        el = time.time() - t0
                        print(f"    {total:,} docs  {el/60:.1f} min  "
                              f"{total/el:.0f} doc/s  decon={sum(decon.values()):,} "
                              f"neardup={len(dup_rows):,}", flush=True)
        per_lang[lang] = n
        print(f"  [{lang}] {n:,} docs", flush=True)

    dup_ids = {d["doc_sha256"] for d in dup_rows}
    dec_ids = {d["doc_sha256"] for d in decon_rows}
    removal = sorted(dup_ids | dec_ids)

    (DATA / "decontamination_hits.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decon_rows))
    (DATA / "near_duplicate_hits.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dup_rows))
    (DATA / "removal_list.txt").write_text("\n".join(removal) + "\n")

    rep = {
        "run_id": "DECON-NEARDEDUP-V1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "params": {"ngram_chars": NGRAM, "corpus_stride": STRIDE, "index_stride": 1,
                   "min_ngram_hits": MIN_HITS, "minhash_permutations": K,
                   "lsh_bands": BANDS, "lsh_rows": ROWS, "max_chars_per_doc": MAXCHARS},
        "documents_scanned": dict(per_lang) | {"total": total},
        "decontamination": {"flagged_by_language": dict(decon),
                            "flagged_total": len(decon_rows),
                            "rate": round(len(decon_rows) / total, 6) if total else None},
        "near_duplicates": {"flagged_total": len(dup_rows),
                            "rate": round(len(dup_rows) / total, 6) if total else None},
        "removal_list": {"path": "data/removal_list.txt", "unique_documents": len(removal),
                         "rate": round(len(removal) / total, 6) if total else None},
        "policy": "flag only; nothing deleted. The tokenisation step must drop these doc_sha256 values.",
        "limitations": [
            "Character 64-grams detect verbatim or near-verbatim overlap only; paraphrased or "
            "translated benchmark items are NOT detected.",
            "Corpus scanned at stride 8 and capped at 40,000 characters per document.",
            "MinHash with 32 permutations and 8x4 banding is tuned for recall of high-similarity "
            "pairs; moderately similar documents will be missed.",
            "LSH keeps the FIRST document seen in each bucket, so which member of a near-duplicate "
            "cluster survives depends on file order.",
        ],
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    (DATA / "decon_neardedup_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n")
    print(f"\n=== DONE {rep['runtime_minutes']} min ===")
    print(f"scanned {total:,} | decon {len(decon_rows):,} | neardup {len(dup_rows):,} "
          f"| removal {len(removal):,} ({100*len(removal)/total:.2f}%)")


if __name__ == "__main__":
    main()
