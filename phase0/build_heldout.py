#!/usr/bin/env python3
"""Phase 0 / Step D -- build the frozen held-out BPB sets.

Streams the source corpus (no full download), selects documents by the shared
HELDOUT-BUCKET-V1 rule, applies quality filters, and writes a hashed set plus a
manifest.

  python3 build_heldout.py --set TH-WEB-HELDOUT --target 2000

The same rule module is imported by the training pipeline to drop the bucket, so
held-out documents can never be trained on -- including documents in shards this
builder never opened.

scientific_evidence_allowed = false (this builds measurement inputs, not results).
"""
from __future__ import annotations
import argparse, json, hashlib, datetime, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "data" / "heldout"
sys.path.insert(0, str(HERE))
from heldout_rule import doc_hash, is_heldout, RULE_ID, TOTAL_BUCKETS, HELDOUT_BUCKETS  # noqa: E402

SETS = {
    "TH-WEB-HELDOUT": {
        "language": "th", "repo": "aisingapore/SEA-PILE-v2",
        # STRATIFIED across the 54 shards. The corpus is ordered chronologically by
        # CommonCrawl dump (shard 0 = CC-MAIN-2020-45, shard 2 = CC-MAIN-2022-05), so taking
        # the first N shards yields a held-out set covering only the OLDEST slice of the
        # corpus. Evaluating a model trained on 2020-2025 text against a 2020-2022-only
        # held-out set is a temporal bias. Spread the sample across the whole range instead.
        "files": [f"th/train-{i:05d}-of-00054.parquet" for i in (0, 13, 27, 40, 53)],
        "text_field": "text", "script_min_ratio": 0.5,
        "script_range": ("฀", "๿"),
    },
    "EN-HELDOUT": {
        "language": "en", "repo": "HuggingFaceFW/fineweb-edu",
        "files": ["sample/10BT/000_00000.parquet"],
        "text_field": "text", "script_min_ratio": 0.5,
        "script_range": ("A", "z"),
    },
}

MIN_CHARS, MAX_CHARS = 500, 20000


def script_ratio(text, lo, hi):
    if not text:
        return 0.0
    return sum(1 for c in text if lo <= c <= hi) / len(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--max-scan", type=int, default=400000)
    a = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    cfg = SETS[a.set]
    rev = HfApi().dataset_info(cfg["repo"]).sha
    print(f"[*] {a.set}: {cfg['repo']} @ {rev}")
    print(f"[*] rule {RULE_ID}: bucket in {sorted(HELDOUT_BUCKETS)} of {TOTAL_BUCKETS} "
          f"({100*len(HELDOUT_BUCKETS)/TOTAL_BUCKETS:.1f}% withheld)")

    # Download shards and read locally. HF streaming was ~100x slower than the
    # download+read path for this corpus and could not finish in reasonable time.
    local = []
    for f in cfg["files"]:
        print(f"    fetching {f} ...", flush=True)
        local.append(hf_hub_download(cfg["repo"], f, repo_type="dataset", revision=rev))

    def rows():
        for path in local:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2000):
                d = batch.to_pydict()
                for i in range(len(d[cfg["text_field"]])):
                    yield {k: d[k][i] for k in d}

    ds = rows()

    lo, hi = cfg["script_range"]
    kept, seen_hashes = [], set()
    stats = dict(scanned=0, in_bucket=0, too_short=0, too_long=0, wrong_script=0, dup=0)

    for row in ds:
        stats["scanned"] += 1
        if stats["scanned"] > a.max_scan:
            break
        t = row[cfg["text_field"]]
        if not is_heldout(t):
            continue
        stats["in_bucket"] += 1
        n = len(t)
        if n < MIN_CHARS:   stats["too_short"] += 1; continue
        if n > MAX_CHARS:   stats["too_long"] += 1; continue
        if script_ratio(t, lo, hi) < cfg["script_min_ratio"]:
            stats["wrong_script"] += 1; continue
        h = doc_hash(t)
        if h in seen_hashes:
            stats["dup"] += 1; continue
        seen_hashes.add(h)
        kept.append({"doc_sha256": h, "chars": n, "utf8_bytes": len(t.encode()),
                     "url": row.get("url"), "dump": row.get("dump"), "text": t})
        if len(kept) >= a.target:
            break
        if len(kept) and len(kept) % 250 == 0:
            print(f"    kept {len(kept)}/{a.target} (scanned {stats['scanned']:,})", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUTDIR / f"{a.set}.jsonl"
    with jsonl.open("w") as fh:
        for d in kept:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    set_hash = hashlib.sha256(
        "".join(d["doc_sha256"] for d in sorted(kept, key=lambda x: x["doc_sha256"])).encode()
    ).hexdigest()

    man = {
        "set_id": a.set,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "exclusion_rule": {
            "rule_id": RULE_ID, "module": "phase0/heldout_rule.py",
            "total_buckets": TOTAL_BUCKETS, "heldout_buckets": sorted(HELDOUT_BUCKETS),
            "contract": "The training pipeline MUST import heldout_rule.is_trainable and drop "
                        "every document for which it returns False. This excludes the whole "
                        "bucket, including documents in shards this builder never opened.",
        },
        "source": {"repo": cfg["repo"], "revision": rev, "files": cfg["files"],
                   "access": "hf_hub_download + pyarrow local read"},
        "filters": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
                    "script_min_ratio": cfg["script_min_ratio"]},
        "counts": {**stats, "kept": len(kept)},
        "totals": {"documents": len(kept),
                   "utf8_bytes": sum(d["utf8_bytes"] for d in kept),
                   "chars": sum(d["chars"] for d in kept)},
        "set_sha256": set_hash,
        "jsonl": str(jsonl.relative_to(HERE.parent)),
        "limitations": [
            "Token counts are NOT recorded here; they depend on the final tokenizer choice "
            "and must be computed after the base model is locked.",
            "Benchmark decontamination against ThaiExam/M3Exam is a SEPARATE step and has not run yet.",
            "Streamed from the first shards only; this is a held-out sample, not a stratified "
            "sample of the whole corpus.",
        ],
    }
    (OUTDIR / f"{a.set}.manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")

    print(f"\n[+] kept {len(kept)} docs, {man['totals']['utf8_bytes']:,} utf8 bytes")
    print(f"[+] set_sha256 = {set_hash}")
    print(f"[+] wrote {jsonl}")
    print(f"[+] stats: {stats}")


if __name__ == "__main__":
    main()
