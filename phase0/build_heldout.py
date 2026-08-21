#!/usr/bin/env python3
"""Build frozen BPB held-out sets with HELDOUT-BUCKET-V1.

Source revisions and shard selections are pinned. EN and CODE use quotas over
range-spread shards so a sorted first shard cannot fill the entire held-out set.
This prepares measurement inputs; scientific_evidence_allowed is false.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "data" / "heldout"
sys.path.insert(0, str(HERE))
from heldout_rule import HELDOUT_BUCKETS, RULE_ID, TOTAL_BUCKETS, doc_hash, is_heldout  # noqa: E402

CODE_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}

SETS = {
    "TH-WEB-HELDOUT": {
        "language": "th",
        "repo": "aisingapore/SEA-PILE-v2",
        "revision": "77573cc84631412a781daa8e6f72cf322d4207f0",
        "files": [f"th/train-{i:05d}-of-00054.parquet" for i in (0, 13, 27, 40, 53)],
        "text_field": "text",
        "script_min_ratio": 0.5,
        "script_range": ("฀", "๿"),
        "min_chars": 500,
        "max_chars": 20000,
        "balanced_files": False,
    },
    "EN-HELDOUT": {
        "language": "en",
        "repo": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        # These source shards are already part of the replay pass. Their dump fields
        # jointly cover CC-MAIN 2013--2024; quotas prevent any one shard/dump dominating.
        "files": [f"sample/10BT/{i:03d}_00000.parquet" for i in (0, 2, 4, 6, 7)],
        "text_field": "text",
        "script_min_ratio": 0.5,
        "script_range": ("A", "z"),
        "min_chars": 500,
        "max_chars": 20000,
        "balanced_files": True,
        "metadata_fields": ("id", "url", "dump", "date", "language"),
    },
    "CODE-HELDOUT": {
        "language": "code",
        "repo": "codeparrot/github-code-clean",
        "revision": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
        "files": [f"data/train-{i:05d}-of-00880.parquet" for i in (0, 251, 502, 753, 879)],
        "text_field": "code",
        "script_min_ratio": 0.0,
        "script_range": ("A", "z"),
        "min_chars": 100,
        "max_chars": 200000,
        "balanced_files": True,
        "allowed_licenses": CODE_LICENSES,
        "metadata_fields": ("repo_name", "path", "language", "license"),
    },
}


def script_ratio(text, lo, hi):
    if not text:
        return 0.0
    return sum(1 for char in text if lo <= char <= hi) / len(text)


def quotas(total, count, balanced):
    if not balanced:
        return [total] * count
    return [total // count + (1 if i < total % count else 0) for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, choices=sorted(SETS))
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--max-scan", type=int, default=2_000_000)
    args = parser.parse_args()
    target = args.target or (1000 if args.set == "CODE-HELDOUT" else 2000)

    import warnings
    warnings.filterwarnings("ignore")
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    cfg = SETS[args.set]
    revision = cfg["revision"]
    HfApi().dataset_info(cfg["repo"], revision=revision)
    print(f"[*] {args.set}: {cfg['repo']} @ {revision}")
    print(
        f"[*] rule {RULE_ID}: bucket in {sorted(HELDOUT_BUCKETS)} of {TOTAL_BUCKETS} "
        f"({100 * len(HELDOUT_BUCKETS) / TOTAL_BUCKETS:.1f}% withheld)"
    )

    local = []
    for source_file in cfg["files"]:
        print(f"    fetching {source_file} ...", flush=True)
        local.append(
            hf_hub_download(cfg["repo"], source_file, repo_type="dataset", revision=revision)
        )

    def rows(path):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2000):
            data = batch.to_pydict()
            for index in range(len(data[cfg["text_field"]])):
                yield {key: data[key][index] for key in data}

    lo, hi = cfg["script_range"]
    kept, seen_hashes = [], set()
    stats = {
        "scanned": 0, "in_bucket": 0, "too_short": 0, "too_long": 0,
        "wrong_script": 0, "disallowed_license": 0, "dup": 0,
    }
    file_quotas = quotas(target, len(local), cfg.get("balanced_files", False))
    per_file = []
    max_scan_hit = False

    for source_file, path, quota in zip(cfg["files"], local, file_quotas):
        file_scanned = file_kept = 0
        for row in rows(path):
            stats["scanned"] += 1
            file_scanned += 1
            if stats["scanned"] > args.max_scan:
                max_scan_hit = True
                break
            text = row[cfg["text_field"]]
            if not is_heldout(text):
                continue
            stats["in_bucket"] += 1
            length = len(text)
            if length < cfg["min_chars"]:
                stats["too_short"] += 1
                continue
            if length > cfg["max_chars"]:
                stats["too_long"] += 1
                continue
            if script_ratio(text, lo, hi) < cfg["script_min_ratio"]:
                stats["wrong_script"] += 1
                continue
            license_name = str(row.get("license", "")).strip().lower()
            if cfg.get("allowed_licenses") and license_name not in cfg["allowed_licenses"]:
                stats["disallowed_license"] += 1
                continue
            digest = doc_hash(text)
            if digest in seen_hashes:
                stats["dup"] += 1
                continue
            seen_hashes.add(digest)
            record = {
                "doc_sha256": digest,
                "chars": length,
                "utf8_bytes": len(text.encode("utf-8")),
                "text": text,
                "source_file": source_file,
                "source_revision": revision,
            }
            for field in cfg.get("metadata_fields", ("url", "dump")):
                if field in row:
                    record[field] = row.get(field)
            kept.append(record)
            file_kept += 1
            if file_kept >= quota or len(kept) >= target:
                break
        per_file.append(
            {"file": source_file, "quota": quota, "scanned": file_scanned, "kept": file_kept}
        )
        if max_scan_hit or len(kept) >= target:
            break
        if file_kept < quota:
            print(f"[!] {source_file}: kept {file_kept}/{quota}", flush=True)

    if len(kept) != target:
        raise RuntimeError(
            f"fail-closed: {args.set} produced {len(kept)}/{target}; "
            f"increase --max-scan or revise the pinned shard selection"
        )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUTDIR / f"{args.set}.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for document in kept:
            handle.write(json.dumps(document, ensure_ascii=False, default=str) + "\n")

    set_hash = hashlib.sha256(
        "".join(document["doc_sha256"] for document in sorted(kept, key=lambda x: x["doc_sha256"])).encode()
    ).hexdigest()
    manifest = {
        "set_id": args.set,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "exclusion_rule": {
            "rule_id": RULE_ID, "module": "phase0/heldout_rule.py",
            "total_buckets": TOTAL_BUCKETS, "heldout_buckets": sorted(HELDOUT_BUCKETS),
            "contract": "Training MUST import heldout_rule.is_trainable and drop false records.",
        },
        "source": {
            "repo": cfg["repo"], "revision": revision, "files": cfg["files"],
            "access": "hf_hub_download + pyarrow local read",
        },
        "filters": {
            "order": ["heldout_bucket", "length", "script_ratio", "code_license_if_applicable", "exact_duplicate"],
            "min_chars": cfg["min_chars"], "max_chars": cfg["max_chars"],
            "script_min_ratio": cfg["script_min_ratio"],
            "allowed_licenses": sorted(cfg.get("allowed_licenses", [])) or None,
        },
        "counts": {**stats, "kept": len(kept)},
        "per_file": per_file,
        "totals": {
            "documents": len(kept),
            "utf8_bytes": sum(document["utf8_bytes"] for document in kept),
            "chars": sum(document["chars"] for document in kept),
        },
        "set_sha256": set_hash,
        "jsonl": str(jsonl.relative_to(HERE.parent)),
        "limitations": [
            "Token counts are not recorded here; BPB scoring tokenizes with the evaluated model.",
            "Benchmark decontamination is a separate step and has not run yet.",
            "Selected from pinned, range-spread source shards; EN/CODE use per-file quotas.",
        ],
    }
    (OUTDIR / f"{args.set}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[+] kept {len(kept)} docs, {manifest['totals']['utf8_bytes']:,} utf8 bytes")
    print(f"[+] set_sha256 = {set_hash}")
    print(f"[+] wrote {jsonl}")
    print(f"[+] stats: {stats}")


if __name__ == "__main__":
    main()
