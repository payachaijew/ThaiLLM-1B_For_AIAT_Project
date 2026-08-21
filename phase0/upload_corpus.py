#!/usr/bin/env python3
"""Upload the cleaned corpus to a PRIVATE HuggingFace dataset repo for handoff.

Private by default and deliberately so: the license and PDPA questions in
LICENSE_COMPLIANCE.md are still open, and this corpus is derived from CommonCrawl.
Making it public is a separate decision that belongs to the project owner.

  python3 upload_corpus.py --repo <user>/<name> --dry-run     # see what would go
  python3 upload_corpus.py --repo <user>/<name>               # upload

Requires a write token:  huggingface-cli login   (or HF_TOKEN in the environment)
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (local path, path inside the repo)
PAYLOAD = [
    ("data/clean_pii/th",            "corpus/th"),
    ("data/clean_replay_v2/en",      "corpus/en"),
    ("data/clean_replay_v2/code",    "corpus/code"),
    ("data/clean_replay_v2/math",    "corpus/math"),
    ("data/heldout",                 "heldout"),
    ("phase0",                       "scripts"),
]
FILES = [
    ("DATASET_CARD.md",                     "README.md"),   # HF renders README.md as the card
    ("LICENSE_COMPLIANCE.md",               "LICENSE_COMPLIANCE.md"),
    ("data/removal_list.txt",               "removal_list.txt"),
    ("data/clean_th_manifest.json",         "manifests/clean_th_manifest.json"),
    ("data/clean_pii_th_manifest.json",     "manifests/clean_pii_th_manifest.json"),
    ("data/clean_replay_v2_manifest.json",  "manifests/clean_replay_v2_manifest.json"),
    ("data/decon_neardedup_report.json",    "manifests/decon_neardedup_report.json"),
    ("data/decontamination_hits.jsonl",     "manifests/decontamination_hits.jsonl"),
    ("data/benchmark_ngrams_meta.json",     "manifests/benchmark_ngrams_meta.json"),
    ("validation/replay_v2_verification.json", "manifests/replay_v2_verification.json"),
    ("validation/VALIDATION_OUTPUT_LOG.md", "VALIDATION_OUTPUT_LOG.md"),
]
IGNORE = ["*.pyc", "__pycache__/*", "overnight_logs/*", "*.sqlite3*", "*.npz"]


def human(n): return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. payachaijew/thaillm-1b-cpt-corpus")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--public", action="store_true",
                    help="DO NOT USE until the license/PDPA questions are resolved")
    a = ap.parse_args()

    total = 0
    print("จะอัปโหลด:")
    for src, dst in PAYLOAD:
        p = ROOT / src
        if not p.exists(): print(f"  ! ไม่พบ {src}"); continue
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()
                 and not any(f.match(g) for g in IGNORE))
        total += sz
        print(f"  {src:34s} -> {dst:16s} {human(sz):>10s}")
    for src, dst in FILES:
        p = ROOT / src
        if not p.exists(): print(f"  ! ไม่พบ {src}"); continue
        total += p.stat().st_size
        print(f"  {src:34s} -> {dst:16s} {human(p.stat().st_size):>10s}")
    print(f"\nรวม {human(total)}")

    if a.public:
        print("\n⛔ --public ถูกปฏิเสธ: LICENSE_COMPLIANCE.md ยังมีคำถาม PDPA ที่ยังไม่มีข้อสรุป")
        sys.exit(1)
    if a.dry_run:
        print("\n(dry run — ยังไม่อัปโหลด)"); return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"\n[*] repo: {a.repo} (private)")
    for src, dst in PAYLOAD:
        p = ROOT / src
        if not p.exists(): continue
        print(f"  uploading {src} ...", flush=True)
        api.upload_folder(folder_path=str(p), path_in_repo=dst, repo_id=a.repo,
                          repo_type="dataset", ignore_patterns=IGNORE)
    for src, dst in FILES:
        p = ROOT / src
        if not p.exists(): continue
        print(f"  uploading {src} ...", flush=True)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dst,
                        repo_id=a.repo, repo_type="dataset")
    print(f"\n[+] เสร็จ -> https://huggingface.co/datasets/{a.repo}")
    print("[!] repo เป็น private — เชิญ mentor เข้าถึงในหน้า Settings > Collaborators")


if __name__ == "__main__":
    main()
