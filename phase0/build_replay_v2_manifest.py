#!/usr/bin/env python3
"""Assemble the immutable Task-B replay-v2 manifest from runner summaries."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _sum_nested(items, key):
    total = collections.Counter()
    for item in items:
        total.update(item.get(key, {}))
    return dict(sorted(total.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-a-manifest", type=Path, default=ROOT / "data/clean_replay_manifest.json")
    parser.add_argument("--scan-summary", type=Path, default=ROOT / "data/scan_replay_v2_final_summary.json")
    parser.add_argument("--rebalance-summary", type=Path, default=ROOT / "data/rebalance_code_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/clean_replay_v2_manifest.json")
    args = parser.parse_args()

    old = json.loads(args.task_a_manifest.read_text())
    scan = json.loads(args.scan_summary.read_text())
    rebalance = json.loads(args.rebalance_summary.read_text())

    before_tokens = collections.Counter()
    for shard in rebalance["candidate_shards"]:
        if shard["source_kind"] == "task_a_scanned":
            before_tokens.update(shard["group_tokens"])
    before_total = sum(before_tokens.values())
    before_percent = {
        key: round(value * 100.0 / before_total, 6)
        for key, value in sorted(before_tokens.items())
    }

    original_code = scan["languages"]["code"]
    new_scans = [x["metrics"] for x in rebalance["candidate_shards"] if x["source_kind"] == "new_raw"]
    all_secret_hits = collections.Counter()
    all_secret_docs = collections.Counter()
    all_pii_hits = collections.Counter()
    all_pii_docs = collections.Counter()
    for language in ("en", "math", "code"):
        row = scan["languages"][language]
        all_secret_hits.update(row.get("secret_hits", {}))
        all_secret_docs.update(row.get("secret_documents", {}))
        all_pii_hits.update(row.get("pii_hits", {}))
        all_pii_docs.update(row.get("pii_documents", {}))
    all_secret_hits.update(_sum_nested(new_scans, "secret_hits"))
    all_secret_docs.update(_sum_nested(new_scans, "secret_documents"))
    all_pii_hits.update(_sum_nested(new_scans, "pii_hits"))
    all_pii_docs.update(_sum_nested(new_scans, "pii_documents"))

    sources = {
        language: {
            "repo": old["sources"][language]["repo"],
            "revision": old["sources"][language]["revision"],
            "subset": old["sources"][language]["subset"],
            "dataset_level_license": old["sources"][language]["dataset_level_license"],
            "additional_terms": old["sources"][language]["additional_terms"],
        }
        for language in ("en", "code", "math")
    }
    sources["code"]["per_row_license_allowlist"] = rebalance["source"]["per_row_license_allowlist"]
    sources["code"]["source_files"] = sorted(rebalance["final"]["source_file_tokens"])

    output_shards = []
    for shard in scan["shards"]:
        if shard["language"] not in {"en", "math"}:
            continue
        m = shard["metrics"]
        output_shards.append({
            "language": shard["language"],
            "output_file": shard["output_file"],
            "output_sha256": shard["output_sha256"],
            "documents": m["output_documents"],
            "utf8_bytes": m["output_bytes"],
            "qwen_tokens": m["output_qwen_tokens"],
        })
    for shard in rebalance["final"]["outputs"]:
        output_shards.append({"language": "code", **shard})

    final_totals = {
        "en": {
            "documents": scan["languages"]["en"]["output_documents"],
            "utf8_bytes": scan["languages"]["en"]["output_bytes"],
            "qwen_tokens": scan["languages"]["en"]["output_qwen_tokens"],
        },
        "math": {
            "documents": scan["languages"]["math"]["output_documents"],
            "utf8_bytes": scan["languages"]["math"]["output_bytes"],
            "qwen_tokens": scan["languages"]["math"]["output_qwen_tokens"],
        },
        "code": {
            "documents": rebalance["final"]["documents"],
            "utf8_bytes": rebalance["final"]["utf8_bytes"],
            "qwen_tokens": rebalance["final"]["qwen_tokens"],
        },
    }

    manifest = {
        "manifest_id": "CLEAN-REPLAY-EN-CODE-MATH-V2",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "exclusion_rule": scan["policy"]["post_redaction"][0],
        "scan_policy": scan["policy"],
        "selected_tokenizer": scan["tokenizer"],
        "sources": sources,
        "scan_input_output": scan["languages"],
        "new_code_source_scan": rebalance["new_source_scan"],
        "all_scanned_secret_hits": dict(sorted(all_secret_hits.items())),
        "all_scanned_secret_documents": dict(sorted(all_secret_docs.items())),
        "all_scanned_pii_hits": dict(sorted(all_pii_hits.items())),
        "all_scanned_pii_documents": dict(sorted(all_pii_docs.items())),
        "code_distribution": {
            "before_task_b_tokens": dict(sorted(before_tokens.items())),
            "before_task_b_percent": before_percent,
            "after_task_b_tokens": rebalance["final"]["group_tokens"],
            "after_task_b_percent": rebalance["final"]["group_percent"],
            "individual_language_tokens": rebalance["final"]["language_tokens"],
            "individual_language_percent": rebalance["final"]["language_percent"],
            "target_fractions": rebalance["selection"]["pre_registered_group_fractions"],
            "target_total_qwen_tokens": rebalance["selection"]["target_total_qwen_tokens"],
        },
        "balance_rationale": rebalance["rationale"],
        "final_totals": final_totals,
        "output_shards": sorted(output_shards, key=lambda x: (x["language"], x["output_file"])),
        "sample_policy": {
            "directory": "data/scan_samples",
            "masked_context_only": True,
            "target_per_detected_type": 20,
            "exception": "When fewer than 20 occurrences exist in the complete scanned pool, retain every available masked example; never fabricate samples.",
        },
        "limitations": [
            *rebalance["limitations"],
            "Secret and PII detectors intentionally favour precision; unsupported formats may remain.",
            "Masked examples are audit aids, not estimates of natural-world prevalence or scientific evidence.",
            "Counts for newly fetched code describe the complete scanned candidate pool, not only selected final rows.",
        ],
    }
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"[PASS] wrote {args.output}")


if __name__ == "__main__":
    main()
