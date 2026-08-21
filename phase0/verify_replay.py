#!/usr/bin/env python3
"""Independently verify replay outputs and EN/CODE held-out isolation."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import random
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from build_replay import ALLOWED_CODE_LICENSES, TOKENIZER_REVISION, artifact_path, sha256_file  # noqa: E402
from heldout_rule import doc_hash, is_heldout  # noqa: E402


def fail(errors, message):
    errors.append(message)
    print(f"[FAIL] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "clean_replay_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "replay_verification.json")
    parser.add_argument("--token-samples", type=int, default=100)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    errors = []
    totals = {language: {"documents": 0, "utf8_bytes": 0, "qwen_tokens": 0} for language in ("en", "code", "math")}
    samples = {language: [] for language in totals}
    rng = random.Random(20260820)

    temp = tempfile.NamedTemporaryFile(prefix="replay_verify_", suffix=".sqlite3", delete=False)
    temp.close()
    unique = sqlite3.connect(temp.name)
    unique.execute("PRAGMA journal_mode=OFF")
    unique.execute("PRAGMA synchronous=OFF")
    unique.execute("CREATE TABLE hashes(digest BLOB PRIMARY KEY, language TEXT)")

    for shard in manifest["output_shards"]:
        language = shard["language"]
        expected_source = manifest["sources"][language]
        path = ROOT / shard["output_file"]
        if not path.is_file():
            fail(errors, f"missing output shard: {path}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != shard["output_sha256"]:
            fail(errors, f"compressed SHA mismatch: {path}")
        shard_totals = {"documents": 0, "utf8_bytes": 0, "qwen_tokens": 0}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                text = row["text"]
                digest = doc_hash(text)
                shard_totals["documents"] += 1
                shard_totals["utf8_bytes"] += len(text.encode("utf-8"))
                shard_totals["qwen_tokens"] += int(row["qwen_tokens"])
                if digest != row["doc_sha256"]:
                    fail(errors, f"doc hash mismatch {path}:{line_number}")
                if is_heldout(text):
                    fail(errors, f"held-out record leaked into replay {path}:{line_number}")
                if int(row["qwen_tokens"]) <= 0:
                    fail(errors, f"non-positive token count {path}:{line_number}")
                if row.get("source_repo") != expected_source["repo"]:
                    fail(errors, f"source repo mismatch {path}:{line_number}")
                if row.get("source_revision") != expected_source["revision"]:
                    fail(errors, f"source revision mismatch {path}:{line_number}")
                if language == "code" and row.get("license") not in ALLOWED_CODE_LICENSES:
                    fail(errors, f"disallowed code license {row.get('license')} {path}:{line_number}")
                try:
                    unique.execute("INSERT INTO hashes VALUES(?,?)", (bytes.fromhex(digest), language))
                except sqlite3.IntegrityError:
                    fail(errors, f"cross-shard duplicate {digest} at {path}:{line_number}")
                seen = totals[language]["documents"] + shard_totals["documents"]
                if len(samples[language]) < args.token_samples:
                    samples[language].append((text, int(row["qwen_tokens"])))
                else:
                    candidate = rng.randrange(seen)
                    if candidate < args.token_samples:
                        samples[language][candidate] = (text, int(row["qwen_tokens"]))
        for key in shard_totals:
            totals[language][key] += shard_totals[key]
        expected = {
            "documents": shard["output_documents"],
            "utf8_bytes": shard["output_bytes"],
            "qwen_tokens": shard["qwen_tokens"],
        }
        if shard_totals != expected:
            fail(errors, f"shard totals mismatch {path}: {shard_totals} != {expected}")
        print(f"[OK] {path.name}: {shard_totals['documents']:,} docs", flush=True)
    unique.commit()

    for language, observed in totals.items():
        source = manifest["sources"][language]
        expected = {
            "documents": source["output"]["documents"],
            "utf8_bytes": source["output"]["utf8_bytes"],
            "qwen_tokens": source["output"]["qwen_tokens"],
        }
        if observed != expected:
            fail(errors, f"aggregate mismatch for {language}: {observed} != {expected}")
        if observed["qwen_tokens"] < source["target_qwen_tokens"] or not source["target_met"]:
            fail(errors, f"target not met for {language}")

    heldout_results = {}
    for set_id in ("EN-HELDOUT", "CODE-HELDOUT"):
        path = ROOT / "data" / "heldout" / f"{set_id}.jsonl"
        held_manifest_path = ROOT / "data" / "heldout" / f"{set_id}.manifest.json"
        held_manifest = json.loads(held_manifest_path.read_text())
        hashes = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                digest = doc_hash(row["text"])
                hashes.append(digest)
                if digest != row["doc_sha256"] or not is_heldout(row["text"]):
                    fail(errors, f"invalid held-out record {path}:{line_number}")
                if unique.execute("SELECT 1 FROM hashes WHERE digest=?", (bytes.fromhex(digest),)).fetchone():
                    fail(errors, f"held-out leakage: {set_id} {digest}")
        set_sha = hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()
        if set_sha != held_manifest["set_sha256"]:
            fail(errors, f"held-out set hash mismatch: {set_id}")
        if len(hashes) != held_manifest["totals"]["documents"]:
            fail(errors, f"held-out count mismatch: {set_id}")
        heldout_results[set_id] = {"documents": len(hashes), "set_sha256": set_sha, "leakage": 0}

    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tokenizer_path = Path(
        hf_hub_download(
            manifest["selected_tokenizer"]["repo"],
            manifest["selected_tokenizer"]["artifact"],
            revision=manifest["selected_tokenizer"]["revision"],
            local_files_only=True,
        )
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_mismatches = {}
    for language, records in samples.items():
        mismatches = 0
        encodings = tokenizer.encode_batch([text for text, _ in records], add_special_tokens=False)
        for encoding, (_, recorded) in zip(encodings, records):
            mismatches += len(encoding.ids) != recorded
        token_mismatches[language] = {"sampled": len(records), "mismatches": mismatches}
        if mismatches:
            fail(errors, f"token recount mismatches for {language}: {mismatches}")

    report = {
        "verification_id": "VERIFY-CLEAN-REPLAY-V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "manifest": artifact_path(args.manifest),
        "manifest_id": manifest["manifest_id"],
        "tokenizer_revision": TOKENIZER_REVISION,
        "totals": totals,
        "heldout": heldout_results,
        "token_recount": token_mismatches,
        "unique_documents": unique.execute("SELECT COUNT(*) FROM hashes").fetchone()[0],
        "errors": errors,
        "verdict": "PASS" if not errors else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    unique.close()
    Path(temp.name).unlink(missing_ok=True)
    print(f"[{report['verdict']}] wrote {args.output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
