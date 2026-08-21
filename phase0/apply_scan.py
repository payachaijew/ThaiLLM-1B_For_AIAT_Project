#!/usr/bin/env python3
"""Apply secret and English/international PII policy to replay corpora.

Inputs are never overwritten.  Processing checkpoints atomically at compressed
input-shard boundaries in SQLite and can be resumed.  The final text is checked
again against HELDOUT-BUCKET-V1 and exact-deduplicated after redaction.

For the Task-B production workflow this script writes EN/math directly to v2;
the scanned code output is used as the candidate input to ``rebalance_code.py``.
This is data preparation, not scientific evidence.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import pii_filter_en  # noqa: E402
import secret_scan  # noqa: E402
from build_replay import (  # noqa: E402
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    artifact_path,
    deterministic_gzip_text,
    encode_lengths,
    load_tokenizer,
    sha256_file,
)
from heldout_rule import RULE_ID, doc_hash, is_heldout  # noqa: E402

SCIENTIFIC_EVIDENCE_ALLOWED = False


class ScanState:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen (
              doc_sha256 TEXT PRIMARY KEY,
              language TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS completed (
              language TEXT NOT NULL,
              input_file TEXT NOT NULL,
              input_sha256 TEXT NOT NULL,
              output_file TEXT NOT NULL,
              output_sha256 TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              PRIMARY KEY(language,input_file)
            );
            CREATE TABLE IF NOT EXISTS samples (
              family TEXT NOT NULL,
              kind TEXT NOT NULL,
              sample_key TEXT NOT NULL,
              record_json TEXT NOT NULL,
              PRIMARY KEY(family,kind,sample_key)
            );
            """
        )
        self.db.commit()

    def completed(self, language: str, input_file: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM completed WHERE language=? AND input_file=?",
            (language, input_file),
        ).fetchone() is not None

    def duplicate(self, digest: str) -> bool:
        return self.db.execute("SELECT 1 FROM seen WHERE doc_sha256=?", (digest,)).fetchone() is not None

    def remember(self, digest: str, language: str) -> None:
        self.db.execute("INSERT INTO seen VALUES(?,?)", (digest, language))

    def sample(
        self,
        family: str,
        kind: str,
        digest: str,
        event_index: int,
        event: Mapping[str, object],
        language: str,
        source_file: str,
        limit: int,
    ) -> None:
        key = hashlib.sha256(f"{digest}:{family}:{kind}:{event_index}".encode()).hexdigest()
        record = {
            "scientific_evidence_allowed": False,
            "family": family,
            "type": kind,
            "language": language,
            "doc_sha256": digest,
            "source_file": source_file,
            "position": {"start": event.get("start"), "end": event.get("end")},
            "masked_context": event.get("context", ""),
        }
        self.db.execute(
            "INSERT OR IGNORE INTO samples VALUES(?,?,?,?)",
            (family, kind, key, json.dumps(record, ensure_ascii=False)),
        )
        self.db.execute(
            """DELETE FROM samples WHERE rowid IN (
                 SELECT rowid FROM samples WHERE family=? AND kind=?
                 ORDER BY sample_key DESC LIMIT -1 OFFSET ?
               )""",
            (family, kind, limit),
        )

    def commit(self, language: str, input_file: str, input_sha: str, output_file: str, output_sha: str, metrics: Mapping[str, object]) -> None:
        self.db.execute(
            "INSERT INTO completed VALUES(?,?,?,?,?,?)",
            (language, input_file, input_sha, output_file, output_sha, json.dumps(metrics, sort_keys=True)),
        )
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def rows(self) -> List[Tuple[object, ...]]:
        return self.db.execute(
            "SELECT language,input_file,input_sha256,output_file,output_sha256,metrics_json "
            "FROM completed ORDER BY language,input_file"
        ).fetchall()

    def write_samples(self, root: Path) -> Mapping[str, int]:
        root.mkdir(parents=True, exist_ok=True)
        rows = self.db.execute(
            "SELECT family,kind,record_json FROM samples ORDER BY family,kind,sample_key"
        ).fetchall()
        grouped: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        for family, kind, record in rows:
            grouped[(family, kind)].append(record)
        counts = {}
        for (family, kind), records in grouped.items():
            path = root / f"{family}_{kind}.jsonl"
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text("\n".join(records) + "\n", encoding="utf-8")
            os.replace(temp, path)
            counts[f"{family}:{kind}"] = len(records)
        return counts


def _counter_dict(counter: collections.Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def process_shard(
    state: ScanState,
    language: str,
    input_path: Path,
    output_path: Path,
    tokenizer,
    sample_limit: int,
) -> Mapping[str, object]:
    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("refusing to overwrite input")
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if temp.exists():
        temp.unlink()
    if output_path.exists() or meta_path.exists():
        raise RuntimeError(f"orphan output exists without checkpoint: {output_path}")

    metrics = collections.Counter()
    secret_hits = collections.Counter()
    secret_docs = collections.Counter()
    pii_hits = collections.Counter()
    pii_docs = collections.Counter()
    drops = collections.Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with gzip.open(input_path, "rt", encoding="utf-8") as source, deterministic_gzip_text(temp) as output:
            for line_no, line in enumerate(source, 1):
                row = json.loads(line)
                text = row.get("text")
                if not isinstance(text, str):
                    raise RuntimeError(f"missing text at {input_path}:{line_no}")
                original_digest = str(row.get("doc_sha256") or doc_hash(text))
                original_tokens = int(row.get("qwen_tokens") or 0)
                metrics["input_documents"] += 1
                metrics["input_bytes"] += len(text.encode("utf-8"))
                metrics["input_qwen_tokens"] += original_tokens

                secret_text, sec_counts, sec_events = secret_scan.redact(text)
                secret_hits.update(sec_counts)
                for kind in sec_counts:
                    secret_docs[kind] += 1
                for index, event in enumerate(sec_events):
                    state.sample("secret", str(event["type"]), original_digest, index, event, language, str(input_path), sample_limit)
                if secret_text is None:
                    drops["private_key_document"] += 1
                    continue

                final_text, this_pii, pii_events = pii_filter_en.redact(secret_text)
                pii_hits.update(this_pii)
                for kind in this_pii:
                    pii_docs[kind] += 1
                for index, event in enumerate(pii_events):
                    state.sample("pii", str(event["type"]), original_digest, index, event, language, str(input_path), sample_limit)

                changed = final_text != text
                final_digest = doc_hash(final_text)
                if changed and is_heldout(final_text):
                    drops["post_redaction_heldout"] += 1
                    continue
                if state.duplicate(final_digest):
                    drops["post_redaction_exact_duplicate"] += 1
                    continue
                if changed:
                    final_tokens = encode_lengths(tokenizer, [final_text])[0]
                else:
                    final_tokens = original_tokens or encode_lengths(tokenizer, [final_text])[0]
                if final_tokens <= 0:
                    drops["zero_tokens"] += 1
                    continue

                state.remember(final_digest, language)
                result = dict(row)
                result.update({
                    "scientific_evidence_allowed": False,
                    "text": final_text,
                    "source_doc_sha256": original_digest,
                    "doc_sha256": final_digest,
                    "utf8_bytes": len(final_text.encode("utf-8")),
                    "qwen_tokens": final_tokens,
                    "scan_policy": "SECRET-PII-EN-V1",
                })
                output.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                metrics["output_documents"] += 1
                metrics["output_bytes"] += result["utf8_bytes"]
                metrics["output_qwen_tokens"] += final_tokens
                if changed:
                    metrics["redacted_documents"] += 1

        os.replace(temp, output_path)
        metrics_record = {
            "scientific_evidence_allowed": False,
            **_counter_dict(metrics),
            "secret_hits": _counter_dict(secret_hits),
            "secret_documents": _counter_dict(secret_docs),
            "pii_hits": _counter_dict(pii_hits),
            "pii_documents": _counter_dict(pii_docs),
            "dropped_documents": _counter_dict(drops),
        }
        meta = {
            "scientific_evidence_allowed": False,
            "language": language,
            "input_file": artifact_path(input_path),
            "input_sha256": sha256_file(input_path),
            "output_file": artifact_path(output_path),
            "output_sha256": sha256_file(output_path),
            "metrics": metrics_record,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state.commit(language, meta["input_file"], meta["input_sha256"], meta["output_file"], meta["output_sha256"], metrics_record)
        return meta
    except BaseException:
        state.rollback()
        if temp.exists():
            temp.unlink()
        raise


def aggregate(state: ScanState) -> Mapping[str, object]:
    languages: Dict[str, Dict[str, object]] = {}
    shards = []
    for language, input_file, input_sha, output_file, output_sha, metrics_json in state.rows():
        metrics = json.loads(metrics_json)
        target = languages.setdefault(language, {
            "input_documents": 0, "input_bytes": 0, "input_qwen_tokens": 0,
            "output_documents": 0, "output_bytes": 0, "output_qwen_tokens": 0,
            "redacted_documents": 0,
            "secret_hits": collections.Counter(), "secret_documents": collections.Counter(),
            "pii_hits": collections.Counter(), "pii_documents": collections.Counter(),
            "dropped_documents": collections.Counter(), "source_shards": 0,
        })
        target["source_shards"] += 1
        for key in ("input_documents", "input_bytes", "input_qwen_tokens", "output_documents", "output_bytes", "output_qwen_tokens", "redacted_documents"):
            target[key] += int(metrics.get(key, 0))
        for key in ("secret_hits", "secret_documents", "pii_hits", "pii_documents", "dropped_documents"):
            target[key].update(metrics.get(key, {}))
        shards.append({
            "language": language, "input_file": input_file, "input_sha256": input_sha,
            "output_file": output_file, "output_sha256": output_sha, "metrics": metrics,
        })
    serial = {}
    for language, values in languages.items():
        serial[language] = {
            key: (_counter_dict(value) if isinstance(value, collections.Counter) else value)
            for key, value in values.items()
        }
    return {"languages": serial, "shards": shards}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=("en", "code", "math"), default=("en", "code", "math"))
    parser.add_argument("--input-root", type=Path, default=ROOT / "data" / "clean_replay")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "clean_replay_v2")
    parser.add_argument("--state-db", type=Path, default=ROOT / "data" / "scan_replay_v2_state.sqlite3")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "scan_replay_v2_summary.json")
    parser.add_argument("--samples", type=Path, default=ROOT / "data" / "scan_samples")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.input_root.resolve() == args.output_root.resolve():
        raise SystemExit("input and output roots must differ")

    tokenizer, tokenizer_path, tokenizer_hash = load_tokenizer(args.offline)
    state = ScanState(args.state_db)
    for language in args.languages:
        paths = sorted((args.input_root / language).glob("*.jsonl.gz"))
        if args.max_shards is not None:
            paths = paths[:args.max_shards]
        if not paths:
            raise RuntimeError(f"no input shards for {language}")
        for input_path in paths:
            input_artifact = artifact_path(input_path)
            if state.completed(language, input_artifact):
                print(f"[resume] {input_artifact}", flush=True)
                continue
            output_path = args.output_root / language / input_path.name
            print(f"[*] scan {input_artifact}", flush=True)
            meta = process_shard(state, language, input_path, output_path, tokenizer, args.sample_limit)
            print(
                f"[+] {language}: {meta['metrics']['output_documents']:,} docs / "
                f"{meta['metrics']['output_qwen_tokens']:,} tokens",
                flush=True,
            )
            state.write_samples(args.samples)

    result = aggregate(state)
    summary = {
        "summary_id": "REPLAY-SECRET-PII-SCAN-V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "policy": {
            "secret": "private-key document drop; other accepted secret matches redact",
            "pii": "redact, do not drop",
            "post_redaction": ["HELDOUT-BUCKET-V1", "exact_dedup", "exact_Qwen_token_recount_if_changed"],
        },
        "tokenizer": {
            "repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": tokenizer_hash, "add_special_tokens": False,
        },
        **result,
        "sample_files": state.write_samples(args.samples),
        "limitations": [
            "Regex/structure-based secret and PII detection cannot guarantee complete recall.",
            "Public IPv4 detection requires network context to avoid decimal-section and version false positives.",
            "Phone detection requires a plus sign, parenthesised area code, or nearby phone cue.",
            "Private-key policy detects PEM armour; unarmoured key material is not detected.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temp = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.summary)
    print(f"[PASS] wrote {args.summary}", flush=True)


if __name__ == "__main__":
    main()
