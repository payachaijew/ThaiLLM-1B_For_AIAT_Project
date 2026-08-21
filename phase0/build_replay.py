#!/usr/bin/env python3
"""Build English, permissive-code and math replay pools for Thai CPT.

The pipeline is CPU-only and downloads parquet shards one at a time at immutable
dataset revisions.  It applies HELDOUT-BUCKET-V1 before every other content
filter, exact-deduplicates with ``heldout_rule.doc_hash``, counts tokens with the
Qwen3-1.7B tokenizer (no special tokens), and checkpoints at source-shard
boundaries in SQLite.  A failed shard is written to a temporary file and can be
reprocessed without duplicating records.

Examples:
  python3 phase0/build_replay.py --languages en code math
  python3 phase0/build_replay.py --languages code --max-files 1 --target code=1000000

This prepares data; it is not scientific evidence.
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
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from heldout_rule import RULE_ID, doc_hash, is_heldout  # noqa: E402

TOKENIZER_REPO = "Qwen/Qwen3-1.7B-Base"
TOKENIZER_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DEFAULT_TARGETS = {"en": 3_500_000_000, "code": 1_000_000_000, "math": 500_000_000}
ALLOWED_CODE_LICENSES = frozenset({"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"})
SCIENTIFIC_EVIDENCE_ALLOWED = False


def stratified_order(n: int, anchors: int = 8) -> List[int]:
    """Return every index once, beginning with indices spread over the range."""
    if n <= 0:
        return []
    anchor_count = min(n, anchors)
    if anchor_count == 1:
        first = [0]
    else:
        first = [round(i * (n - 1) / (anchor_count - 1)) for i in range(anchor_count)]
    seen = set(first)
    return first + [i for i in range(n) if i not in seen]


@dataclass(frozen=True)
class SourceSpec:
    language: str
    repo: str
    revision: str
    subset: str
    dataset_license: str
    additional_terms: str
    files: Tuple[str, ...]
    text_field: str
    min_chars: int
    max_chars: int
    metadata_fields: Tuple[str, ...]


_EN_FILES = tuple(f"sample/10BT/{i:03d}_00000.parquet" for i in stratified_order(14))
_CODE_FILES = tuple(f"data/train-{i:05d}-of-00880.parquet" for i in stratified_order(880))
_MATH_FILES = tuple(
    f"finemath-4plus/train-{i:05d}-of-00064.parquet" for i in stratified_order(64)
)

SOURCES: Dict[str, SourceSpec] = {
    "en": SourceSpec(
        language="en",
        repo="HuggingFaceFW/fineweb-edu",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        subset="sample/10BT",
        dataset_license="ODC-By-1.0",
        additional_terms="CommonCrawl Terms of Use",
        files=_EN_FILES,
        text_field="text",
        min_chars=500,
        max_chars=100_000,
        metadata_fields=("id", "url", "dump", "date", "language", "score", "int_score"),
    ),
    "code": SourceSpec(
        language="code",
        repo="codeparrot/github-code-clean",
        revision="c48d40f9e70f0196f8236901ee35807f7d6c44c0",
        subset="data",
        dataset_license="Apache-2.0 (dataset packaging only)",
        additional_terms=(
            "Underlying files retain their repository licenses; only MIT, Apache-2.0, "
            "BSD-2-Clause and BSD-3-Clause rows are retained."
        ),
        files=_CODE_FILES,
        text_field="code",
        min_chars=100,
        max_chars=200_000,
        metadata_fields=("repo_name", "path", "language", "license"),
    ),
    "math": SourceSpec(
        language="math",
        repo="HuggingFaceTB/finemath",
        revision="e92b25a616738fe95dc186b64dfb19f9c8525594",
        subset="finemath-4plus",
        dataset_license="ODC-By-1.0",
        additional_terms="CommonCrawl Terms of Use",
        files=_MATH_FILES,
        text_field="text",
        min_chars=200,
        max_chars=100_000,
        metadata_fields=("url", "crawl", "language", "score", "int_score"),
    ),
}


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


@contextlib.contextmanager
def deterministic_gzip_text(path: Path):
    raw = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(gz, encoding="utf-8")
    try:
        yield text
    finally:
        text.close()
        raw.close()


def artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalise_license(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def pre_token_filter(
    text: object,
    spec: SourceSpec,
    row: Mapping[str, object],
    duplicate: bool = False,
) -> Optional[str]:
    """Return a drop reason. HELDOUT is deliberately the first content test."""
    if not isinstance(text, str) or not text:
        return "missing_text"
    if is_heldout(text):
        return "heldout_bucket"
    n = len(text)
    if n < spec.min_chars:
        return "too_short"
    if n > spec.max_chars:
        return "too_long"
    if spec.language == "code" and normalise_license(row.get("license")) not in ALLOWED_CODE_LICENSES:
        return "disallowed_license"
    if duplicate:
        return "exact_duplicate"
    return None


def parse_targets(values: Sequence[str]) -> Dict[str, int]:
    result = dict(DEFAULT_TARGETS)
    for value in values:
        try:
            language, count = value.split("=", 1)
            count_i = int(count)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid --target {value!r}; use en=3500000000") from exc
        if language not in SOURCES or count_i <= 0:
            raise argparse.ArgumentTypeError(f"invalid --target {value!r}")
        result[language] = count_i
    return result


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
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
              source_file TEXT NOT NULL,
              output_file TEXT NOT NULL,
              output_sha256 TEXT NOT NULL,
              input_documents INTEGER NOT NULL,
              input_bytes INTEGER NOT NULL,
              output_documents INTEGER NOT NULL,
              output_bytes INTEGER NOT NULL,
              qwen_tokens INTEGER NOT NULL,
              dropped_json TEXT NOT NULL,
              dropped_bytes_json TEXT NOT NULL,
              PRIMARY KEY(language, source_file)
            );
            CREATE TABLE IF NOT EXISTS drop_samples (
              language TEXT NOT NULL,
              reason TEXT NOT NULL,
              sample_key TEXT NOT NULL,
              record_json TEXT NOT NULL,
              PRIMARY KEY(language, reason, sample_key)
            );
            """
        )
        self.db.commit()

    def is_completed(self, language: str, source_file: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM completed WHERE language=? AND source_file=?", (language, source_file)
        ).fetchone()
        return row is not None

    def is_duplicate(self, digest: str) -> bool:
        return self.db.execute("SELECT 1 FROM seen WHERE doc_sha256=?", (digest,)).fetchone() is not None

    def remember(self, digest: str, language: str) -> None:
        self.db.execute("INSERT INTO seen(doc_sha256,language) VALUES(?,?)", (digest, language))

    def sample_drop(
        self, language: str, reason: str, digest: str, text: str, row: Mapping[str, object], limit: int
    ) -> None:
        record = {
            "scientific_evidence_allowed": False,
            "language": language,
            "reason": reason,
            "doc_sha256": digest,
            "chars": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
            "source_metadata": {k: row.get(k) for k in ("url", "repo_name", "path", "language", "license") if k in row},
            "text_preview": text[:400],
        }
        self.db.execute(
            "INSERT OR IGNORE INTO drop_samples VALUES(?,?,?,?)",
            (language, reason, digest, json.dumps(record, ensure_ascii=False)),
        )
        self.db.execute(
            """DELETE FROM drop_samples WHERE rowid IN (
                 SELECT rowid FROM drop_samples WHERE language=? AND reason=?
                 ORDER BY sample_key DESC LIMIT -1 OFFSET ?
               )""",
            (language, reason, limit),
        )

    def totals(self, language: str) -> Mapping[str, int]:
        row = self.db.execute(
            """SELECT COALESCE(SUM(input_documents),0),COALESCE(SUM(input_bytes),0),
                      COALESCE(SUM(output_documents),0),COALESCE(SUM(output_bytes),0),
                      COALESCE(SUM(qwen_tokens),0),COUNT(*)
               FROM completed WHERE language=?""",
            (language,),
        ).fetchone()
        return dict(zip(("input_documents", "input_bytes", "output_documents", "output_bytes", "qwen_tokens", "source_shards"), row))

    def commit_file(self, meta: Mapping[str, object]) -> None:
        self.db.execute(
            """INSERT INTO completed VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                meta["language"], meta["source_file"], meta["output_file"], meta["output_sha256"],
                meta["input_documents"], meta["input_bytes"], meta["output_documents"],
                meta["output_bytes"], meta["qwen_tokens"], json.dumps(meta["dropped_documents"]),
                json.dumps(meta["dropped_bytes"]),
            ),
        )
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def manifest_rows(self) -> List[Tuple[object, ...]]:
        return self.db.execute(
            """SELECT language,source_file,output_file,output_sha256,input_documents,input_bytes,
                      output_documents,output_bytes,qwen_tokens,dropped_json,dropped_bytes_json
               FROM completed ORDER BY language,source_file"""
        ).fetchall()

    def write_drop_samples(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        rows = self.db.execute(
            "SELECT language,reason,record_json FROM drop_samples ORDER BY language,reason,sample_key"
        ).fetchall()
        grouped: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        for language, reason, record in rows:
            grouped[(language, reason)].append(record)
        for (language, reason), records in grouped.items():
            path = root / f"{language}-{reason}.jsonl"
            path.write_text("\n".join(records) + "\n", encoding="utf-8")


def load_tokenizer(offline: bool):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = Path(
        hf_hub_download(
            TOKENIZER_REPO,
            "tokenizer.json",
            revision=TOKENIZER_REVISION,
            local_files_only=offline,
        )
    )
    tokenizer = Tokenizer.from_file(str(path))
    return tokenizer, path, sha256_file(path)


def iter_parquet(path: Path, columns: Sequence[str], batch_size: int) -> Iterator[Dict[str, object]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    required = [c for c in columns if c in available]
    if not required or columns[0] not in available:
        raise RuntimeError(f"required text column {columns[0]!r} missing in {path}: {sorted(available)}")
    for batch in pf.iter_batches(columns=required, batch_size=batch_size):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            yield {key: data[key][i] for key in data}


def encode_lengths(tokenizer, texts: Sequence[str]) -> List[int]:
    # Explicitly disable special tokens.  A previous screen silently counted BOS.
    return [len(enc.ids) for enc in tokenizer.encode_batch(list(texts), add_special_tokens=False)]


def process_file(
    state: State,
    spec: SourceSpec,
    source_path: Path,
    source_file: str,
    output_file: Path,
    tokenizer,
    batch_size: int,
    sample_limit: int,
) -> Mapping[str, object]:
    temp = output_file.with_suffix(output_file.suffix + ".tmp")
    meta_path = output_file.with_suffix(output_file.suffix + ".meta.json")
    if temp.exists():
        temp.unlink()
    if output_file.exists() or meta_path.exists():
        raise RuntimeError(
            f"orphan output exists without committed state: {output_file}; "
            "move it aside before retrying (fail-closed)"
        )

    counters = collections.Counter()
    drop = collections.Counter()
    drop_bytes = collections.Counter()
    pending: List[Tuple[str, str, Mapping[str, object], int]] = []
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def record_drop(reason: str, digest: str, text: str, row: Mapping[str, object], byte_count: int) -> None:
        drop[reason] += 1
        drop_bytes[reason] += byte_count
        state.sample_drop(spec.language, reason, digest, text, row, sample_limit)

    def flush(out) -> None:
        if not pending:
            return
        texts = [item[1] for item in pending]
        lengths = encode_lengths(tokenizer, texts)
        for (digest, text, row, byte_count), token_count in zip(pending, lengths):
            if token_count <= 0:
                record_drop("zero_tokens", digest, text, row, byte_count)
                continue
            state.remember(digest, spec.language)
            result = {
                "text": text,
                "doc_sha256": digest,
                "utf8_bytes": byte_count,
                "qwen_tokens": token_count,
                "source_repo": spec.repo,
                "source_revision": spec.revision,
                "source_file": source_file,
            }
            for key in spec.metadata_fields:
                if key in row and row[key] is not None:
                    result[key] = row[key]
            out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            counters["output_documents"] += 1
            counters["output_bytes"] += byte_count
            counters["qwen_tokens"] += token_count
        pending.clear()

    try:
        with deterministic_gzip_text(temp) as out:
            columns = (spec.text_field,) + spec.metadata_fields
            batch_seen: set = set()
            for row in iter_parquet(source_path, columns, batch_size):
                text = row.get(spec.text_field)
                counters["input_documents"] += 1
                byte_count = len(text.encode("utf-8")) if isinstance(text, str) else 0
                counters["input_bytes"] += byte_count
                digest = doc_hash(text) if isinstance(text, str) else hashlib.sha256(b"").hexdigest()

                # Held-out is checked inside pre_token_filter before length/license/dedup.
                basic_reason = pre_token_filter(text, spec, row, duplicate=False)
                if basic_reason:
                    record_drop(basic_reason, digest, text if isinstance(text, str) else "", row, byte_count)
                    continue
                duplicate = digest in batch_seen or state.is_duplicate(digest)
                if duplicate:
                    record_drop("exact_duplicate", digest, text, row, byte_count)
                    continue
                batch_seen.add(digest)
                pending.append((digest, text, row, byte_count))
                if len(pending) >= batch_size:
                    flush(out)
                    batch_seen.clear()
            flush(out)

        os.replace(str(temp), str(output_file))
        meta = {
            "scientific_evidence_allowed": False,
            "language": spec.language,
            "source_file": source_file,
            "output_file": artifact_path(output_file),
            "output_sha256": sha256_file(output_file),
            "input_documents": counters["input_documents"],
            "input_bytes": counters["input_bytes"],
            "output_documents": counters["output_documents"],
            "output_bytes": counters["output_bytes"],
            "qwen_tokens": counters["qwen_tokens"],
            "dropped_documents": dict(drop),
            "dropped_bytes": dict(drop_bytes),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state.commit_file(meta)
        return meta
    except BaseException:
        state.rollback()
        if temp.exists():
            temp.unlink()
        raise


def build_manifest(
    state: State,
    targets: Mapping[str, int],
    tokenizer_path: Path,
    tokenizer_hash: str,
    manifest_path: Path,
) -> Mapping[str, object]:
    source_results: Dict[str, Dict[str, object]] = {}
    aggregate_drop: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    aggregate_drop_bytes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    shard_records = []
    for row in state.manifest_rows():
        language, source_file, output_file, output_sha, in_docs, in_bytes, out_docs, out_bytes, tokens, drops, drop_b = row
        aggregate_drop[language].update(json.loads(drops))
        aggregate_drop_bytes[language].update(json.loads(drop_b))
        shard_records.append({
            "language": language, "source_file": source_file, "output_file": output_file,
            "output_sha256": output_sha, "input_documents": in_docs, "input_bytes": in_bytes,
            "output_documents": out_docs, "output_bytes": out_bytes, "qwen_tokens": tokens,
        })
    for language, spec in SOURCES.items():
        totals = dict(state.totals(language))
        source_results[language] = {
            "repo": spec.repo,
            "revision": spec.revision,
            "subset": spec.subset,
            "dataset_level_license": spec.dataset_license,
            "additional_terms": spec.additional_terms,
            "target_qwen_tokens": targets[language],
            "target_met": totals["qwen_tokens"] >= targets[language],
            "filters": {
                "order": ["heldout_bucket", "length", "code_license_if_applicable", "exact_duplicate"],
                "min_chars": spec.min_chars,
                "max_chars": spec.max_chars,
                "code_license_allowlist": sorted(ALLOWED_CODE_LICENSES) if language == "code" else None,
            },
            "input": {"documents": totals["input_documents"], "utf8_bytes": totals["input_bytes"], "source_shards": totals["source_shards"]},
            "output": {"documents": totals["output_documents"], "utf8_bytes": totals["output_bytes"], "qwen_tokens": totals["qwen_tokens"]},
            "dropped_documents": dict(aggregate_drop[language]),
            "dropped_bytes": dict(aggregate_drop_bytes[language]),
        }
    manifest = {
        "manifest_id": "CLEAN-REPLAY-EN-CODE-MATH-V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "exclusion_rule": RULE_ID,
        "selected_tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "artifact": "tokenizer.json",
            "tokenizer_json_sha256": tokenizer_hash,
            "add_special_tokens": False,
        },
        "sources": source_results,
        "output_shards": shard_records,
        "limitations": [
            "Exact deduplication only; near-duplicate removal is not applied.",
            "FineWeb-Edu and FineMath derive from CommonCrawl and retain source-content risks.",
            "The code dataset card is Apache-2.0, but source files retain per-repository licenses; only the recorded permissive allowlist is retained.",
            "No PII or secrets scanner is applied to replay data in this task.",
            "Token counts are exact for the pinned Qwen3-1.7B tokenizer, but training may sample only a capped subset of each pool.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--target", action="append", default=[], metavar="LANG=TOKENS")
    parser.add_argument("--max-files", type=int, default=None, help="per-language smoke/debug cap")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-drops", type=int, default=20)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "clean_replay")
    parser.add_argument("--state-db", type=Path, default=ROOT / "data" / "clean_replay_state.sqlite3")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "clean_replay_manifest.json")
    args = parser.parse_args()
    targets = parse_targets(args.target)

    from huggingface_hub import hf_hub_download

    tokenizer, tokenizer_path, tokenizer_hash = load_tokenizer(args.offline)
    state = State(args.state_db)
    print(f"[*] tokenizer {TOKENIZER_REPO}@{TOKENIZER_REVISION} sha256={tokenizer_hash}", flush=True)
    for language in args.languages:
        spec = SOURCES[language]
        files_considered = 0
        for source_index, source_file in enumerate(spec.files):
            if state.totals(language)["qwen_tokens"] >= targets[language]:
                break
            if state.is_completed(language, source_file):
                continue
            if args.max_files is not None and files_considered >= args.max_files:
                break
            files_considered += 1
            print(f"[*] {language}: fetch {source_file} @ {spec.revision}", flush=True)
            local = Path(hf_hub_download(
                spec.repo, source_file, repo_type="dataset", revision=spec.revision,
                local_files_only=args.offline,
            ))
            output_index = state.totals(language)["source_shards"]
            output = args.output_root / language / f"{language}-{output_index:05d}.jsonl.gz"
            meta = process_file(state, spec, local, source_file, output, tokenizer, args.batch_size, args.sample_drops)
            total = state.totals(language)["qwen_tokens"]
            print(
                f"[+] {language} shard: {meta['output_documents']:,} docs / "
                f"{meta['qwen_tokens']:,} tokens; cumulative={total:,}/{targets[language]:,}",
                flush=True,
            )
            build_manifest(state, targets, tokenizer_path, tokenizer_hash, args.manifest)
            state.write_drop_samples(ROOT / "data" / "replay_dropped_samples")

    manifest = build_manifest(state, targets, tokenizer_path, tokenizer_hash, args.manifest)
    state.write_drop_samples(ROOT / "data" / "replay_dropped_samples")
    for language in args.languages:
        source = manifest["sources"][language]
        print(
            f"[{('PASS' if source['target_met'] else 'PENDING')}] {language}: "
            f"{source['output']['qwen_tokens']:,}/{source['target_qwen_tokens']:,} Qwen tokens",
            flush=True,
        )


if __name__ == "__main__":
    main()
