#!/usr/bin/env python3
"""Build a secret/PII-redacted, language-balanced code replay pool.

The builder keeps Task-A data immutable, imports its separately scanned code
shards as candidates, then downloads unused GitHub-Code-Clean shards in a
deterministic farthest-first spread until every pre-registered language-group
quota is available.  Selection is token-based and source-round-robin.

This is data preparation, not scientific evidence.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import pii_filter_en  # noqa: E402
import secret_scan  # noqa: E402
from apply_scan import ScanState  # noqa: E402
from build_replay import (  # noqa: E402
    ALLOWED_CODE_LICENSES,
    SOURCES,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    artifact_path,
    deterministic_gzip_text,
    encode_lengths,
    iter_parquet,
    load_tokenizer,
    normalise_license,
    sha256_file,
)
from heldout_rule import RULE_ID, doc_hash, is_heldout  # noqa: E402

SCIENTIFIC_EVIDENCE_ALLOWED = False
SOURCE = SOURCES["code"]
TOTAL_TARGET = 1_000_000_000

# The two 15% groups are deliberately pre-registered at 14.5%.  This remains
# within the task's ±3 pp tolerance and leaves a safety margin below the hard
# 15% cap after whole-document quota overshoot.
GROUP_TARGET_FRACTIONS: Mapping[str, float] = {
    "python": 0.145,
    "javascript_typescript": 0.145,
    "java": 0.120,
    "c_cpp": 0.120,
    "csharp": 0.040,
    "long_tail": 0.260,
    "markdown": 0.050,
    "html": 0.050,
    "css_config": 0.070,
}
GROUP_TARGETS = {key: int(TOTAL_TARGET * value) for key, value in GROUP_TARGET_FRACTIONS.items()}

CONFIG_LANGUAGES = {
    "css", "makefile", "dockerfile", "yaml", "yml", "toml", "cmake", "json",
    "xml", "ini", "hcl", "nix", "gradle", "properties",
}


def normalise_language(value: object) -> str:
    return re.sub(r"[\s_.#-]+", "", str(value or "").strip().lower())


def code_group(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"c#", "c sharp", "c-sharp"}:
        return "csharp"
    lang = normalise_language(value)
    if lang == "python":
        return "python"
    if lang in {"javascript", "typescript"}:
        return "javascript_typescript"
    if lang == "java":
        return "java"
    if lang in {"c", "c++", "cpp"}:
        return "c_cpp"
    if lang in {"csharp", "cs"}:
        return "csharp"
    if lang == "markdown":
        return "markdown"
    if lang == "html":
        return "html"
    if lang in {normalise_language(item) for item in CONFIG_LANGUAGES}:
        return "css_config"
    return "long_tail"


def source_index(source_file: str) -> int:
    match = re.search(r"train-(\d{5})-of-00880\.parquet$", source_file)
    if not match:
        raise ValueError(f"unrecognised source shard {source_file!r}")
    return int(match.group(1))


def farthest_first_indices(used: Iterable[int], total: int = 880) -> List[int]:
    """Deterministic full-range order; every next shard maximises source spread."""
    selected = set(int(x) for x in used)
    remaining = set(range(total)) - selected
    order: List[int] = []
    while remaining:
        if not selected:
            chosen = 0
        else:
            chosen = max(
                remaining,
                key=lambda candidate: (min(abs(candidate - old) for old in selected), -candidate),
            )
        order.append(chosen)
        selected.add(chosen)
        remaining.remove(chosen)
    return order


class RebalanceState:
    def __init__(self, scan_state: ScanState):
        self.scan_state = scan_state
        self.db = scan_state.db
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidate_seen (
              doc_sha256 TEXT PRIMARY KEY,
              source_file TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidate_completed (
              source_kind TEXT NOT NULL,
              source_file TEXT NOT NULL,
              candidate_file TEXT NOT NULL,
              candidate_sha256 TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              group_tokens_json TEXT NOT NULL,
              PRIMARY KEY(source_kind,source_file)
            );
            CREATE TABLE IF NOT EXISTS selected_completed (
              code_group TEXT PRIMARY KEY,
              output_file TEXT NOT NULL,
              output_sha256 TEXT NOT NULL,
              documents INTEGER NOT NULL,
              qwen_tokens INTEGER NOT NULL,
              utf8_bytes INTEGER NOT NULL,
              language_tokens_json TEXT NOT NULL,
              source_tokens_json TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    def candidate_done(self, source_kind: str, source_file: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM candidate_completed WHERE source_kind=? AND source_file=?",
            (source_kind, source_file),
        ).fetchone() is not None

    def duplicate(self, digest: str) -> bool:
        return self.db.execute("SELECT 1 FROM candidate_seen WHERE doc_sha256=?", (digest,)).fetchone() is not None

    def remember(self, digest: str, source_file: str) -> None:
        self.db.execute("INSERT INTO candidate_seen VALUES(?,?)", (digest, source_file))

    def commit_candidate(
        self,
        source_kind: str,
        source_file: str,
        candidate_file: str,
        candidate_sha: str,
        metrics: Mapping[str, object],
        group_tokens: Mapping[str, int],
    ) -> None:
        self.db.execute(
            "INSERT INTO candidate_completed VALUES(?,?,?,?,?,?)",
            (source_kind, source_file, candidate_file, candidate_sha,
             json.dumps(metrics, sort_keys=True), json.dumps(group_tokens, sort_keys=True)),
        )
        self.db.commit()

    def candidate_rows(self) -> List[Tuple[object, ...]]:
        return self.db.execute(
            "SELECT source_kind,source_file,candidate_file,candidate_sha256,metrics_json,group_tokens_json "
            "FROM candidate_completed ORDER BY source_kind,source_file"
        ).fetchall()

    def group_totals(self) -> collections.Counter:
        result = collections.Counter()
        for row in self.db.execute("SELECT group_tokens_json FROM candidate_completed"):
            result.update(json.loads(row[0]))
        return result

    def selected_done(self, group: str) -> bool:
        return self.db.execute("SELECT 1 FROM selected_completed WHERE code_group=?", (group,)).fetchone() is not None

    def commit_selected(
        self, group: str, output_file: str, output_sha: str, documents: int,
        tokens: int, byte_count: int, languages: Mapping[str, int], sources: Mapping[str, int],
    ) -> None:
        self.db.execute(
            "INSERT INTO selected_completed VALUES(?,?,?,?,?,?,?,?)",
            (group, output_file, output_sha, documents, tokens, byte_count,
             json.dumps(languages, sort_keys=True), json.dumps(sources, sort_keys=True)),
        )
        self.db.commit()

    def selected_rows(self) -> List[Tuple[object, ...]]:
        return self.db.execute(
            "SELECT code_group,output_file,output_sha256,documents,qwen_tokens,utf8_bytes,"
            "language_tokens_json,source_tokens_json FROM selected_completed ORDER BY code_group"
        ).fetchall()

    def rollback(self) -> None:
        self.db.rollback()


def _counter(value: Mapping[str, int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(value.items())}


def _write_candidate_record(output, row: Mapping[str, object], group: str) -> None:
    result = dict(row)
    result["scientific_evidence_allowed"] = False
    result["code_group"] = group
    result["rebalance_policy"] = "CODE-LANGUAGE-MIX-V1"
    output.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def import_scanned_candidate(
    state: RebalanceState,
    input_path: Path,
    output_path: Path,
) -> Mapping[str, object]:
    source_file_key = artifact_path(input_path)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    if output_path.exists():
        raise RuntimeError(f"orphan candidate exists: {output_path}")
    metrics = collections.Counter()
    groups = collections.Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(input_path, "rt", encoding="utf-8") as source, deterministic_gzip_text(temp) as output:
            for line_no, line in enumerate(source, 1):
                row = json.loads(line)
                text = row.get("text")
                if not isinstance(text, str):
                    raise RuntimeError(f"missing text at {input_path}:{line_no}")
                metrics["input_documents"] += 1
                metrics["input_qwen_tokens"] += int(row["qwen_tokens"])
                digest = doc_hash(text)
                if digest != row.get("doc_sha256"):
                    raise RuntimeError(f"hash mismatch at {input_path}:{line_no}")
                if is_heldout(text):
                    metrics["drop_heldout"] += 1
                    continue
                if normalise_license(row.get("license")) not in ALLOWED_CODE_LICENSES:
                    metrics["drop_disallowed_license"] += 1
                    continue
                if state.duplicate(digest):
                    metrics["drop_exact_duplicate"] += 1
                    continue
                state.remember(digest, source_file_key)
                group = code_group(row.get("language"))
                tokens = int(row["qwen_tokens"])
                groups[group] += tokens
                metrics["output_documents"] += 1
                metrics["output_qwen_tokens"] += tokens
                metrics["output_bytes"] += int(row["utf8_bytes"])
                _write_candidate_record(output, row, group)
        os.replace(temp, output_path)
        metrics_record = {"scientific_evidence_allowed": False, **_counter(metrics)}
        state.commit_candidate(
            "task_a_scanned", source_file_key, artifact_path(output_path), sha256_file(output_path),
            metrics_record, _counter(groups),
        )
        return {"metrics": metrics_record, "group_tokens": _counter(groups)}
    except BaseException:
        state.rollback()
        if temp.exists():
            temp.unlink()
        raise


def process_raw_candidate(
    state: RebalanceState,
    source_path: Path,
    source_file: str,
    output_path: Path,
    tokenizer,
    sample_limit: int,
    batch_size: int,
) -> Mapping[str, object]:
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    if output_path.exists():
        raise RuntimeError(f"orphan candidate exists: {output_path}")
    metrics = collections.Counter()
    groups = collections.Counter()
    secret_hits = collections.Counter()
    secret_docs = collections.Counter()
    pii_hits = collections.Counter()
    pii_docs = collections.Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = (SOURCE.text_field,) + SOURCE.metadata_fields
    pending: List[Mapping[str, object]] = []

    def process_batch(rows: Sequence[Mapping[str, object]], output) -> None:
        texts = [row.get(SOURCE.text_field) if isinstance(row.get(SOURCE.text_field), str) else "" for row in rows]
        raw_lengths = encode_lengths(tokenizer, texts)
        for row, raw_tokens in zip(rows, raw_lengths):
            text = row.get(SOURCE.text_field)
            metrics["input_documents"] += 1
            metrics["input_qwen_tokens"] += raw_tokens
            metrics["input_bytes"] += len(text.encode("utf-8")) if isinstance(text, str) else 0
            if not isinstance(text, str) or not text:
                metrics["drop_missing_text"] += 1
                continue

            # HELDOUT-BUCKET-V1 is the first content eligibility decision.
            if is_heldout(text):
                metrics["drop_heldout"] += 1
                continue
            if len(text) < SOURCE.min_chars:
                metrics["drop_too_short"] += 1
                continue
            if len(text) > SOURCE.max_chars:
                metrics["drop_too_long"] += 1
                continue
            if normalise_license(row.get("license")) not in ALLOWED_CODE_LICENSES:
                metrics["drop_disallowed_license"] += 1
                continue
            original_digest = doc_hash(text)
            if state.duplicate(original_digest):
                metrics["drop_exact_duplicate"] += 1
                continue

            secret_text, sec_counts, sec_events = secret_scan.redact(text)
            secret_hits.update(sec_counts)
            for kind in sec_counts:
                secret_docs[kind] += 1
            for index, event in enumerate(sec_events):
                state.scan_state.sample("secret", str(event["type"]), original_digest, index, event, "code", source_file, sample_limit)
            if secret_text is None:
                metrics["drop_private_key_document"] += 1
                continue
            final_text, these_pii, pii_events = pii_filter_en.redact(secret_text)
            pii_hits.update(these_pii)
            for kind in these_pii:
                pii_docs[kind] += 1
            for index, event in enumerate(pii_events):
                state.scan_state.sample("pii", str(event["type"]), original_digest, index, event, "code", source_file, sample_limit)

            final_digest = doc_hash(final_text)
            if final_text != text and is_heldout(final_text):
                metrics["drop_post_redaction_heldout"] += 1
                continue
            if final_digest != original_digest and state.duplicate(final_digest):
                metrics["drop_post_redaction_exact_duplicate"] += 1
                continue
            final_tokens = raw_tokens if final_text == text else encode_lengths(tokenizer, [final_text])[0]
            if final_tokens <= 0:
                metrics["drop_zero_tokens"] += 1
                continue
            state.remember(final_digest, source_file)
            result = {
                "scientific_evidence_allowed": False,
                "text": final_text,
                "source_doc_sha256": original_digest,
                "doc_sha256": final_digest,
                "utf8_bytes": len(final_text.encode("utf-8")),
                "qwen_tokens": final_tokens,
                "source_repo": SOURCE.repo,
                "source_revision": SOURCE.revision,
                "source_file": source_file,
                "scan_policy": "SECRET-PII-EN-V1",
            }
            for field in SOURCE.metadata_fields:
                if row.get(field) is not None:
                    result[field] = row[field]
            group = code_group(result.get("language"))
            groups[group] += final_tokens
            metrics["output_documents"] += 1
            metrics["output_qwen_tokens"] += final_tokens
            metrics["output_bytes"] += result["utf8_bytes"]
            _write_candidate_record(output, result, group)

    try:
        with deterministic_gzip_text(temp) as output:
            for row in iter_parquet(source_path, columns, batch_size):
                pending.append(row)
                if len(pending) >= batch_size:
                    process_batch(pending, output)
                    pending.clear()
            if pending:
                process_batch(pending, output)
        os.replace(temp, output_path)
        metrics_record = {
            "scientific_evidence_allowed": False,
            **_counter(metrics),
            "secret_hits": _counter(secret_hits),
            "secret_documents": _counter(secret_docs),
            "pii_hits": _counter(pii_hits),
            "pii_documents": _counter(pii_docs),
        }
        state.commit_candidate(
            "new_raw", source_file, artifact_path(output_path), sha256_file(output_path),
            metrics_record, _counter(groups),
        )
        return {"metrics": metrics_record, "group_tokens": _counter(groups)}
    except BaseException:
        state.rollback()
        if temp.exists():
            temp.unlink()
        raise


def quotas_met(totals: Mapping[str, int]) -> bool:
    return all(int(totals.get(group, 0)) >= target for group, target in GROUP_TARGETS.items())


def _iter_group(path: Path, group: str) -> Iterator[Mapping[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("code_group") == group:
                yield row


def round_robin_group(paths: Sequence[Path], group: str) -> Iterator[Mapping[str, object]]:
    active = [iter(_iter_group(path, group)) for path in paths]
    while active:
        next_active = []
        for iterator in active:
            try:
                yield next(iterator)
                next_active.append(iterator)
            except StopIteration:
                pass
        active = next_active


def select_group(
    state: RebalanceState,
    group: str,
    target: int,
    candidate_paths: Sequence[Path],
    output_path: Path,
) -> Mapping[str, object]:
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    if output_path.exists():
        raise RuntimeError(f"orphan selected output exists: {output_path}")
    documents = tokens = byte_count = 0
    languages = collections.Counter()
    sources = collections.Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with deterministic_gzip_text(temp) as output:
            for row in round_robin_group(candidate_paths, group):
                output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                row_tokens = int(row["qwen_tokens"])
                documents += 1
                tokens += row_tokens
                byte_count += int(row["utf8_bytes"])
                languages[str(row.get("language") or "UNKNOWN")] += row_tokens
                sources[str(row.get("source_file") or "UNKNOWN")] += row_tokens
                if tokens >= target:
                    break
        if tokens < target:
            raise RuntimeError(f"candidate exhaustion for {group}: {tokens:,}/{target:,}")
        os.replace(temp, output_path)
        output_sha = sha256_file(output_path)
        state.commit_selected(
            group, artifact_path(output_path), output_sha, documents, tokens, byte_count,
            _counter(languages), _counter(sources),
        )
        return {"documents": documents, "qwen_tokens": tokens, "utf8_bytes": byte_count}
    except BaseException:
        state.rollback()
        if temp.exists():
            temp.unlink()
        raise


def build_summary(state: RebalanceState) -> Mapping[str, object]:
    candidate_rows = []
    candidate_totals = collections.Counter()
    new_files = []
    scan_secret = collections.Counter()
    scan_secret_docs = collections.Counter()
    scan_pii = collections.Counter()
    scan_pii_docs = collections.Counter()
    for source_kind, source_file, candidate_file, candidate_sha, metrics_json, groups_json in state.candidate_rows():
        metrics = json.loads(metrics_json)
        groups = json.loads(groups_json)
        candidate_totals.update(groups)
        if source_kind == "new_raw":
            new_files.append(source_file)
            scan_secret.update(metrics.get("secret_hits", {}))
            scan_secret_docs.update(metrics.get("secret_documents", {}))
            scan_pii.update(metrics.get("pii_hits", {}))
            scan_pii_docs.update(metrics.get("pii_documents", {}))
        candidate_rows.append({
            "source_kind": source_kind, "source_file": source_file,
            "candidate_file": candidate_file, "candidate_sha256": candidate_sha,
            "metrics": metrics, "group_tokens": groups,
        })

    final_groups = {}
    final_languages = collections.Counter()
    final_sources = collections.Counter()
    total_docs = total_tokens = total_bytes = 0
    outputs = []
    for group, output_file, output_sha, docs, tokens, byte_count, languages_json, sources_json in state.selected_rows():
        langs = json.loads(languages_json)
        sources = json.loads(sources_json)
        final_groups[group] = int(tokens)
        final_languages.update(langs)
        final_sources.update(sources)
        total_docs += int(docs)
        total_tokens += int(tokens)
        total_bytes += int(byte_count)
        outputs.append({
            "code_group": group, "output_file": output_file, "output_sha256": output_sha,
            "documents": docs, "qwen_tokens": tokens, "utf8_bytes": byte_count,
        })

    return {
        "candidate_group_tokens": _counter(candidate_totals),
        "candidate_shards": candidate_rows,
        "new_source_files": sorted(new_files, key=source_index),
        "new_source_scan": {
            "secret_hits": _counter(scan_secret), "secret_documents": _counter(scan_secret_docs),
            "pii_hits": _counter(scan_pii), "pii_documents": _counter(scan_pii_docs),
        },
        "final": {
            "documents": total_docs, "qwen_tokens": total_tokens, "utf8_bytes": total_bytes,
            "group_tokens": _counter(final_groups),
            "group_percent": {k: round(v / total_tokens * 100, 6) for k, v in sorted(final_groups.items())} if total_tokens else {},
            "language_tokens": _counter(final_languages),
            "language_percent": {k: round(v / total_tokens * 100, 6) for k, v in sorted(final_languages.items())} if total_tokens else {},
            "source_file_tokens": _counter(final_sources),
            "outputs": outputs,
        },
    }


def validate_final(summary: Mapping[str, object]) -> None:
    final = summary["final"]
    total = int(final["qwen_tokens"])
    if total < TOTAL_TARGET:
        raise RuntimeError(f"code target not met: {total:,}/{TOTAL_TARGET:,}")
    group_percent = final["group_percent"]
    for group, expected in GROUP_TARGET_FRACTIONS.items():
        actual = float(group_percent.get(group, 0.0))
        task_target = {
            "python": 15, "javascript_typescript": 15, "java": 12, "c_cpp": 12,
            "csharp": 4, "long_tail": 25, "markdown": 5, "html": 5, "css_config": 7,
        }[group]
        if abs(actual - task_target) > 3.0:
            raise RuntimeError(f"group {group} outside tolerance: {actual:.3f}% vs {task_target}%")
    if float(group_percent.get("long_tail", 0)) < 20.0:
        raise RuntimeError("long-tail share below 20%")
    for language, share in final["language_percent"].items():
        if float(share) > 15.0:
            raise RuntimeError(f"individual language {language} exceeds 15%: {share}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanned-code-root", type=Path, default=ROOT / "data" / "scanned_replay_stage" / "code")
    parser.add_argument("--candidate-root", type=Path, default=ROOT / "data" / "rebalance_code_candidates")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "clean_replay_v2" / "code")
    parser.add_argument("--state-db", type=Path, default=ROOT / "data" / "scan_replay_v2_state.sqlite3")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "rebalance_code_summary.json")
    parser.add_argument("--samples", type=Path, default=ROOT / "data" / "scan_samples")
    parser.add_argument("--max-new-shards", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    tokenizer, tokenizer_path, tokenizer_hash = load_tokenizer(args.offline)
    scan_state = ScanState(args.state_db)
    state = RebalanceState(scan_state)

    scanned_paths = sorted(args.scanned_code_root.glob("*.jsonl.gz"))
    if not scanned_paths:
        raise RuntimeError(f"no scanned Task-A code shards in {args.scanned_code_root}")
    for input_path in scanned_paths:
        key = artifact_path(input_path)
        if state.candidate_done("task_a_scanned", key):
            print(f"[resume] candidate {key}", flush=True)
            continue
        out = args.candidate_root / f"base-{input_path.name}"
        print(f"[*] import scanned candidate {key}", flush=True)
        import_scanned_candidate(state, input_path, out)

    used_indices = {
        source_index(str(row.get("source_file")))
        for path in scanned_paths
        for row in [json.loads(gzip.open(path, "rt", encoding="utf-8").readline())]
    }
    order = farthest_first_indices(used_indices)
    from huggingface_hub import hf_hub_download

    fetched_this_run = 0
    for index in order:
        totals = state.group_totals()
        if quotas_met(totals):
            break
        if fetched_this_run >= args.max_new_shards:
            break
        source_file = f"data/train-{index:05d}-of-00880.parquet"
        if state.candidate_done("new_raw", source_file):
            continue
        fetched_this_run += 1
        print(
            f"[*] fetch {source_file}; deficits="
            + json.dumps({k: max(0, GROUP_TARGETS[k]-totals.get(k, 0)) for k in GROUP_TARGETS}),
            flush=True,
        )
        local = Path(hf_hub_download(
            SOURCE.repo, source_file, repo_type="dataset", revision=SOURCE.revision,
            local_files_only=args.offline,
        ))
        output = args.candidate_root / f"new-{index:05d}.jsonl.gz"
        result = process_raw_candidate(
            state, local, source_file, output, tokenizer, args.sample_limit, args.batch_size,
        )
        print(f"[+] candidate {source_file}: {result['metrics'].get('output_qwen_tokens',0):,} tokens", flush=True)
        scan_state.write_samples(args.samples)

    totals = state.group_totals()
    if not quotas_met(totals):
        deficits = {k: GROUP_TARGETS[k] - int(totals.get(k, 0)) for k in GROUP_TARGETS if totals.get(k, 0) < GROUP_TARGETS[k]}
        raise RuntimeError(f"candidate quotas not met after {fetched_this_run} new shards: {deficits}")

    candidate_paths = [ROOT / str(row[2]) for row in state.candidate_rows()]
    for group, target in GROUP_TARGETS.items():
        if state.selected_done(group):
            print(f"[resume] selected {group}", flush=True)
            continue
        output = args.output_root / f"code-{group}.jsonl.gz"
        print(f"[*] select {group}: target={target:,}", flush=True)
        result = select_group(state, group, target, candidate_paths, output)
        print(f"[+] {group}: {result['qwen_tokens']:,} tokens", flush=True)

    summary = {
        "summary_id": "CODE-REBALANCE-V1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "source": {
            "repo": SOURCE.repo, "revision": SOURCE.revision,
            "dataset_level_license": SOURCE.dataset_license,
            "per_row_license_allowlist": sorted(ALLOWED_CODE_LICENSES),
        },
        "heldout_rule": RULE_ID,
        "tokenizer": {
            "repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": tokenizer_hash, "add_special_tokens": False,
        },
        "selection": {
            "target_total_qwen_tokens": TOTAL_TARGET,
            "pre_registered_group_fractions": GROUP_TARGET_FRACTIONS,
            "whole_document_selection": True,
            "source_order": "round-robin across deterministic farthest-first source shards",
        },
        "rationale": {
            "reason": "Correct a source-sampling artifact so the replay slice better reflects broad real-world code activity; not benchmark optimisation.",
            "external_reference": {
                "title": "GitHub Octoverse 2025",
                "url": "https://github.blog/news-insights/octoverse/octoverse-a-new-developer-joins-github-every-second-as-ai-leads-typescript-to-1/",
                "finding": "TypeScript, Python, JavaScript, Java and C# lead GitHub contributor activity; nearly 80% of new repositories use Python, JavaScript, TypeScript, Java, C++ or C#.",
                "accessed": "2026-08-21",
            },
            "explicitly_not_used": "HumanEval or any downstream benchmark composition",
        },
        **build_summary(state),
        "limitations": [
            "Target shares are policy choices informed by broad activity rankings, not estimates of a unique true token distribution.",
            "Per-row licenses are inherited from dataset metadata and are not independently revalidated against every repository.",
            "Exact deduplication only; near-duplicate and generated-code filtering remain open.",
            "Selected files are grouped by language bucket; downstream packing must shuffle globally.",
        ],
    }
    validate_final(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temp = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.summary)
    scan_state.write_samples(args.samples)
    print(f"[PASS] wrote {args.summary}", flush=True)


if __name__ == "__main__":
    main()
