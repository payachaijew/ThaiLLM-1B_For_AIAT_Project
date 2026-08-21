#!/usr/bin/env python3
"""Unit tests for replay filtering invariants (no network, no model weights)."""
from __future__ import annotations

import unittest
import sys
import gzip
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_replay
from heldout_rule import is_heldout


def text_for_bucket(heldout: bool) -> str:
    index = 0
    while True:
        text = ("deterministic fixture text " + str(index) + " ") * 30
        if is_heldout(text) is heldout:
            return text
        index += 1


class ReplayPipelineTests(unittest.TestCase):
    def test_stratified_order_is_complete_and_spread(self):
        order = build_replay.stratified_order(14, anchors=5)
        self.assertEqual(sorted(order), list(range(14)))
        self.assertEqual(order[:5], [0, 3, 6, 10, 13])

    def test_heldout_precedes_code_license_and_length(self):
        spec = build_replay.SOURCES["code"]
        text = text_for_bucket(True)
        reason = build_replay.pre_token_filter(text, spec, {"license": "gpl-3.0"}, duplicate=True)
        self.assertEqual(reason, "heldout_bucket")

    def test_code_license_fail_closed(self):
        spec = build_replay.SOURCES["code"]
        text = text_for_bucket(False)
        self.assertEqual(
            build_replay.pre_token_filter(text, spec, {"license": "gpl-3.0"}),
            "disallowed_license",
        )
        self.assertIsNone(build_replay.pre_token_filter(text, spec, {"license": "MIT"}))
        self.assertEqual(
            build_replay.pre_token_filter(text, spec, {"license": None}),
            "disallowed_license",
        )

    def test_duplicate_is_last_filter(self):
        spec = build_replay.SOURCES["en"]
        text = text_for_bucket(False)
        self.assertEqual(build_replay.pre_token_filter(text, spec, {}, duplicate=True), "exact_duplicate")

    def test_targets_reject_bad_values(self):
        with self.assertRaises(Exception):
            build_replay.parse_targets(["code=0"])
        self.assertEqual(build_replay.parse_targets(["math=123"])["math"], 123)

    def test_process_file_counts_tokens_and_resumes_at_file_boundary(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        train_text = text_for_bucket(False)
        heldout_text = text_for_bucket(True)
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.parquet"
            pq.write_table(
                pa.table(
                    {
                        "code": [heldout_text, train_text, train_text, train_text + "x"],
                        "repo_name": ["r"] * 4,
                        "path": ["a", "b", "c", "d"],
                        "language": ["Python"] * 4,
                        "license": ["gpl-3.0", "mit", "mit", "gpl-3.0"],
                    }
                ),
                source,
            )
            state = build_replay.State(root / "state.sqlite3")
            output = root / "code-00000.jsonl.gz"
            meta = build_replay.process_file(
                state, build_replay.SOURCES["code"], source, "fixture.parquet", output,
                tokenizer, batch_size=2, sample_limit=20,
            )
            self.assertEqual(meta["output_documents"], 1)
            self.assertGreater(meta["qwen_tokens"], 0)
            self.assertEqual(meta["dropped_documents"]["heldout_bucket"], 1)
            self.assertEqual(meta["dropped_documents"]["exact_duplicate"], 1)
            self.assertEqual(meta["dropped_documents"]["disallowed_license"], 1)
            self.assertTrue(state.is_completed("code", "fixture.parquet"))
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual(row["license"], "mit")
            self.assertGreater(row["qwen_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
