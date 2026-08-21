#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pii_filter_en
import secret_scan
from apply_scan import ScanState, process_shard
from heldout_rule import doc_hash, is_heldout
from rebalance_code import GROUP_TARGET_FRACTIONS, code_group, farthest_first_indices


class FakeEncoding:
    def __init__(self, text: str):
        self.ids = list(text.encode("utf-8"))


class FakeTokenizer:
    def encode_batch(self, texts, add_special_tokens=False):
        assert add_special_tokens is False
        return [FakeEncoding(text) for text in texts]


class DetectorTests(unittest.TestCase):
    def test_secret_types_and_placeholders(self):
        realish = "TOKEN=A1b2C3d4E5f6G7h8I9j0\nkey=" + "ghp" + "_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
        out, counts, events = secret_scan.redact(realish)
        self.assertEqual(counts, {"env_secret": 1, "github_token": 1})
        self.assertNotIn("A1b2C3d4E5f6", out)
        self.assertNotIn("A1b2C3d4E5f6", json.dumps(events))
        placeholders = "TOKEN=<TOKEN>\nAPI_KEY=your_api_key_here\nAWS=" + "AKI" + "AIOSFODNN7EXAMPLE"
        self.assertEqual(secret_scan.redact(placeholders)[1], {})
        self.assertEqual(secret_scan.scan("postgres://user:[PASSWORD]@localhost/db"), {})
        self.assertEqual(secret_scan.scan("TOKENS_FILE=/run/secrets/tokens"), {})
        self.assertEqual(secret_scan.scan("MS_LIVE_PASSWORD=<MS live password>"), {})
        self.assertEqual(secret_scan.scan(r"MS_LIVE_PASSWORD=\<MS live password\>"), {})
        self.assertEqual(secret_scan.scan("mongodb://%s:%s@localhost/db"), {})
        self.assertEqual(secret_scan.scan("mysql://[user[:pass]@host/db"), {})

    def test_private_key_drops_document(self):
        text = "header\n-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n"
        out, counts, events = secret_scan.redact(text)
        self.assertIsNone(out)
        self.assertEqual(counts, {"private_key": 1})
        self.assertEqual(events[0]["context"], "[REDACTED_PRIVATE_KEY_DOCUMENT]")

    def test_pii_positive_and_false_positive_controls(self):
        text = (
            "Email ada@example.org phone +44 20 7946 0958; SSN: 123-45-6789; "
            "credit card 4111 1111 1111 1111; endpoint http://8.8.8.8:53"
        )
        out, counts, _ = pii_filter_en.redact(text)
        self.assertEqual(set(counts), {"email", "phone", "us_ssn", "credit_card", "public_ip"})
        self.assertNotIn("ada@example.org", out)
        controls = (
            "version 2.3.4.5; Function.prototype.bind section 15.4.5.1; "
            "ISSN 123-45-6789; time 12:30:45; private host 192.168.1.1; id 123-45-6789"
        )
        self.assertEqual(pii_filter_en.redact(controls)[1], {})


class RebalanceTests(unittest.TestCase):
    def test_language_groups(self):
        expected = {
            "Python": "python", "JavaScript": "javascript_typescript", "TypeScript": "javascript_typescript",
            "Java": "java", "C": "c_cpp", "C++": "c_cpp", "C#": "csharp",
            "Markdown": "markdown", "HTML": "html", "CSS": "css_config", "Dockerfile": "css_config",
            "Rust": "long_tail", "GO": "long_tail",
        }
        self.assertEqual({key: code_group(key) for key in expected}, expected)
        self.assertAlmostEqual(sum(GROUP_TARGET_FRACTIONS.values()), 1.0)

    def test_shard_order_excludes_used_and_spreads(self):
        used = {0, 126, 251, 377, 502, 628, 753, 879, 1}
        order = farthest_first_indices(used)
        self.assertEqual(len(order), 880 - len(used))
        self.assertFalse(set(order) & used)
        self.assertEqual(len(set(order)), len(order))
        self.assertGreater(max(order[:8]) - min(order[:8]), 600)


class PipelineTest(unittest.TestCase):
    def _trainable(self, prefix: str) -> str:
        for index in range(10000):
            text = f"{prefix} {index} " + ("content " * 30)
            if not is_heldout(text):
                return text
        raise AssertionError("could not find trainable fixture")

    def test_process_shard_redacts_drops_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "input.jsonl.gz"
            output = root / "out" / "output.jsonl.gz"
            state = ScanState(root / "state.sqlite3")
            normal = self._trainable("contact ada@example.org")
            private = self._trainable("-----BEGIN RSA PRIVATE KEY----- fake")
            rows = []
            for text in (normal, private):
                rows.append({
                    "text": text, "doc_sha256": doc_hash(text), "qwen_tokens": len(text.encode()),
                    "utf8_bytes": len(text.encode()), "source_file": "fixture",
                })
            with gzip.open(inp, "wt", encoding="utf-8") as out:
                for row in rows:
                    out.write(json.dumps(row) + "\n")
            meta = process_shard(state, "en", inp, output, FakeTokenizer(), 20)
            self.assertEqual(meta["metrics"]["input_documents"], 2)
            self.assertEqual(meta["metrics"]["dropped_documents"], {"private_key_document": 1})
            with gzip.open(output, "rt", encoding="utf-8") as source:
                result = json.loads(next(source))
                self.assertIn("[EMAIL]", result["text"])
                self.assertFalse(result["scientific_evidence_allowed"])
                self.assertEqual(result["doc_sha256"], doc_hash(result["text"]))
                self.assertFalse(is_heldout(result["text"]))
                with self.assertRaises(StopIteration):
                    next(source)


if __name__ == "__main__":
    unittest.main()
