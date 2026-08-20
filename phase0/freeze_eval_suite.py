#!/usr/bin/env python3
"""Phase 0 / Step B -- freeze the evaluation suite BEFORE any baseline is measured.

Anything not written here cannot later be presented as a planned metric.
The output carries a sha256 over the normalised spec; that hash must be quoted
in every run record that claims to use this suite.

scientific_evidence_allowed = false (this file defines measurement, it is not a result).
"""
from __future__ import annotations
import json, hashlib, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent / "eval_suite_frozen.json"

SPEC = {
    "suite_id": "THAILLM-EVAL-FROZEN-V1",
    "frozen_at": "2026-08-18",
    "freeze_rule": (
        "Benchmarks, few-shot counts, prompt templates, metrics and decision thresholds are "
        "fixed here BEFORE any baseline number is produced. Changing any field requires a new "
        "suite_id and an explicit note in VALIDATION_OUTPUT_LOG.md explaining what changed and why."
    ),
    "model_stage": "base_only",
    "evaluation_mode": "few_shot_and_likelihood_only_no_chat_template",
    "rationale": "Base checkpoints have no chat template; instruct-style prompting would measure the template, not the model.",

    "thai_acquisition": [
        {"task": "thaiexam",  "subsets": ["onet", "ic", "a_level", "tgat", "tpat"],
         "shots": 5, "metric": "accuracy", "role": "primary",
         "note": "Primary Thai benchmark used by Typhoon 2 (arXiv:2412.13702)."},
        {"task": "m3exam_th", "shots": 5, "metric": "accuracy", "role": "primary",
         "note": "Second primary Thai benchmark used by Typhoon 2."},
        {"task": "belebele_th", "shots": 5, "metric": "accuracy", "role": "secondary"},
        {"task": "thai_bpb_heldout", "shots": 0, "metric": "bits_per_byte", "role": "primary_continuous",
         "note": "Tracked every N steps during CPT. BPB not perplexity: perplexity is tokenizer-dependent "
                 "and not comparable across candidate bases."},
    ],

    "retention": [
        {"task": "mmlu",      "shots": 5, "metric": "accuracy",     "role": "primary_guardrail"},
        {"task": "hellaswag", "shots": 10, "metric": "accuracy_norm", "role": "guardrail"},
        {"task": "arc_challenge", "shots": 25, "metric": "accuracy_norm", "role": "guardrail"},
        {"task": "english_bpb_heldout", "shots": 0, "metric": "bits_per_byte", "role": "primary_continuous"},
        {"task": "code_bpb_heldout",    "shots": 0, "metric": "bits_per_byte", "role": "primary_continuous"},
        {"task": "humaneval", "shots": 0, "metric": "pass@1", "role": "optional",
         "note": "Base models score low on HumanEval; treat as directional only."},
    ],

    "heldout_bpb_sets": {
        "construction_rule": (
            "Sampled from the SAME source distributions as training, then REMOVED from the training "
            "pool by document hash before any training run. Never re-generated after training starts."
        ),
        "sets": [
            {"set_id": "TH-WEB-HELDOUT",  "language": "th", "source": "SEA-PILE-v2 th", "target_docs": 2000},
            {"set_id": "TH-ENC-HELDOUT",  "language": "th", "source": "Thai encyclopedic", "target_docs": 1000},
            {"set_id": "EN-HELDOUT",      "language": "en", "source": "FineWeb-Edu", "target_docs": 2000},
            {"set_id": "CODE-HELDOUT",    "language": "code", "source": "permissive code replay", "target_docs": 1000},
        ],
        "requirements": [
            "Disjoint from training by document hash.",
            "Frozen and hashed before the first training step.",
            "Decontaminated against every benchmark listed in this suite.",
            "Token counts recorded under the FINAL selected tokenizer, not a dataset-card figure.",
        ],
    },

    "decision_thresholds": {
        "note": "Mirrors scientific_thresholds in configs/experiment_parameters.json. Frozen here so the "
                "baseline cannot be re-interpreted after the fact.",
        "thai_bpb_relative_improvement_min": 0.02,
        "thai_downstream_min_points": 1.0,
        "max_english_or_code_aggregate_regression_points": 1.0,
        "bootstrap_resamples": 10000,
        "confidence_level": 0.95,
        "multiplicity_correction": "holm_within_metric_family",
    },

    "headroom_gate": {
        "purpose": "Decide whether the chosen base has enough Thai headroom to justify CPT at all.",
        "rule": (
            "If the baseline already sits within 1 point of the best openly reported ~1-2B Thai result on "
            "the primary tasks, the expected CPT gain is small and the base choice must be revisited "
            "before any GPU budget is committed."
        ),
        "evaluated_in": "Phase 0 Step C",
    },

    "harness": {
        "preferred": "lm-evaluation-harness",
        "pin_requirement": "Record the harness commit sha in every run record. Results are not comparable across harness versions.",
        "batch_size": "auto",
        "dtype": "bfloat16",
        "seed": 0,
    },

    "candidates_to_baseline": [
        {"repo": "Qwen/Qwen3-1.7B-Base", "role": "proposed_main_base"},
        {"repo": "google/gemma-4-E2B",   "role": "challenger_better_thai_tokenizer"},
    ],
}


def main():
    blob = json.dumps(SPEC, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    doc = {
        "spec_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "spec": SPEC,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    print(f"[+] wrote {OUT}")
    print(f"[+] spec_sha256 = {doc['spec_sha256']}")
    print("[!] Quote this hash in every run record that claims to use THAILLM-EVAL-FROZEN-V1.")


if __name__ == "__main__":
    main()
