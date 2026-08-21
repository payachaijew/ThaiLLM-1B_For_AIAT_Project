#!/usr/bin/env python3
"""Plan C step 1 -- build the n-gram index of every benchmark in THAILLM-EVAL-FROZEN-V1.

Any training document that contains one of these n-grams has seen part of an exam we intend
to score on, which would make that score memorisation rather than ability.

Uniform character 64-grams are used for every language. Thai has no word boundaries, so a
word-based n-gram would need a segmenter and would behave differently across languages; a
character window is language-agnostic and directly comparable.

scientific_evidence_allowed = false.
"""
from __future__ import annotations
import json, re, sys, unicodedata, datetime
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmark_ngrams.npz"
NGRAM = 64
_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", str(t)).lower()
    return _WS.sub(" ", t).strip()


MASK = (1 << 63) - 1


def hashes(text: str, n: int = NGRAM, stride: int = 1) -> np.ndarray:
    """Hashes of sliding character windows.

    Uses Python's built-in str hash (C speed) rather than a numpy rolling polynomial: the
    polynomial version costs n*64 multiply-adds per document and is far too slow at corpus
    scale. REQUIRES PYTHONHASHSEED=0 for reproducibility across processes.

    The benchmark index is built with stride 1 and the corpus is scanned with a larger stride.
    That asymmetry is deliberate: a document containing a verbatim benchmark span still matches,
    because the index covers every alignment.
    """
    t = norm(text)
    if len(t) < n:
        return np.empty(0, dtype=np.uint64)
    return np.fromiter((hash(t[i:i + n]) & MASK for i in range(0, len(t) - n + 1, stride)),
                       dtype=np.uint64)


SPECS = [
    ("mmlu", "cais/mmlu", {"name": "all", "split": "test"},
     lambda r: " ".join([r["question"]] + list(r["choices"]))),
    ("hellaswag", "Rowan/hellaswag", {"split": "validation"},
     lambda r: " ".join([r["ctx"]] + list(r["endings"]))),
    ("arc_challenge", "allenai/ai2_arc", {"name": "ARC-Challenge", "split": "test"},
     lambda r: " ".join([r["question"]] + list(r["choices"]["text"]))),
    ("humaneval", "openai/openai_humaneval", {"split": "test"},
     lambda r: " ".join([r["prompt"], r["canonical_solution"], r["test"]])),
    ("belebele_th", "facebook/belebele", {"name": "tha_Thai", "split": "test"},
     lambda r: " ".join([r["flores_passage"], r["question"],
                         r["mc_answer1"], r["mc_answer2"], r["mc_answer3"], r["mc_answer4"]])),
    ("belebele_en", "facebook/belebele", {"name": "eng_Latn", "split": "test"},
     lambda r: " ".join([r["flores_passage"], r["question"],
                         r["mc_answer1"], r["mc_answer2"], r["mc_answer3"], r["mc_answer4"]])),
    ("m3exam_th", "chiayewken/m3exam", {"name": "thai", "split": "test"},
     lambda r: " ".join([str(r.get("question_text", ""))] + [str(x) for x in (r.get("options") or [])])),
    # ThaiExam ships as five separate configs. These are exactly the five subsets named in
    # THAILLM-EVAL-FROZEN-V1, so all five must be in the index.
    ("thaiexam_onet", "scb10x/thai_exam", {"name": "onet", "split": "test"}, None),
    ("thaiexam_ic", "scb10x/thai_exam", {"name": "ic", "split": "test"}, None),
    ("thaiexam_tgat", "scb10x/thai_exam", {"name": "tgat", "split": "test"}, None),
    ("thaiexam_tpat1", "scb10x/thai_exam", {"name": "tpat1", "split": "test"}, None),
    ("thaiexam_a_level", "scb10x/thai_exam", {"name": "a_level", "split": "test"}, None),
]


def main():
    import warnings; warnings.filterwarnings("ignore")
    from datasets import load_dataset
    allh, per = [], {}
    for name, repo, kw, fn in SPECS:
        try:
            ds = load_dataset(repo, **kw)
        except Exception as e:
            print(f"  ❌ {name:14s} {type(e).__name__}: {str(e)[:90]}", flush=True)
            per[name] = {"status": "failed", "error": f"{type(e).__name__}"}
            continue
        hs, items = [], 0
        for r in ds:
            try:
                t = fn(r) if fn else " ".join(
                    str(v) for v in r.values() if isinstance(v, str) and v.strip())
            except Exception:
                continue
            h = hashes(t)
            if h.size:
                hs.append(h); items += 1
        arr = np.unique(np.concatenate(hs)) if hs else np.empty(0, dtype=np.uint64)
        allh.append(arr)
        per[name] = {"status": "ok", "repo": repo, "items": items, "unique_ngrams": int(arr.size)}
        print(f"  ✅ {name:14s} items={items:6,}  ngrams={arr.size:9,}", flush=True)

    idx = np.unique(np.concatenate(allh)) if allh else np.empty(0, dtype=np.uint64)
    np.savez_compressed(OUT, ngrams=idx)
    meta = {
        "index_id": "BENCH-NGRAM-V1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "ngram_chars": NGRAM,
        "normalisation": "NFKC + lowercase + whitespace collapse",
        "eval_suite": "THAILLM-EVAL-FROZEN-V1",
        "total_unique_ngrams": int(idx.size),
        "per_benchmark": per,
        "limitations": [
            "Character 64-grams detect verbatim or near-verbatim overlap only. Paraphrased or "
            "translated benchmark items are NOT detected.",
            "A benchmark that failed to load is NOT covered by this index.",
        ],
    }
    (OUT.parent / "benchmark_ngrams_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"\n[+] unique n-grams: {idx.size:,}  -> {OUT}")


if __name__ == "__main__":
    main()
