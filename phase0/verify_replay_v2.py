#!/usr/bin/env python3
"""Independent verification of the v2 replay corpus (post secret/PII scan + code rebalance).

Re-derives every headline number from the data itself rather than trusting any summary file.

scientific_evidence_allowed = false.
"""
from __future__ import annotations
import gzip, json, glob, sys, random, datetime, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from heldout_rule import doc_hash, is_heldout

ROOT = HERE.parent / "data" / "clean_replay_v2"
SAMPLE_PER_LANG = 100
DEDUP_SCAN = 250_000

GROUPS = {
    "python": ["Python"], "javascript_typescript": ["JavaScript", "TypeScript"],
    "java": ["Java"], "c_cpp": ["C", "C++"], "csharp": ["C#"],
    "markdown": ["Markdown"], "html": ["HTML"],
}


def main():
    import warnings; warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    random.seed(20260821)

    out = {
        "verification_id": "VERIFY-CLEAN-REPLAY-V2",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "method": "every number re-derived from data/clean_replay_v2; no summary file trusted",
        "totals": {}, "token_recount": {}, "integrity": {}, "code_distribution": {},
        "redaction": {}, "errors": [],
    }

    for lang in ("en", "code", "math"):
        files = sorted(glob.glob(str(ROOT / lang / "*.jsonl.gz")))
        if not files:
            out["errors"].append(f"no shards for {lang}"); continue
        docs = toks = byts = 0
        seen = set(); dup = 0; leak = 0; scanned = 0
        red_docs = 0
        grp_tok = collections.Counter()
        pool = []
        for f in files:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    docs += 1; toks += r.get("qwen_tokens", 0); byts += r.get("utf8_bytes", 0)
                    if lang == "code":
                        grp_tok[str(r.get("language", "?"))] += r.get("qwen_tokens", 0)
                    if "[REDACTED_SECRET]" in r["text"]:
                        red_docs += 1
                    if scanned < DEDUP_SCAN:
                        scanned += 1
                        h = doc_hash(r["text"])
                        if h in seen: dup += 1
                        else: seen.add(h)
                        if is_heldout(r["text"]): leak += 1
                    if len(pool) < 4000 and random.random() < 0.01:
                        pool.append(r)
        out["totals"][lang] = {"documents": docs, "qwen_tokens": toks, "utf8_bytes": byts,
                               "shards": len(files)}
        out["integrity"][lang] = {"scanned": scanned, "exact_duplicates": dup,
                                  "heldout_leakage": leak}
        out["redaction"][lang] = {"documents_with_redacted_secret": red_docs,
                                  "rate": round(red_docs / docs, 6) if docs else None}
        # exact token recount
        sample = random.sample(pool, min(SAMPLE_PER_LANG, len(pool)))
        mism = sum(1 for r in sample
                   if len(tok(r["text"], add_special_tokens=False)["input_ids"]) != r.get("qwen_tokens"))
        out["token_recount"][lang] = {"sampled": len(sample), "mismatches": mism}
        if lang == "code":
            T = sum(grp_tok.values())
            merged = {g: sum(grp_tok[x] for x in xs) for g, xs in GROUPS.items()}
            merged["long_tail"] = T - sum(merged.values()) - sum(
                grp_tok[x] for x in grp_tok if x in ("CSS",) )
            out["code_distribution"] = {
                "total_tokens": T,
                "by_group_pct": {g: round(100 * v / T, 3) for g, v in merged.items() if v > 0},
                "max_single_language_pct": round(100 * max(grp_tok.values()) / T, 3),
                "max_single_language": max(grp_tok, key=grp_tok.get),
            }
        print(f"[{lang}] docs={docs:,} tokens={toks:,} dup={dup} leak={leak} "
              f"recount_mismatch={mism} redacted_docs={red_docs:,}", flush=True)

    ok = (all(v["exact_duplicates"] == 0 and v["heldout_leakage"] == 0 for v in out["integrity"].values())
          and all(v["mismatches"] == 0 for v in out["token_recount"].values())
          and not out["errors"])
    out["verdict"] = "PASS" if ok else "FAIL"
    p = HERE.parent / "validation" / "replay_v2_verification.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\nverdict={out['verdict']}  -> {p}")


if __name__ == "__main__":
    main()
