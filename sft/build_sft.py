#!/usr/bin/env python3
"""Build the Thai SFT set: licence filter, decontamination, held-out split.

Mirrors what phase0 did for the pretraining corpus, because instruction data needs the same
guarantees and gets them less often. Three things happen here, in this order, and every one
of them reports how much it removed:

  1. Per-row licence filter. Aggregated instruction sets carry different terms row by row -
     WangchanThaiInstruct is 86.6% non-commercial - so a set cannot be judged as a whole.
  2. Decontamination against the frozen benchmark index. Instruction data is FAR more likely
     to contain benchmark items than web text is: these sets are assembled from exam questions
     and QA pairs, which is exactly what the benchmarks are made of. Skipping this step here
     would be worse than skipping it on the corpus.
  3. Held-out split via phase0/heldout_rule.py - the same HELDOUT-BUCKET-V1 used for the
     corpus. Sharing the rule is the point: an example cannot land in SFT training and corpus
     evaluation at the same time, because the bucket depends only on the text.

  python3 build_sft.py --out ../data/sft --limit 200      # smoke
  python3 build_sft.py --out ../data/sft                  # full

scientific_evidence_allowed = false.
"""
from __future__ import annotations
import argparse, datetime, gzip, hashlib, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase0"))

from sources import SOURCES, licence_allowed          # noqa: E402
import heldout_rule                                    # noqa: E402
from build_benchmark_index import hashes, NGRAM        # noqa: E402

# The corpus scan used stride 8 because it had 4.5M documents to get through. Instruction rows
# are short and there are two orders of magnitude fewer of them, so every alignment is checked.
STRIDE = 1
MIN_HITS = 1
MAXCHARS = 20000


def render(instruction, context, output):
    """One flat text per example. Also what the held-out bucket is computed from, so the same
    example always lands in the same bucket no matter which source supplied it."""
    parts = [str(instruction or "").strip()]
    if context:
        parts.append(str(context).strip())
    parts.append(str(output or "").strip())
    return "\n\n".join(p for p in parts if p)


def load_rows(src, limit=None):
    from datasets import load_dataset
    for split in src["splits"]:
        ds = load_dataset(src["repo"], src.get("config"), split=split)
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            yield split, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/sft")
    ap.add_argument("--index", default="../data/benchmark_ngrams.npz")
    ap.add_argument("--limit", type=int, default=None, help="rows per split, for smoke runs")
    ap.add_argument("--min-output-chars", type=int, default=10)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = np.load(a.index)["ngrams"]; idx.sort()
    print(f"[*] benchmark index: {idx.size:,} n-grams", flush=True)

    stats = {}
    seen = set()
    train_f = gzip.open(out / "sft_train.jsonl.gz", "wt")
    held_f = gzip.open(out / "sft_heldout.jsonl.gz", "wt")
    decon_f = open(out / "sft_decontamination_hits.jsonl", "w")
    n_train = n_held = 0
    t0 = time.time()

    for src in SOURCES:
        s = dict(seen_rows=0, dropped_licence=0, dropped_filter=0, dropped_rating=0,
                 dropped_empty=0, dropped_dup=0, dropped_decon=0, kept_train=0, kept_heldout=0)
        licences = {}
        try:
            rows = load_rows(src, a.limit)
        except Exception as e:
            print(f"  ! {src['name']}: โหลดไม่ได้ ({type(e).__name__}: {e})", flush=True)
            stats[src["name"]] = {"error": str(e)}
            continue

        for split, r in rows:
            s["seen_rows"] += 1

            if "row_keep" in src and not src["row_keep"](r):
                s["dropped_filter"] += 1; continue

            lic = r.get(src["row_licence_field"]) if src["row_licence_field"] else \
                src.get("fixed_licence")
            licences[str(lic)] = licences.get(str(lic), 0) + 1
            if not licence_allowed(lic):
                s["dropped_licence"] += 1; continue

            if src.get("min_rating") is not None:
                rat = r.get("rating")
                if rat is None or float(rat) < src["min_rating"]:
                    s["dropped_rating"] += 1; continue

            instruction, context, output = src["map"](r)
            if not instruction or not output or len(str(output)) < a.min_output_chars:
                s["dropped_empty"] += 1; continue

            text = render(instruction, context, output)
            h = hashlib.sha256(heldout_rule.normalise(text).encode()).hexdigest()
            if h in seen:
                s["dropped_dup"] += 1; continue
            seen.add(h)

            g = hashes(text[:MAXCHARS], NGRAM, STRIDE)
            if g.size:
                pos = np.searchsorted(idx, g); pos[pos >= idx.size] = 0
                hits = int(np.count_nonzero(idx[pos] == g))
                if hits >= MIN_HITS:
                    s["dropped_decon"] += 1
                    decon_f.write(json.dumps({"source": src["name"], "doc_sha256": h,
                                              "ngram_hits": hits,
                                              "preview": text[:200]}, ensure_ascii=False) + "\n")
                    continue

            rec = {"instruction": str(instruction).strip(),
                   "input": (str(context).strip() if context else ""),
                   "output": str(output).strip(),
                   "source": src["name"], "licence": str(lic),
                   "provenance": src["provenance"], "doc_sha256": h}

            if heldout_rule.is_heldout(text):
                held_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                s["kept_heldout"] += 1; n_held += 1
            else:
                train_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                s["kept_train"] += 1; n_train += 1

        s["licence_values_seen"] = licences
        stats[src["name"]] = s
        print(f"  {src['name']:<28} เห็น {s['seen_rows']:>7,} | licence -{s['dropped_licence']:>6,}"
              f" | rating -{s['dropped_rating']:>5,} | decon -{s['dropped_decon']:>4,}"
              f" | ซ้ำ -{s['dropped_dup']:>4,} | เหลือ train {s['kept_train']:>6,}"
              f" heldout {s['kept_heldout']:>5,}", flush=True)

    train_f.close(); held_f.close(); decon_f.close()

    report = {
        "build_id": "SFT-BUILD-V1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "heldout_rule": heldout_rule.RULE_ID,
        "benchmark_index": str(a.index),
        "decontamination": {"ngram_chars": NGRAM, "stride": STRIDE, "min_hits": MIN_HITS,
                            "note": "stride 1 on both sides; instruction sets are built from "
                                    "exam and QA items, so contamination risk is higher here "
                                    "than in web text"},
        "limit": a.limit,
        "totals": {"train": n_train, "heldout": n_held},
        "per_source": stats,
        "wall_minutes": round((time.time() - t0) / 60, 2),
    }
    (out / "sft_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n[+] train {n_train:,} · held-out {n_held:,} · {out}/sft_build_report.json",
          flush=True)
    if a.limit:
        print("[!] ใช้ --limit อยู่ ตัวเลขนี้ไม่ใช่ชุดเต็ม", flush=True)


if __name__ == "__main__":
    main()
