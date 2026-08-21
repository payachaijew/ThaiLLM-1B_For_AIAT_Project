#!/usr/bin/env python3
"""Build a packed training stream from the per-language token pools.

This is the file that makes the Track 2 comparison honest. S0, D1 and D2 all read the SAME
stream file, so "identical data, identical order" is true by construction rather than by
convention, and the manifest sha256 lets any reviewer verify it after the fact.

It is also what makes the replay-ratio ablation cheap: changing the mixture rebuilds a stream
from the existing pools in minutes, with no re-tokenisation.

  # main run, the mixture frozen in configs/experiment_parameters.json
  python3 build_training_stream.py --name main --budget 10e9 --th 0.50 --en 0.35 --code 0.10 --math 0.05

  # ablation arms
  python3 build_training_stream.py --name mix70 --budget 2e9 --th 0.70 --en 0.21 --code 0.06 --math 0.03

scientific_evidence_allowed = false.
"""
from __future__ import annotations
import argparse, json, hashlib, datetime, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
POOLS = DATA / "tokens"
OUT = DATA / "streams"
LANGS = ("th", "en", "code", "math")


def load_pool(lang):
    b, i, m = POOLS/f"{lang}.bin", POOLS/f"{lang}.idx", POOLS/f"{lang}.meta.json"
    if not b.exists():
        sys.exit(f"FATAL: ยังไม่มี {b} — รัน tokenize_corpus.py ก่อน")
    return (np.memmap(b, dtype=np.uint32, mode="r"),
            np.fromfile(i, dtype=np.uint64),
            json.loads(m.read_text()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget", type=float, required=True, help="total tokens, e.g. 10e9")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20260821)
    for l in LANGS:
        ap.add_argument(f"--{l}", type=float, required=True)
    a = ap.parse_args()

    ratios = {l: getattr(a, l) for l in LANGS}
    s = sum(ratios.values())
    if abs(s - 1.0) > 1e-6:
        sys.exit(f"FATAL: สัดส่วนรวมได้ {s}, ต้องเป็น 1.0")
    budget = int(a.budget)
    dst = OUT / a.name
    dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print(f"[*] stream '{a.name}'  budget {budget/1e9:.2f}B  seq_len {a.seq_len}  seed {a.seed}")

    picked = []          # (lang, doc_index) in the order they will be written
    report = {}
    for lang in LANGS:
        want = int(round(budget * ratios[lang]))
        if want == 0:
            report[lang] = {"target_tokens": 0, "used_tokens": 0, "documents": 0}
            continue
        _, idx, meta = load_pool(lang)
        lens = (idx[1:] - idx[:-1]).astype(np.int64)
        order = rng.permutation(len(lens))
        cum = np.cumsum(lens[order])
        n = int(np.searchsorted(cum, want) + 1)
        n = min(n, len(order))
        used = int(cum[n-1]) if n else 0
        picked += [(lang, int(d)) for d in order[:n]]
        short = want - used
        report[lang] = {
            "target_tokens": want, "used_tokens": used, "documents": n,
            "pool_tokens": meta["tokens"], "pool_documents": meta["documents_kept"],
            "epochs": round(used / meta["tokens"], 4),
            "shortfall": short if short > 0 else 0,
        }
        flag = "  ⚠️ ไม่พอ" if short > 0 else ""
        print(f"  {lang:5s} target {want/1e9:5.3f}B  ใช้จริง {used/1e9:5.3f}B  "
              f"{n:>9,} docs  {report[lang]['epochs']:.3f} epoch{flag}")

    # interleave languages so no single language dominates any contiguous span
    rng.shuffle(picked)

    pools = {l: load_pool(l) for l in LANGS if report.get(l, {}).get("documents")}
    binp = dst / "train.bin"
    h = hashlib.sha256()
    written = seqs = 0
    buf = np.empty(0, dtype=np.uint32)
    with open(binp, "wb") as f:
        for lang, di in picked:
            arr, idx, _ = pools[lang]
            doc = np.asarray(arr[idx[di]:idx[di+1]], dtype=np.uint32)
            buf = np.concatenate([buf, doc]) if buf.size else doc
            while buf.size >= a.seq_len:
                chunk = buf[:a.seq_len]
                b = chunk.tobytes(); f.write(b); h.update(b)
                written += a.seq_len; seqs += 1
                buf = buf[a.seq_len:]
    # the tail shorter than one sequence is discarded rather than padded, so every
    # sequence in the file is full length and no padding token enters the loss
    dropped_tail = int(buf.size)

    man = {
        "stream_id": a.name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "seed": a.seed,
        "sequence_length": a.seq_len,
        "requested_mixture": ratios,
        "budget_tokens": budget,
        "per_language": report,
        "achieved_mixture": {l: round(report[l]["used_tokens"] / max(written, 1), 4) for l in LANGS},
        "sequences": seqs,
        "tokens_written": written,
        "tail_tokens_discarded": dropped_tail,
        "train_bin_sha256": h.hexdigest(),
        "dtype": "uint32",
        "contract": "S0, D1 and D2 MUST read this exact file. Identical data and identical "
                    "order across conditions is guaranteed by using one stream, and is "
                    "verifiable by comparing train_bin_sha256.",
        "limitations": [
            "Documents are sampled without replacement until the per-language budget is met; "
            "a language whose pool is smaller than its target is reported as a shortfall.",
            "Sequences cross document boundaries; EOS marks each boundary.",
            "The final partial sequence is discarded, not padded.",
        ],
    }
    (dst / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
    print(f"\n[+] {seqs:,} sequences · {written/1e9:.3f}B tokens · "
          f"{binp.stat().st_size/1e9:.1f} GB")
    print(f"[+] sha256 {h.hexdigest()[:32]}...")
    print(f"[+] {binp}")


if __name__ == "__main__":
    main()
