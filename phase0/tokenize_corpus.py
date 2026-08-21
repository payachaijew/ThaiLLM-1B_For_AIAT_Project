#!/usr/bin/env python3
"""Tokenise the cleaned corpus into per-language token pools.

Per-language on purpose. Keeping the pools separate means the training mixture is chosen at
STREAM-BUILD time, not baked in here, so the replay-ratio ablation (50/70/90 percent Thai)
costs nothing extra and the expensive tokenisation runs exactly once.

Two exclusions are applied, and both are belt-and-braces:
  1. every doc_sha256 in data/removal_list.txt  (benchmark contamination + near-duplicates)
  2. anything is_heldout() still returns True for, even though the corpora were already
     filtered. If this second guard ever fires it means an earlier stage leaked, so the
     count is reported rather than silently absorbed.

Output per language:
  data/tokens/<lang>.bin        uint32 token ids, documents concatenated, EOS between docs
  data/tokens/<lang>.idx        uint64 start offset of each document
  data/tokens/<lang>.meta.json  counts, drop reasons, sha256 of the .bin

scientific_evidence_allowed = false.
"""
from __future__ import annotations
import gzip, json, glob, sys, time, hashlib, datetime, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = DATA / "tokens"
sys.path.insert(0, str(HERE))
from heldout_rule import is_heldout  # noqa: E402

TOKENIZER = "Qwen/Qwen3-1.7B-Base"
TOK_REV = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
BATCH = 512

SOURCES = [
    ("th",   DATA / "clean_pii" / "th"),
    ("en",   DATA / "clean_replay_v2" / "en"),
    ("code", DATA / "clean_replay_v2" / "code"),
    ("math", DATA / "clean_replay_v2" / "math"),
]


def main():
    import warnings; warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer

    only = sys.argv[1:] or None
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOK_REV)
    eos = tok.eos_token_id
    if eos is None:
        sys.exit("FATAL: tokenizer has no eos_token_id")

    removal = set()
    p = DATA / "removal_list.txt"
    if p.exists():
        removal = {l.strip() for l in p.read_text().splitlines() if l.strip()}
    print(f"[*] removal list: {len(removal):,} doc_sha256")
    print(f"[*] tokenizer: {TOKENIZER} @ {TOK_REV[:8]}  eos={eos}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for lang, root in SOURCES:
        if only and lang not in only:
            continue
        files = sorted(glob.glob(str(root / "*.jsonl.gz")))
        if not files:
            print(f"  ! no shards for {lang}"); continue
        binp, idxp = OUT / f"{lang}.bin", OUT / f"{lang}.idx"
        if binp.exists():
            print(f"  [{lang}] มีอยู่แล้ว ข้าม (ลบไฟล์ถ้าต้องการทำใหม่)"); continue

        drop = collections.Counter()
        kept = ntok = 0
        offsets = [0]
        t0 = time.time()
        h = hashlib.sha256()

        with open(binp, "wb") as fb:
            batch, ids_seen = [], 0
            def flush(batch):
                nonlocal ntok, kept
                if not batch: return
                encs = tok(batch, add_special_tokens=False)["input_ids"]
                for e in encs:
                    a = np.asarray(e + [eos], dtype=np.uint32)
                    b = a.tobytes(); fb.write(b); h.update(b)
                    ntok += a.size; kept += 1
                    offsets.append(offsets[-1] + a.size)

            for f in files:
                with gzip.open(f, "rt") as fh:
                    for line in fh:
                        r = json.loads(line)
                        did = r.get("doc_sha256")
                        if did and did in removal:
                            drop["removal_list"] += 1; continue
                        if is_heldout(r["text"]):
                            drop["heldout_guard_fired"] += 1; continue
                        batch.append(r["text"]); ids_seen += 1
                        if len(batch) >= BATCH:
                            flush(batch); batch = []
                            if kept % 100000 < BATCH:
                                el = time.time() - t0
                                print(f"    [{lang}] {kept:,} docs  {ntok/1e9:.3f}B tok  "
                                      f"{el/60:.1f} min  {ntok/el/1e6:.2f}M tok/s", flush=True)
            flush(batch)

        np.asarray(offsets, dtype=np.uint64).tofile(idxp)
        meta = {
            "language": lang,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scientific_evidence_allowed": False,
            "tokenizer": {"repo": TOKENIZER, "revision": TOK_REV, "eos_token_id": eos,
                          "add_special_tokens": False},
            "source_dir": str(root.relative_to(DATA.parent)),
            "documents_in": kept + sum(drop.values()),
            "documents_kept": kept,
            "dropped": dict(drop),
            "tokens": ntok,
            "bytes_on_disk": binp.stat().st_size,
            "bin_sha256": h.hexdigest(),
            "dtype": "uint32",
            "layout": "documents concatenated, one EOS appended after each document",
            "index": f"{lang}.idx contains uint64 start offsets, length documents_kept+1",
            "runtime_minutes": round((time.time() - t0) / 60, 1),
        }
        (OUT / f"{lang}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        print(f"  [{lang}] kept {kept:,} docs · {ntok/1e9:.3f}B tokens · "
              f"dropped {dict(drop)} · {meta['runtime_minutes']} min", flush=True)

    # roll-up
    metas = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.meta.json"))}
    if metas:
        tot = sum(m["tokens"] for m in metas.values())
        roll = {
            "manifest_id": "TOKENS-V1",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scientific_evidence_allowed": False,
            "tokenizer": {"repo": TOKENIZER, "revision": TOK_REV},
            "removal_list_applied": {"path": "data/removal_list.txt", "entries": len(removal)},
            "per_language": {k.replace(".meta", ""): {"documents": m["documents_kept"],
                                                      "tokens": m["tokens"],
                                                      "share": round(m["tokens"] / tot, 4),
                                                      "bin_sha256": m["bin_sha256"]}
                             for k, m in metas.items()},
            "total_tokens": tot,
            "note": "Pools are per-language on purpose; the training mixture is chosen by "
                    "build_training_stream.py so the replay-ratio ablation needs no re-tokenisation.",
        }
        (DATA / "tokens_manifest.json").write_text(json.dumps(roll, ensure_ascii=False, indent=2) + "\n")
        print(f"\n=== รวม {tot/1e9:.3f}B tokens ===")
        for k, v in roll["per_language"].items():
            print(f"  {k:6s} {v['tokens']/1e9:6.3f}B  {100*v['share']:5.1f}%")


if __name__ == "__main__":
    main()
