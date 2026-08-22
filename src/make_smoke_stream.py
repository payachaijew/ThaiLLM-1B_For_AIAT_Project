#!/usr/bin/env python3
"""Build the tiny stream used to smoke-test the training pipeline.

Takes the first N*SEQ tokens of a real stream and re-chunks them to a short sequence length so
train_cpt.py and preflight.py can be exercised end to end on a laptop.

This deliberately ignores document boundaries: the chunks cut through documents, so the losses
and BPB values it produces are meaningless. It exists to prove the pipeline runs, resumes, and
saves — never to measure anything.

  python3 make_smoke_stream.py --src ../data/streams/main --out ../data/streams/_smoke
"""
import argparse, datetime, hashlib, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/streams/main")
    ap.add_argument("--out", default="../data/streams/_smoke")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--sequences", type=int, default=64)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    need = a.seq_len * a.sequences
    arr = np.memmap(src / "train.bin", dtype=np.uint32, mode="r")
    if arr.size < need:
        raise SystemExit(f"{src} holds {arr.size:,} tokens, need {need:,}")
    np.array(arr[:need]).tofile(dst / "train.bin")

    sha = hashlib.sha256((dst / "train.bin").read_bytes()).hexdigest()
    parent = json.loads((src / "manifest.json").read_text())
    (dst / "manifest.json").write_text(json.dumps({
        "stream_id": dst.name,
        "derived_from": parent["stream_id"],
        "note": "Smoke fixture. Re-chunked without respecting document boundaries. "
                "Never use for measurement.",
        "scientific_evidence_allowed": False,
        "seed": parent["seed"],
        "sequence_length": a.seq_len,
        "sequences": a.sequences,
        "tokens_written": need,
        "train_bin_sha256": sha,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"[+] {dst}  {a.sequences} x {a.seq_len} = {need:,} tokens  sha {sha[:16]}")


if __name__ == "__main__":
    main()
