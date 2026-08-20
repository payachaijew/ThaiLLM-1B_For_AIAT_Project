#!/usr/bin/env python3
"""Phase 0 / Step C -- baseline measurement on the frozen suite.

Answers the question that gates all GPU spend: DOES THE CHOSEN BASE HAVE THAI
HEADROOM LEFT? Run this before committing any CPT budget.

  # smoke test on the frozen 14-doc screen sample (CPU/MPS, minutes)
  python3 measure_baseline.py --model Qwen/Qwen3-0.6B-Base --smoke

  # real run against the frozen held-out sets (GPU)
  python3 measure_baseline.py --model Qwen/Qwen3-1.7B-Base \
      --heldout ../data/heldout/TH-WEB-HELDOUT.jsonl --device cuda

BPB (bits per byte) is used instead of perplexity on purpose: perplexity depends
on the tokenizer, so it cannot be compared across candidate bases. BPB divides by
UTF-8 bytes, which are tokenizer-invariant.

Downstream few-shot accuracy is NOT computed here -- use lm-evaluation-harness with
the task list in eval_suite_frozen.json and record its commit sha.

scientific_evidence_allowed = false for --smoke runs. Always.
"""
from __future__ import annotations
import argparse, json, hashlib, math, datetime, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE / "eval_suite_frozen.json"


def load_smoke_texts():
    src = HERE.parent / "base_selection" / "tokenizer_screen.json"
    samples = json.loads(src.read_text())["sample_set"]["samples"]
    bad = [s["sample_id"] for s in samples
           if hashlib.sha256(s["text"].encode()).hexdigest() != s["sha256"]]
    if bad:
        sys.exit(f"FATAL: frozen sample integrity failure: {bad}")
    out = {}
    for s in samples:
        out.setdefault(s["language"], []).append(s["text"])
    return out


def load_heldout(path):
    texts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


@__import__("torch").no_grad()
def bpb(model, tok, texts, device, max_len):
    """Sum token NLL over the corpus, divide by total UTF-8 bytes."""
    import torch
    import time
    nats, nbytes, ntok = 0.0, 0, 0
    t0 = time.time()
    for i, t in enumerate(texts):
        if i and i % 100 == 0:
            r = i / (time.time() - t0)
            print(f"      {i}/{len(texts)} docs  {r:.1f} doc/s  eta {(len(texts)-i)/r/60:.1f} min", flush=True)
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        logits = model(ids).logits          # [1, seq, vocab], model dtype
        tgt = ids[:, 1:]
        # Predict token i+1 from position i, scored in CHUNKS over sequence positions.
        #
        # Why chunked: a full float32 copy of [1, 2048, 151936] is ~1.2 GB, and
        # log_softmax needs another. On a 16 GB unified-memory machine that drove the
        # process into memory thrashing -- observed at 6.6% CPU and under 100 documents
        # in 8 minutes. Chunking bounds the transient to CHUNK x vocab x 4 bytes
        # (~155 MB at 256) while keeping fp32 accumulation, so the metric is unchanged.
        #
        # An F.cross_entropy variant was also tried; it did not help speed and returned
        # an fp16-rounded sum (rel. error ~4e-4 per document), so it was rejected.
        CHUNK = 256
        pos = logits.shape[1] - 1
        for a0 in range(0, pos, CHUNK):
            b0 = min(a0 + CHUNK, pos)
            lp = torch.log_softmax(logits[:, a0:b0].float(), dim=-1)
            nats += -lp.gather(-1, tgt[:, a0:b0].unsqueeze(-1)).sum().item()
            del lp
        ntok += tgt.numel()
        nbytes += len(t.encode())
    if not nbytes:
        return None
    return {
        "bits_per_byte": round(nats / (math.log(2) * nbytes), 6),
        "nats_total": round(nats, 3),
        "utf8_bytes": nbytes,
        "scored_tokens": ntok,
        "documents": len(texts),
        "note": "First token of each document is unscored (no left context).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true", help="use the frozen 14-doc screen sample")
    ap.add_argument("--heldout", action="append", default=[], metavar="LANG=PATH.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="deterministic subsample per set")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi

    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    rev = a.revision or HfApi().model_info(a.model).sha
    print(f"[*] {a.model} @ {rev}  device={dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=rev,
        # MPS: fp16 measured ~1.5x faster than bf16 (0.79 vs 0.52 doc/s on Qwen3-1.7B).
        # Log-softmax is computed in fp32 regardless, so this affects speed, not the metric.
        dtype=torch.float32 if dev == "cpu" else (torch.float16 if dev == "mps" else torch.bfloat16),
    ).to(dev).eval()

    if a.smoke:
        groups = load_smoke_texts()
    else:
        groups = {}
        for spec in a.heldout:
            lang, _, p = spec.partition("=")
            groups[lang] = load_heldout(p)
        if not groups:
            sys.exit("give --smoke or at least one --heldout LANG=PATH.jsonl")

    if a.limit:
        # deterministic subsample: sort by doc order as written (already hash-ordered by source)
        groups = {k: v[:a.limit] for k, v in groups.items()}
        print(f"[!] SUBSAMPLE: first {a.limit} docs per set. The frozen set is larger; "
              f"record this as a subsample, not the full-set number.", flush=True)
    results = {}
    for lang, texts in groups.items():
        print(f"[*] {lang}: {len(texts)} docs ...", flush=True)
        results[lang] = bpb(model, tok, texts, dev, a.max_len)
        print(f"    bpb={results[lang]['bits_per_byte']}", flush=True)

    suite = json.loads(SUITE.read_text()) if SUITE.exists() else {}
    rec = {
        "run_id": f"PHASE0-C-BASELINE-{a.model.replace('/', '_')}",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "phase0_baseline_bpb",
        "scientific_evidence_allowed": False,
        "eval_suite_id": suite.get("spec", {}).get("suite_id"),
        "eval_suite_sha256": suite.get("spec_sha256"),
        "model": {"repo": a.model, "revision": rev,
                  "parameters": sum(p.numel() for p in model.parameters())},
        "measurement": {"device": dev, "dtype": str(model.dtype), "max_length": a.max_len,
                        "subsample_limit": a.limit,
                        "text_source": "frozen_14doc_screen_sample" if a.smoke else "frozen_heldout_sets"},
        "bpb_by_language": results,
        "limitations": [
            "BPB only. No downstream accuracy here -- run lm-evaluation-harness for the frozen task list.",
            "Smoke runs use 14 short authored documents. They verify the PIPELINE, not the model.",
            "If subsample_limit is set, this is NOT the full frozen held-out set. BPB converges "
            "quickly but the full-set number must be produced on GPU before it is quoted.",
            "BPB is comparable ACROSS MODELS for the same language. It is NOT comparable ACROSS "
            "LANGUAGES: Thai is ~3 UTF-8 bytes per character and English ~1, so Thai BPB is "
            "mechanically lower. Never read 'Thai BPB < English BPB' as 'the model is better at Thai'.",
        ],
        "verdict": "pipeline_smoke_only" if a.smoke else "baseline_recorded",
    }
    out = Path(a.out) if a.out else HERE / f"baseline_{a.model.replace('/', '_')}{'_smoke' if a.smoke else ''}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
