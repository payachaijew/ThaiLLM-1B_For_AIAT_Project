#!/bin/zsh
# Phase 0 overnight batch -- all CPU/MPS, no rented GPU required.
# Steps are ordered most-valuable-first so a failure still leaves the important work done.
set -u
cd "/Users/prince/Documents/Research for AIAT/research/thai-llm-1b-attnres/phase0"
LOG_DIR="../phase0/overnight_logs"; mkdir -p "$LOG_DIR"
say() { echo "\n=========== $1  [$(date '+%H:%M:%S')] ===========" }

say "STEP 1/5  download all 54 Thai shards (22.2 GB)"
python3 - > "$LOG_DIR/01_download.log" 2>&1 <<'PY'
import warnings; warnings.filterwarnings("ignore")
from huggingface_hub import hf_hub_download
for i in range(54):
    f = f"th/train-{i:05d}-of-00054.parquet"
    try:
        hf_hub_download("aisingapore/SEA-PILE-v2", f, repo_type="dataset")
        print(f"ok   {f}", flush=True)
    except Exception as e:
        print(f"FAIL {f}: {type(e).__name__}: {e}", flush=True)
PY
echo "  -> $(grep -c '^ok' "$LOG_DIR/01_download.log") / 54 shards"

say "STEP 2/5  rebuild TH-WEB-HELDOUT with STRATIFIED shards (fixes temporal bias)"
python3 build_heldout.py --set TH-WEB-HELDOUT --target 2000 --max-scan 2000000 \
  > "$LOG_DIR/02_heldout.log" 2>&1
tail -4 "$LOG_DIR/02_heldout.log"

say "STEP 3/5  corpus audit over 1,000,000 docs"
python3 corpus_audit.py 1000000 > "$LOG_DIR/03_audit.log" 2>&1
tail -14 "$LOG_DIR/03_audit.log"

say "STEP 4/5  baseline Qwen3-1.7B-Base on the NEW held-out set"
python3 measure_baseline.py --model Qwen/Qwen3-1.7B-Base \
  --heldout th=../data/heldout/TH-WEB-HELDOUT.jsonl --device mps \
  --out baseline_Qwen3-1.7B-Base_TH-WEB-HELDOUT-v2.json > "$LOG_DIR/04_bl17.log" 2>&1
grep -E "bpb=|bits_per_byte" "$LOG_DIR/04_bl17.log" | tail -2

say "STEP 5/5  baseline Qwen3-0.6B-Base (sanity: smaller model must be WORSE)"
python3 measure_baseline.py --model Qwen/Qwen3-0.6B-Base \
  --heldout th=../data/heldout/TH-WEB-HELDOUT.jsonl --device mps \
  --out baseline_Qwen3-0.6B-Base_TH-WEB-HELDOUT-v2.json > "$LOG_DIR/05_bl06.log" 2>&1
grep -E "bpb=|bits_per_byte" "$LOG_DIR/05_bl06.log" | tail -2

say "SUMMARY"
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("baseline_*TH-WEB-HELDOUT-v2.json")):
    d = json.load(open(f))
    th = d["bpb_by_language"].get("th", {})
    print(f"  {d['model']['repo']:24s} thai_bpb={th.get('bits_per_byte')} docs={th.get('documents')}")
try:
    m = json.load(open("../data/heldout/TH-WEB-HELDOUT.manifest.json"))
    print(f"  heldout set_sha256 = {m['set_sha256']}")
    print(f"  shards used        = {m['source']['files']}")
except Exception as e:
    print("  (manifest unreadable)", e)
PY
echo "\nDONE $(date)"
