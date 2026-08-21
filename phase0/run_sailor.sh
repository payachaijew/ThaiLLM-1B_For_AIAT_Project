#!/bin/zsh
cd "/Users/prince/Documents/Research for AIAT/research/thai-llm-1b-attnres/phase0"
exec python3 measure_baseline.py --model sail/Sailor2-1B \
  --heldout th=../data/heldout/TH-WEB-HELDOUT.jsonl --device mps \
  --out baseline_Sailor2-1B_TH-WEB-HELDOUT-v2.json
