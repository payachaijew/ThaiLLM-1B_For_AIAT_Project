#!/bin/zsh
# สร้าง git history ตามลำดับ C1..C12
# รันจาก repo root:  ./commits/make_commits.sh
set -e
cd "$(dirname "$0")/.."
[[ -d .git ]] || { git init -q && git branch -M main; }

commit_set() {  # $1 = manifest, $2 = message
  local files=()
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    [[ -e "$f" ]] && files+=("$f") || echo "  (ข้าม ไม่พบ: $f)"
  done < "$1"
  [[ ${#files[@]} -eq 0 ]] && { echo "  (ไม่มีไฟล์ ข้าม)"; return; }
  git add -- "${files[@]}"
  git diff --cached --quiet && { echo "  (ไม่มีการเปลี่ยนแปลง ข้าม)"; return; }
  git commit -q -m "$2"
  echo "  -> $(git rev-parse --short HEAD)  $2"
}

echo "C01"; commit_set commits/C01_scaffolding.txt   "chore: repo scaffolding, ignore rules and dependencies"
echo "C02"; commit_set commits/C02_plans.txt         "docs: research plan, build plan and compute/storage plan"
echo "C03"; commit_set commits/C03_configs.txt       "docs: experiment parameters, data manifest and run-record templates"
echo "C04"; commit_set commits/C04_sources.txt       "docs: source registry of verified papers, models, datasets and venue"
echo "C05"; commit_set commits/C05_novelty.txt       "docs: novelty audit, nearest-work matrix, pre-mortem and decision memo"
echo "C06"; commit_set commits/C06_base_selection.txt "docs: base-model screen with tokenizer, license and port audit"
echo "C07"; commit_set commits/C07_tokenizer_ext.txt "feat(phase0): extend tokenizer screen to live base candidates"
echo "C08"; commit_set commits/C08_eval_freeze.txt   "feat(phase0): freeze evaluation suite THAILLM-EVAL-FROZEN-V1"
echo "C09"; commit_set commits/C09_heldout.txt       "feat(phase0): held-out bucket rule and stratified builder"
echo "C10"; commit_set commits/C10_baseline.txt      "feat(phase0): baseline Thai BPB for Qwen3 1.7B and 0.6B"
echo "C11"; commit_set commits/C11_corpus_audit.txt  "feat(phase1): SEA-PILE-v2 Thai corpus quality audit"
echo "C12"; commit_set commits/C12_clean_and_log.txt "feat(phase1): corpus cleaning pipeline, validation log and compliance notes"

echo "\n=== ผลลัพธ์ ==="
git log --oneline
echo "\nขนาด repo: $(git count-objects -vH | grep size-pack)"
