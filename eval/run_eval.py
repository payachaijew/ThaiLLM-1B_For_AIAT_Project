#!/usr/bin/env python3
"""Run the frozen evaluation suite.

Task list and shot counts are read from phase0/eval_suite_frozen.json rather than typed here,
so an evaluation cannot quietly drift from what was frozen before the first baseline. The
suite's sha256 and the lm-eval commit are written into every result file.

  python3 eval/run_eval.py --model Qwen/Qwen3-1.7B-Base --out eval/results/base
  python3 eval/run_eval.py --model runs/main_s0/hf --out eval/results/main_s0 --device cuda

scientific_evidence_allowed = false until it runs against a real trained checkpoint.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, datetime, importlib.metadata as md
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUITE = ROOT / "phase0" / "eval_suite_frozen.json"

# frozen task name -> lm-eval task name. Anything unmapped is reported, never silently skipped.
MAP = {
    "thaiexam": "thaiexam",
    "m3exam_th": "m3exam_th",
    "belebele_th": "belebele_tha_Thai",
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
    "arc_challenge": "arc_challenge",
    "humaneval": "humaneval",
}
BPB_ONLY = {"thai_bpb_heldout", "english_bpb_heldout", "code_bpb_heldout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="smoke only; never for a reported result")
    ap.add_argument("--dtype", default="bfloat16")
    # HumanEval scores by EXECUTING model-generated code. lm-eval refuses unless this is set,
    # and that refusal is correct: run it in a disposable environment such as the rented GPU
    # box, not on a personal machine.
    ap.add_argument("--allow-code-exec", action="store_true",
                    help="required for humaneval; executes model-generated code")
    a = ap.parse_args()

    suite = json.loads(SUITE.read_text())
    spec = suite["spec"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tasks, shots, skipped = [], {}, []
    for group in ("thai_acquisition", "retention"):
        for t in spec[group]:
            name = t["task"]
            if name in BPB_ONLY:
                skipped.append({"task": name, "why": "BPB is measured by src/train_cpt.py "
                                                     "and phase0/measure_baseline.py, not lm-eval"})
                continue
            if name not in MAP:
                skipped.append({"task": name, "why": "no lm-eval mapping"})
                continue
            tasks.append(MAP[name]); shots[MAP[name]] = t["shots"]

    if not tasks:
        sys.exit("no runnable tasks")

    # lm-eval takes one --num_fewshot for the whole call, so group tasks by shot count
    groups = {}
    for t in tasks:
        groups.setdefault(shots[t], []).append(t)

    results = {}
    for n, ts in sorted(groups.items()):
        margs = f"pretrained={a.model},dtype={a.dtype}"
        if a.revision:
            margs += f",revision={a.revision}"
        rf = out / f"shots{n}.json"
        cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", margs,
               "--tasks", ",".join(ts), "--num_fewshot", str(n),
               "--device", a.device, "--batch_size", str(a.batch_size),
               "--include_path", str(HERE / "tasks"),
               "--output_path", str(rf)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        print(f"\n=== {n}-shot: {', '.join(ts)} ===", flush=True)
        env = dict(os.environ)
        if a.allow_code_exec:
            env["HF_ALLOW_CODE_EVAL"] = "1"
        elif "humaneval" in ts:
            print("  [!] ข้าม humaneval: ต้องใส่ --allow-code-exec (รันโค้ดที่โมเดลสร้าง)")
            results[f"{n}shot"] = {"tasks": ts, "returncode": "skipped_needs_allow_code_exec"}
            continue
        rc = subprocess.run(cmd, env=env).returncode
        results[f"{n}shot"] = {"tasks": ts, "returncode": rc, "output": str(rf)}

    rep = {
        "eval_id": f"EVAL-{Path(a.model).name}",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "model": {"path": a.model, "revision": a.revision, "dtype": a.dtype},
        "eval_suite": {"suite_id": spec["suite_id"], "spec_sha256": suite["spec_sha256"]},
        "lm_eval_version": md.version("lm_eval"),
        "limit": a.limit,
        "runs": results,
        "not_run_by_lm_eval": skipped,
        "custom_tasks": ["thaiexam (5 subsets)", "m3exam_th"],
        "warning": ("--limit was set, so these numbers are a smoke check and must not be "
                    "reported") if a.limit else None,
    }
    (out / "eval_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n")
    print(f"\n[+] {out/'eval_report.json'}")
    for s in skipped:
        print(f"[!] ไม่ได้รันผ่าน lm-eval: {s['task']} — {s['why']}")


if __name__ == "__main__":
    main()
