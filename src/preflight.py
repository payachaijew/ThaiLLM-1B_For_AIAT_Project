#!/usr/bin/env python3
"""One-GPU preflight. Measures; does not produce a model.

Trains each condition for a short run and throws the weights away. The point is to replace
estimates with measurements before any real budget is committed:

  1. tokens/second       -> the MFU-40%% estimate is a guess that can be off by 2x; this is not
  2. peak VRAM           -> largest batch that fits, and whether the card is big enough
  3. step-time spread    -> spikes and instability show up here, not at hour 40
  4. checkpoint + resume -> a spot instance can be reclaimed at any moment. If resume is
                            broken the whole rented run is lost, and the only cheap moment
                            to discover that is now
  5. S0 vs D1 vs D2      -> the routing overhead ratio, which is the axis Track 2 lives on

  python3 preflight.py --stream ../data/streams/mix50 --out preflight_out

scientific_evidence_allowed = false. Two hundred steps says nothing about model quality.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time, datetime, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, log):
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def read_log(run_dir: Path):
    f = run_dir / "log_rank0.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", default="preflight_out")
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--measure-steps", type=int, default=200)
    ap.add_argument("--conditions", default="S0,D1,D2")
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--routing-heads", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    total = a.warmup_steps + a.measure_steps
    report = {
        "preflight_id": "PREFLIGHT-V1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scientific_evidence_allowed": False,
        "purpose": "measurement only; the trained weights are discarded",
        "config": {"stream": a.stream, "warmup_steps": a.warmup_steps,
                   "measure_steps": a.measure_steps, "micro_batch": a.micro_batch,
                   "grad_accum": a.grad_accum, "seq_len": a.seq_len, "model": a.model},
        "conditions": {}, "resume_test": {}, "errors": [],
    }

    base = [sys.executable, str(HERE / "train_cpt.py"), "--stream", a.stream,
            "--model", a.model, "--seq-len", str(a.seq_len),
            "--micro-batch", str(a.micro_batch), "--grad-accum", str(a.grad_accum),
            "--routing-heads", str(a.routing_heads), "--device", a.device,
            "--max-steps", str(total), "--save-every", str(10**9),
            "--eval-every", str(10**9), "--warmup", str(a.warmup_steps)]

    for cond in a.conditions.split(","):
        cond = cond.strip()
        d = out / f"run_{cond}"
        print(f"\n=== {cond} : {total} steps ===", flush=True)
        t0 = time.time()
        rc = run(base + ["--condition", cond, "--out", str(d)], out / f"{cond}.log")
        wall = time.time() - t0
        recs = [r for r in read_log(d) if "tokens_per_s" in r and r["step"] > a.warmup_steps]
        if rc != 0 or not recs:
            report["errors"].append(f"{cond}: rc={rc}, {len(recs)} measured steps")
            report["conditions"][cond] = {"status": "FAILED", "returncode": rc,
                                          "log": f"{cond}.log"}
            print(f"  FAILED rc={rc} — ดู {out/(cond+'.log')}", flush=True)
            continue
        tps = [r["tokens_per_s"] for r in recs]
        st = [r["step_time_s"] for r in recs]
        vram = [r["peak_vram_gb"] for r in recs if "peak_vram_gb" in r]
        report["conditions"][cond] = {
            "status": "ok",
            "measured_steps": len(recs),
            "tokens_per_s_median": round(statistics.median(tps), 1),
            "tokens_per_s_mean": round(statistics.mean(tps), 1),
            "step_time_s_median": round(statistics.median(st), 4),
            "step_time_s_p90": round(sorted(st)[int(0.9 * len(st))], 4),
            "step_time_cv": round(statistics.pstdev(st) / statistics.mean(st), 4),
            "peak_vram_gb": max(vram) if vram else None,
            "wall_seconds": round(wall, 1),
            "final_loss": recs[-1]["loss"],
        }
        c = report["conditions"][cond]
        print(f"  {c['tokens_per_s_median']:,.0f} tok/s median · "
              f"VRAM {c['peak_vram_gb']} GB · step CV {c['step_time_cv']}", flush=True)

    ok = {k: v for k, v in report["conditions"].items() if v.get("status") == "ok"}
    if "S0" in ok:
        s0 = ok["S0"]["tokens_per_s_median"]
        for k, v in ok.items():
            v["throughput_vs_S0"] = round(v["tokens_per_s_median"] / s0, 4)
        report["routing_overhead"] = {
            "note": "arXiv:2607.27230 reports 0.55-0.88x of baseline for the AttnRes family "
                    "even with fused Triton kernels. Compare.",
            "ratios": {k: v["throughput_vs_S0"] for k, v in ok.items()},
        }

    # ---- resume test: the cheap check that saves a whole rented run
    print("\n=== resume test (S0) ===", flush=True)
    rd = out / "resume_test"
    half = max(a.warmup_steps + 10, 30)
    rc1 = run(base + ["--condition", "S0", "--out", str(rd), "--max-steps", str(half),
                      "--save-every", str(half)], out / "resume_a.log")
    ck = rd / f"ckpt_{half}.pt"
    if rc1 == 0 and ck.exists():
        rc2 = run(base + ["--condition", "S0", "--out", str(rd / "cont"),
                          "--max-steps", str(half + 10), "--resume", str(ck)],
                  out / "resume_b.log")
        fresh = [r for r in read_log(rd) if r.get("step") == half]
        cont = [r for r in read_log(rd / "cont") if r.get("step") == half + 10]
        report["resume_test"] = {
            "status": "ok" if rc2 == 0 and cont else "FAILED",
            "checkpoint": str(ck.name),
            "loss_before": fresh[0]["loss"] if fresh else None,
            "loss_after_resume": cont[0]["loss"] if cont else None,
            "interpretation": "A run that cannot resume will lose everything if a spot "
                              "instance is reclaimed. This must be ok before the main run.",
        }
    else:
        report["resume_test"] = {"status": "FAILED", "checkpoint_written": ck.exists(),
                                 "returncode": rc1}
    print(f"  {report['resume_test']['status']}", flush=True)

    # ---- budget projection from MEASURED throughput
    if "S0" in ok:
        tps = ok["S0"]["tokens_per_s_median"]
        report["budget_projection"] = {
            "basis": "measured S0 median tokens/s on THIS machine, single process",
            "tokens_per_s": tps,
            "gpu_hours_per_billion_tokens": round(1e9 / tps / 3600, 2),
            "for_budgets": {f"{b}B": round(b * 1e9 / tps / 3600, 1) for b in (1, 3, 6, 10)},
            "note": "Multiply by GPU count only if you scale the job; total GPU-hours stay "
                    "roughly constant, so cost is driven by tokens, not by card count.",
        }
        bp = report["budget_projection"]
        print(f"\n=== งบจากตัวเลขจริง ===")
        print(f"  {tps:,.0f} tok/s  ->  {bp['gpu_hours_per_billion_tokens']} GPU-hr ต่อ 1B tokens")
        for k, v in bp["for_budgets"].items():
            print(f"    {k:>4s}: {v:6.1f} GPU-hr")

    (out / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n[+] {out/'preflight_report.json'}")
    if report["errors"] or report["resume_test"].get("status") != "ok":
        print("[!] มีข้อผิดพลาด — อย่าเพิ่งเริ่มรอบหลัก")
        sys.exit(1)


if __name__ == "__main__":
    main()
