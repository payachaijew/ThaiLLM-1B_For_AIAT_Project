#!/usr/bin/env python3
"""Continued pretraining for ThaiLLM-1B.

Reads a packed token stream produced by phase0/build_training_stream.py. All three
architecture conditions read the SAME stream file, so "identical data, identical order"
holds by construction and is verifiable through the stream manifest's sha256.

  # single GPU
  python3 train_cpt.py --stream ../data/streams/main --condition S0 --out runs/main_s0

  # multi GPU
  torchrun --nproc_per_node 2 train_cpt.py --stream ../data/streams/main --condition S0 --out runs/main_s0

Conditions
  S0  standard residual, untouched base model
  D1  Delta Block AttnRes, one routing head
  D2  Delta Block AttnRes + MHAR, --routing-heads heads

Resume is exact: the data order is a deterministic permutation of sequence indices seeded by
--seed, so restarting at global_step N replays precisely the sequences a fresh run would have
seen. Nothing about the order depends on how many times the job was interrupted.
"""
from __future__ import annotations
import argparse, json, math, os, time, sys, gzip, datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------- data
class StreamDataset:
    """Flat uint32 file of fixed-length sequences. Order is a seeded permutation."""

    def __init__(self, stream_dir: Path, seq_len: int, seed: int):
        self.bin = stream_dir / "train.bin"
        man = json.loads((stream_dir / "manifest.json").read_text())
        if man["sequence_length"] != seq_len:
            raise ValueError(f"stream seq_len {man['sequence_length']} != --seq-len {seq_len}")
        self.seq_len = seq_len
        self.manifest = man
        self.arr = np.memmap(self.bin, dtype=np.uint32, mode="r")
        self.n = self.arr.size // seq_len
        self.order = np.random.default_rng(seed).permutation(self.n)

    def batch(self, step: int, batch_size: int, rank: int, world: int) -> np.ndarray:
        """Sequences for one optimiser step on one rank. Wraps around if the stream runs out."""
        start = step * batch_size * world + rank * batch_size
        idx = self.order[[(start + i) % self.n for i in range(batch_size)]]
        return np.stack([self.arr[i*self.seq_len:(i+1)*self.seq_len] for i in idx])


def load_heldout(path: Path, limit: int | None = None):
    op = gzip.open if str(path).endswith(".gz") else open
    out = []
    with op(path, "rt") as fh:
        for line in fh:
            out.append(json.loads(line)["text"])
            if limit and len(out) >= limit:
                break
    return out


@torch.no_grad()
def eval_bpb(model, tok, texts, device, max_len=2048, chunk=256):
    """Bits per byte. Chunked over sequence positions: a full float32 copy of
    [1, seq, vocab] is ~1.2 GB at this vocabulary and caused memory thrashing when done
    in one piece. Chunking bounds the transient and leaves the metric unchanged."""
    was = model.training
    model.eval()
    nats = nbytes = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        logits = model(ids).logits
        tgt = ids[:, 1:]
        pos = logits.shape[1] - 1
        for a in range(0, pos, chunk):
            b = min(a + chunk, pos)
            lp = torch.log_softmax(logits[:, a:b].float(), -1)
            nats += -lp.gather(-1, tgt[:, a:b].unsqueeze(-1)).sum().item()
            del lp
        nbytes += len(t.encode())
    model.train(was)
    return nats / (math.log(2) * nbytes) if nbytes else None


# ---------------------------------------------------------------- model
def autocast_dtype(a, device: str):
    """bf16 everywhere except MPS, which has patchy bf16 support and was measured faster in
    fp16 on this project's laptop (0.79 vs 0.52 doc/s)."""
    if a.precision == "bf16":
        return None                      # weights are already bf16; no autocast needed
    return torch.float16 if device.startswith("mps") else torch.bfloat16


def build_model(a, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    # Mixed precision keeps the master weights in fp32 and casts only the forward pass.
    # Pure bf16 has an 8-bit mantissa, so at lr 2e-5 a typical update is smaller than one ulp
    # of the weight it is applied to and is rounded away entirely: the run looks healthy and
    # the model barely moves. exp_avg_sq suffers worse, since it holds squared gradients.
    # The cost is memory - fp32 master, fp32 grads and fp32 Adam moments roughly double it -
    # which is why the card has to be sized for this choice rather than the other way round.
    param_dtype = torch.bfloat16 if a.precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=param_dtype)

    routed = None
    if a.condition != "S0":
        from routing import DeltaAttnResAdapter
        heads = 1 if a.condition == "D1" else a.routing_heads
        routed = DeltaAttnResAdapter(model, block_size_layers=a.block_size, num_heads=heads,
                                     null_logit_init=a.null_logit_init)
        model = routed

    # Enabled for EVERY condition on purpose. The archived adapter refused gradient
    # checkpointing, so the baseline ran in a different memory regime from the routed arms
    # and the GPU-hour comparison measured the wrapper rather than the architecture.
    if a.grad_checkpoint:
        base = routed.base_model if routed is not None else model
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
            if hasattr(base, "config"):
                base.config.use_cache = False
    return model.to(device), tok, routed



def assert_routers_learn(model, routed, ds, dev, seq_len, amp_dtype):
    """One forward/backward before training, to prove the routers actually receive gradient.

    The archived adapter multiplied the routed mixture by a scale that started at zero, which
    made routing-parameter gradients identically zero — the run trained happily and produced a
    clean-looking negative result that was an artefact of the wrapper. That failure is silent,
    so it is checked rather than trusted.

    Routers in the FIRST block are inert by construction: there is no earlier block for them to
    route over. They are expected to stay at their init values forever, and are reported here so
    nobody later reads an unchanged first-block router as this bug returning.
    """
    b = ds.batch(0, 1, 0, 1)
    ids = torch.from_numpy(b.astype(np.int64)).to(dev)
    if amp_dtype is None:
        model(input_ids=ids, labels=ids).loss.backward()
    else:
        with torch.autocast(device_type=dev.split(":")[0], dtype=amp_dtype):
            model(input_ids=ids, labels=ids).loss.backward()

    inert, live, dead, gated = [], 0, [], 0
    for n, prm in model.named_parameters():
        if ".router." not in n:
            continue
        layer = int(n.split(".layers.")[1].split(".")[0])
        grad = 0.0 if prm.grad is None else float(prm.grad.abs().max())
        if layer < routed.layers_without_sources:
            inert.append(n)          # first block: nothing earlier to route over
        elif n.endswith(".norm.weight") and grad == 0.0:
            # Expected, and only at init. The logit is query . norm(x), so d/d(norm.weight)
            # carries a factor of query, which the paper initialises to exactly zero. Measured:
            # norm.weight grad 0.000e+00 at init, 1.05e-05 after query moves by 1e-3.
            gated += 1
        elif grad > 0:
            live += 1
        else:
            dead.append(n)
    model.zero_grad(set_to_none=True)

    print(f"[*] routers: {live} receive gradient, {gated} gated by the zero-init query "
          f"(live once query moves), {len(inert)} inert in the first block "
          f"(no earlier block to route over)", flush=True)
    if dead:
        raise RuntimeError(
            f"{len(dead)} routing parameters outside the first block received no gradient, "
            f"e.g. {dead[:3]}. Training would silently measure a disabled treatment. "
            f"See src/routing.py FIX 1 (route_scale).")
    if live == 0:
        raise RuntimeError("no routing parameter received gradient at all")


def lr_at(step, total, peak, warmup, floor_ratio=0.1):
    if step < warmup:
        return peak * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return peak * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="S0", choices=["S0", "D1", "D2"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--revision", default="ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    ap.add_argument("--routing-heads", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--null-logit-init", type=float, default=2.0,
                    help="softly gates the routed mixture at init; see DeltaRouter docstring. "
                         "0=paper default (disruptive), 2=default here, >=4 risks a router "
                         "that learns too slowly to matter")
    ap.add_argument("--precision", default="mixed", choices=["mixed", "bf16"],
                    help="mixed = fp32 master weights + autocast forward (default, and what a "
                         "multi-billion-token run needs). bf16 = pure bf16, roughly half the "
                         "memory but updates below one ulp are silently lost.")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--schedule-steps", type=int, default=None,
                    help="horizon of the cosine LR schedule. Defaults to --max-steps. Set it "
                         "explicitly when a run will be stopped early and resumed, so the "
                         "schedule does not silently change shape between legs.")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--log-every", type=int, default=10,
                    help="preflight sets this to 1 so throughput statistics are computed from "
                         "every step rather than from a handful of sampled ones")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-docs", type=int, default=200)
    ap.add_argument("--heldout", action="append", default=[], metavar="LANG=PATH")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local)

    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if ddp:
        dev = f"cuda:{local}"
    is_main = rank == 0

    torch.manual_seed(a.seed + rank)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ds = StreamDataset(Path(a.stream), a.seq_len, a.seed)
    tok_per_step = a.micro_batch * a.grad_accum * a.seq_len * world
    total_steps = a.max_steps or (ds.n * a.seq_len) // tok_per_step
    # The cosine denominator, kept separate from the stopping point. A run stopped at step 30
    # and resumed to 40 must follow the SAME schedule as a run that went straight to 40;
    # otherwise every leg after an interruption trains at a different learning rate and the
    # resumed run is not the run you meant to do. This is recorded in the checkpoint and
    # checked on resume, because the divergence is invisible in the loss curve.
    sched_steps = a.schedule_steps or total_steps
    if is_main:
        print(f"[*] stream {a.stream}  {ds.n:,} sequences  sha {ds.manifest['train_bin_sha256'][:16]}", flush=True)
        print(f"[*] condition {a.condition}  device {dev}  world {world}", flush=True)
        print(f"[*] {tok_per_step:,} tokens/step  ->  {total_steps:,} steps", flush=True)

    amp_dtype = autocast_dtype(a, dev)
    model, tok, routed = build_model(a, dev)
    if is_main:
        n = sum(p.numel() for p in model.parameters())
        bytes_per = 4 if a.precision == "mixed" else 2
        # master + grads + exp_avg + exp_avg_sq
        gb = n * bytes_per * 4 / 1e9
        print(f"[*] precision {a.precision}  autocast {amp_dtype}  {n/1e9:.2f}B params  "
              f"~{gb:.1f} GB for weights+grads+optimizer (activations on top)", flush=True)
    if routed is not None and is_main:
        assert_routers_learn(model, routed, ds, dev, a.seq_len, amp_dtype)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8)

    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        (model.module if ddp else model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        prev = ck.get("schedule_steps")
        if prev is not None and prev != sched_steps:
            raise RuntimeError(
                f"checkpoint was trained on a {prev}-step LR schedule but this run uses "
                f"{sched_steps}. Pass --schedule-steps {prev} to continue the same schedule, "
                f"or accept the change deliberately.")
        if is_main:
            print(f"[*] resumed at step {step}  (LR horizon {sched_steps})", flush=True)

    held = {}
    for spec in a.heldout:
        lang, _, p = spec.partition("=")
        held[lang] = load_heldout(Path(p), a.eval_docs)

    logf = (out / f"log_rank{rank}.jsonl").open("a")
    t0 = time.time(); tokens_done = 0
    model.train()

    while step < total_steps:
        lr = lr_at(step, sched_steps, a.lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tstep = time.time()
        loss_sum = 0.0

        for micro in range(a.grad_accum):
            gs = step * a.grad_accum + micro
            b = ds.batch(gs, a.micro_batch, rank, world)
            ids = torch.from_numpy(b.astype(np.int64)).to(dev)
            if amp_dtype is None:
                out_ = model(input_ids=ids, labels=ids)
            else:
                with torch.autocast(device_type=dev.split(":")[0], dtype=amp_dtype):
                    out_ = model(input_ids=ids, labels=ids)
            loss = out_.loss.float() / a.grad_accum
            loss.backward()
            loss_sum += loss.detach().item() * a.grad_accum

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        step += 1
        tokens_done += tok_per_step
        dt = time.time() - tstep

        if is_main and (step % a.log_every == 0 or step == 1):
            rec = {"step": step, "loss": round(loss_sum / a.grad_accum, 5), "lr": lr,
                   "grad_norm": float(gn), "step_time_s": round(dt, 4),
                   "tokens_per_s": round(tok_per_step / dt, 1),
                   "elapsed_min": round((time.time() - t0) / 60, 2)}
            if dev.startswith("cuda"):
                rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"  step {step}/{total_steps}  loss {rec['loss']:.4f}  "
                  f"{rec['tokens_per_s']:,.0f} tok/s  {rec.get('peak_vram_gb','-')} GB", flush=True)

        if is_main and held and step % a.eval_every == 0:
            m = (model.module if ddp else model)
            ev = {"step": step, "event": "eval"}
            for lang, texts in held.items():
                ev[f"bpb_{lang}"] = eval_bpb(m, tok, texts, dev, chunk=256)
            logf.write(json.dumps(ev) + "\n"); logf.flush()
            print(f"  [eval] {ev}", flush=True)

        if is_main and step % a.save_every == 0:
            ckp = out / f"ckpt_{step}.pt"
            torch.save({"step": step, "model": (model.module if ddp else model).state_dict(),
                        "optimizer": opt.state_dict(), "args": vars(a),
                        "schedule_steps": sched_steps,
                        "stream_sha256": ds.manifest["train_bin_sha256"]}, ckp)
            print(f"  [saved] {ckp}", flush=True)

    if is_main:
        (out / "final.json").write_text(json.dumps({
            "run": a.out, "condition": a.condition, "steps": step,
            "precision": a.precision, "autocast_dtype": str(amp_dtype),
            "tokens_seen": tokens_done, "wall_minutes": round((time.time() - t0) / 60, 1),
            "stream": a.stream, "stream_sha256": ds.manifest["train_bin_sha256"],
            "args": vars(a), "scientific_evidence_allowed": False,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2) + "\n")
        print("[+] done", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
