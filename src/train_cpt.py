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
def build_model(a, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16)

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
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
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
    if is_main:
        print(f"[*] stream {a.stream}  {ds.n:,} sequences  sha {ds.manifest['train_bin_sha256'][:16]}", flush=True)
        print(f"[*] condition {a.condition}  device {dev}  world {world}", flush=True)
        print(f"[*] {tok_per_step:,} tokens/step  ->  {total_steps:,} steps", flush=True)

    model, tok, routed = build_model(a, dev)
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
        if is_main:
            print(f"[*] resumed at step {step}", flush=True)

    held = {}
    for spec in a.heldout:
        lang, _, p = spec.partition("=")
        held[lang] = load_heldout(Path(p), a.eval_docs)

    logf = (out / f"log_rank{rank}.jsonl").open("a")
    t0 = time.time(); tokens_done = 0
    model.train()

    while step < total_steps:
        lr = lr_at(step, total_steps, a.lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tstep = time.time()
        loss_sum = 0.0

        for micro in range(a.grad_accum):
            gs = step * a.grad_accum + micro
            b = ds.batch(gs, a.micro_batch, rank, world)
            ids = torch.from_numpy(b.astype(np.int64)).to(dev)
            out_ = model(input_ids=ids, labels=ids)
            loss = out_.loss / a.grad_accum
            loss.backward()
            loss_sum += float(loss) * a.grad_accum

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        step += 1
        tokens_done += tok_per_step
        dt = time.time() - tstep

        if is_main and (step % 10 == 0 or step == 1):
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
                        "stream_sha256": ds.manifest["train_bin_sha256"]}, ckp)
            print(f"  [saved] {ckp}", flush=True)

    if is_main:
        (out / "final.json").write_text(json.dumps({
            "run": a.out, "condition": a.condition, "steps": step,
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
