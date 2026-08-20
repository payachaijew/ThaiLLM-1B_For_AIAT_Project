# Compute Estimate — 4×A100

**Estimate ID:** `BASE-SCREEN-2026-08-18-COMPUTE`
**Date:** 2026-08-18
**Evidence class:** analytic model. **Nothing here was measured on a GPU.**
**`scientific_evidence_allowed`:** `false`

## 0. Read this before using any number below

Every figure in this document is arithmetic from published config values plus one assumed
efficiency constant. No A100 was touched. The `A100-PREFLIGHT-*` run required by
`pre_train_validation_plan.md` is what turns these into evidence. Treat them as **budget planning
inputs with roughly ±30 % uncertainty**, not as results.

**Unresolved hardware question — must be answered before scheduling.** LANTA's GPU nodes are
4×A100 **40 GB**, but this was not verified for this project's allocation. The memory tables below
report both. It happens not to change the verdict (see §3), but it changes the achievable
micro-batch, and therefore the step count, and therefore the wall-clock.

## 1. Assumptions, stated so they can be attacked

| Assumption | Value | Basis |
|---|---|---|
| Sustained bf16 throughput per A100 | 120 TFLOPS | ~38 % MFU of 312 TFLOPS peak, typical for a ~1B dense decoder with FlashAttention-2. **Assumed.** |
| GPUs | 4 | Given |
| FLOPs per token, fwd+bwd | `6·N_total + 12·L·seq·d` | Standard accounting. `N_total` includes embeddings because the `lm_head` matmul is real compute |
| Gradient checkpointing overhead | ×1.33 | Full activation recompute |
| Optimizer | AdamW, bf16 params + fp32 master | 16 bytes/param unsharded |
| Sharding | ZeRO-2 (weights replicated, grad+master+moments sharded ÷4) | Recommended regime for this size |
| Routed-arm throughput | **0.55–0.88×** of standard | AR03 (arXiv:2607.27230) published range. **Different hardware and model — this is the weakest assumption in the document.** |

## 2. Parameters ≠ token budget

These are two independent axes and the plan must not conflate them.

**Axis 1 — model size (fixed by the base choice).** Exact counts, verified against each repo's
`safetensors` metadata at the pinned revision:

| Model | Total | Embedding | Non-embedding | Emb % | Layers × d |
|---|---:|---:|---:|---:|---|
| `Qwen/Qwen3-0.6B-Base` | 596,049,920 | 155.6 M (tied) | 440.5 M | 26.1 % | 28 × 1024 |
| `google/gemma-3-1b-pt` | 999,885,952 | 302.0 M (tied) | 697.9 M | 30.2 % | 26 × 1152 |
| `meta-llama/Llama-3.2-1B` | 1,235,814,400 | 262.7 M (tied) | 973.1 M | 21.3 % | 16 × 2048 |
| `allenai/OLMo-2-0425-1B` | **1,484,916,736** | 411.0 M (untied) | 1,073.8 M | 27.7 % | 16 × 2048 |
| `Qwen/Qwen2.5-1.5B` | 1,543,714,304 | 233.4 M (tied) | 1,310.3 M | 15.1 % | 28 × 1536 |

Two corrections to `configs/experiment_parameters.json` fall out of this:
`allenai/OLMo-2-0425-1B` was recorded as 1.0 B and is **1.485 B** — a 48 % error that would have
mis-sized the whole OLMo arm. `Qwen/Qwen3-0.6B-Base` was recorded as 600 M and is 596.0 M (minor).

**Axis 2 — token budget (fixed by the tokenizer, *not* by the model size).** The Thai corpus is a
fixed number of *bytes*. How many *tokens* that is depends entirely on the tokenizer, and cost
scales with tokens:

| Model | Thai bytes/token | ×tokens vs Gemma 3 | SEA-PILE-v2 `th` becomes |
|---|---:|---:|---:|
| `google/gemma-3-1b-pt` | 7.778 | 1.00× | 6.50 B tokens |
| `meta-llama/Llama-3.2-1B` | 5.982 | 1.30× | 8.45 B tokens |
| `Qwen/Qwen2.5-1.5B`, `Qwen3-0.6B-Base` | 5.047 | 1.54× | 10.02 B tokens |
| `allenai/OLMo-2-0425-1B` | 2.971 | **2.62×** | 17.01 B tokens |
| `tiiuae/Falcon3-1B-Base` | 2.024 | **3.84×** | 24.98 B tokens |

Anchor: the SEA-PILE-v2 card reports the `th` subset as 6.5 B tokens *under the Gemma 3 tokenizer*.
Multipliers are the ratio of measured Thai bytes/token from `tokenizer_screen.json`.

**The two axes multiply.** OLMo 2 is 1.20× Llama's parameters *and* 2.01× Llama's token count for
the same Thai text, so the same corpus costs it ≈2.35× the GPU-hours. That compounding — not the
tokenizer number alone — is what makes OLMo 2 uneconomical as the Thai base.

## 3. Memory on 4×A100 (seq 4096, micro-batch 4, bf16, grad-ckpt on)

| Model | Weights+optim, plain DDP (16N) | Weights+optim, ZeRO-2 | Activations | Delta buffer (D1/D2) | **ZeRO-2 total** |
|---|---:|---:|---:|---:|---:|
| `Qwen3-0.6B-Base` | 8.9 GB | 3.05 GB | 1.13 GB | 0.22 GB | **≈ 4.4 GB** |
| `gemma-3-1b-pt` | 14.9 GB | 5.12 GB | 1.19 GB | 0.46 GB | **≈ 6.8 GB** |
| `Llama-3.2-1B` | 18.4 GB | 6.33 GB | 1.50 GB | 0.25 GB | **≈ 8.1 GB** |
| `OLMo-2-0425-1B` | 22.1 GB | 7.61 GB | 1.50 GB | 0.25 GB | **≈ 9.4 GB** |
| `Qwen2.5-1.5B` | 23.0 GB | 7.91 GB | 1.69 GB | 0.33 GB | **≈ 9.9 GB** |

**Verdict: memory is not a constraint for any candidate, on 40 GB or 80 GB.** Even naive DDP with
no sharding fits every candidate on a 40 GB A100 with >15 GB spare. ZeRO-2 leaves so much headroom
that micro-batch can be raised well above 4 to improve utilisation.

The **delta buffer** is the only AttnRes-specific memory cost: `n_sources × batch × seq × d × 2`
bytes. At block size 4 it is 0.22–0.46 GB — under 5 % of the ZeRO-2 footprint. **AttnRes does not
create a memory problem at this scale.** Its cost is throughput, not VRAM.

## 4. Throughput and sequence length

Standard-residual (S0) figures, seq 4096, grad-ckpt on:

| Model | GFLOPs/token | tokens/s (4 GPUs) | GPU-h per 100 M tokens |
|---|---:|---:|---:|
| `Qwen3-0.6B-Base` | 4.99 | ≈ 72,400 | **1.53** |
| `gemma-3-1b-pt` | 7.47 | ≈ 48,300 | **2.30** |
| `Llama-3.2-1B` | 9.03 | ≈ 40,000 | **2.78** |
| `OLMo-2-0425-1B` | 10.52 | ≈ 34,300 | **3.24** |
| `Qwen2.5-1.5B` | 11.38 | ≈ 31,700 | **3.50** |

**Recommended pilot sequence length: 2048.** Reasoning, in order of weight:

1. Thai CPT documents are mostly short. At Llama-3.2's measured Thai fertility, a 4096-token window
   holds ≈24 kB of Thai text — far longer than a typical web or encyclopedic document, so seq 4096
   is mostly packing unrelated documents together.
2. The attention term `12·L·seq·d` is 18 % of Llama-3.2's per-token FLOPs at seq 4096 and 9 % at
   2048 — a direct ~9 % throughput gain on the arm being measured.
3. Halving seq doubles the number of optimizer steps for a fixed token budget, which gives the
   routers more updates within the 50–100 M-token pilot. Given `ROUTE_SCALE_GATE`, more router
   steps is exactly what the pilot needs to avoid a false negative.
4. **Constraint:** OLMo 2's context is only 4096, so 2048 is the only length that is legal on every
   candidate. Sequence length must be identical across all arms and all families
   (`research_plan.md` §7).

Final CPT can move to 4096 once the pilot has settled the routing question, but then it must be
4096 for **every** arm.

## 5. Pilot feasibility (50–100 M observed tokens per condition)

Using the AR03 0.55–0.88× routed penalty. `research_plan.md` §7 requires an **optimised native S0
path**, not a routed wrapper with routing switched off, so S0 carries no penalty.

**Pilot on `Qwen3-0.6B-Base` @ 100 M tokens/condition, 1 seed:**

| Condition | GPU-hours | Wall-clock on one 4×A100 node |
|---|---:|---:|
| S0 | 1.53 | 23 min |
| D1 | 1.74 – 2.78 | 26 – 42 min |
| D2 | 1.74 – 2.78 | 26 – 42 min |
| **Total (3 conditions)** | **5.0 – 7.1** | **1.3 – 1.8 h** |
| **Total (3 conditions × 2 seeds)** | **10.0 – 14.2** | **2.5 – 3.6 h** |

**Pilot on `Llama-3.2-1B` @ 100 M tokens/condition, 1 seed:** 9.1 – 12.9 GPU-hours, 2.3 – 3.2 h wall.

**Verdict: the pilot is trivially affordable on either model.** At these costs the choice should
*not* be driven by compute. Two consequences worth acting on:

- Run the pilot at the **full 100 M** end of the 50–100 M band, not 50 M. The extra cost is under
  4 GPU-hours and it roughly doubles the router's update count.
- Seriously consider running the pilot on **both** `Qwen3-0.6B-Base` and `Llama-3.2-1B`
  (≈15–20 GPU-hours combined). That converts "does the pilot transfer to the 1B base?" from an
  assumption into a measurement, and it directly addresses the reviewer objection that a 0.6 B
  screening result was extrapolated to 1.2 B. This is the single highest-value use of pilot budget.

## 6. Final CPT feasibility

Budget definition, stated explicitly so it is not confused with parameter count:

- Thai: one pass over SEA-PILE-v2 `th` ≈ 8.45 B Llama-3.2 tokens
- Mixture 90 % Thai / 5 % English / 5 % code (`experiment_parameters.json`)
- ⇒ total observed tokens ≈ 8.45 / 0.90 ≈ 9.4 B → **round to 10 B observed tokens per condition**

On `meta-llama/Llama-3.2-1B`, 4×A100, seq 4096:

| Condition | GPU-hours | Wall-clock, one node |
|---|---:|---:|
| S0 | 278 | 69 h (2.9 days) |
| D1 (routed) | 316 – 505 | 79 – 126 h |
| D2 (routed) | 316 – 505 | 79 – 126 h |
| **S0 + one promoted arm** | **594 – 783** | **6.2 – 8.1 days** |
| **All three arms at full scale** | **910 – 1,288** | **9.5 – 13.4 days** |

| Base | 10 B-token CPT, S0 only |
|---|---|
| `gemma-3-1b-pt` (would be 7.7 B tokens for the same corpus) | ≈ 178 GPU-h |
| `Llama-3.2-1B` | ≈ 278 GPU-h |
| `Qwen2.5-1.5B` (11.6 B tokens for the same corpus) | ≈ 414 GPU-h |
| `OLMo-2-0425-1B` (19.7 B tokens for the same corpus) | ≈ 638 GPU-h |

**Verdict: the final CPT is feasible for Llama 3.2 1B and Gemma 3 1B; affordable but noticeably
more expensive for Qwen2.5-1.5B; and poor value for OLMo 2** — 2.3× Llama's cost to train an
English-only base on Thai.

Recommended sequencing: **do not fund all three arms at 10 B tokens.** Run S0 + the pilot-promoted
arm at full scale (≈600–780 GPU-h) and keep the third arm at pilot scale as an ablation. If the
pilot promotes nothing, `research_plan.md` §8 already says to build ThaiLLM-1B on S0 alone — that
is 278 GPU-h and the engineering deliverable still ships.

Instruct-tuning (`ThaiLLM-1B-Instruct`) is not modelled here. SFT on a few hundred thousand
examples is normally <5 % of CPT cost and is not a scheduling risk.

## 7. What would change these numbers

- **The 120 TFLOPS assumption.** If real MFU is 25 % rather than 38 %, every wall-clock figure
  rises ~50 %. The 200-step preflight settles this and should be the first GPU work done.
- **The 0.55–0.88× routed penalty.** Measured on other hardware and another model. If routing
  costs more than 0.55× here, the GPU-hour-matched comparison — the paper's core claim — gets
  harder to win, and the budget in §6 is optimistic.
- **`E4` from the static audit.** Gradient checkpointing is currently hard-disabled on routed arms
  only. Until that is fixed the arms are in different memory regimes and *no* GPU-hour comparison
  is interpretable. This must be closed before the preflight, not after.
- **Real Thai fertility.** The bytes/token multipliers come from 14 screening documents, not from
  SEA-PILE-v2 itself. Re-measure on ≥100 MB of the actual corpus before committing the token budget.
- **Corpus size.** If the Thai mixture grows beyond one pass over SEA-PILE-v2 `th`, §6 scales
  linearly.
