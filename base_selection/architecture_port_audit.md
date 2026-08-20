# Architecture Port Audit — Delta Block AttnRes (D1) and MHAR (D2)

**Audit ID:** `BASE-SCREEN-2026-08-18-PORT-AUDIT`
**Date:** 2026-08-18
**Evidence class:** static source audit. No weights loaded, no training run.
**`scientific_evidence_allowed`:** `false`

## 0. What was actually inspected

| Artifact | Source | Pin |
|---|---|---|
| `modeling_llama.py` | huggingface/transformers | tag `v4.57.1` |
| `modeling_qwen3.py` | huggingface/transformers | tag `v4.57.1` |
| `modeling_gemma3.py` | huggingface/transformers | tag `v4.57.1` |
| `modular_olmo2.py` | huggingface/transformers | tag `v4.57.1` |
| `config.json` (all candidates) | Hugging Face | pinned commit SHAs in `candidate_matrix.csv` |

Qwen2.5 (`Qwen2DecoderLayer`) was **not** re-read from source in this pass; it is treated as
structurally equal to Llama/Qwen3 pre-norm on the strength of its config plus the well-known
`Qwen2` layer definition. That is an **assumption, not a verified reading** — see §7.

The reference implementation `CODE01`
(`github.com/wdlctc/delta-attention-residuals-code`, MIT) was **not** cloned or read in this
pass. Its contents are taken from the description already recorded in `source_registry.csv`.
Everything in §5 that depends on it is therefore *expected*, not *verified*.

## 1. The only thing that matters: where the residual is written

Delta Block AttnRes routes over **block deltas** — the quantity added to the residual stream at
each write point. So the port question reduces to: *is the delta a clean, isolable tensor?*

Verbatim structure from the pinned sources:

**Llama 3.2 / Qwen3 — canonical pre-norm (2 clean write points per layer)**
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(hidden_states, ...)
hidden_states = residual + hidden_states          # delta_attn = self_attn(ln(x))   <-- CLEAN
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states          # delta_mlp  = mlp(ln(x))         <-- CLEAN
```

**Gemma 3 — sandwich norm (2 write points, but delta includes a post-norm)**
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
position_embeddings = position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global
hidden_states, _ = self.self_attn(hidden_states, position_embeddings=position_embeddings, ...)
hidden_states = self.post_attention_layernorm(hidden_states)   # <-- norm INSIDE the delta
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden_states)))
hidden_states = residual + hidden_states
```

**OLMo 2 — reordered post-norm (`input_layernorm` is deleted)**
```python
residual = hidden_states
hidden_states, _ = self.self_attn(hidden_states, ...)          # attention reads the RAW residual
hidden_states = self.post_attention_layernorm(hidden_states)   # norm INSIDE the delta
hidden_states = residual + hidden_states
```

**Consequence.** All four families expose a well-defined delta, so D1 is *definable* everywhere.
But only Llama/Qwen3 give a delta that is the bare sublayer output. In Gemma 3 and OLMo 2 the
delta is already normalised, which means the router's key `normalize(delta)` is operating on an
almost-unit-norm vector — the RMSNorm has largely destroyed the magnitude signal the router would
otherwise use to distinguish blocks. This is not a blocker, but it is a **silent behavioural
change** to the method being studied, and it would have to be reported as a deviation.

## 2. Per-candidate port assessment

### 2.1 `meta-llama/Llama-3.2-1B` — **lowest port risk of any ≈1B candidate**

| Dimension | Finding |
|---|---|
| Edit points | `LlamaDecoderLayer.forward` only; `LlamaModel.forward` to thread the delta buffer |
| Exact identity conversion | **Yes.** Router output enters additively; with `null_logit → +∞` (or paper-faithful bounded init) the block reduces to `residual + delta` bit-exactly |
| Block partitioning | 16 layers → block sizes 2, 4, 8 all divide evenly |
| MHAR subspaces | `d=2048`; /4 = 512, /8 = 256 — both clean |
| KV cache | Unaffected. Routing acts on the residual stream, never on K/V. Cache layout untouched |
| Gradient checkpointing | `LlamaDecoderLayer` already subclasses `GradientCheckpointingLayer`; delta buffer must be passed as an explicit arg, not closed over, or recompute will diverge |
| FSDP/DDP | Wrap policy is per-`LlamaDecoderLayer`; the router is tiny and can live inside the layer. Cross-block delta buffer must be a plain tensor list, not a module attribute, to avoid FSDP flattening it |
| `torch.compile` | Softmax over a variable-length source list causes recompiles. Fix by padding the source axis to `L/block_size` and masking — then the graph is static |
| Positional/attention quirks | None. Single RoPE, no sliding window, no QK-norm, no softcapping |
| Save/load | Router params are new keys; `strict=False` load from base, then a self-check that non-router keys match the base exactly |
| **Estimated effort** | **3–5 engineer-days** for D1 + D2 + identity tests |
| **Implementation risk** | **Low** |

### 2.2 `Qwen/Qwen3-0.6B-Base` — **lowest absolute risk (pilot)**

Structurally identical to Llama plus QK-norm, which sits *inside* attention and does not touch the
residual write points. 28 layers → block sizes 2, 4, 7, 14. `d=1024` divides by 4 and 8.

The decisive advantage: `CODE01` reportedly ships `modeling_qwen3_attnres.py`, i.e. a reference
port against this exact architecture. **Estimated effort 1–2 engineer-days**, risk **very low** —
*conditional on that file being what the registry says it is, which this audit did not verify.*

### 2.3 `Qwen/Qwen2.5-1.5B` — low risk

Pre-norm like Llama. Only quirk is `bias=True` on Q/K/V projections, which is irrelevant to
residual routing. 28 layers → 2, 4, 7, 14. `d=1536` → /4 = 384, /8 = 192. **4–6 engineer-days**,
risk **low**. (Subject to the §7 caveat that the layer source was not re-read.)

### 2.4 `google/gemma-3-1b-pt` — **high port risk, and the risk lands on the science**

Five compounding problems:

1. **Heterogeneous attention within a routing block.** `sliding_window=512`,
   `sliding_window_pattern=6` — five local layers to one global. A Delta Block router mixing deltas
   from local-512 and global-32768 layers is mixing tensors produced under *different receptive
   fields*. Any routing pattern found could be an artefact of the local/global alternation rather
   than of depth. This is a genuine confound for the paper's mechanism story, not just an
   engineering cost.
2. **Block/pattern misalignment.** 26 layers. Divisors are 2, 13, 26 — **block size 4 is
   impossible**, so the block size cannot be held constant across Llama and Gemma arms. Worse,
   aligning blocks to the 6-layer sliding pattern is impossible too, since 6 ∤ 26 (26 = 4×6 + 2).
   Whatever block size is chosen, some blocks straddle the local/global boundary unevenly.
3. **Dual RoPE plumbing.** The layer takes `position_embeddings_global` *and*
   `position_embeddings_local`; any wrapper must thread both.
4. **Delta is post-normed** (see §1), altering the router key distribution.
5. **MQA with 1 KV head and 4 query heads.** Not a correctness problem for routing, but it makes
   Gemma's attention cost profile so different from the others that GPU-hour matching across
   families becomes hard to interpret.

Identity conversion is still exact and MHAR subspaces are fine (`d=1152`; /4 = 288, /8 = 144).
**Estimated effort 10–15 engineer-days**, risk **high**.

### 2.5 `allenai/OLMo-2-0425-1B` — medium risk, high scientific value, Thai-fatal

The reordered post-norm topology is *the* reason to keep OLMo 2 in the study: it is a genuinely
different residual structure, so "does delta routing help?" replicating across pre-norm **and**
post-norm is a much stronger claim than replicating across two pre-norm models. Deltas are clean
and isolable; 16 layers partition well; `d=2048` divides fine.

Costs: no `input_layernorm` means a Llama-shaped adapter cannot be reused as-is; MHA (16/16) not
GQA; context is only 4096. **5–8 engineer-days**, risk **medium**.

This does not rescue it as a Thai base — see the tokenizer screen.

## 3. Exact / identity-preserving conversion

Requirement from `experiment_parameters.json`: `max_logit_error` 1e-5, `max_loss_error` 1e-6.

For **Delta Block AttnRes (D1)** the conversion is exact on every candidate, because the routed
output is `residual + mixture(deltas)` and the mixture can be made to select the current block's
delta with weight 1. Two admissible initialisations:

- **Paper-faithful (AR02):** learned zero-init query, keys = `normalize(values)`, `null_logit`
  initialised so the softmax puts ~all mass on the null/identity path. Gives a *bounded
  perturbation*, not bit-exact identity, and the router gradient is non-zero at step 1.
- **Team strict-identity (archived harness):** an extra multiplicative `route_scale` initialised
  at 0. Gives bit-exact identity (`max_logit_abs_error = 0.0`, recorded in
  `VAL-2026-08-18-LOCAL-STATIC-AUDIT`) **but zeroes the router gradient at step 1**.

**The `ROUTE_SCALE_GATE` finding already in the validation log is a port-selection input, not just
a bug.** `out = residual + s·mixture(w)` ⇒ `∂L/∂w = (∂L/∂out)·s·(∂mixture/∂w)`, which is
identically zero at `s = 0`. Choosing bit-exact identity buys a clean conversion test and pays for
it with a router that cannot learn early. **Recommendation: adopt the paper-faithful bounded
init for all training arms, and keep `route_scale` only as a unit-test fixture** for proving the
conversion path. This must be settled before any GPU hours are spent, on every candidate equally —
it is family-independent.

For **Original/cumulative Block AttnRes**, exact conversion is *not* available (per `research_plan.md`
§6), which is why it stays out of the CPT arms.

## 4. MHAR (D2) specifics

MHAR splits the residual width into `H_r` routed subspaces. All five shortlisted candidates satisfy
`d mod 4 == 0` and `d mod 8 == 0`, so `H_r ∈ {4, 8}` is available everywhere. No candidate is
selected or rejected on MHAR grounds.

**Parameter cost is negligible and this is good news for the fair-comparison rules.** Per routing
point the router is a query matrix `[H_r, d]` plus `H_r` null logits. For Llama 3.2 1B with
`H_r = 8` at all 16 layers: 16 × 8 × 2048 ≈ 2.6 × 10⁵ params ≈ **0.02 % of 1.236 B**. The
`research_plan.md` §7 requirement for a *parameter-matched control* is therefore satisfied trivially —
router parameters are not material, and a reviewer asking "is the gain just extra parameters?"
can be answered arithmetically rather than with an extra training run.

**D2 does not currently exist.** `VAL-2026-08-18-LOCAL-STATIC-AUDIT` finding `E1` records that the
archived harness contains only `IdentityDeltaRouter` + `HFDepthRouterAdapter`. MHAR must be written
from scratch (or taken from `CODE01`) and validated before it can be a condition. Budget this
separately from the family port: it is roughly **+3–4 engineer-days on top of** the per-family
numbers in §2, and it is the same cost whichever base is chosen.

## 5. Systems compatibility

| Concern | Assessment |
|---|---|
| **BF16** | Safe for all. One caution: the router softmax should be computed in fp32 and cast back, or routing weights over many sources will quantise visibly in bf16 |
| **KV cache** | Unaffected on all candidates. Routing never touches K/V. Gemma 3's hybrid cache is untouched too — but the wrapper must not break its `cache_implementation="hybrid"` path |
| **DDP** | Fine. Router params are tiny; no bucket tuning needed |
| **FSDP** | Wrap per decoder layer. The cross-block delta buffer must stay a plain list of tensors outside module state, otherwise FSDP flattening will corrupt it |
| **DeepSpeed ZeRO-2/3** | Fine. ZeRO-2 is the recommended regime (see `compute_estimate.md`) |
| **`torch.compile`** | Needs a static source axis (pad + mask). Without it, expect a recompile per block index. Gemma 3's dual-RoPE + hybrid mask is the hardest to keep in one graph |
| **Gradient checkpointing** | **This is the highest-risk item in the whole study, and it is not a model-family problem.** `VAL-2026-08-18-LOCAL-STATIC-AUDIT` finding `E4` records that the archived harness *hard-disables* gradient checkpointing on routed arms only. That puts S0 and D1/D2 in different memory regimes, which biases the GPU-hour axis against routing — the exact axis the paper claims to measure. Finding `E2` (routing weights retaining the autograd graph) biases the same direction. Both must be fixed before the preflight |
| **Save/load** | New router keys; load base with `strict=False`, then assert non-router state-dict equality against the base checkpoint. Round-trip already proven on the toy fixture |

## 6. Port-gate verdict

| Candidate | Port gate | Effort (D1+D2, incl. ~3–4 d MHAR) | Risk |
|---|---|---|---|
| `Qwen/Qwen3-0.6B-Base` | **PASS** | 4–6 d | very low |
| `meta-llama/Llama-3.2-1B` | **PASS** | 6–9 d | low |
| `Qwen/Qwen2.5-1.5B` | **PASS** (assumption, §7) | 7–10 d | low |
| `allenai/OLMo-2-0425-1B` | **PASS** | 8–12 d | medium |
| `google/gemma-3-1b-pt` | **PASS WITH RESERVATION** | 13–19 d | high — and the reservation is scientific, not only engineering |

Gemma 3 is not rejected on port grounds. It is *flagged*: its local/global alternation means a
routing result on Gemma would be materially harder to attribute to depth routing than the same
result on Llama 3.2, and its 26 layers prevent holding block size constant across arms.

## 7. What this audit did **not** establish

- **`Qwen2DecoderLayer` was not re-read from source.** Its "low risk" rating rests on the config
  plus general knowledge of the Qwen2 layer, not on a pinned reading like the other four.
- **`CODE01` was not cloned or inspected.** The 1–2 day Qwen3 estimate and the claim that a
  reference `modeling_qwen3_attnres.py` exists both come from `source_registry.csv`, not from
  reading the repository. If that file does not exist or does not match AR02's equations, the Qwen3
  pilot estimate rises to roughly the Llama figure.
- **No conversion was executed on any real pretrained checkpoint.** Every "exact identity" claim
  here is an argument from the algebra plus the tiny-random-model fixture already logged. The
  `max_logit_error ≤ 1e-5` gate on a real 1B checkpoint remains **unproven**.
- **Engineer-day estimates are judgement, not measurement**, and assume one engineer already
  familiar with the archived harness.
- **No throughput was measured.** The routed-arm slowdown used in `compute_estimate.md` is taken
  from AR03's published 0.55–0.88× range, on different hardware and a different model.
