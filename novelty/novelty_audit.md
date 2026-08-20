# Novelty Audit — ThaiLLM-1B + Attention Residuals

**Audit date:** 2026-08-18
**Auditor role:** external reviewer (senior NLP researcher)
**Target venue:** iSAI-NLP 2026
**Status of this document:** desk evidence only — `scientific_evidence_allowed=false`

---

## 0. Executive summary

| Question | Answer |
|---|---|
| Do the five AttnRes papers in `source_registry.csv` actually exist? | Yes — all five verified against arXiv API metadata. One title is stale (see §2.1). |
| Is the exact intersection (Thai CPT × AttnRes family × token- **and** GPU-hour-matched × retention) occupied? | **No** — not in the searched record. |
| Is the intersection *wide enough* to carry a paper? | **Marginally.** Three of the four planned contributions are partly pre-empted (§3). |
| Is the biggest risk novelty? | **No.** The biggest risks are (a) a **14-day** submission deadline and (b) a **measured implementation defect** that would invalidate the D1/D2 comparison (§5). |
| Verdict | **CONDITIONAL_GO**, with a mandatory scope change. See `decision_memo_th.md`. |

---

## 1. Search methodology

All searches were run on **2026-08-18**. Two independent channels were used so that no
conclusion rests on a single query, per the project's own rule.

### 1.1 Channel A — arXiv API (structured, exhaustive within query)

Endpoint: `http://export.arxiv.org/api/query`, `sortBy=submittedDate&sortOrder=descending`.

| Query ID | `search_query` string | Results returned |
|---|---|---|
| Q1 | `all:"attention residuals"` | 60 (capped) |
| Q2 | `ti:"attention residual"` | 35 |
| Q3 | `abs:"depth routing" OR abs:"depthwise routing" OR abs:"residual routing"` | 19 |
| Q4 | `ti:"DenseFormer" OR ti:"DeepCrossAttention" OR ti:"Hyper-Connections" OR abs:"hyper-connections"` | 60 (capped) |
| T1 | `abs:"Thai" AND (abs:"continued pretraining" OR abs:"continual pre-training" OR abs:"continued pre-training" OR abs:"language adaptation")` | 8 |
| T2 | `ti:"Thai" AND (ti:"language model" OR ti:"LLM")` | 22 |
| T3 | `abs:"compute-matched" OR abs:"GPU-hour" OR abs:"GPU hours" AND abs:"continued pretraining"` | 40 (capped) |
| **T4** | `(all:"attention residuals" OR all:"hyper-connections") AND (all:"multilingual" OR all:"low-resource" OR all:"cross-lingual")` | **1** |

Unique arXiv identifiers surfaced across Q1–Q4: **144**.

### 1.2 Channel B — direct source verification

Every claim-bearing work was opened at its primary source (arXiv abstract page, arXiv HTML
full text, GitHub repository, or conference site) rather than being cited from a search snippet.

### 1.3 The decisive query

**T4 is the load-bearing result of this audit.** Restricting the entire AttnRes +
hyper-connection literature to anything multilingual, cross-lingual, or low-resource returns
**exactly one** document: `arXiv:2608.09119` (*Motif 3 Technical Report*), which is a
314B-parameter frontier MoE system report, not a controlled architecture study, and which does
not isolate Thai or any low-resource language.

This is the strongest available evidence that the language-adaptation axis of the AttnRes
family is genuinely unoccupied. It is **still not proof of non-existence** — it cannot see
non-arXiv venues, industry work, or papers using different vocabulary (e.g. "cross-layer
attention", "layer aggregation"). The search must be repeated immediately before paper freeze.

---

## 2. Corrections to the existing `source_registry.csv`

### 2.1 AR05 title is stale

`AR05` is recorded as *"WAV Multi-Resolution Block Residual Routing"*.
`arXiv:2606.06564` **v1** (2026-06-04) was indeed titled
*"WAV: Multi-Resolution Block Residual Routing for Deep Decoder-Only Transformers"*, but
**v2** (2026-06-17) renamed the method to **HAARES: Half-Split Residual Basis Routing for Deep
Transformers** (author: Kehan Wang). Citing "WAV" in a 2026 submission would look like the
authors did not read past v1. Registry updated.

### 2.2 License status of the Delta AttnRes code was recorded incorrectly

Both `src/README.md` and the archived `routing.py` docstring state that the team wrote its own
implementation because the official repository is **"unlicensed"**.

This is no longer true. `https://github.com/wdlctc/delta-attention-residuals-code` is released
under the **MIT license**, and ships `modeling_qwen3_attnres.py`, DDP training scripts for
220M–1B, FSDP scripts for 7B+, fine-tuning entry points, downstream evaluation, and a
`WANDB_RUNS.md` recording the exact configuration of every paper experiment.

**Consequence:** the stated justification for maintaining a divergent in-house implementation
has disappeared. MIT permits use, modification and redistribution with attribution. Reusing the
official implementation for D1/D2 removes the fidelity risk documented in §5 at essentially
zero legal cost, and the released W&B configurations give a reproduction target.

### 2.3 Works absent from the registry that a reviewer will know

`arXiv:2607.18730` (Dual AttnRes), `arXiv:2608.01075` (Role-Decoupled AttnRes),
`arXiv:2606.13168` (Causal Probes on Block AttnRes), `arXiv:2605.17887` (Attention Sinks in
AttnRes), `arXiv:2605.09850` (Routing-Conditional Calibration), `arXiv:2607.24653` (Kimi K3),
`arXiv:2608.09119` (Motif 3), plus the pre-2026 ancestors DenseFormer, DeepCrossAttention,
Hyper-Connections and mHC. All added to the registry.

---

## 3. Contribution-by-contribution overlap assessment

`plans/research_plan.md` §2 lists four intended contributions. Assessed against the matrix:

| # | Intended contribution | Overlap status | Evidence |
|---|---|---|---|
| 1 | Controlled token-matched **and** GPU-hour-matched evaluation at ~1B | **Partly open.** MHAR already publishes throughput ratios (0.55–0.88× baseline with fused Triton kernels) and LR-AttnRes already frames a validation-vs-FLOPs tradeoff. Nobody publishes a *quality-per-GPU-hour Pareto frontier for a language-adaptation run*. | NW03, NW04 |
| 2 | A non-trivial Thai or compute-normalised finding | **Open.** No AttnRes work touches Thai or any low-resource language (T4 = 1 hit, a model report). | NW19, T4 |
| 3 | Reproducible conversion / profiling / evaluation infrastructure | **Largely closed.** The official Delta repository is MIT-licensed and already ships conversion, training at our scale, evaluation, and per-experiment W&B configs. | §2.2 |
| 4 | Mechanism analysis via interventions, not just heatmaps | **Severely pre-empted.** `arXiv:2606.13168` performs exactly this — causal routing-ablation on Block AttnRes — **on Qwen3-0.6B, our designated pilot model**, and reports the headline result that routing mass dissociates from causal importance. | NW08 |

**Net:** contributions 3 and 4 can no longer be sold as contributions. Contribution 2 is the
only fully open one. Contribution 1 is open only in its *language-adaptation* framing.

The paper therefore has **one and a half** live contributions, not four.

---

## 4. Contribution sentence

Written in the required form, and constrained to what the evidence above actually supports:

> **Prior work shows** that routing over layer-wise deltas (Delta AttnRes, 220M–7.6B) and over
> per-subspace heads (MHAR, 100M–8B) improves validation loss, and that pretrained checkpoints
> can be converted into these variants by ordinary fine-tuning — **under the setting** of
> general-domain, predominantly English, from-scratch or mid-training runs evaluated per
> observed token. **What is not known** is whether any of this survives the setting that
> actually governs Thai LLM work: compute-limited continued pretraining from a fixed
> ~1B checkpoint, where the routed variants train at a measured 0.55–0.88× of baseline
> throughput and must therefore *pay for themselves in wall-clock*, and where gains on the
> target language must be weighed against regression in the source languages the base model
> already has. **We therefore** convert one fixed pretrained decoder into Standard Residual,
> Delta Block AttnRes and Delta Block MHAR under identical data, identical document order and
> identical optimizer state, and compare them on Thai acquisition, English/code retention and
> routing behaviour on **both** a token-matched and a GPU-hour-matched axis. **We report**
> where the general-domain ranking of these variants does and does not transfer to Thai
> compute-limited CPT — including the outcome in which it does not transfer, which is the
> result a Thai lab planning a CPT budget most needs.

The last clause is deliberate: it makes a null result publishable rather than fatal.

### 4.1 Why "first AttnRes for Thai" is rejected

Per the project's own hard rule (`first_in_thai_is_sufficient_novelty: false`), and because
it is independently indefensible: AttnRes is already deployed in production frontier systems
(Kimi K3, NW20) and in a multilingual 314B MoE (Motif 3, NW19). "First to apply X to Thai" is
an availability claim, not a knowledge claim. The knowledge claim must be about the
**interaction** between routed residuals and the compute-limited CPT regime.

---

## 5. Local pre-LANTA validation results

All results below are engineering diagnostics on tiny random models.
**`scientific_evidence_allowed=false`** — none of this is evidence about Thai, about 1B models,
or about the scientific hypothesis. It can only *reject* implementations, never support a claim.

The archived harness was copied to a scratchpad before execution; the archived directory at
`../thai-llm-five-to-two/depth_routing/` was **not modified** (verified by checksum).

### 5.1 Config and schema validation — PASS

`configs/experiment_parameters.json`, `configs/data_manifest.template.json` and
`validation/run_record.template.json` all parse as valid JSON.
`sources/source_registry.csv` parses as well-formed CSV, 13 rows, 8 columns, no duplicate
`source_id`, no ragged rows.

### 5.2 Archived unit tests — 25/25 PASS

`python3 -m unittest discover -s tests -p "test_depth_routing.py"` → **Ran 25 tests, OK**
(0.091 s). `test_hf_integration.py` could not run: `transformers` is not installed in the local
environment (as the existing log entry already records). Environment: Python 3.9.6, torch 2.8.0,
numpy 2.0.2, no pytest, no transformers.

### 5.3 Identity conversion on tiny/random model — PASS

Bitwise-exact: `torch.equal(logits_before, logits_after) == True`, max logit error `0.0`,
at `block_size_layers` 1 and 2. Checkpoint round-trip passes. This meets the
`max_logit_error: 1e-5` threshold in `experiment_parameters.json` with room to spare.

### 5.4 Equation-level comparison against the official Delta formulation — MOSTLY FAITHFUL

Checked the archived `IdentityDeltaRouter` against the equations in `arXiv:2605.18855` HTML:

| Aspect | Official Delta AttnRes | Archived team implementation | Match |
|---|---|---|---|
| Query | learned parameter vector `w_l ∈ R^d`, **zero-initialised**, not derived from the current hidden state | `nn.Linear(hidden, 1, bias=False)` with `nn.init.zeros_`, not derived from current hidden state | ✅ |
| Keys | `K = norm(V)` | `self.query(self.norm(values))` | ✅ |
| Values | per-sublayer deltas `v_i = h_{i+1} − h_i`, or block deltas `Δ_b` | block deltas accumulated every `block_size_layers` | ✅ |
| Output form | **additive**: `ĥ_l = h̃_l + Σ α_i v_i` (residual preserved by default) | `residual + route_scale * mixture` | ⚠️ see below |
| Init behaviour | uniform softmax → **bounded perturbation** | **exact identity** via extra `route_scale = 0` | ❌ **deviation** |

Probe P1 confirmed empirically that routing weights do **not** depend on the current residual
(`max_abs_weight_delta = 0.0` when the residual is changed 5×+ with sources held fixed) — this
is *correct* behaviour and matches the paper's learned-query design.

### 5.5 **BLOCKING FINDING** — the `route_scale` gate suppresses router learning

The archived implementation adds a scalar `route_scale`, initialised to zero, that does not
exist in the paper. Its stated purpose (in the module docstring) is to make checkpoint
migration *mathematically exact* rather than a bounded perturbation.

Because the output is `out = residual + s · mixture(w)`, the gradient with respect to every
routing parameter is **exactly proportional to `s`**:

```
∂L/∂w = (∂L/∂out) · s · (∂mixture/∂w)
```

At `s = 0` this is identically zero. This is analytic, not merely empirical.

Measured (probe P2, tiny LM, cross-entropy loss, at migration init):

| Parameter | Gradient at init |
|---|---|
| `query.weight` (all layers) | **0.0** |
| `null_logit` (all layers) | **0.0** |
| `route_scale` | 0.093 – 0.150 (non-zero) |

Measured (probe P2b, surrogate loss, AdamW lr=1e-3), query gradient as a fraction of the
paper-faithful (`s = 1`, zero-init query) reference:

| Step | `route_scale` | query grad / paper-faithful grad |
|---|---|---|
| 1 | 0.000000 | 0.0 |
| 2 | 0.001000 | 0.006 % |
| 5 | 0.003938 | 0.022 % |
| 10 | 0.008177 | 0.047 % |
| 20 | 0.010959 | 0.064 % |
| 50 | 0.016297 | **0.095 %** |

After 50 steps the router's query is still receiving roughly **one thousandth** of the gradient
signal the paper-faithful initialisation gives it. The paper-faithful variant instead starts at
a relative perturbation of 0.386 with a query gradient of 1.105 from step 1.

Caveat stated honestly: the step counts come from a synthetic fixed-batch surrogate loss and
will not transfer numerically to real CPT. The **mechanism** — a multiplicative zero gate makes
router gradients vanish — is exact and will hold at any scale.

**Why this is blocking.** It is a direct confound for the study's central comparison. If D1 and
D2 underperform S0, we would be unable to distinguish "delta routing does not help Thai CPT"
(the scientific finding) from "our extra gate prevented the router from learning within the
pilot budget" (an implementation artefact). A reviewer who reads the code will raise exactly
this, and the paper is not defensible without it fixed.

### 5.6 Other engineering findings

| ID | Finding | Severity |
|---|---|---|
| E1 | **MHAR (condition D2) does not exist.** The archive contains only `IdentityDeltaRouter` and `HFDepthRouterAdapter`; no multi-head router class (probe P6). D2 must be written from scratch and validated. | High — schedule |
| E2 | **`last_routing` retains the autograd graph.** Stored `weights` tensors have `requires_grad=True` and a live `grad_fn` after the forward returns (probe P3), pinning activation memory for every layer until the next forward. `input_norm` is detached; `weights` is not. At 1B × 16 layers × long sequences this is a real memory regression that will distort the very throughput/VRAM numbers the study is built on. | High — invalidates efficiency measurements |
| E3 | **Block-size semantics are ambiguous.** With `block_size_layers=4` on an 8-layer model, probe P5 shows layers 0–3 receive **0** routing sources and layers 4–7 receive **1**. The first block is never routed and its router parameters are dead weight. The paper's "B=4 default" is ambiguous between block *count* and block *size*; this must be pinned against the official config before any run. | Medium — fidelity |
| E4 | **Gradient checkpointing is hard-disabled** (`HFDepthRouterAdapter` raises if `is_gradient_checkpointing`), because the routing state is not re-entrant. At 1B on a single A100 this materially constrains sequence length and batch size — and the constraint applies only to the routed arms, which biases the GPU-hour-matched comparison against them. | High — biases the primary axis |
| E5 | The archived `evaluate_engineering` gate still enforces the **fixed 10 % overhead kill rule** (`test_engineering_rejects_paper_reported_20pct_overhead` passes). `README.md` and `experiment_parameters.json` explicitly repudiate this rule (`fixed_overhead_kill_threshold: null`). Reused code must not silently re-import it. | Medium — governance |

Note that E2 and E4 both bias measurements **against** the routed conditions, while E5's
inherited rule would kill them outright. Combined with MHAR's published 0.55–0.88× throughput,
the GPU-hour-matched axis is currently stacked so heavily against D1/D2 that a "routing loses
per GPU-hour" result would be uninterpretable.

---

## 6. Contribution type classification

| Type | Fit | Assessment |
|---|---|---|
| Method paper | ✗ | No new method is proposed. D1 and D2 are other people's methods. |
| **Empirical architecture study** | **✓ best fit** | The claim is about *how known architectures behave in an unstudied regime*. This is what the evidence can support, and it tolerates a null result. |
| Systems / efficiency study | ~ partial | The GPU-hour axis is real and under-served, but E2/E4 must be fixed first or the numbers are not trustworthy. Best used as a *secondary* axis inside the empirical study. |
| Dataset / evaluation contribution | ✗ | Nothing new is being released here; SEA-PILE-v2 and FineWeb-Edu are existing corpora. Could become a fallback (a frozen Thai/English/code-switch routing-diagnostic suite), but it is not the current plan. |
| Application paper | ✗ | Would collapse into "first-in-Thai", which the project already forbids. |

**Recommended:** an **empirical architecture study** with a clearly-labelled systems component.
This is the most defensible framing for iSAI-NLP 2026 and the only one the available evidence
can carry.

---

## 7. Feasibility against the venue calendar — the dominant constraint

Verified at `https://isai-nlp2026.aiat.or.th/` on 2026-08-18:

| Milestone | Date |
|---|---|
| Paper submission (extended) | **2026-09-01** |
| Notification (extended) | 2026-10-03 |
| Camera-ready (extended) | 2026-10-10 |
| Conference | 2026-11-19 – 21, Bangkok (in-person only) |

**Today is 2026-08-18. There are 14 days to the deadline.**

Against that, the current plan requires, in sequence: fix §5.5 and E2/E4 → implement MHAR from
scratch → re-run Tier A/B/C validation → base-model screen across five candidates → data
provenance and license audit → one-A100 preflight (200 steps × 3 conditions) → premise pilot
(50–100M tokens × 3 conditions) → second seed for promoted conditions → full 1B CPT × 3
conditions → evaluation → write-up.

With no LANTA run started, no GPU allocation confirmed, D2 unimplemented, and a blocking
correctness defect open, **the planned 1B study cannot be completed by 2026-09-01.** This is not
a novelty problem and no amount of literature work changes it.

---

## 8. What is genuinely left for us

Stated precisely, so it can be defended line by line:

1. **The regime.** Compute-limited continued pretraining from a fixed pretrained checkpoint.
   Every AttnRes paper either trains from scratch (NW01, NW05, NW06, NW07) or does
   general-domain mid-training (NW02, NW03). None studies *language adaptation*.
2. **The outcome pair.** Target-language acquisition measured jointly with source-language
   retention. No work in the matrix measures retention at all — the entire column is empty.
3. **The cost axis, correctly posed.** Not "does the router add FLOPs" (NW04 has that) and not
   "what is the throughput ratio" (NW03 has that), but "**at equal wall-clock on fixed
   hardware, which residual topology buys more target-language quality**". That question only
   becomes interesting in a compute-limited setting, which is why nobody with frontier compute
   has asked it.
4. **Language-conditioned routing behaviour.** NW08 already owns causal routing probes in
   general. It does **not** ask whether routing distributions differ systematically between
   Thai, English, and controlled code-switch text. That specific question is still open — but it
   is now a *sub-result*, not a headline contribution.

Items 1–3 together are enough for a regional-venue empirical paper. They are **not** enough for
a top-tier venue, and they are not enough at all without a working, faithful implementation.

---

## 9. Required re-check before paper freeze

The AttnRes field produced at least seven papers between 2026-05 and 2026-08 — roughly one every
ten days. Two of the most threatening (NW07 on 2026-08-02, NW19 on 2026-08-10) appeared in the
**last three weeks**. Re-run queries Q1–Q4 and T4 within 48 hours of submission and record the
result as a fresh `NOVELTY-DESK` entry. A paper landing on "AttnRes for cross-lingual
adaptation" before 2026-09-01 would remove contribution 2, the only fully open one.

---

## 10. Sources

All verified from primary sources on 2026-08-18.

- [arXiv:2603.15031 — Attention Residuals (Kimi Team)](https://arxiv.org/abs/2603.15031)
- [arXiv:2605.18855 — Delta Attention Residuals](https://arxiv.org/abs/2605.18855) · [HTML full text](https://arxiv.org/html/2605.18855v1) · [code (MIT)](https://github.com/wdlctc/delta-attention-residuals-code)
- [arXiv:2607.27230 — Multi-Head Attention Residuals](https://arxiv.org/abs/2607.27230)
- [arXiv:2607.09694 — Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694)
- [arXiv:2606.06564 — HAARES (v1: WAV)](https://arxiv.org/abs/2606.06564)
- [arXiv:2607.18730 — Dual Attention Residuals](https://arxiv.org/abs/2607.18730)
- [arXiv:2608.01075 — Role-Decoupled Attention Residuals](https://arxiv.org/abs/2608.01075)
- [arXiv:2606.13168 — When Does Routing Become Interpretable?](https://arxiv.org/abs/2606.13168)
- [arXiv:2605.17887 — Attention Sinks and Outliers in Attention Residuals](https://arxiv.org/abs/2605.17887)
- [arXiv:2605.09850 — Probing Routing-Conditional Calibration](https://arxiv.org/abs/2605.09850)
- [arXiv:2402.02622 — DenseFormer](https://arxiv.org/abs/2402.02622)
- [arXiv:2502.06785 — DeepCrossAttention](https://arxiv.org/abs/2502.06785)
- [arXiv:2409.19606 — Hyper-Connections](https://arxiv.org/abs/2409.19606)
- [arXiv:2512.24880 — mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [arXiv:2607.24653 — Kimi K3](https://arxiv.org/abs/2607.24653)
- [arXiv:2608.09119 — Motif 3 Technical Report](https://arxiv.org/abs/2608.09119)
- [arXiv:2412.13702 — Typhoon 2](https://arxiv.org/abs/2412.13702)
- [arXiv:2504.05747 — SEA-LION](https://arxiv.org/abs/2504.05747)
- [arXiv:2510.08620 — JAI-1](https://arxiv.org/abs/2510.08620)
- [arXiv:2507.14664 — Mangosteen](https://arxiv.org/abs/2507.14664)
- [iSAI-NLP 2026 official site](https://isai-nlp2026.aiat.or.th/)
