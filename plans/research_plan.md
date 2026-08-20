# Research Plan

**Created:** 2026-08-18  
**Target:** iSAI-NLP 2026  
**Mode:** Model-first ThaiLLM-1B plus controlled architecture study

> **⚠️ เอกสารนี้เป็นแผนตั้งต้น (2026-08-18) หลายส่วนถูกแทนที่แล้ว**
>
> | หัวข้อ | ในเอกสารนี้ | สถานะจริง (2026-08-19) |
> |---|---|---|
> | Base model | Llama 3.2 1B เป็นตัวเต็ง | **Qwen3-1.7B-Base** (Llama gated + บังคับชื่อ derivative) |
> | สัดส่วนข้อมูล | ไทย 90% | **ไทย 50 / อังกฤษ 35 / code 10 / math 5** |
> | Compute | รอ LANTA | **เช่า GPU** (LANTA อนุมัติช้า) |
> | Deliverable | Base + Instruct | **Base อย่างเดียวก่อน** |
> | ลำดับความสำคัญ | research-first | **engineering-first** ตามมติ mentor |
>
> ดูสถานะปัจจุบันที่ [`thaillm_1b_build_plan.md`](thaillm_1b_build_plan.md),
> [`compute_and_storage_plan.md`](compute_and_storage_plan.md) และ
> [`../configs/experiment_parameters.json`](../configs/experiment_parameters.json)
> เก็บไฟล์นี้ไว้เพื่อบันทึกเหตุผลตั้งต้นและ novelty gate ที่ยังบังคับใช้อยู่

## 1. Research claim

The eligible claim is a compute-controlled empirical finding:

> Under matched Thai token exposure and matched GPU-hours, compare how Standard
> Residual, Delta Block Attention Residual and Delta Block MHAR affect Thai
> quality, English/code retention and systems efficiency at approximately 1B
> parameters.

This project does not claim to invent Attention Residuals.

## 2. What would make the study publishable

At least two contributions are required:

1. Controlled token-matched and GPU-hour-matched evaluation at approximately 1B.
2. A non-trivial Thai or compute-normalized finding.
3. Reproducible conversion, profiling and evaluation infrastructure.
4. Mechanism analysis supported by interventions, not only routing heatmaps.

Examples of eligible findings:

- a variant wins per observed token but loses per GPU-hour;
- the best variant for Thai CPT differs from the general-language papers;
- a routed condition improves the Thai acquisition–retention Pareto;
- routing changes systematically across Thai, English and controlled code-switch
  text and causal clamping affects predictions in the expected direction.

## 3. Novelty gate

Existing work already includes Original/Block AttnRes, Delta AttnRes, MHAR,
Low-Rank AttnRes and WAV. Earlier depth aggregation such as DenseFormer,
DeepCrossAttention and hyper-connections must also be in the final matrix.

Preliminary public search through 2026-08-18 found no exact Thai study covering
Thai CPT, multiple AttnRes variants, GPU-hour matching, retention and systems
efficiency. This is a search result, not proof of absolute non-existence.

Reject or pivot when:

- the contribution is only first-in-Thai;
- an existing work covers the same observation, intervention and evaluation;
- the result only repeats the general Delta ranking;
- gains vanish after compute, data, parameters or data order are controlled;
- variance cannot be estimated with replicated runs.

## 4. Base-model strategy

Do not lock the final base until all candidates are evaluated on one immutable
Thai/English/code screening set.

| Candidate | Role | Main reason | Risk |
|---|---|---|---|
| Llama 3.2 1B | Final frontrunner | 1.23B and Thai officially supported | Custom license and new port |
| Gemma 3 1B PT | Exact-size fallback | 1.0B and multilingual | Gated/custom terms and new port |
| Qwen2.5 1.5B | Apache fallback | Thai supported and mature stack | 1.54B; mentor must accept size |
| OLMo 2 1B | Replication family | Fully open research stack | Primarily English |
| Qwen3 0.6B Base | Cheap pilot | Existing scaffold and close to Delta setup | Not final 1B deliverable |

Selection measurements:

- immutable revision and license acceptance;
- exact parameter count;
- Thai bytes/token, characters/token and tokens/document;
- frozen Thai, English and code BPB;
- throughput, peak VRAM and latency;
- port effort and exact conversion;
- DDP/FSDP, BF16, compile and checkpointing compatibility.

Expected choice before results: Llama 3.2 1B if license and port pass; otherwise
Gemma 3 1B. This expectation is not a locked decision.

## 5. Data plan

Primary Thai source:

- SEA-PILE-v2 subset `th`, reported as 6.5B Thai tokens using the Gemma 3
  tokenizer; ODC-By 1.0 plus CommonCrawl terms.

Candidate supplements:

- ThaiLLM data repository components after per-source provenance/license audit;
- pinned Thai Wikimedia data with attribution/share-alike compliance;
- licensed Thai government, legal, educational, news or book sources.

Initial controlled mixture to test:

```text
90% Thai CPT text
 5% English replay
 5% permissively licensed code replay
```

The ratio is preregistration input, not a known optimum. Every architecture
condition must receive the same documents, order and observed token counts.

Before training, freeze disjoint Thai web, Thai encyclopedic, English and code
validation sets plus downstream Thai/English controls and a controlled
Thai–English code-switch diagnostic set.

## 6. Core experimental conditions

| ID | Condition | Question |
|---|---|---|
| `S0` | Standard Residual | Optimized no-routing baseline |
| `D1` | Delta Block AttnRes, one routing head | Does additive block-delta routing help? |
| `D2` | Delta Block AttnRes + MHAR | Does subspace-specific depth routing add value? |

Original cumulative Block AttnRes is optional in pretrained CPT because its
conversion can create an initialization confound. Include it only after a valid
conversion or as a separate from-scratch ablation.

## 7. Fair-comparison requirements

- identical base checkpoint and tokenizer;
- identical data and data order;
- identical optimizer, schedule, sequence length and global batch;
- two seeds for promoted conditions;
- both token-matched and GPU-hour-matched curves;
- parameter and FLOP accounting;
- parameter-matched control if router parameters are material;
- optimized native Standard path, not a routed wrapper with routing disabled;
- frozen metrics and thresholds before pilot results are opened.

## 8. Promotion and stop rules

Promote a routed condition only when:

- Thai BPB improves by about 2% relative, or a frozen downstream aggregate
  improves by about 1 point;
- the gain remains in GPU-hour-matched analysis;
- English/code regression is at most about 1 aggregate point, or the Pareto
  frontier is clearly superior;
- direction repeats across two seeds;
- routing is neither collapsed nor effectively uniform;
- no contamination, silent fallback or baseline weakness is found.

If no routed condition passes, continue the ThaiLLM-1B engineering track using
Standard Residual and activate the backup research topic.

