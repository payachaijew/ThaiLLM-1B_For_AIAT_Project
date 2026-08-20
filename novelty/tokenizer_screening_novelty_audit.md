# Novelty Audit — Thai Tokenizer / Base-Model Screening Study

**Audit date:** 2026-08-18
**Trigger:** proposed fallback paper topic — "which tokenizer/base model is most compute-efficient for Thai"
**Status:** desk evidence only — `scientific_evidence_allowed=false`
**Result:** ⛔ **NOT RECOMMENDED as a standalone paper.** The genre is saturated and the
nearest work covers Thai, ~1.5B controlled training, and GPU-hours simultaneously.

---

## 1. Search methodology

arXiv API, `sortBy=submittedDate&sortOrder=descending`, run 2026-08-18, followed by
primary-source verification of every threatening hit.

| ID | `search_query` | Hits |
|---|---|---|
| K1 | `abs:"tokenizer" AND (abs:"fertility" OR abs:"token premium" OR abs:"bytes per token" OR abs:"tokens per word")` | 40 (capped) |
| K2 | `ti:"tokenizer" AND (ti:"choice" OR ti:"evaluation" OR ti:"comparison" OR ti:"impact" OR ti:"matters")` | 40 (capped) |
| K3 | `abs:"tokenizer" AND (abs:"compute cost" OR abs:"cost" OR abs:"inequity" OR abs:"unfairness" OR abs:"efficiency") AND abs:"multilingual"` | 40 (capped) |
| K4 | `(abs:"Thai" OR ti:"Thai") AND (abs:"tokenizer" OR abs:"tokenization" OR abs:"subword")` | 20 |
| K5 | `abs:"tokenizer" AND (abs:"continued pretraining" OR abs:"vocabulary expansion" OR abs:"language adaptation" OR abs:"cross-lingual transfer")` | 40 (capped) |
| K6 | `(abs:"base model selection" OR abs:"model selection" OR abs:"screening") AND abs:"continued pretraining"` | 5 |

**Contrast with the AttnRes audit:** the decisive AttnRes query (T4) returned **1** hit.
Here, three separate queries hit the 40-result cap. That difference is the entire finding.

---

## 2. The genre is saturated

There is an established and crowded 2025–2026 paper genre — informally "the **X Tax**"
papers — that measures exactly "how much more does language X cost to tokenize than English".

| arXiv | Date | Title | Languages | Thai? |
|---|---|---|---|---|
| 2608.12278 | **2026-08-12** | Structural Silence: When AI Infrastructure Fails Speakers of Underrepresented Languages | Bengali | ✗ |
| 2608.09046 | **2026-08-10** | Measuring the Tokenization Premium: A Cost Audit for Underserved Language Communities | bn, hi, ar, ta, yo | ✗ |
| 2607.24276 | 2026-07-27 | The Tokenizer Tax: Quantifying and Explaining the Cross-Lingual Cost of Subword Tokenization | 14 Indian languages | ✗ |
| 2606.24460 | 2026-06-23 | The African Language Tax: Quantifying the Cost, Latency, and Context Penalty | African | ✗ |
| **2606.15044** | **2026-06-13** | **Equity with Efficiency: An Empirical Study of Tokenizers for Multilingual LLMs** | **11 SEA languages** | ✅ **YES** |
| 2605.24718 | 2026-05-23 | The Tokenizer Tax Across 25 European Languages | 25 European | ✗ |
| 2605.01188 | 2026-05-02 | Compute Optimal Tokenization | — | ? |
| 2602.11174 | 2026-01-19 | The Script Tax: Measuring Tokenization-Driven Efficiency and Latency Disparities | multi-script | ? |
| 2601.13328 | 2026-01-19 | Reducing Tokenization Premiums for Low-Resource Languages | low-resource | ? |
| 2512.20757 | 2025-12-23 | TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior | — | ? |
| 2510.21909 | 2025-10-24 | Explaining and Mitigating Crosslingual Tokenizer Inequities | multi | ? |
| 2509.05486 | 2025-09-05 | The Token Tax: Systematic Bias in Multilingual Tokenization | multi | ? |

**Two of these were posted within the last 10 days.** The genre is actively producing.

---

## 3. The blocking nearest work

### `arXiv:2606.15044` — *Equity with Efficiency* (2026-06-13)

Verified from the HTML full text, not from a search snippet:

> "we compare them across eleven Southeast Asian (SEA) languages: English, Burmese,
> Chinese, Indonesian, Khmer, Lao, Malay, Tagalog, Tamil, **Thai**, and Vietnamese."

| Axis | What the paper already does |
|---|---|
| Thai | ✅ explicitly one of the 11 languages |
| Controlled LM training | ✅ **1.5B-parameter decoder-only, based on OLMo-2-1B** (one of our own base candidates) |
| Compute cost reported | ✅ tokenizer training 3 h CPU (BPE) → 50 h (MYTE) → 33 h on 4×H100 (BLT); LM training **68–300 hours on 8×H200 equivalents** |
| Data | 100M sentences / 203 GB FineWeb2 |
| Tokenizers compared | byte-level BPE, Parity-aware BPE, MYTE, BLT |
| Framing | "first systematic comparison of equitable tokenizers on a unified benchmark" |

This single paper simultaneously occupies **Thai + ~1.5B controlled training + GPU-hour
accounting + tokenizer comparison** — i.e. the four axes that made the proposed study look
attractive.

### `arXiv:2406.14670` — *Exploring Design Choices for Building Language-Specific LLMs* (2024)

Explicitly studies "base model selection, vocabulary extension, and continued pretraining"
for low-resource languages, and finds the optimal adaptation method is highly
language-dependent. This occupies the **base-model-selection-for-CPT** framing directly.

---

## 4. What is actually left

Narrow, and thinner than the AttnRes gap:

1. **Cost is defined as inference cost almost everywhere, not training cost.** Confirmed
   explicitly: `arXiv:2607.24276` "does not measure latency, training compute, or GPU-hours";
   `arXiv:2608.09046` defines cost as "API cost, latency, and usable context length".
   Nobody converts tokenizer fertility into *GPU-hours of a continued-pretraining run*.
2. **`arXiv:2606.15044` compares tokenizer *algorithms* trained from scratch** (BPE / parity
   BPE / MYTE / BLT) — **not the tokenizers of existing off-the-shelf pretrained checkpoints**
   (Llama 3.2 1B, Gemma 3 1B, Qwen2.5 1.5B, Sailor2 1B, Falcon3, SmolLM2…). The practical
   question "given I must pick an existing 1B checkpoint to continue-pretrain on Thai, which
   one and why" is not what it answers.
3. **Thai-specific depth.** The SEA study treats Thai as one of eleven; nobody has done a
   Thai-focused screen with per-source Thai text types.

**Assessment:** items 1–3 are real but thin. They amount to a *well-executed engineering
report*, not a finding. A reviewer who knows `arXiv:2606.15044` will ask what we add beyond
it, and "we used off-the-shelf checkpoints instead of trained tokenizers" is a weak answer
for a standalone paper.

---

## 5. Verdict and correct home for this work

⛔ **Do not pursue as a standalone paper.**

✅ **Correct home: a section inside the ThaiLLM-1B model paper**, justifying the base-model
choice. Model/resource papers are *expected* to contain exactly this analysis, it is not
required to be novel there, and the work has to be done anyway to pick a base. The measured
numbers already collected (e.g. SmolLM2 at 0.583 Thai chars/token with only 2 vocab pieces
containing Thai; Falcon3 at 0.737) are good material for that section.

**Comparative ranking of the three candidate papers after this audit:**

| Rank | Paper | Novelty headroom | Evidence |
|---|---|---|---|
| 1 | **AttnRes controlled study** | **Widest** | decisive query returned **1** hit |
| 2 | **ThaiLLM-1B model paper** | N/A — judged on artifact quality, not novelty | resource-paper norms |
| 3 | Tokenizer / screening study | **Narrowest** | 3 queries capped at 40; nearest work covers Thai + 1.5B + GPU-hours |

This **reverses** the provisional ranking suggested before this audit was run, which had
placed the tokenizer study first on the assumption that it was uncontested. It is contested.

---

## 6. Sources

- [arXiv:2606.15044 — Equity with Efficiency](https://arxiv.org/abs/2606.15044) · [HTML](https://arxiv.org/html/2606.15044v1)
- [arXiv:2607.24276 — The Tokenizer Tax](https://arxiv.org/abs/2607.24276)
- [arXiv:2608.09046 — Measuring the Tokenization Premium](https://arxiv.org/abs/2608.09046)
- [arXiv:2608.12278 — Structural Silence](https://arxiv.org/abs/2608.12278)
- [arXiv:2606.24460 — The African Language Tax](https://arxiv.org/abs/2606.24460)
- [arXiv:2605.24718 — The Tokenizer Tax Across 25 European Languages](https://arxiv.org/abs/2605.24718)
- [arXiv:2406.14670 — Exploring Design Choices for Building Language-Specific LLMs](https://arxiv.org/abs/2406.14670)
- [arXiv:2605.01188 — Compute Optimal Tokenization](https://arxiv.org/abs/2605.01188)
- [arXiv:2512.20757 — TokSuite](https://arxiv.org/abs/2512.20757)
