# แผนสร้าง ThaiLLM (Track 1 — Product)

**วันที่:** 2026-08-18
**ปรับปรุง:** rev.3 — ข้อจำกัดขนาดเปลี่ยนเป็น "ขั้นต่ำ 1B เน้นฉลาด" + ตรวจ config จริงของผู้สมัครทุกตัว
**หลักฐาน:** desk research จากต้นฉบับ + config.json จริง — `scientific_evidence_allowed=false`

## ข้อจำกัดที่ยืนยันแล้ว

| ข้อ | คำตอบ | ผลต่อแผน |
|---|---|---|
| Deliverable | **Base อย่างเดียวก่อน** | SFT/DPO ออกจาก critical path |
| ปล่อย weight | **ใช่ บน HuggingFace** | **license ต้องเป็น Apache-2.0** |
| ขนาด | **ขั้นต่ำ 1B บวกลบได้ เน้นฉลาด** | เปิดช่วง 1.5–2B |
| งบ GPU | **ยังไม่ยืนยัน** | วางแผนแบบ staged (§7) |

---

## 0. ข้อเสนอหลัก

# 🏆 Base model: **Qwen3-1.7B-Base**

| หัวข้อ | ข้อเสนอ |
|---|---|
| **Base** | **Qwen3-1.7B-Base** (Apache-2.0, 1.7B, 28 ชั้น, dense) |
| Tokenizer | ไม่ขยาย vocab ไทย |
| Data mix | **ไทย 50% / อังกฤษ 35% / code 10% / math 5%** |
| วิธีเทรน | Full fine-tuning |
| Context | 8192 (base รองรับ 32768) |
| งบ token | เริ่ม **10B** แล้วประเมิน |

---

## 1. ⛔ ปัญหาใหญ่ที่สุดของ config ปัจจุบัน: สัดส่วนข้อมูล

| งาน | สัดส่วนที่ไม่ใช่ภาษาเป้าหมาย |
|---|---|
| **Typhoon 2** (ไทย, 1B–70B) | **English 50%** |
| **Racka** (ฮังการี, HPC มหาลัย) | English 24% + German 21% + code 11% = **~56%** |
| **EstLLM** (เอสโตเนีย) | English replay + code + math + instruction-like |
| **`experiment_parameters.json` ปัจจุบัน** | **English 5% + code 5% = 10%** |

Typhoon 2 ระบุเหตุผลตรง ๆ ว่าเลือก 50% English เพราะ
*"inspired by prior work on catastrophic forgetting mitigation"*

**ทำไมอันตราย:** โมเดลระดับ 1–2B capacity น้อย ลืมง่ายกว่าโมเดลใหญ่มาก
ไทย 90% เสี่ยงได้โมเดลที่ไทยขยับนิดเดียวแต่อังกฤษกับ reasoning พัง
→ **ขัดกับเป้าหมาย "เน้นฉลาด" โดยตรง และปล่อยขึ้น HuggingFace ไม่ได้**

> 💡 ทำ **ablation 1–2B tokens × 3 สัดส่วน (50 / 70 / 90% ไทย)** ก่อนลงทุนหนัก
> ถูกมาก กันความผิดพลาดที่แพงที่สุด **และเป็น section ที่ดีใน model paper**

---

## 2. Tokenizer: ไม่ขยาย

**Typhoon 2 ตัดสินใจไม่ขยาย vocab ไทย** เหตุผล: การเพิ่ม token
*"can degrade overall performance"* แม้ช่วย generation efficiency
(Racka เลือกตรงข้าม → หลักฐานไม่เอกฉันท์ แต่สำหรับเราควรไม่ขยาย)

1. **งบ compute ไม่พอ** — embedding ใหม่ต้องใช้ token มหาศาลกว่าจะเรียนรู้
2. **เพิ่มความเสี่ยง** — ต้องเลือกและ validate วิธี init (FOCUS/HYPEROFA/LGSE) = หลายสัปดาห์
3. **ทำลาย Track 2** — เปลี่ยน tokenizer = เพิ่มตัวแปรปนในการเทียบ S0/D1/D2

**แต่ยังต้องวัดและรายงาน** Thai chars/token → เข้า model paper (§6)

---

## 3. Base model — ผลตรวจ config.json จริง

### ผู้สมัครที่ผ่านตัวกรอง Apache-2.0 + ≥1B + มี Base checkpoint

| โมเดล | ขนาด | ชั้น | สถาปัตยกรรมจริง (จาก config) | Context | ประเมิน |
|---|---|---|---|---|---|
| **Qwen3-1.7B-Base** | 1.7B | **28** | `Qwen3ForCausalLM` — **dense decoder มาตรฐาน** | 32,768 | ✅ **เลือกตัวนี้** |
| Qwen3.5-2B-Base | 2B | 24 | `Qwen3_5ForConditionalGeneration` — **18 ชั้น linear_attention + 6 ชั้น full_attention** + multimodal + MTP | 262,144 | ⚠️ เสี่ยงสูง |
| Gemma-4-E2B | 2.3B eff. | — | `Gemma4ForConditionalGeneration` — **MatFormer + per-layer embeddings + audio encoder + vision** | 128K | ❌ ซับซ้อนเกิน |
| Sailor2-1B | 988M | 48 | `Qwen2ForCausalLM` — dense | **4,096** | ❌ ต่ำกว่า 1B + context แค่ 4K |
| OLMo-2-1B | 1.0B | — | dense | — | ⚠️ อังกฤษเป็นหลัก |

### ทำไม Qwen3-1.7B-Base

**1. ฉลาดจริงในขนาดนี้** — pretrain ด้วย **36T tokens ครอบคลุม 119 ภาษา**
(Qwen2.5 ครอบคลุมราว 1 ใน 3 ของจำนวนนี้) → ตรงกับโจทย์ "เน้นฉลาด"

**2. เป็น dense decoder มาตรฐาน** — `Qwen3ForCausalLM` เรียบ ๆ ไม่มี linear attention,
ไม่มี MoE, ไม่มี multimodal, ไม่มี MatFormer
→ recipe CPT ทุกอันจากงานที่อ้างถึงใช้ได้ตรง ๆ, tooling สมบูรณ์, ความเสี่ยงต่ำสุดสำหรับทีม 3 คน

**3. 🎯 ตัวชี้ขาด — repo ทางการของ Delta AttnRes ส่ง `modeling_qwen3_attnres.py` มาให้เลย**
Track 2 แทบไม่ต้องเขียนโค้ดใหม่ และ MIT license ใช้ได้ทันที
**นี่คือเหตุผลเดียวที่ทำให้สองสายใช้ base ตัวเดียวกันได้อย่างสมเหตุสมผล**

**4. 28 ชั้น** — ความลึกพอเหมาะสำหรับ block routing ของ Track 2

**5. Context 32K** — เทียบกับ Sailor2 ที่มีแค่ 4K

**6. Apache-2.0 + มี Base checkpoint** — ปล่อยบน HF ได้ไม่มีเงื่อนไข

### ทำไมไม่เอา Qwen3.5-2B-Base (แม้จะใหม่กว่าและน่าจะฉลาดกว่า)

config เผยว่า `layer_types` เป็น **linear_attention 18 ชั้น สลับ full_attention 6 ชั้น**
(อัตรา 3:1, `full_attention_interval: 4`) พร้อม multi-token prediction และ multimodal head

- **Track 2 พังทันที** — AttnRes ออกแบบบนสมมติฐาน dense attention decoder
  การเอาไปใส่โมเดล hybrid linear-attention เป็นงานวิจัยอีกชิ้นหนึ่งเลย
- **CPT ยากขึ้น** — hybrid linear attention มี training dynamics ต่างออกไป และ tooling ยังใหม่
- **multimodal ไม่ได้ใช้** — แบกน้ำหนักและความซับซ้อนฟรี ๆ สำหรับงาน text-only

> ถ้าภายหลังงบ GPU เยอะและมีเวลาเหลือ **Qwen3-4B-Base** เป็น upgrade path ที่ปลอดภัยกว่า
> (dense เหมือนกัน) — ต้นทุน CPT ราว 2.4× ของ 1.7B

---

## 4. Training recipe (Base only)

| พารามิเตอร์ | ค่า | เหตุผล |
|---|---|---|
| วิธี | **Full fine-tuning** | Typhoon 2 ใช้ full FT ที่ ≤8B; LoRA เฉพาะ 70B |
| Context | **8192** | ตาม Typhoon 2 (base รองรับ 32K แต่ 8K ประหยัดกว่ามาก) |
| Precision | BF16 | มาตรฐาน |
| Optimizer | AdamW | — |
| LR | **1e-5 – 3e-5**, warmup + cosine | CPT ต้องใช้ LR ต่ำกว่า pretraining มาก ไม่งั้นลืมหนัก |
| Parallelism | DDP | 1.7B + optimizer state ใส่ A100 80GB ได้ |
| Gradient checkpointing | **เปิด** | Track 1 ใช้ได้ (Track 2 มีข้อจำกัด E4) |
| Token budget | เริ่ม 10B → ประเมิน | §7 |

**Stage 2–3 (SFT/DPO) เลื่อนออกไปก่อน** ตามคำตอบ "Base อย่างเดียว"
แต่ควรออกแบบ pipeline ให้ต่อยอดได้ — ทั้ง Typhoon 2, EstLLM, PureTC-1B ใช้ครบสาม stage
(PureTC-1B ใช้ DPO ลด non-target-language token ได้ **51.3% relative**)

---

## 5. Evaluation (base model → few-shot + BPB)

### Thai
| Benchmark | บทบาท |
|---|---|
| **ThaiExam** (ONET, IC, A-Level, TGAT, TPAT) | ตัวหลัก — Typhoon 2 ใช้ |
| **M3Exam** (th) | ตัวหลัก — Typhoon 2 ใช้ |
| Belebele (th) | reading comprehension |
| **Thai BPB** บน held-out ที่ freeze | ตัวชี้วัดต่อเนื่องระหว่างเทรน |

### Retention (ห้ามข้าม — เป็นหัวใจของ "ฉลาด")
- English: MMLU, HellaSwag, ARC
- Code: HumanEval / MBPP
- **BPB อังกฤษ + code** บนชุด freeze

### ต้องมี
- **วัด baseline ของ Qwen3-1.7B-Base ก่อนเทรนแม้แต่ step เดียว** → รู้ headroom จริง
- freeze benchmark และ threshold ทั้งหมด **ก่อน**เปิดผล

---

## 6. สิ่งที่ตัดไปตีพิมพ์ได้ (model paper)

| # | Section | ได้จาก |
|---|---|---|
| 1 | Base-model + tokenizer screening สำหรับไทย | §3 |
| 2 | **Replay-ratio ablation: ไทยกี่ % จึงเหมาะกับ CPT ที่ ~2B** | §1 |
| 3 | Thai data pipeline + provenance/decontamination | ต้องทำอยู่แล้ว |
| 4 | ผลประเมินเต็ม + retention | §5 |

> ⚠️ ข้อ 1 **ห้ามทำเป็น paper เดี่ยว** — ดู `novelty/tokenizer_screening_novelty_audit.md`

---

## 7. งบ GPU ยังไม่ยืนยัน → แผน staged

### ประมาณการสำหรับ 1.7B (ต้องแทนด้วยผล preflight)

```
FLOPs ≈ 6 × N × D,  N = 1.7e9
A100 BF16 peak ~312 TFLOPS × MFU 0.40 ≈ 125 TFLOPS/GPU ; 4 GPU ≈ 500 TFLOPS
```

| Token budget | 4×A100 (ประมาณ) |
|---|---|
| 10B | **~2.5 วัน** |
| 20B | **~5 วัน** |
| 30B | **~7 วัน** |

**ตัวเลขนี้เป็นทฤษฎีล้วน** MFU จริงมักต่ำกว่า และ dataloader/checkpointing กินเพิ่ม

### Decision tree ตามงบจริง

| งบเช่า GPU | แผน |
|---|---|
| **~$250** | 6B tokens, ไทย 50%, รอบเดียว, ablation ย่อเหลือ 2 สัดส่วน |
| **~$400–700** ⭐ | ablation 3 สัดส่วน @1–2B → CPT 6–10B ด้วยสัดส่วนที่ชนะ |
| **> $700** | ablation เต็ม → พิจารณา Qwen3-4B-Base → เริ่ม Track 2 ขนาน |

> รายละเอียดใน [`compute_and_storage_plan.md`](compute_and_storage_plan.md) — LANTA ถูกแทนที่ด้วยการเช่า GPU

---

## 8. ลำดับงาน

| # | งาน | ต้องใช้ GPU? |
|---|---|---|
| 1 | **วัด baseline Qwen3-1.7B-Base บน eval ที่ freeze** → รู้ headroom ไทย | น้อยมาก (inference) |
| 2 | Data pipeline + license/decontamination audit | **ไม่** |
| 3 | Freeze eval suite + threshold | **ไม่** |
| 4 | Preflight throughput → แทนตัวเลข §7 | นิดเดียว |
| 5 | **Replay-ratio ablation 1–2B × 3 สัดส่วน** | ปานกลาง |
| 6 | CPT รอบหลัก 10B tokens | เยอะ |
| 7 | Eval + ตัดสินใจขยาย budget | น้อย |
| 8 | เขียน model paper | ไม่ |

**ขั้น 1–3 เริ่มได้ทันทีโดยไม่ต้องรอ GPU** และ **Track 2 ใช้ผลขั้น 2–4 ต่อได้ฟรีทั้งหมด**

---

## 9. ความเสี่ยงหลัก

| ความเสี่ยง | ผลกระทบ | การรับมือ |
|---|---|---|
| **ไทย 90% ทำให้ลืมหนัก** | ขัดเป้าหมาย "ฉลาด" | ablation ขั้นที่ 5 |
| Qwen3 เห็นไทยมาพอแล้ว → headroom น้อย | CPT ไม่ขยับ | **วัด baseline ก่อน (ขั้น 1)** |
| งบ GPU ไม่มา | ทำไม่ได้ | ขั้น 1–3 เดินได้โดยไม่ใช้ GPU |
| ข้อมูลไทยไม่ผ่าน audit | ต้องหาแหล่งใหม่ | เริ่ม audit วันแรก |
| Token budget ไม่พอเห็นผล | CPT ไม่ขยับ | checkpoint ประเมินที่ 10B |

---

## 10. แหล่งอ้างอิง

- [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) — Apache-2.0, 36T tokens, 119 ภาษา
- [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) — hybrid linear attention (ไม่เลือก)
- [gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) — MatFormer + multimodal (ไม่เลือก)
- [arXiv:2412.13702 — Typhoon 2](https://arxiv.org/abs/2412.13702) · [HTML](https://arxiv.org/html/2412.13702v1)
- [arXiv:2502.12982 — Sailor2](https://arxiv.org/abs/2502.12982)
- [arXiv:2601.01244 — Racka (Hungarian, academic HPC)](https://arxiv.org/abs/2601.01244)
- [arXiv:2603.02041 — EstLLM](https://arxiv.org/abs/2603.02041)
- [arXiv:2510.01616 — PureTC-1B](https://arxiv.org/abs/2510.01616)
- [arXiv:2510.25947 — Revisiting Multilingual Data Mixtures](https://arxiv.org/abs/2510.25947)
- [arXiv:2507.14664 — Mangosteen (Thai corpus)](https://arxiv.org/abs/2507.14664)
- [delta-attention-residuals-code (MIT)](https://github.com/wdlctc/delta-attention-residuals-code) — มี `modeling_qwen3_attnres.py`
