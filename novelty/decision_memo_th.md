# บันทึกคำตัดสิน (Decision Memo) — ThaiLLM-1B + Attention Residuals

**วันที่:** 2026-08-18
**ปรับปรุง:** 2026-08-18 (rev.2 — สะท้อน framing engineering-first ของ mentor + ผล tokenizer audit)
**ผู้ตรวจ:** Senior NLP Researcher / Reviewer (ตรวจในฐานะบุคคลภายนอก)
**สถานะหลักฐาน:** desk audit + local engineering probe เท่านั้น — `scientific_evidence_allowed=false`

> **rev.2 เปลี่ยนอะไร:** mentor ระบุว่างานนี้เป็น **engineering-first** (สร้าง ThaiLLM-1B ก่อน
> แล้วค่อยตัดบางส่วนไปตีพิมพ์) → เส้นตาย iSAI-NLP 2026 (1 ก.ย.) **ไม่ใช่เส้นตายบังคับอีกต่อไป**
> และผล novelty audit ของหัวข้อ tokenizer (ดู `tokenizer_screening_novelty_audit.md`)
> **ทำให้อันดับตัวเลือก paper พลิกกลับ** — รายละเอียดใน §6

---

## 1. คำตัดสิน — แยกเป็นสองสาย

| สาย | เนื้องาน | คำตัดสิน | ความมั่นใจ |
|---|---|---|---|
| **Track 1 — Product** | ThaiLLM-1B Base/Instruct ด้วย **Standard Residual** | **`GO`** | สูง |
| **Track 2 — Research** | controlled study S0/D1/D2 (AttnRes) | **`CONDITIONAL_GO`** | ปานกลาง |

**สรุปประโยคเดียว:** หัวข้อไม่ตาย และไม่เคยตาย — สิ่งที่ตายคือ *แผนที่บังคับให้ทำ full 1B CPT
สามรอบให้เสร็จก่อน 1 ก.ย.* เมื่อ mentor ยืนยันว่างานคือ engineering-first แรงกดดันนั้นหายไป
เหลือแค่สองเรื่องที่ต้องจัดการ คือ **บั๊กใน router** และ **นาฬิกาที่วรรณกรรมเป็นคนตั้ง ไม่ใช่เราตั้ง**

---

## 2. หลักฐานที่มีแล้ว

### 2.1 ด้าน novelty — ผ่าน (แต่แคบ)

- ยืนยันแล้วว่า paper AttnRes ทั้ง 5 ฉบับใน `source_registry.csv` **มีอยู่จริง** (ตรวจผ่าน arXiv API)
- คำค้นชี้ขาดคือ **T4**: intersect วรรณกรรม AttnRes + hyper-connections ทั้งหมด
  เข้ากับ multilingual / low-resource / cross-lingual → ได้ **1 รายการ**
  คือ Motif 3 (`arXiv:2608.09119`) ซึ่งเป็น technical report ของ MoE 314B ไม่ใช่ controlled study
  → **แกน language adaptation ของตระกูล AttnRes ยังว่างอยู่จริง**
- **ไม่มีงานใดในตาราง nearest work วัด retention เลย — คอลัมน์นี้ว่างทั้งคอลัมน์**
  นี่คือไพ่ใบที่แข็งที่สุดที่เรามี

### 2.2 ด้าน engineering — ผ่านบางส่วน

- JSON configs valid, CSV registry well-formed
- unit tests เดิม **ผ่าน 25/25**
- **identity conversion แม่นระดับ bitwise** — `max_logit_error = 0.0` (threshold คือ `1e-5`)
- ตรวจสมการแล้ว: query เป็น learned vector zero-init, `K = norm(V)`, output additive
  → **ทั้งหมดตรงกับ paper Delta จริง**

---

## 3. ปัญหาที่ต้องปิด

### 3.1 ⛔ BLOCKING — `route_scale` ทำให้ router เรียนรู้ไม่ได้

โค้ดทีมเพิ่ม `route_scale` init = 0 ซึ่ง **ไม่มีใน paper** เพราะ `out = residual + s · mixture(w)` จึงได้
`∂L/∂w = (∂L/∂out) · s · (∂mixture/∂w)` → **gradient ของ router แปรผันตรงกับ `s` และเป็นศูนย์พอดีที่ init**
(ข้อเท็จจริงเชิงพีชคณิต ไม่ใช่แค่ผลวัด)

| step | `route_scale` | gradient เทียบกับแบบตาม paper |
|---|---|---|
| 1 | 0.000000 | **0 %** |
| 10 | 0.008177 | 0.047 % |
| 50 | 0.016297 | **0.095 %** |

**ทำไมร้ายแรง:** ถ้า D1/D2 แพ้ S0 เราจะ **แยกไม่ออก**ว่าเป็นผลวิจัยหรือเป็นบั๊ก
**และมันทำลายทางออกสำรองด้วย** — เพราะ negative result จะมีค่าก็ต่อเมื่อ implementation ถูกต้อง

### 3.2 ปัญหา engineering อื่น

| ID | ปัญหา | ผลกระทบ |
|---|---|---|
| E1 | **MHAR (D2) ยังไม่มีโค้ดเลย** | ต้องเขียนใหม่ |
| E2 | `last_routing` **ค้าง autograd graph** | กิน memory เฉพาะฝั่ง routed → ตัวเลข efficiency เพี้ยน |
| E3 | `block_size_layers=4` บน 8 ชั้น → ชั้น 0–3 ได้ **0 sources** | ต้องเทียบ config กับ official |
| E4 | **ปิด gradient checkpointing** เฉพาะ routed arms | S0 กับ D1/D2 คนละ memory regime → ไม่ใช่การทดลองที่ควบคุมแล้ว |
| E5 | gate เก่ายังบังคับกฎ **10% overhead kill** ที่ยกเลิกไปแล้ว | ระวังติดกลับมาตอน reuse |

⚠️ **E2 กับ E4 ลำเอียงต่อต้านฝั่ง routed ทั้งคู่** เมื่อรวมกับ throughput ของ AttnRes ที่
`arXiv:2607.27230` รายงานไว้ที่ **0.55–0.88×** → ผล "routing แพ้" จะตีความไม่ได้เลย

**💡 ทางลัดที่แนะนำ:** repo ทางการเป็น **MIT** (`github.com/wdlctc/delta-attention-residuals-code`)
มี `modeling_qwen3_attnres.py`, DDP 220M–1B, FSDP 7B+, fine-tuning, eval และ `WANDB_RUNS.md`
→ **สลับไปใช้ของเขา ปิด §3.1 + E1 + E2 ได้พร้อมกัน** และได้ target ให้ reproduce ด้วย

### 3.3 ข้อมูลที่แก้ไขใน registry

- **AR05:** `arXiv:2606.06564` v1 ชื่อ "WAV" แต่ **v2 เปลี่ยนเป็น "HAARES"** → ต้องอ้างชื่อ v2
- **license:** repo ทางการเป็น **MIT** ไม่ใช่ "unlicensed" ตามที่ `src/README.md` เขียนไว้
- เพิ่มงานที่หายไป **7 ชิ้น** (2 ชิ้นจาก 3 สัปดาห์ล่าสุด)

---

## 4. ตัววัดของ Track 2 (S0 / D1 / D2)

**⚠️ ข้อควรระวังเชิงกรอบคิด:** สี่ตระกูลนี้คือ *แกนที่วัด* **ไม่ใช่ contribution**
paper ที่บอกว่า "เราวัด 4 อย่าง" คือ **รายงาน**
paper ที่บอกว่า "สองแกนนี้ให้คำตอบขัดกัน และนี่คือผลที่ตามมา" คือ **contribution**

### A. Thai acquisition
- **Thai BPB** บน held-out ที่ freeze ไว้ แยก web / encyclopedic
- Thai downstream aggregate (freeze benchmark ก่อนเปิดผล)
- วัดเป็นเส้นโค้งเทียบ **observed tokens** และเทียบ **GPU-hours**

> **ต้องใช้ BPB ไม่ใช่ perplexity** — perplexity ผูกกับ tokenizer และ tokenizer แต่ละตัวตัดคำไทย
> ไม่เท่ากันเลย (SmolLM2 = 0.583 chars/token) → เทียบข้ามโมเดลไม่ได้ ส่วน BPB หารด้วย byte
> ซึ่งคงที่ → เทียบได้จริง

### B. English/code retention ← **แกนที่ไม่มีใครวัดเลย**
- English BPB, code BPB, downstream aggregates
- **`retention_at_matched_acquisition()`** — "ณ จุดที่ทุก condition ได้ไทยเพิ่มเท่ากัน
  แต่ละตัวเสียอังกฤษไปเท่าไหร่" (มีใน harness แล้ว)

### C. Training efficiency
- tokens/sec, การกระจาย step-time, peak VRAM (allocated + reserved)
- **GPU-hours ที่ใช้ไปถึง Thai BPB เป้าหมายเดียวกัน** ← ตัวเลขที่มีค่าที่สุด
- inference latency

### D. Routing behavior
- mean entropy, **mean max weight** (Delta paper: max weight ≈ 0.2 = collapse)
- source usage ตามความลึก
- **JS divergence ระหว่าง routing ตอนอ่านไทย กับตอนอ่านอังกฤษ** ← ส่วนที่ยังเปิด
- code-switch boundary, clamp effect

### รูปหลักของ paper — มีแค่ 2 รูป
1. **Thai BPB (y) vs GPU-hours (x)** — 3 เส้น *ถ้าเส้นตัดกันคือได้ผล*
2. **Thai ดีขึ้น (x) vs English แย่ลง (y)** — Pareto frontier

**⚠️ จุดอ่อนที่ reviewer จะจี้:** แผนใช้ **2 seeds** ซึ่งประมาณ variance ไม่ได้จริง
ส่วน RD-AttnRes (`arXiv:2608.01075`, ส.ค. 2026) ใช้ **5 seeds** ต่อ scale

---

## 5. ประโยค contribution ที่อนุญาตให้ใช้

> **งานเดิมแสดงว่า** การ route ผ่าน layer-wise deltas (Delta AttnRes, 220M–7.6B) และผ่าน
> per-subspace heads (MHAR, 100M–8B) ช่วยลด validation loss และแปลง pretrained checkpoint
> ได้ด้วย fine-tuning ธรรมดา — **ภายใต้ setting** ของงาน general-domain ภาษาอังกฤษเป็นหลัก
> ที่เทรนจากศูนย์หรือ mid-training และวัดผลต่อ observed token
> **แต่ยังไม่ทราบว่า** ผลนี้เหลืออยู่หรือไม่ใน setting ที่กำหนดงาน Thai LLM จริง คือ
> continued pretraining แบบจำกัด compute จาก checkpoint ที่ตายตัว ซึ่ง routed variants
> เทรนได้เพียง 0.55–0.88× ของ baseline จึง **ต้องคุ้มค่าในหน่วย wall-clock ให้ได้เอง**
> และซึ่งผลได้ในภาษาเป้าหมายต้องชั่งกับการถดถอยของภาษาที่โมเดลฐานมีอยู่แล้ว
> **เราจึง** แปลง pretrained decoder ตัวเดียวกันเป็น S0 / D1 / D2 ภายใต้ข้อมูล ลำดับเอกสาร
> และ optimizer เดียวกัน แล้วเทียบบน Thai acquisition, English/code retention และ routing
> behaviour **ทั้งแกน token-matched และ GPU-hour-matched**
> **และรายงานว่า** ลำดับความดีของ variants ในงาน general-domain ถ่ายทอดมาสู่ Thai
> compute-limited CPT ตรงไหนและไม่ตรงไหน — **รวมถึงกรณีที่ถ่ายทอดไม่ได้**

---

## 6. ⚠️ อันดับตัวเลือก paper — พลิกกลับหลัง tokenizer audit

รัน novelty audit หัวข้อ tokenizer ด้วยมาตรฐานเดียวกัน (6 queries + ตรวจต้นฉบับ) ผลคือ:

| | AttnRes audit | Tokenizer audit |
|---|---|---|
| คำค้นชี้ขาด | **1 รายการ** | **3 queries ชน cap 40** |
| แปลว่า | พื้นที่ว่าง | **พื้นที่แน่น** |

**`arXiv:2606.15044` (Equity with Efficiency, 13 มิ.ย. 2026) ปิดประตูหัวข้อ tokenizer:**
ครอบคลุม **ภาษาไทย** (1 ใน 11 SEA languages), **controlled 1.5B บน OLMo-2-1B**,
และรายงาน **GPU-hours จริง** (68–300 ชม. บน 8×H200) — กินครบทั้ง 4 แกนพร้อมกัน

### อันดับหลังตรวจ

| อันดับ | Paper | novelty headroom |
|---|---|---|
| **1** | **AttnRes controlled study** | **กว้างที่สุด** (1 hit) |
| 2 | ThaiLLM-1B model paper | ไม่ตัดสินที่ novelty แต่ตัดสินที่คุณภาพ artifact |
| 3 | Tokenizer / screening study | **แคบที่สุด** → **ไม่ควรทำเป็น paper เดี่ยว** |

**งาน tokenizer ที่ทำไว้แล้วยังมีค่า** — แต่ที่ที่ถูกต้องคือ **เป็น section ใน ThaiLLM-1B model
paper** (justify การเลือก base) ซึ่ง model paper ถูกคาดหวังให้มีอยู่แล้วและไม่ต้อง novel ตรงนั้น

---

## 7. โครงสร้างสองสาย

```
Track 1 (Product)  ThaiLLM-1B + Standard Residual
                   ├─ base-model screen  ──┐
                   ├─ data pipeline      ──┼──► ใช้ร่วมกับ Track 2 ได้ฟรี
                   ├─ eval harness       ──┘
                   └─ tokenizer analysis ───► section ใน model paper

Track 2 (Research) controlled study S0/D1/D2
                   └─ ใช้ data + eval จาก Track 1 → ต้นทุนเพิ่มเฉพาะ GPU ของ pilot
```

| | Track 1 — Product | Track 2 — Research |
|---|---|---|
| ของที่ได้ | ThaiLLM-1B Base/Instruct | controlled study S0/D1/D2 |
| สถาปัตยกรรม | **Standard Residual** (ปลอดภัย) | AttnRes variants |
| ขนาด | 1B | เล็กกว่าได้ (0.6B) |
| ความเสี่ยง | ต่ำ | รับได้ — ไม่กระทบ product |
| นาฬิกา | ตามงาน engineering | **วรรณกรรมเป็นคนตั้ง** (ดู §8) |

**❗ อย่าเอา AttnRes ใส่ลงใน product** — เอาความเสี่ยงงานวิจัยไปแขวนกับของที่ต้องส่งมอบ
(`README.md` เขียนกฎข้อนี้ไว้เองอยู่แล้ว)

### 💡 base model: สองสายต้องการคนละแบบ — อย่าให้การตัดสินใจเดียวรับใช้ทั้งสองสาย

- **Track 1 (product)** อยากได้จุดเริ่มที่ **ไทยเก่งที่สุดเท่าที่หาได้** — โมเดลที่ผ่าน Thai CPT
  มาแล้วคือ **ข้อได้เปรียบ**
- **Track 2 (study)** อยากได้ base ที่ **ยังมี headroom ภาษาไทยเหลือ** ไม่งั้นวัด Thai
  acquisition ไม่ขึ้นเพราะอิ่มตัวไปแล้ว

> เหตุผลที่ตัด Sailor2 ออกเพราะ "ผ่าน Thai CPT มาแล้ว" — **ถูกต้องสำหรับ Track 2
> แต่กลับกันเลยสำหรับ Track 1**

---

## 8. ⏰ Track 2 ไม่ใช่ "ตัวสำรอง" — มันคือของที่หมดอายุได้

**ข้อเข้าใจผิดที่ต้องแก้:** ถ้ามอง Track 2 เป็น "เผื่อไว้กรณี model ตีพิมพ์ไม่ได้"
แล้วเริ่มทำก็ต่อเมื่อ Track 1 ล้มเหลว → **จะเริ่มสายเกินไป**

เหตุผล:
1. **Track 2 มี novelty headroom กว้างที่สุดในสามตัวเลือก** (§6) มันไม่ใช่ของสำรอง
2. **ช่องว่างของมันปิดได้เอง** — สนาม AttnRes ผลิต paper ราว **1 ชิ้นทุก 10 วัน**
   ถ้ารอ 8 เดือน query T4 อาจได้ 5 hits แทนที่จะเป็น 1
3. **Track 1 ไม่หมดอายุ** — โมเดลที่ดีก็ยังดีในอีกปีหนึ่ง

→ **เรียงงานที่เน่าเสียได้ไว้ท้ายสุด คือเรียงกลับด้าน**

**สิ่งที่ต้องทำเพื่อ "รักษาสิทธิ์" ไว้ (ถูกมาก):**

| งาน | ต้นทุน | ทำไม |
|---|---|---|
| ค้น novelty ซ้ำ (Q1–Q4 + T4) ทุก 4–6 สัปดาห์ | ~30 นาที | รู้ทันทีถ้าช่องว่างปิด |
| สลับไป repo MIT + ปิด E2/E4 | 2–3 วัน | รักษา option ให้ยังใช้ได้ |
| ออกแบบ data pipeline ของ Track 1 ให้ Track 2 ใช้ต่อได้ | ~0 | ทำอยู่แล้ว |

---

## 9. Experiment ขั้นต่ำก่อนล็อก Track 2

| # | เงื่อนไข | ต้องได้ |
|---|---|---|
| M1 | ใช้ implementation ทางการ (MIT) **หรือ** ลบ `route_scale` แล้วพิสูจน์ว่า query gradient ≠ 0 ที่ step 1 | **บังคับ** |
| M2 | ปิด E2 และ E4 ให้ทั้ง 3 conditions อยู่ใน memory regime เดียวกัน | **บังคับ** |
| M3 | D2/MHAR มีโค้ด + ผ่าน identity conversion + router gradient ≠ 0 | **บังคับ** |
| M4 | one-A100 preflight 200 steps × 3 conditions | **บังคับ** |
| M5 | pilot 50–100M tokens × 3 conditions ได้ Thai BPB curve ทั้ง 2 แกน | **บังคับ** |
| M6 | routing ไม่ collapse และไม่ uniform | **บังคับ** |
| M7 | ทิศทางผลซ้ำใน seed ที่ 2 (ถ้าได้ 3+ seeds ยิ่งดี — คู่แข่งใช้ 5) | บังคับก่อน promote |
| M8 | ค้น novelty ซ้ำภายใน 48 ชม. ก่อนส่ง | **บังคับ** |

**M1–M3 ต้องเสร็จก่อนใช้ GPU แม้แต่ชั่วโมงเดียว**

---

## 10. Kill conditions

### Track 1
1. ไม่มี base model ที่ license ใช้ได้จริง
2. ข้อมูลไทยไม่ผ่าน provenance/license audit

### Track 2
1. หลังแก้ router แล้ว routing ยัง collapse หรือ uniform ใน pilot
2. **มี paper ครอบ "AttnRes สำหรับ cross-lingual / language adaptation" ออกมา**
   → contribution ข้อเดียวที่เปิดอยู่หายไป *(ตรวจทุก 4–6 สัปดาห์)*
3. ผลเป็นแค่การยืนยันลำดับของ Delta paper ซ้ำ โดยไม่มีข้อใดใน 3 ข้อนี้:
   ranking กลับด้านต่อ GPU-hour / magnitude ต่างใน CPT / acquisition-retention แยกทาง
4. แยก effect ของ architecture ออกจาก data/compute/parameters ไม่ได้
5. ประมาณ variance ด้วย replicated runs ไม่ได้

---

## 11. Claim ที่อนุญาต vs. ห้ามใช้

### ✅ อนุญาต

- "ภายใต้ Thai token exposure, ลำดับข้อมูล และ GPU-hours ที่ควบคุมเท่ากัน เราวัดผลต่างของ
  S0 / D1 / D2 บนโมเดลขนาด **[ระบุขนาดที่รันจริง]**"
- "เท่าที่ค้นได้ถึง **[วันที่]** ด้วย query **[ระบุ]** ยังไม่พบงานที่ศึกษา AttnRes ภายใต้
  language-adaptation CPT"
- "งานเดิมไม่มีชิ้นใดวัด retention ของภาษาต้นทาง"
- "routed variants ชนะ/แพ้ต่อ observed token แต่ให้ผลตรงข้าม/เหมือนกันต่อ GPU-hour"
- "เป็นผลจาก pilot ขนาด [X] ยังไม่ได้ยืนยันที่ 1B"

### ❌ ห้ามใช้เด็ดขาด

- ❌ "งานแรกที่ใช้ AttnRes กับภาษาไทย" (AttnRes อยู่ใน Kimi K3 และ Motif 3 multilingual แล้ว)
- ❌ "ไม่มีใครทำมาก่อน" → ต้องเป็น "ไม่พบในการค้นเมื่อ [วันที่] ด้วย query [ระบุ]"
- ❌ "A Controlled 1B Study" ถ้ารันจริงต่ำกว่า 1B
- ❌ อ้าง mechanism analysis เป็นของใหม่ โดยไม่อ้าง `arXiv:2606.13168`
- ❌ อ้างว่า implementation ของทีมเป็น reproduction ของ Delta AttnRes ทางการ
- ❌ ใช้ผล smoke test / tiny model เป็นหลักฐานวิทยาศาสตร์
- ❌ อ้างผล 1B ก่อนมีผล 1B จริง
- ❌ อ้าง "WAV" เป็นชื่อวิธีของ `arXiv:2606.06564` (v2 = HAARES)
- ❌ เสนอ tokenizer/screening เป็น paper เดี่ยวโดยไม่อ้าง `arXiv:2606.15044`

---

## 12. สคริปต์คุย mentor

> "ผมตรวจ novelty ใหม่ถึงวันนี้แล้วครับ **หัวข้อ AttnRes ยังมีช่องว่างจริง** — ผมค้น arXiv
> โดย intersect วรรณกรรม AttnRes ทั้งหมดกับ multilingual/low-resource ได้ผลลัพธ์เดียว
> และเป็น technical report ไม่ใช่ controlled study แล้วก็**ไม่มี paper ไหนวัด retention เลย**
>
> ตาม framing engineering-first ที่อาจารย์ให้มา ผมเสนอแยกเป็นสองสายครับ —
> **Track 1 ทำ ThaiLLM-1B ด้วย Standard Residual** ไม่เอาสถาปัตยกรรมวิจัยมาเสี่ยงกับ product
> ส่วน **Track 2 คือ AttnRes study** ใช้ data pipeline กับ eval ร่วมกับ Track 1 ได้เลย
> ต้นทุนเพิ่มแค่ GPU ของ pilot
>
> **มีสองเรื่องที่ต้องรายงานตรง ๆ ครับ**
> หนึ่ง — ผมเจอบั๊กใน router ที่ทีมเขียนเอง ตัว `route_scale` ที่ init เป็น 0 ทำให้ gradient
> ของ router เป็นศูนย์พอดีตอนเริ่ม หลัง 50 steps ยังได้แค่ 0.1% ของที่ควรจะเป็น
> ถ้ารันทั้งอย่างนี้แล้วได้ผลลบ เราจะแยกไม่ออกว่าเป็นผลวิจัยหรือบั๊ก
> ข่าวดีคือ repo ทางการเป็น MIT license ใช้ของเขาได้เลย ปัญหาหายทันที
>
> สอง — ผมลองตรวจหัวข้อ tokenizer ที่คิดว่าจะเอามาตีพิมพ์แยก **ปรากฏว่าสนามนั้นแน่นมาก**
> มี paper ชื่อ Equity with Efficiency เมื่อ มิ.ย. ที่ครอบคลุมภาษาไทย เทรน 1.5B จริง
> และรายงาน GPU-hours ครบ → ผมเสนอให้ยุบงาน tokenizer ไปเป็น section ใน model paper แทน
>
> **สิ่งที่ผมอยากขออนุมัติคือ อย่าเลื่อน Track 2 ไปไว้ท้ายสุดครับ** เพราะช่องว่างของมันปิดได้เอง
> — สนามนี้ออก paper ราวทุก 10 วัน ผมขอเวลา 2–3 วันแก้ router ให้ถูกไว้ก่อน
> แล้วค้น novelty ซ้ำทุกเดือน จะได้รู้ทันถ้าช่องว่างปิด"

**หลักการพูดที่ห้ามพลาด:**

1. **นำด้วยหลักฐาน** — บอก query, วันที่, จำนวนผลลัพธ์
2. **พูดข่าวร้ายก่อนถูกถาม** — บั๊ก `route_scale` และผล tokenizer audit
3. **ห้ามพูดว่า "ยังไม่มีใครทำ"** → "เท่าที่ค้นได้ถึง [วันที่] ยังไม่พบ"
4. **ห้ามสัญญาว่าจะได้ตีพิมพ์** — ยังไม่มีผล model pilot แม้แต่ตัวเดียว
5. **ขายผลลบล่วงหน้า** — "ถ้า routing ไม่คุ้มต่อ GPU-hour นั่นคือผลที่ lab ไทยได้ใช้จริง
   และเราจะ pre-register คำทำนายนี้ไว้ก่อน"
6. **เน้นว่า Track 2 ต้นทุนต่ำ** เพราะใช้ของจาก Track 1 ซ้ำได้เกือบหมด
