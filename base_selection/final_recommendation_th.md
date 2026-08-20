# คำแนะนำการเลือก Base Model — ThaiLLM-1B + Attention Residuals

**รหัสเอกสาร:** `BASE-SCREEN-2026-08-18-DECISION`
**วันที่:** 2026-08-18
**สถานะ:** `conditional — ยังไม่ล็อกตัวเลือกสุดท้าย`
**`scientific_evidence_allowed`:** `false`

---

## 0. บทสรุปผู้บริหาร

| บทบาท | โมเดล | Revision (immutable) | สถานะ |
|---|---|---|---|
| **Main base (เงื่อนไข)** | `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | **ยังล็อกไม่ได้** — ติด gate 2 ข้อ |
| **Fallback** | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | พร้อมใช้ทันที ต้องขออนุมัติเรื่องขนาด |
| **Pilot** | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | พร้อมใช้ทันที |
| **Cross-family replication (ไทย)** | `Qwen/Qwen2.5-1.5B` | เดียวกับ fallback | พร้อมใช้ |
| **Cross-family replication (สถาปัตยกรรม, อังกฤษเท่านั้น)** | `allenai/OLMo-2-0425-1B` | `a1847dff35000b4271fa70afc5db10fd29fedbdf` | ใช้ได้เฉพาะแกนสถาปัตยกรรม |
| **ทางเลือกที่ tokenizer ดีที่สุด** | `google/gemma-3-1b-pt` | `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` | เก็บไว้เป็นทางเลือก ไม่แนะนำเป็นหลัก |

**ข้อความสำคัญที่สุดของเอกสารนี้: ยังล็อก final model ไม่ได้ และไม่ควรล็อก**
เพราะ license gate ยังไม่ผ่าน ไม่ใช่เพราะข้อมูลไม่พอ รายละเอียดอยู่ใน §1

---

## 1. Fail-closed: gate ที่ยังไม่ผ่าน (สำคัญที่สุด)

`experiment_parameters.json` กำหนด `fail_closed_on_missing_license_or_revision: true`
และโจทย์กำหนดว่า *ห้ามเลือก final model ก่อน license, tokenizer และ port gate ผ่าน*
สถานะจริง ณ ตอนนี้:

### Gate A — การเข้าถึงถูกปิด (ยังไม่ผ่าน)

`meta-llama/Llama-3.2-1B` และ `google/gemma-3-1b-pt` มีสถานะ `gated: manual` บน Hugging Face
และในเครื่องนี้ **ไม่มี HF token** ผลคือ:

- ดึง `config.json` และ `tokenizer.json` จาก repo ต้นฉบับโดยตรง **ไม่ได้** (401/403)
- ตัวเลข tokenizer และ architecture ของสองรุ่นนี้ในรายงาน มาจาก **ungated mirror**
  (`unsloth/Llama-3.2-1B` rev `9535bd9b…`, `unsloth/gemma-3-1b-pt` rev `34a98bf3…`)
- **ยังไม่ได้พิสูจน์ว่า artifact จาก mirror ตรงกับต้นฉบับแบบ byte-identical**

ตัวเลขทั้งหมดของ Llama/Gemma ในเอกสารชุดนี้จึงเป็น **provisional**
ต้องรัน tokenizer screen ซ้ำกับ revision ต้นฉบับหลังได้รับสิทธิ์ ก่อนใช้เป็นหลักฐาน

> ตัวเลข parameter ของ mirror ตรงกับที่รายงานทุกตัว (Llama 1,235,814,400 / Gemma 999,885,952)
> และ config ก็สอดคล้องกับ model card สาธารณะ จึงมีเหตุผลเชื่อว่า mirror ถูกต้อง
> แต่ *"มีเหตุผลเชื่อ"* ไม่ใช่ *"ตรวจสอบแล้ว"* — ตามกฎ fail-closed ต้องนับว่ายังไม่ผ่าน

### Gate B — เงื่อนไขการตั้งชื่อ กระทบชื่อ deliverable โดยตรง (ยังไม่ผ่าน)

นี่เป็นข้อค้นพบที่กระทบแผนงานมากที่สุด และยังไม่ปรากฏในเอกสารเดิม:

| License | อนุญาตทำ derivative? | เงื่อนไขการตั้งชื่อ |
|---|---|---|
| Llama 3.2 Community | ได้ | **ชื่อ derivative ต้องขึ้นต้นด้วย "Llama"** + ต้องแสดง "Built with Llama" + ต้องส่งต่อ AUP |
| Gemma Terms | ได้ | **ชื่อ derivative ต้องขึ้นต้นด้วย "Gemma"** + ต้องส่งต่อ Gemma Terms และ Prohibited Use Policy |
| Apache-2.0 (Qwen, OLMo) | ได้ | **ไม่มีเงื่อนไข** |

แปลว่า **ถ้าใช้ Llama หรือ Gemma จะเผยแพร่ในชื่อ `ThaiLLM-1B-Base` ไม่ได้**
ต้องเป็น `Llama-ThaiLLM-1B-Base` หรือ `Gemma-ThaiLLM-1B-Base`
ซึ่งขัดกับชื่อ deliverable ที่ระบุไว้ใน `experiment_parameters.json` และ `README.md`

**นี่เป็นการตัดสินใจของ mentor/โครงการ ไม่ใช่ของ engineer** และเป็นเหตุผลหลักที่
fallback ในเอกสารนี้ถูกเปลี่ยนจากแผนเดิม (ดู §4)

### Gate C — tokenizer gate: **ผ่าน**

วัดจริงด้วย `tokenizers` 0.22.2 บน frozen sample set 14 เอกสาร
(`sha256 = 6abc31ea71544ca227b329554e6a296b54c0a6a1f8ad9dfdb60283d757e3a963`)
ทุก tokenizer round-trip ไม่มีการสูญเสีย, U+FFFD = 0, `<unk>` = 0 → ไม่มีใครตกด้วยเหตุผลความถูกต้อง
คัดกรองด้วย *ต้นทุน* แทน (ดู §2)

### Gate D — port gate: **ผ่านทุกตัว** แต่ต้นทุนต่างกันมาก

ไม่มี candidate ไหนที่ทำ Delta Block AttnRes ไม่ได้ (ดู `architecture_port_audit.md`)
ต่างกันที่ effort 4–19 engineer-days และระดับความเสี่ยงที่ conversion จะกลายเป็น confound

---

## 2. ผลการวัด Tokenizer (หลักฐานจริง ไม่ใช่การคาดเดา)

วัดบน sample เดียวกันทุกโมเดล ค่าที่สูงกว่า = ดีกว่า สำหรับ chars/token

| โมเดล | vocab | **ไทย chars/token** | ไทย bytes/token | EN bytes/token | code bytes/token | ไทย/อังกฤษ token ratio |
|---|---:|---:|---:|---:|---:|---:|
| `gemma-3-1b-pt` | 262,144 | **2.83** | 7.78 | 5.91 | 3.14 | **1.45×** |
| `Llama-3.2-1B` | 128,256 | **2.18** | 5.98 | 6.03 | 3.98 | 2.33× |
| `Qwen2.5-1.5B` / `Qwen3-0.6B-Base` | 151,936 | 1.84 | 5.05 | 6.03 | 3.90 | 2.86× |
| `OLMo-2-0425-1B` | 100,352 | **1.08** | 2.97 | 6.03 | 3.98 | 5.24× |
| `Falcon3-1B-Base` | 131,072 | **0.74** | 2.02 | 5.87 | 2.90 | 7.90× |
| `SmolLM2-1.7B` | 49,152 | **0.58** | 1.60 | 5.87 | 3.11 | 9.81× |

### ทำไมห้ามตัดสินจาก vocabulary size อย่างเดียว — มีหลักฐานยืนยันในข้อมูลนี้เอง

- **Qwen มี vocab ใหญ่กว่า Llama (151,936 vs 128,256) และมีชิ้นส่วนที่มีอักษรไทยมากกว่า
  (2,570 vs 1,391) แต่ตัดคำไทยได้ *แย่กว่า*** (1.84 vs 2.18 chars/token)
- **Gemma มีชิ้นส่วนไทยน้อยกว่า Qwen (2,177 vs 2,570) แต่ดีกว่าอย่างมาก** (2.83 vs 1.84)

ดังนั้นทั้ง vocab size และจำนวนชิ้นส่วนไทย ต่างก็ **ทำนายคุณภาพการตัดคำไทยไม่ได้**
สิ่งที่ต่างจริงคือคุณภาพของ merge:

| ข้อความ | Gemma 3 | Llama 3.2 | Qwen | OLMo 2 |
|---|---|---|---|---|
| `การสังเคราะห์ด้วยแสง` | **5 tokens** `การ\|สัง\|เคราะห์\|ด้วย\|แสง` | 6 | 11 | 18 |
| `ก็ไม่รู้เหมือนกัน` | **5 tokens** `ก็\|ไม่\|รู้\|เหมือน\|กัน` | 9 | 10 | 14 |
| `ปัญญาประดิษฐ์` | **6 tokens** | 7 | 9 | 17 |
| `๒๕๖๘` (เลขไทย) | **4 tokens** (ตัวละ token) | 6 | 8 | 8 |

Gemma 3 เป็น tokenizer **เดียว** ในชุดนี้ที่ตัดภาษาไทยออกมาเป็นหน่วยคล้ายหน่วยคำจริง
ตัวอื่นแตกเป็นเศษ byte ระดับต่ำกว่าพยางค์

### ผลต่อ compute โดยตรง

corpus ไทยชุดเดียวกัน (SEA-PILE-v2 `th`) กลายเป็นจำนวน token ต่างกัน → ชั่วโมง GPU ต่างกัน:

| โมเดล | token ที่ได้ | เทียบ Gemma | GPU-h สำหรับ CPT 10B token |
|---|---:|---:|---:|
| `gemma-3-1b-pt` | 6.50 B | 1.00× | ≈178 |
| `Llama-3.2-1B` | 8.45 B | 1.30× | ≈278 |
| `Qwen2.5-1.5B` | 10.02 B | 1.54× | ≈414 |
| `OLMo-2-0425-1B` | 17.01 B | **2.62×** | ≈638 |

---

## 3. ตารางคะแนน 100 คะแนน

| โมเดล | ไทย/tok 25 | AttnRes 20 | License 15 | Compute 15 | EN/code 10 | Ecosystem 10 | ขนาด 5 | **รวม** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `meta-llama/Llama-3.2-1B` | 20 | 19 | 9 | 14 | 8 | 10 | 5 | **85** |
| `sail/Sailor2-1B` | 18 | 18 | 15 | 13 | 6 | 9 | 5 | **84 → ถูกตัด** |
| `Qwen/Qwen3-0.6B-Base` | 16 | 20 | 15 | 15 | 5 | 10 | 1 | **82 → pilot เท่านั้น** |
| `Qwen/Qwen2.5-1.5B` | 16 | 18 | 15 | 10 | 9 | 10 | 3 | **81** |
| `google/gemma-3-1b-pt` | 24 | 11 | 8 | 15 | 6 | 8 | 5 | **77** |
| `allenai/OLMo-2-0425-1B` | 4 | 17 | 15 | 6 | 7 | 10 | 3 | **62 → ถูกตัด** |
| `HuggingFaceTB/SmolLM2-1.7B` | 1 | 18 | 15 | 6 | 7 | 10 | 3 | **60 → ถูกตัด** |
| `tiiuae/Falcon3-1B-Base` | 2 | 18 | 7 | 6 | 7 | 9 | 4 | **53 → ถูกตัด** |

### กฎ "คะแนนรวมกลบ fatal blocker ไม่ได้" — เห็นผลจริงในตารางนี้

**`sail/Sailor2-1B` ได้ 84 คะแนน สูงเป็นอันดับ 2 แต่ถูกตัดออก**
เพราะมันผ่าน Thai/SEA continued-pretraining มาแล้ว การนำมาเป็น base
จะทำให้วัด *Thai acquisition* ไม่ได้ — ตัวแปรต้นถูกปนเปื้อนตั้งแต่ต้น
คะแนน 84 ไม่มีความหมายเมื่อโมเดลตอบคำถามวิจัยไม่ได้

เช่นเดียวกัน **`Qwen3-0.6B-Base` ได้ 82 แต่เป็น pilot ไม่ใช่ final** เพราะ 0.596 B ไม่ใช่ "ประมาณ 1B"

---

## 4. คำตัดสินแยกตามบทบาท

### 4.1 Final main base — `meta-llama/Llama-3.2-1B` (มีเงื่อนไข)

**Revision:** `4e20de362430cd3b72f300e6b0f18e50e7166e08`
**Parameters:** 1,235,814,400 (embedding 262.7 M tied 21.3 % / non-embedding 973.1 M)
**สถาปัตยกรรม:** 16 layers × d 2048, GQA 32q/8kv, head_dim 64, FFN 8192, vocab 128,256,
RoPE θ=5e5 + llama3 NTK scaling, RMSNorm pre-norm, SwiGLU, tied embeddings, ctx 131,072

เลือกเพราะ **สมดุล** ที่ดีที่สุด ไม่ใช่เพราะชนะด้านใดด้านหนึ่ง:

- tokenizer ไทยดีเป็นอันดับ 2 (2.18 chars/token) และ **ภาษาไทยอยู่ในรายการภาษาที่ Meta รองรับอย่างเป็นทางการ**
  (`language: [en, de, fr, it, pt, hi, es, th]` บน model card) — เป็นตัวเดียวในกลุ่มที่ไทยเป็นภาษาทางการที่ประกาศไว้
- **สถาปัตยกรรมสะอาดที่สุดสำหรับ AttnRes** — pre-norm มาตรฐาน, delta เป็น output ของ sublayer ตรง ๆ,
  16 layers หารด้วย 2/4/8 ลงตัว, ไม่มี sliding window, ไม่มี QK-norm, ไม่มี softcapping
- port ต่ำสุดในกลุ่ม ≈1B (3–5 วัน + MHAR อีก 3–4 วัน)

**เงื่อนไขที่ต้องผ่านก่อนล็อก:** Gate A (ได้สิทธิ์ gated + verify tokenizer กับต้นฉบับ)
และ Gate B (ยอมรับชื่อ `Llama-ThaiLLM-1B-Base`)

### 4.2 Final fallback — `Qwen/Qwen2.5-1.5B` **(เปลี่ยนจากแผนเดิม)**

แผนเดิมใน `research_plan.md` ระบุ Gemma 3 1B เป็น fallback **ขอเสนอให้เปลี่ยน**

เหตุผล: fallback ที่ดีต้อง **พังด้วยเหตุผลคนละแบบกับตัวหลัก**
ถ้า Llama ล้มเพราะ Gate A หรือ Gate B → **Gemma ก็ล้มด้วยเหตุผลเดียวกันเป๊ะ**
(gated:manual เหมือนกัน, บังคับตั้งชื่อเหมือนกัน) จึงไม่ใช่ fallback จริง

`Qwen2.5-1.5B` เป็น Apache-2.0, ไม่ gate, ไม่มีเงื่อนไขตั้งชื่อ →
**เผยแพร่ในชื่อ `ThaiLLM-1B-Base` ได้ตรงตามแผน** และไทยอยู่ในภาษาที่ประกาศรองรับ
สถาปัตยกรรม pre-norm สะอาด (ต่างจาก Llama แค่ QKV bias)

ราคาที่ต้องจ่าย: 1.544 B (มากกว่า "ประมาณ 1B" 25 %) และ compute สูงกว่า Llama ≈49 %
→ **ต้องขออนุมัติ mentor เรื่องขนาด**

### 4.3 Cheap pilot — `Qwen/Qwen3-0.6B-Base`

- port risk ต่ำที่สุด: `CODE01` (MIT) มี `modeling_qwen3_attnres.py` ตรงสถาปัตยกรรมนี้
- ถูกที่สุด: 100 M token/condition = 1.53 GPU-h (S0); 3 conditions ≈ 5–7 GPU-h
- **tokenizer เหมือน `Qwen2.5-1.5B` แบบ byte-identical (sha256 ตรงกัน)** → ถ้าตกไปใช้ fallback
  ผล pilot จะโอนได้แบบ token-invariant

**ข้อควรระวัง 2 ข้อ:**
1. `AR08` (arXiv:2606.13168) ทำ causal routing probe บน Qwen3-0.6B ไปแล้ว — ตรงกับ pilot เรา
   ตาม `VAL-2026-08-18-NOVELTY-MATRIX` mechanism analysis ต้องลดเป็น supporting result อยู่แล้ว
   pilot บนโมเดลเดียวกันจึงไม่เพิ่มความเสี่ยง novelty แต่ก็ไม่ช่วย
2. tokenizer ของ Qwen **ไม่เหมือน** Llama → ผล pilot ไม่โอนแบบ token-for-token ไปยัง main base
   **ข้อเสนอ: รัน pilot ทั้งบน Qwen3-0.6B และ Llama-3.2-1B (รวม ≈15–20 GPU-h)**
   เพื่อไม่ต้องสมมติว่าผลจาก 0.6B โอนไป 1.2B ได้

### 4.4 Cross-family replication — แยกเป็นสองแกน

- **แกนภาษาไทย:** `Qwen/Qwen2.5-1.5B` — คนละตระกูล, tokenizer คนละแบบ, ไทยรองรับจริง
- **แกนสถาปัตยกรรม (อังกฤษเท่านั้น):** `allenai/OLMo-2-0425-1B` — เป็น **post-norm แบบ reordered**
  (ไม่มี `input_layernorm` เลย) ซึ่งเป็น residual topology ที่ต่างจริง
  ถ้าผล delta routing ซ้ำได้ทั้งบน pre-norm และ post-norm ข้อสรุปจะแข็งกว่ามาก
  แต่ **ห้ามใช้กับแกนไทย** (ดู §4.5)

### 4.5 Rejected candidates

| โมเดล | คะแนน | เหตุผลที่ตัด |
|---|---:|---|
| `allenai/OLMo-2-0425-1B` | 62 | **Thai fatal (ในบทบาท base)** — model card ระบุ "Language(s): English" ชัดเจน; ไทย 1.08 chars/token ≈ byte-level; corpus เดียวกันแพงขึ้น 2.62× และ **พารามิเตอร์จริง 1.485 B ไม่ใช่ 1.0 B ตามที่บันทึกไว้เดิม**; ctx 4096 |
| `sail/Sailor2-1B` | **84** | **Confound fatal** — ผ่าน Thai/SEA CPT มาแล้ว วัด Thai acquisition ไม่ได้ (ตัวอย่างชัดเจนว่าคะแนนสูงไม่ช่วยอะไร) |
| `tiiuae/Falcon3-1B-Base` | 53 | **Thai fatal** — รองรับแค่ en/fr/es/pt; ไทย 0.74 chars/token (แพงขึ้น 3.84×); vocab มีชิ้นส่วนไทยที่ยาวเกิน 1 อักษรแค่ **1 ชิ้น**; license custom |
| `HuggingFaceTB/SmolLM2-1.7B` | 60 | **Thai fatal** — อังกฤษล้วน; ไทย 0.58 chars/token แย่ที่สุด; vocab มีชิ้นส่วนที่มีอักษรไทยแค่ **2 ชิ้น** |
| `ibm-granite/granite-3.3-2b-base` | — | ขนาด 2.53 B เกินกรอบ "ประมาณ 1B" มาก; มี `residual_multiplier`, `attention_multiplier`, `logits_scaling` ที่ไปยุ่งกับ residual stream โดยตรง → เสี่ยงเป็น confound กับ AttnRes |
| `aisingapore/Gemma-SEA-LION-v3-9B` | — | 9.24 B ผิดขนาดโดยสิ้นเชิง (ตรวจเพื่อยืนยันขนาดเท่านั้น) |
| `google/gemma-3-1b-pt` | 77 | **ไม่ตัด แต่ไม่แนะนำเป็นหลัก** — ดู §4.6 |

### 4.6 กรณี Gemma 3 1B — ทำไม tokenizer ดีที่สุดถึงไม่ได้เป็นตัวหลัก

Gemma 3 ชนะขาดด้าน tokenizer ไทย (2.83 chars/token, ถูกกว่า Llama 30 %) และขนาดตรงเป๊ะ 1.0 B
แต่แพ้ที่ **ความเสี่ยงต่อการทดลอง**:

1. **local/global attention สลับ 5:1 (`sliding_window=512`, `pattern=6`)** — router ที่ผสม delta
   จาก layer local-512 กับ global-32768 กำลังผสมเทนเซอร์ที่มา *คนละ receptive field*
   pattern การ route ที่เจออาจเป็นผลของการสลับ local/global ไม่ใช่ของ depth routing
   **นี่คือ confound ต่อ mechanism claim โดยตรง ไม่ใช่แค่ต้นทุน engineering**
2. **26 layers หารด้วย 4 ไม่ลงตัว** (หารลงตัวแค่ 2, 13, 26) → คุม block size ให้เท่ากันข้ามตระกูลไม่ได้
   และ 6 ∤ 26 ทำให้ block คร่อมขอบ local/global ไม่เท่ากันไม่ว่าจะเลือก block size เท่าไร
3. **sandwich norm** — delta ถูก normalize ก่อนบวกเข้า residual ทำให้ key ของ router
   เกือบเป็น unit norm สูญเสียสัญญาณ magnitude ที่ router ควรใช้
4. **MQA 1 KV head + 4 query heads** — cost profile ต่างจากตัวอื่นมากจนการ match GPU-hour ข้ามตระกูลตีความยาก
5. **697.9 M non-embedding เท่านั้น** (30.2 % ของพารามิเตอร์เป็น embedding) → ความสามารถ EN/code อ่อนสุดในกลุ่ม ≥1B
6. license มี Gate A + Gate B เหมือน Llama ทุกประการ (ไม่ได้ลดความเสี่ยงเลย)

port effort 10–15 วัน สูงกว่า Llama 3 เท่า
**ถ้าเป้าหมายเป็นแค่ ThaiLLM ที่ดีที่สุด Gemma 3 คือคำตอบ แต่เป้าหมายคือการวัดผลของ residual routing
ให้ตีความได้ — สถาปัตยกรรมของ Gemma 3 ทำให้ผลตีความยากที่สุด**

---

## 5. ทำไมถึงเลือกแบบนี้ ไม่ใช่เพราะ benchmark สูง

**เอกสารนี้ไม่ได้ใช้ benchmark score ใด ๆ ในการตัดสินเลย** ไม่มีการวัด MMLU, ไม่มี BPB, ไม่มี downstream
เพราะยังไม่มีสิทธิ์รันบน weights จริง เกณฑ์ที่ใช้ทั้งหมดเป็นคุณสมบัติที่ตรวจสอบได้จากต้นทาง:

1. **ความเหมาะกับภาษาไทย** — วัดจาก tokenizer fertility จริงบน sample ที่ freeze แล้ว
   ไม่ใช่จากคำโฆษณาว่า "รองรับ 140 ภาษา" Gemma อ้าง 140+ ภาษาและวัดแล้วดีจริง
   แต่ OLMo อ้างอังกฤษอย่างเดียวและวัดแล้วก็แย่จริง — **การวัดยืนยันทั้งสองทิศทาง**
2. **ความเหมาะกับ architecture experiment** — น้ำหนักสูงเป็นพิเศษ เพราะ deliverable วิจัย
   คือ *การเปรียบเทียบที่ตีความได้* ไม่ใช่โมเดลที่เก่งที่สุด base ที่ tokenizer ดีกว่า 30 %
   แต่ทำให้ผลการทดลองตีความไม่ได้ ถือว่าแย่กว่าสำหรับโครงการนี้
3. **ความเสี่ยง** — ให้ค่ากับ fallback ที่ล้มคนละแบบกับตัวหลัก มากกว่า fallback ที่คะแนนสูง
4. **License** — Gate B (เงื่อนไขตั้งชื่อ) กระทบชื่อ deliverable ที่เขียนไว้ใน README โดยตรง
   นี่เป็นข้อจำกัดจริงต่อการเผยแพร่ ไม่ใช่รายละเอียดทางกฎหมายที่มองข้ามได้
5. **Compute** — แยก token budget ออกจาก parameter count อย่างชัดเจน (ดู `compute_estimate.md` §2)
   OLMo แพงขึ้น 2.35× จากการ *คูณกัน* ของสองแกน ไม่ใช่จากแกนใดแกนหนึ่ง
6. **การเผยแพร่ checkpoint และ paper artifacts** — sample set ของ tokenizer screen เขียนขึ้นเองทั้งหมด
   จึงแจกพร้อม paper ได้โดยไม่ติดลิขสิทธิ์บุคคลที่สาม

---

## 6. สิ่งที่ยังพิสูจน์ไม่ได้ (รายงานตรงไปตรงมา)

- **ไม่มีการโหลด weights ใด ๆ เลย** — cache ในเครื่องมีแต่ไฟล์ `refs` ขนาด 4 KB ไม่มี weights จริง
  จึง **ไม่มี frozen Thai/English/code BPB, ไม่มี inference smoke, ไม่มีการวัด RAM/latency**
  ตามข้อ 4 ของโจทย์ ทางเลือกเดียวคือดาวน์โหลด weights หลาย GB ซึ่งขัดข้อจำกัด "ห้ามดาวน์โหลดโดยไม่จำเป็น"
  และสองตัวที่สำคัญที่สุดก็ถูก gate อยู่ดี → **fail closed**
- **ตัวเลข Llama/Gemma มาจาก mirror ไม่ใช่ต้นฉบับ** — ยังไม่ยืนยัน byte-identity
- **ตัวเลข compute ทั้งหมดเป็นการคำนวณ ไม่ใช่การวัด** — ยึดสมมติฐาน 120 TFLOPS/GPU (MFU ≈38 %)
  ถ้า MFU จริงเป็น 25 % เวลาทั้งหมดเพิ่ม ≈50 %
- **ตัวเลข routed slowdown 0.55–0.88× มาจาก AR03** ซึ่งวัดบนฮาร์ดแวร์และโมเดลอื่น
- **ยังไม่ได้อ่าน `CODE01` จริง** — ประมาณการ pilot 1–2 วันอิงจากที่บันทึกใน registry เท่านั้น
- **ยังไม่ได้อ่าน source ของ `Qwen2DecoderLayer`** — เรต "low risk" เป็นการอนุมานจาก config
- **tokenizer screen ใช้ 14 เอกสาร** ไม่ใช่ corpus จริง ค่าสัมบูรณ์จะเปลี่ยนบน SEA-PILE-v2
  (คาดว่าลำดับไม่เปลี่ยนเพราะช่องว่างกว้างมาก) ต้องวัดซ้ำบน corpus จริง ≥100 MB ก่อนล็อก token budget
- **`ROUTE_SCALE_GATE` ยังไม่ถูกแก้** — `VAL-2026-08-18-LOCAL-STATIC-AUDIT` ยังเป็น
  `blocked_pending_router_fidelity_fix` และ **ปัญหานี้ไม่ขึ้นกับการเลือก base model**
  การเลือก base ไม่ปลด block นี้

---

## 7. สิ่งที่ต้องถาม mentor ก่อนล็อกตัวเลือก

1. **ยอมรับชื่อ `Llama-ThaiLLM-1B-Base` / `Llama-ThaiLLM-1B-Instruct` ได้หรือไม่**
   ถ้ายืนยันว่าต้องเป็น `ThaiLLM-1B-Base` เฉย ๆ → **ต้องเปลี่ยนไปใช้ `Qwen2.5-1.5B` ทันที**
   นี่เป็นคำถามที่ต้องตอบก่อนคำถามอื่นทั้งหมด เพราะมันเปลี่ยนคำตอบสุดท้าย
2. **ยอมรับ 1.544 B ว่าเป็น "ประมาณ 1B" ได้หรือไม่** (เงื่อนไขของ fallback)
   ถ้าไม่ได้ และ Gate B ก็ไม่ผ่าน → เหลือแค่ Gemma 3 1B ซึ่งมีความเสี่ยงตาม §4.6
3. **ใครมีสิทธิ์ gated access ของ Meta และ Google และจะขอในนามใคร**
   (บัญชีบุคคล vs บัญชีองค์กร ส่งผลต่อสิทธิ์การเผยแพร่ checkpoint ในนามหน่วยงาน)
4. **LANTA จัดสรร A100 40 GB หรือ 80 GB** — ไม่เปลี่ยนคำตัดสิน แต่เปลี่ยน micro-batch และ wall-clock
5. **อนุมัติงบ pilot คู่ (Qwen3-0.6B + Llama-3.2-1B ≈15–20 GPU-h) หรือไม่**
   เพื่อเลี่ยงข้อโจมตีว่า extrapolate ผลจาก 0.6B ไป 1.2B
6. **การใช้ OLMo 2 เป็น replication แกนสถาปัตยกรรมบนภาษาอังกฤษอย่างเดียว ถือว่ายอมรับได้หรือไม่**
   หรือ mentor ต้องการ replication บนภาษาไทยเท่านั้น (ซึ่งจะทำให้ OLMo ตกไปทั้งหมด)
7. **ให้ block size ต้องเท่ากันข้ามทุกตระกูลหรือไม่** — ถ้าใช่ Gemma 3 (26 layers) ตกทันที
   เพราะหารด้วย 4 ไม่ลงตัว

---

## 8. ลำดับงานที่แนะนำถัดไป

1. ขอ gated access ทั้ง Meta และ Google **พร้อมกัน** (ใช้เวลารออนุมัติ)
2. ถาม mentor ข้อ 1 และ 2 ใน §7 — สองข้อนี้กำหนดคำตอบสุดท้าย
3. แก้ `ROUTE_SCALE_GATE`, `E2`, `E4` (ไม่ต้องรอคำตอบเรื่อง base — เป็นอิสระจากกัน)
4. เขียนและ validate MHAR (D2) ซึ่ง **ยังไม่มีอยู่จริง** (`E1`)
5. เมื่อได้สิทธิ์แล้ว → รัน tokenizer screen ซ้ำบน revision ต้นฉบับ ปิด Gate A
6. รัน one-A100 preflight 200 steps → เปลี่ยนตัวเลขใน `compute_estimate.md` เป็นค่าที่วัดจริง
7. ล็อก base model **หลังจาก** ขั้นตอน 2, 5 และ 6 ผ่านเท่านั้น

---

## เอกสารประกอบ

- [`candidate_matrix.csv`](candidate_matrix.csv) — 8 candidates × 54 คอลัมน์
- [`tokenizer_screen.json`](tokenizer_screen.json) — ผลวัดเต็ม + sample set ที่ freeze แล้ว
- [`architecture_port_audit.md`](architecture_port_audit.md) — การพอร์ต D1/D2 รายตระกูล
- [`compute_estimate.md`](compute_estimate.md) — memory, throughput, token budget
