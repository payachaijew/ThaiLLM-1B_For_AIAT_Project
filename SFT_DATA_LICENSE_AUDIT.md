# Thai SFT Data — สำรวจและตรวจสัญญาอนุญาต

**วันที่:** 2026-08-23
**สถานะ:** หลักฐานจากแหล่งต้นฉบับ · `scientific_evidence_allowed=false` (ยังไม่ได้เทรนอะไร)
**วิธี:** ค้น HuggingFace API ด้วย 6 คำค้น แล้วเปิด dataset card ของทุกตัวที่เข้ารอบ
อ่านเงื่อนไขจาก LICENSE ต้นทางโดยตรง ไม่เชื่อ license tag บนหน้า repo

> **หลักที่ใช้ตลอดเอกสารนี้:** license tag บน HuggingFace คือสิ่งที่**คนอัปโหลดพิมพ์ลงไป**
> ไม่ใช่สิ่งที่ข้อมูลอนุญาตจริง ถ้าข้อมูลสร้างจากโมเดลอื่น เงื่อนไขของโมเดลต้นทาง
> **ติดมากับ output เสมอ** ไม่ว่าคนอัปโหลดจะติดป้ายว่าอะไร

---

## 1. ข้อค้นพบหลัก — การเลือก Qwen เป็น base ปลดล็อกข้อมูลที่โมเดลอื่นใช้ไม่ได้

`airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k` (118,898 ตัวอย่าง)
สร้างด้วย **Qwen2-72B-Instruct** ซึ่งอยู่ภายใต้ **Tongyi Qianwen License** ที่ระบุว่า:

> "You can not use the Materials or any output therefrom to improve any other large
> language model (excluding Tongyi Qianwen or derivative works thereof)."

อ่านตามตัวอักษร: ห้ามเอา output ไปพัฒนา LLM **อื่น** — **ยกเว้น Qwen และงานสืบทอดของ Qwen**

**โมเดลของเราคืองานสืบทอดของ Qwen3-1.7B-Base** ข้อยกเว้นจึงครอบคลุมเรา
ถ้าเราเลือก Llama หรือ Gemma เป็น base ชุดข้อมูลนี้จะใช้ไม่ได้เลย

⚠️ **นี่คือการอ่านเงื่อนไข ไม่ใช่ความเห็นทางกฎหมาย** ต้องให้อาจารย์หรือฝ่ายกฎหมาย
ของสถาบันยืนยันก่อนใช้จริง โดยเฉพาะประเด็นว่า "derivative works of Tongyi Qianwen"
ครอบคลุมงานที่ต่อยอดจาก Qwen3 (ซึ่งเป็น Apache-2.0) หรือไม่

สังเกตว่า AIResearch ติดป้ายชุดนี้ว่า **MIT** ซึ่งเป็นป้ายของงานที่พวกเขาสร้าง
แต่**ไม่ได้ลบเงื่อนไขของ Qwen ที่ติดมากับ output** — เป็นตัวอย่างตรง ๆ ของหลักด้านบน

---

## 2. ตารางสรุป

| ชุดข้อมูล | ขนาด | ที่มา | ป้าย license | ใช้ได้จริงไหม |
|---|---|---|---|---|
| `airesearch/WangchanThaiInstruct` | 75,014 | **มนุษย์เขียน 100%** | cc-by-sa-4.0 **+ NC บางแถว** | ⚠️ ต้องกรองรายแถว |
| `airesearch/wangchanx-seed-free-...-120k` | 118,898 | Qwen2-72B-Instruct | MIT (ทับเงื่อนไข Qwen) | ✅ **เฉพาะเพราะ base เป็น Qwen** |
| `CohereForAI/aya_dataset` | 65 ภาษา รวมไทย | **มนุษย์เขียน** | apache-2.0 | ✅ สะอาดที่สุด |
| `airesearch/WangchanX-Legal-ThaiCCL-RAG` | 10K–100K | กฎหมายไทย | mit | ✅ เฉพาะโดเมนกฎหมาย |
| `pythainlp/thai-local-instruction-v2` | 10K–100K | ความรู้ท้องถิ่นไทย | cc-by-4.0 | ✅ |
| `OpenAssistant/oasst2` (คัดเฉพาะไทย) | น้อย | **มนุษย์เขียน** | apache-2.0 | ✅ ปริมาณน้อย |
| `airesearch/WangchanX-FLAN-v6.1` | 3.62M | รวม 27+ แหล่ง | **other** | ⚠️ license รายแถว ต้องกรอง |
| `Suraponn/thai_instruction_sft` | 132K | **ไม่ระบุที่มาเลย** | apache-2.0 | ❌ อ้างไม่ได้ |
| `iapp/Thai-R1-Distill-SFT` | 10,000 | ไม่ระบุโมเดล | **ไม่มี license** | ❌ ไม่มี license = ไม่ได้รับอนุญาต |
| `SEACrowd/thai_alpaca` | — | แปลจาก Alpaca | cc-by-**nc**-4.0 | ❌ NC + ต้นทาง GPT |
| `saillab/alpaca-thai-cleaned` | 10K–100K | แปลจาก Alpaca | **ไม่มี** | ❌ ต้นทาง GPT |
| `ping98k/lmsys-chat-1m-thai-filtered` | <1K | LMSYS-Chat-1M | ไม่มี | ❌ output จากหลายโมเดลปนกัน |

---

## 3. สามกับดักที่เจอ

**ก. license รายแถว ไม่ใช่รายชุด**

`WangchanThaiInstruct` เป็นข้อมูลที่**มนุษย์เขียนทั้งหมด** ซึ่งมีค่ามาก แต่การ์ดระบุว่า
"each row has its license according to its source" และมีบางแถวเป็น **NC (ห้ามใช้เชิงพาณิชย์)**
ถ้าโหลดทั้งชุดมาเทรนแล้วปล่อยโมเดลแบบเปิด **เราจะละเมิดเงื่อนไขของแถวที่เป็น NC โดยไม่รู้ตัว**
ต้องกรองด้วยฟิลด์ license ก่อนใช้เสมอ

`WangchanX-FLAN-v6.1` หนักกว่านั้น — รวม 27+ แหล่งที่มี license ต่างกัน 7 แบบ
(cc-by-sa-4.0, MIT, Apache-2.0, cc-by-4.0, CC0-1.0, cc-by-sa-3.0, **LGPL-3.0**)

**ข. "ไม่มี license" ไม่ได้แปลว่า "ใช้ได้อิสระ"**

`iapp/Thai-R1-Distill-SFT` และ `saillab/alpaca-thai-cleaned` ไม่ระบุ license เลย
ตามค่าเริ่มต้นของลิขสิทธิ์ **การไม่ระบุคือไม่ได้ให้สิทธิ์** ไม่ใช่ให้สิทธิ์ทั้งหมด

**ค. ชุดที่แปลมาจาก Alpaca พา ToS ของ OpenAI ติดมาด้วย**

Alpaca ดั้งเดิมสร้างจาก `text-davinci-003` ของ OpenAI ซึ่ง ToS ห้ามใช้ output
ไปพัฒนาโมเดลที่แข่งกับ OpenAI การแปลเป็นภาษาไทยไม่ได้ลบเงื่อนไขนี้
`SEACrowd/thai_alpaca` ยังติด **cc-by-nc-4.0** ซ้อนอีกชั้น

---

## 4. ชุดที่แนะนำ — พร้อมตัวเลขจริงที่นับแล้ว

**แก้ไขจากฉบับแรก:** ตอนแรกผมประเมินว่าจะได้ 150,000–200,000 ตัวอย่าง
หลังนับ license รายแถวจริงแล้ว ตัวเลขเปลี่ยนไปมาก

| ชุด | แถวทั้งหมด | **ใช้ได้จริง** | ที่มา |
|---|---|---|---|
| seed-free 120k | 118,898 | **118,898** ⚠️ | Qwen2-72B — ขึ้นกับการตีความ |
| thai-local-instruction-v2 | 39,829 | **39,829** | cc-by-4.0 |
| WangchanThaiInstruct | 75,014 | **5,014** | มนุษย์เขียน แต่ 86.6% เป็น NC |
| Aya (เฉพาะไทย) | 724 จาก 202,362 | **724** | มนุษย์เขียน apache-2.0 |
| **รวม** | | **~164,000** | |

### ตัวเลขที่ต้องเน้น: WangchanThaiInstruct ใช้ได้แค่ 6.7%

นับจาก datasets-server statistics ทั้ง 4 split:

| split | รวม | cc-by-sa-4.0 | **cc-by-nc-4.0** |
|---|---|---|---|
| default/train | 32,207 | 5,014 (15.6%) | 27,193 |
| default/test | 7,793 | **0** | 7,793 |
| paper/train | 28,098 | 5,014 (17.8%) | 23,084 |
| paper/test | 6,916 | **0** | 6,916 |

**86.6% เป็น NC** และ 5,014 แถวที่ใช้ได้ยังปรากฏซ้ำในทั้ง config `default` และ `paper`
**ของจริงจึงเหลือ 5,014 แถว ไม่ใช่ 10,028 และไม่ใช่ 75,014**

ชุดข้อมูลไทยที่มนุษย์เขียนชิ้นสำคัญที่สุดที่มีอยู่ ให้เราใช้ได้ **6.7%**

### ความเสี่ยงที่ต้องรู้

**118,898 จาก 164,000 (72%) ขึ้นอยู่กับการตีความ Tongyi Qianwen ข้อเดียว**
ถ้าอาจารย์หรือฝ่ายกฎหมายตีความว่างานที่ต่อจาก Qwen3 ไม่นับเป็น "derivative works of
Tongyi Qianwen" **เราจะเหลือข้อมูลราว 45,000 ตัวอย่าง** ซึ่งยังพอเทรน SFT ได้
แต่บางลงมากและเสียความหลากหลายของประเภทงานไป

---

## 5. DPO — ไม่มีข้อมูลไทยที่ใช้ได้เลย

ค้นด้วย 4 คำค้น (`thai dpo`, `thai preference`, `thai rlhf`, `thai reward`) พบ 7 ชุด
**ไม่มีชุดไหนใช้ได้**

| ชุด | ขนาด | ปัญหา |
|---|---|---|
| `SEACrowd/thai_hh_rlhf` | 161K | แปลด้วย Google Translate จาก Anthropic/hh-rlhf |
| `siamaids/Magpie-DPO-Thai-76K` | 76K | **ไม่มี license** · วิธี Magpie ดึงจากโมเดล aligned |
| `Stalemartyr/mt-thai-dpo-v1..v5` | 10K–1M | **ไม่มี license** · `mt` = machine translation |
| `iapp/dpo_thai_tutorial` | <1K | apache-2.0 แต่เป็นตัวอย่างสอน ไม่ใช่ชุดใช้งาน |

`thai_hh_rlhf` น่าสนใจที่สุดเพราะมี license (mit) แต่การ์ดของมันเองเตือนไว้ว่า
ข้อมูลนี้ "are not meant for supervised training of dialogue agents" และ
"training dialogue agents on these data is likely to lead to harmful models"
— เพราะ hh-rlhf เป็นข้อมูล red-teaming ที่ฝั่ง rejected **จงใจให้เป็นคำตอบที่เป็นอันตราย**
เอามาทำ DPO ตรง ๆ จะได้โมเดลที่แย่ลง ไม่ใช่ดีขึ้น

และการแปลด้วยเครื่องเป็นปัญหาซ้อนอีกชั้น — **preference ขึ้นกับความละเอียดอ่อนของภาษา
ซึ่งเป็นสิ่งแรกที่หายไปตอนแปลด้วยเครื่อง** คู่ chosen/rejected ที่ต่างกันด้วยน้ำเสียง
หรือความสุภาพ จะกลายเป็นคู่ที่แยกไม่ออกหลังแปล

### ทางเลือกสำหรับ DPO

1. **ข้ามไปก่อนใน v1** — ปล่อย SFT-only ซึ่งก็ "ตอบรู้เรื่อง" แล้ว DPO เป็นการขัดเงา
   ไม่ใช่ขั้นที่ทำให้มันคุยได้ **แนะนำทางนี้**
2. **สร้างเอง** — ใช้โมเดลที่ได้จาก SFT สร้างคำตอบหลายแบบต่อหนึ่งคำสั่ง แล้วให้โมเดล
   ตัดสิน เงื่อนไขคือ **license ของโมเดลผู้ตัดสินจะติดมากับผลลัพธ์** เหมือนกรณี Qwen
   ถ้าใช้ Qwen เป็นผู้ตัดสินก็อยู่ในข้อยกเว้นเดียวกัน
3. **ให้คนไทยจริงตัดสิน** — คุณภาพดีที่สุด แต่ต้องใช้คนและเวลา อาจทำได้เฉพาะชุดเล็ก
   ไว้ใช้เป็นชุดวัดผลมากกว่าชุดเทรน

---

## 6. เครื่องมือที่สร้างแล้ว

| ไฟล์ | ทำอะไร |
|---|---|
| `sft/sources.py` | ทะเบียนแหล่งข้อมูล + ตัวกรอง license พร้อมตัวเลขที่วัดแล้ว |
| `sft/build_sft.py` | กรอง license รายแถว → decontamination → แบ่ง held-out |

**ตัวกรอง license** ปฏิเสธทุกอย่างที่ไม่อยู่ในรายการอนุญาต รวมถึงค่าว่างและ `other`
เพราะ **การไม่ระบุ license คือไม่ได้ให้สิทธิ์**

**decontamination** ใช้ benchmark index เดิม (13.7M n-gram, `BENCH-NGRAM-V1`) แต่เปลี่ยนเป็น
**stride 1 ทั้งสองฝั่ง** ต่างจากตอนสแกน corpus ที่ใช้ stride 8 เพราะข้อมูล instruct สั้นกว่ามาก
และมีจำนวนน้อยกว่าสองระดับ จึงตรวจได้ทุกตำแหน่ง **ความเสี่ยง contamination สูงกว่า
เว็บเท็กซ์มาก เพราะชุด instruct ประกอบขึ้นจากข้อสอบและคู่ถาม-ตอบ ซึ่งเป็นสิ่งเดียวกับ
ที่ benchmark ทำมาจาก**

ทดสอบด้วย `--limit 60` แล้ว: **จับ contamination ได้ 1 รายการจาก 60 แถวของ seed-free**
(12 n-gram hits ข้อความเกี่ยวกับสถาบันเทคโนโลยีจิตรลดาที่มาจาก Wikipedia ไทย)
ถ้าอัตรานี้คงอยู่ในชุดเต็ม จะมีราว 2,000 แถวที่ต้องตัดออก

**held-out** ใช้ `HELDOUT-BUCKET-V1` ตัวเดียวกับ corpus โดยเจตนา — bucket คำนวณจากตัวข้อความ
อย่างเดียว ตัวอย่างหนึ่ง ๆ จึงไม่มีทางอยู่ใน SFT training และ corpus evaluation พร้อมกัน

**ข้อควรระวังของ `--limit`:** Aya เรียงตามภาษา แถวไทยไม่ได้อยู่ต้นไฟล์ การใส่ `--limit`
จึงได้ 0 แถวจาก Aya ซึ่งถูกต้องตามที่สั่ง ไม่ใช่บั๊ก แต่รันเต็มเท่านั้นที่ได้ครบ

---

## 7. สิ่งที่ต้องทำต่อ

| # | งาน | สถานะ |
|---|---|---|
| 1 | **ให้อาจารย์ยืนยันการตีความ Tongyi Qianwen** | ⛔ รออยู่ — ชี้เป็นชี้ตาย 72% ของข้อมูล |
| 2 | กรอง license รายแถว | ✅ `sft/build_sft.py` |
| 3 | decontamination ชุด SFT | ✅ `sft/build_sft.py` |
| 4 | ขยาย held-out rule ให้ครอบคลุม SFT | ✅ ใช้ `HELDOUT-BUCKET-V1` ร่วมกัน |
| 5 | ข้อมูล DPO | ✅ สำรวจแล้ว — ไม่มีที่ใช้ได้ แนะนำข้ามไปก่อน |
| 6 | รัน `build_sft.py` แบบเต็ม | ⏳ รอข้อ 1 ก่อน จะได้ไม่ต้องรันซ้ำ |
| 7 | เลือก chat template | ⏳ ยังไม่ได้ตัดสิน |
| 8 | เขียน `train_sft.py` | ⏳ |

**ข้อ 6 รอข้อ 1 โดยตั้งใจ** — ถ้าการตีความไม่ผ่าน แหล่งข้อมูลจะเปลี่ยนไปทั้งชุด
รันตอนนี้ก็ต้องรันใหม่อยู่ดี

---

## 8. แหล่งอ้างอิง

- [WangchanThaiInstruct](https://huggingface.co/datasets/airesearch/WangchanThaiInstruct)
- [wangchanx-seed-free-synthetic-instruct-thai-120k](https://huggingface.co/datasets/airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k)
- [Qwen2-72B-Instruct LICENSE](https://huggingface.co/Qwen/Qwen2-72B-Instruct/raw/main/LICENSE)
- [Aya Dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset)
- [WangchanX-FLAN-v6.1](https://huggingface.co/datasets/airesearch/WangchanX-FLAN-v6.1)
- [Suraponn/thai_instruction_sft](https://huggingface.co/datasets/Suraponn/thai_instruction_sft)
- [iapp/Thai-R1-Distill-SFT](https://huggingface.co/datasets/iapp/Thai-R1-Distill-SFT)
- [SEACrowd/thai_hh_rlhf](https://huggingface.co/datasets/SEACrowd/thai_hh_rlhf)
- [pythainlp/thai-local-instruction-v2](https://huggingface.co/datasets/pythainlp/thai-local-instruction-v2)
- ตัวเลข license รายแถว: datasets-server statistics endpoint, ดึงเมื่อ 2026-08-23
