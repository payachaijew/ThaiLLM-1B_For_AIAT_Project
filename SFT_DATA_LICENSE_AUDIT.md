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

## 4. ชุดที่แนะนำ

**ระดับ A — สะอาด ใช้ได้ทันที**

| ชุด | ได้อะไร |
|---|---|
| Aya (เฉพาะไทย) | มนุษย์เขียน Apache-2.0 ไม่มีเงื่อนไขแฝง |
| seed-free 120k | ปริมาณมากที่สุดที่ใช้ได้ · ครอบคลุม 5 ประเภทงาน |
| WangchanThaiInstruct (**กรองเอาเฉพาะ cc-by-sa**) | มนุษย์เขียน · 4 โดเมน 7 ประเภทงาน |
| thai-local-instruction-v2 | ความรู้ท้องถิ่นไทยที่โมเดลต่างชาติไม่มี |
| OASST2 (คัดไทย) | บทสนทนาหลายเทิร์นที่มนุษย์เขียน |

รวมประมาณ **150,000–200,000 ตัวอย่าง** ซึ่งเพียงพอสำหรับ SFT ของโมเดล 1.7B
(งานส่วนใหญ่ใช้ 10K–100K)

**ระดับ B — ใช้ได้ถ้ากรอง** WangchanX-FLAN-v6.1, Legal-ThaiCCL-RAG

**ระดับ C — ไม่แนะนำ** ทุกตัวที่แปลจาก Alpaca, ชุดที่ไม่ระบุที่มา, ชุดที่ไม่มี license

---

## 5. สิ่งที่ต้องทำต่อ

1. **ให้อาจารย์ยืนยันการตีความ Tongyi Qianwen** — ว่างานที่ต่อยอดจาก Qwen3
   นับเป็น "derivative works of Tongyi Qianwen" หรือไม่ **ถ้าไม่ใช่ ชุด 120k ตกทันที**
   และเราจะเหลือข้อมูลไม่ถึงครึ่ง
2. **เขียนสคริปต์กรอง license รายแถว** สำหรับ WangchanThaiInstruct และ FLAN
   พร้อมนับจำนวนที่ถูกตัดออก แบบเดียวกับที่ทำใน `phase0/clean_corpus.py`
3. **decontamination รอบสอง** — ข้อมูล SFT ต้องผ่านการตรวจซ้ำกับ benchmark
   ด้วย `phase0/build_benchmark_index.py` เหมือน corpus เพราะชุด instruct
   มักมีข้อสอบปนอยู่โดยตรง
4. **ขยาย held-out rule ให้ครอบคลุม SFT** — `HELDOUT-BUCKET-V1` ใช้กับ corpus
   ยังไม่ครอบคลุมข้อมูล instruct
5. **ยังไม่มีข้อมูล preference สำหรับ DPO** — ที่เจอมีแต่ `gamepollakrit/WangchanThaiInstruct_DPO`
   (4 downloads ไม่มีเอกสาร) ถ้าจะทำ DPO อาจต้องสร้างเอง

---

## 6. แหล่งอ้างอิง

- [WangchanThaiInstruct](https://huggingface.co/datasets/airesearch/WangchanThaiInstruct)
- [wangchanx-seed-free-synthetic-instruct-thai-120k](https://huggingface.co/datasets/airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k)
- [Qwen2-72B-Instruct LICENSE](https://huggingface.co/Qwen/Qwen2-72B-Instruct/raw/main/LICENSE)
- [Aya Dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset)
- [WangchanX-FLAN-v6.1](https://huggingface.co/datasets/airesearch/WangchanX-FLAN-v6.1)
- [Suraponn/thai_instruction_sft](https://huggingface.co/datasets/Suraponn/thai_instruction_sft)
- [iapp/Thai-R1-Distill-SFT](https://huggingface.co/datasets/iapp/Thai-R1-Distill-SFT)
