# Model Card — ThaiLLM-1B-Base  🚧 ร่าง

> **⚠️ ยังไม่มีโมเดล** เอกสารนี้เป็นโครงที่กรอกเฉพาะส่วนที่รู้แน่แล้ว
> ช่องที่เป็น `TBD` ต้องรอผลจริงหลังเทรน **ห้ามเผยแพร่จนกว่าจะกรอกครบ**

---

## Model Details

| | |
|---|---|
| Base model | [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) rev `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| Architecture | dense decoder-only, 28 layers, hidden 2048, GQA 16Q/8KV, vocab 151,936 |
| Parameters | ~1.7B |
| Context | 32,768 (เทรนที่ 8,192) |
| Tokenizer | **ไม่ขยาย vocab** — ใช้ของ Qwen3 เดิม |
| License | Apache-2.0 (สืบทอดจาก base) |
| Training | continued pretraining · TBD tokens |

## Intended Use

**ตั้งใจให้ใช้:** งานวิจัยภาษาไทย · เป็น base สำหรับ fine-tune ต่อ · ศึกษา language adaptation

**❌ ไม่เหมาะกับ:**
- ใช้ตรง ๆ เป็นแชตบอต (นี่คือ **base model** ไม่ได้ผ่าน instruction tuning)
- งานที่ต้องการความถูกต้องสูงโดยไม่มีคนตรวจ — การแพทย์ กฎหมาย การเงิน
- งานที่ตัดสินใจเกี่ยวกับบุคคล

---

## Training Data

| | |
|---|---|
| แหล่งไทย | [SEA-PILE-v2](https://huggingface.co/datasets/aisingapore/SEA-PILE-v2) subset `th` rev `77573cc84631412a781daa8e6f72cf322d4207f0` |
| License ข้อมูล | **ODC-By 1.0** + CommonCrawl Terms of Use |
| ที่มาเนื้อหา | CommonCrawl 2020–2022+ (เว็บไทยสาธารณะ) |
| replay | อังกฤษ 35% · code 10% · math 5% *(แหล่ง TBD)* |

### การทำความสะอาด — ตัวเลขจริง

| | |
|---|---|
| เอกสารตั้งต้น | **16,428,048** |
| เหลือหลัง clean | **4,567,214 (27.8%)** · 29.53 GB · ~5.85B tokens |

| กฎที่ตัด | เอกสาร | % ของ corpus (bytes) |
|---|---|---|
| ซ้ำแบบตรงตัว | 2,008,876 | 15.0% |
| สั้นกว่า 500 ตัวอักษร | 8,851,360 | 12.4% |
| เนื้อหาพนัน | 735,393 | 10.6% |
| หวย | 82,039 | 1.6% |
| กันไว้เป็นชุดวัด | 165,175 | 1.0% |
| adult | 12,216 | 0.3% |

### ⚠️ ข้อจำกัดของข้อมูลที่เรารู้ตัว

- **ไม่ได้ทำ near-duplicate dedup** ทำแค่ซ้ำแบบตรงตัว — ยังมี near-duplicate เหลืออยู่
- **ตัวกรองเป็น regex ไม่ใช่ classifier** ตัดเกินและตัดไม่หมดทั้งคู่
  ตรวจตัวอย่างจริงแล้วพบว่า **ตัวกรองหวยมี false positive** (เว็บแฟรนไชส์ เว็บกีฬา ที่บังเอิญมีคำว่า "หวย")
- **PII redaction รันแล้ว** — แก้ 108,237 เอกสาร (2.37%): Line ID 104,121 · social handle 18,578 ·
  เบอร์โทร 18,127 · บัญชีธนาคาร 4,163 · อีเมล 575 · **เลขบัตรประชาชน 420** (ตรวจ checksum ก่อนลบ)
  ลบไปเพียง ~1 MB จาก 29.5 GB เพราะแทนที่ตัวเลข ไม่ทิ้งเอกสาร
- **PII detector จับชื่อคน ที่อยู่ วันเกิด ไม่ได้** — เป็น regex ไม่ใช่ NER model
- **ยังไม่ได้ decontaminate** กับ ThaiExam / M3Exam
- dataset card ของต้นทางระบุเองว่า *"harmful, toxic, or private content may still pass through"*

---

## Evaluation

ชุดประเมิน **แช่แข็งไว้ก่อนวัดผลใด ๆ** — `THAILLM-EVAL-FROZEN-V1`
sha256 `1fae436e05fa99cee8b5b878e72f45c95aa8d51b3c1177e49c1c5a98c565cb19`

### Baseline ก่อนเทรน (Thai BPB, TH-WEB-HELDOUT 2,000 docs)

held-out set sha256 `48aaf8623e7f1a7ece19cbfe28f3eeb5ae4f35f6870213b91ed6ca771651c631`
สร้างจาก shard 0/13/27/40/53 แบบกระจายทั้งช่วงเวลา

| model | Thai BPB |
|---|---|
| Qwen3-1.7B-Base *(base ของเรา)* | **0.454218** |
| Qwen3-0.6B-Base *(sanity check)* | 0.521386 |

**headroom = 16.8% relative** เทียบ Sailor2 = **8.4 เท่าของเกณฑ์ 2%**
⚠️ เป็นขอบบน — Sailor2 อาจเคยเห็นเอกสารใน held-out เพราะทั้งคู่มาจาก CommonCrawl
| Sailor2-1B *(อ้างอิง: ผ่าน Thai CPT 500B tokens)* | **0.378051** |

### หลังเทรน

| | ก่อน | หลัง |
|---|---|---|
| ThaiExam / M3Exam-th / Belebele-th | TBD | TBD |
| MMLU / HellaSwag / ARC *(retention)* | TBD | TBD |

---

## Risks, Bias and Limitations

### ✋ อ่านก่อนใช้

1. **โมเดลอาจสร้างข้อมูลเท็จ** — ไม่มีกลไกตรวจสอบข้อเท็จจริง
2. **อาจมีข้อมูลส่วนบุคคลหลงเหลือ** — เทรนจากเว็บสาธารณะที่กรองอัตโนมัติ ไม่มีคนอ่านทุกเอกสาร
3. **อาจสร้างเนื้อหาไม่เหมาะสม** — ตัวกรองเป็น regex ไม่ได้จับได้ทุกกรณี
4. **อคติ** — สะท้อนอคติของเว็บไทย ข้อมูลกระจุกที่ข่าว/ไลฟ์สไตล์/ฟอรัม เนื้อหาวิชาการน้อย
5. **ข้อมูลเก่า** — crawl ปี 2020–2022+ ไม่รู้เหตุการณ์หลังจากนั้น
6. **ตอบปนภาษา** — โมเดลขนาดนี้มักตอบไทยปนอังกฤษ

### ⚠️ ยังไม่ได้ประเมินความปลอดภัย

**ยังไม่ได้รัน safety evaluation ใด ๆ** *(วางแผนใช้ ThaiSafetyBench, arXiv:2603.04992)*
→ ห้ามใช้ในงานที่ผู้ใช้ปลายทางเป็นบุคคลทั่วไป จนกว่าจะประเมินเสร็จ

---

## Attribution

```
โมเดลนี้ผ่าน continued pretraining บนข้อมูลจาก SEA-PILE-v2 (AI Singapore)
subset ภาษาไทย เผยแพร่ภายใต้ ODC-By 1.0
https://huggingface.co/datasets/aisingapore/SEA-PILE-v2
เนื้อหาต้นทางมาจาก CommonCrawl ภายใต้ CommonCrawl Terms of Use

Base model: Qwen3-1.7B-Base (Alibaba Cloud), Apache-2.0
```

## Reproducibility

ไม่ได้เผยแพร่ตัว corpus **แต่สร้างซ้ำได้เป๊ะ** จากสคริปต์ใน repo + revision ที่ pin ไว้
(กฎเลือกชุดวัดเป็น deterministic hash bucket — `phase0/heldout_rule.py`)

## Contact / Reporting

พบปัญหา เนื้อหาไม่เหมาะสม หรือข้อมูลส่วนบุคคลในผลลัพธ์ของโมเดล แจ้งได้ที่ *TBD*
