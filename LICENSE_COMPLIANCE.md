# License Compliance — SEA-PILE-v2 และข้อมูลอื่น

**⚠️ เอกสารนี้ไม่ใช่คำแนะนำทางกฎหมาย** เป็นการสรุปว่าเงื่อนไขเขียนว่าอะไร
และมีคำถามอะไรที่ต้องให้ mentor / ผู้ดูแลสถาบันตัดสิน

---

## 1. เงื่อนไขที่ผูกอยู่กับ SEA-PILE-v2

| ชั้น | เงื่อนไข | ความหมายโดยย่อ |
|---|---|---|
| **ตัว dataset** | **ODC-By 1.0** (Open Data Commons Attribution) | ใช้ / แก้ไข / เผยแพร่ต่อได้ **แต่ต้องให้เครดิต** |
| **เนื้อหาข้างใน** | **CommonCrawl Terms of Use** | เป็นข้อความที่ crawl มาจากเว็บ **ลิขสิทธิ์ยังเป็นของเจ้าของเว็บเดิม** |

**จุดที่ต้องเข้าใจ:** ODC-By คุ้มครอง *ตัวฐานข้อมูล* ไม่ได้ให้สิทธิ์ใน *เนื้อหา* แต่ละชิ้น
เนื้อหาข้างในยังเป็นของ thairath, pantip, thaipbs ฯลฯ

---

## 2. เราทำอะไรกับข้อมูลบ้าง — และแต่ละอย่างมีความเสี่ยงต่างกัน

| การกระทำ | สถานะเรา | ระดับความเสี่ยง |
|---|---|---|
| ดาวน์โหลดมาใช้วิจัย | ✅ ทำแล้ว | ต่ำ — เป็นการใช้งานตามปกติของ dataset สาธารณะ |
| ทำความสะอาด / แปลงรูป | ✅ ทำแล้ว | ต่ำ |
| **เทรนโมเดล** | 🔜 กำลังจะทำ | ต่ำ–กลาง — เป็น practice มาตรฐาน (Typhoon, SEA-LION, OpenThaiGPT ทำแบบเดียวกัน) |
| **ปล่อย weight บน HuggingFace** | 🔜 ตั้งใจจะทำ | **กลาง — ต้องมี attribution** |
| **ปล่อย corpus ที่ clean แล้ว** | ❌ **ยังไม่ตัดสินใจ** | **สูงสุด — ต้องตัดสินใจก่อน** |

---

## 3. สิ่งที่ต้องทำแน่ ๆ (ทำได้เลย ไม่ต้องรอใคร)

### 3.1 Attribution — ODC-By บังคับ

ใส่ใน **model card + paper + repo README**:

```
โมเดลนี้ผ่าน continued pretraining บนข้อมูลจาก
SEA-PILE-v2 (AI Singapore), subset ภาษาไทย
เผยแพร่ภายใต้ ODC-By 1.0
https://huggingface.co/datasets/aisingapore/SEA-PILE-v2
เนื้อหาต้นทางมาจาก CommonCrawl และอยู่ภายใต้ CommonCrawl Terms of Use
```

### 3.2 บันทึก provenance ให้ครบ

`configs/data_manifest.template.json` มีช่องอยู่แล้ว — ต้องกรอกให้ครบ:
revision ที่ pin ไว้ (`77573cc84631412a781daa8e6f72cf322d4207f0`), license, hash

### 3.3 ไม่ต้องปล่อย corpus ก็ทำงานได้

`.gitignore` กัน `data/clean/` ไว้แล้ว และ **bucket rule เป็น deterministic**
→ คนอื่นสร้างข้อมูลชุดเดิมซ้ำได้จากสคริปต์ + revision ที่ pin ไว้
**ไม่ต้อง redistribute ข้อมูลเลย** ซึ่งเลี่ยงคำถามยากที่สุดไปได้ทั้งหมด

---

## 4. ❓ คำถามที่ต้องให้ mentor ตัดสิน

| # | คำถาม | ทำไมสำคัญ |
|---|---|---|
| 1 | **จะปล่อย corpus ที่ clean แล้วหรือไม่** | ถ้าไม่ปล่อย เรื่อง license ง่ายลงมาก |
| 2 | สถาบัน/AIAT มีนโยบายเรื่องข้อมูล CommonCrawl อยู่แล้วหรือไม่ | Typhoon / OpenThaiGPT เจอคำถามนี้มาก่อน น่าจะมีแนวปฏิบัติอยู่แล้ว |
| 3 | **PDPA** — ข้อมูลเว็บไทยอาจมีข้อมูลส่วนบุคคล | dataset card เขียนเองว่า *"harmful, toxic, or private content may still pass through"* และเรา**ยังไม่ได้ทำ PII filter** |
| 4 | จะปล่อย weight ภายใต้ license อะไร | Apache-2.0 ของ Qwen3 ไม่บังคับ แต่เราควรเลือกให้ชัด |
| 5 | ต้องขออนุญาต AI Singapore เพิ่มหรือไม่ | ตาม ODC-By ไม่ต้อง แต่การแจ้งให้ทราบเป็นมารยาทที่ดี |

---

## 5. ⚠️ ช่องโหว่ที่ยังเปิดอยู่ ณ วันนี้

| ช่องโหว่ | สถานะ | ความเร่งด่วน |
|---|---|---|
| **PII filter** | ✅ **รันแล้ว 2026-08-19** — redact 108,237 docs รวม **เลขบัตรประชาชน 420 ใบ** | ปิดแล้ว |
| PII: ชื่อคน / ที่อยู่ / วันเกิด | ❌ ยังจับไม่ได้ (regex ไม่ใช่ NER) | **กลาง** — ยังเหลือความเสี่ยง PDPA |
| **ยังไม่ decontaminate** กับ ThaiExam/M3Exam | ❌ ยังไม่ทำ | สูง (เรื่องความถูกต้องของผล ไม่ใช่ license) |
| ยังไม่ได้ยืนยันว่า Mangosteen / ThaiLLM Repo license เป็นอย่างไร | ❌ | ต่ำ — ยังไม่ได้ใช้ |

---

## 6. สรุปสั้นสำหรับคุยกับ mentor

> **"ข้อมูลเป็น ODC-By 1.0 ครับ ใช้และเทรนได้ ขอแค่ให้เครดิต
> เราตั้งใจ**ไม่**ปล่อยตัว corpus — ปล่อยแค่สคริปต์กับ revision ที่ pin ไว้
> ซึ่งคนอื่นสร้างข้อมูลชุดเดียวกันซ้ำได้เป๊ะ จึงไม่ต้องแตะคำถามเรื่อง redistribution เลย
>
> เรื่อง **PDPA** เราทำ PII redaction ไปแล้วครับ ลบเลขบัตรประชาชนที่ผ่าน checksum จริง 420 ใบ
> เบอร์โทร 18,127 บัญชีธนาคาร 4,163 และ Line ID อีกแสนกว่ารายการ
> แต่ตัวจับเป็น regex จึง**ยังจับชื่อคน ที่อยู่ และวันเกิดไม่ได้**
> อยากทราบว่าสถาบันมีแนวปฏิบัติเดิมเรื่องนี้อยู่แล้วหรือเปล่าครับ"**

---

## 7. แหล่งอ้างอิง

- [SEA-PILE-v2 dataset card](https://huggingface.co/datasets/aisingapore/SEA-PILE-v2)
- [ODC-By 1.0 ฉบับเต็ม](https://opendatacommons.org/licenses/by/1-0/)
- [CommonCrawl Terms of Use](https://commoncrawl.org/terms-of-use)
- [PDPA ประเทศไทย (PDPC)](https://www.pdpc.or.th/)

---

## 8. Replay data ที่เลือกใช้จริง (2026-08-20)

| บทบาท | แหล่ง | revision | เงื่อนไข |
|---|---|---|---|
| English | FineWeb-Edu `sample/10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By 1.0 + CommonCrawl ToU |
| Math | FineMath `finemath-4plus` | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By 1.0 + CommonCrawl ToU |
| Code | GitHub Code Clean | `c48d40f9e70f0196f8236901ee35807f7d6c44c0` | dataset card Apache-2.0; source file ใช้ license ต่อ repository |

Code pipeline เก็บ `license` ไว้ทุก record และยอมรับเฉพาะ `mit`, `apache-2.0`,
`bsd-2-clause`, `bsd-3-clause` เท่านั้น อย่างไรก็ตาม metadata นี้ยังไม่ได้ตรวจย้อนกับ upstream
repository ทั้ง 700,129 records จึงต้องรายงานเป็นข้อจำกัด ไม่ใช่อ้างว่า license risk เป็นศูนย์
