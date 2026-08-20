# ThaiLLM-1B

**Continued pretraining ของ Qwen3-1.7B ให้เป็นโมเดลภาษาไทยแบบ generalist**

เป้าหมายคือโมเดลที่ใช้ภาษาไทยได้ดีขึ้นอย่างมีนัยสำคัญ **โดยไม่ทิ้งความสามารถภาษาอังกฤษ
การให้เหตุผล และโค้ด** ที่ base model มีอยู่แล้ว

> **กฎของ repo นี้:** ทุกตัวเลขมาพร้อมสคริปต์ที่สร้างมัน, revision hash ของโมเดลและข้อมูลที่ใช้
> และข้อจำกัดที่รู้ตัว — รวมถึง **บั๊กที่เจอระหว่างทาง** ซึ่งบันทึกไว้ครบใน
> [`validation/VALIDATION_OUTPUT_LOG.md`](validation/VALIDATION_OUTPUT_LOG.md)

---

## สถานะ: Phase 0 เสร็จ — **ยังไม่ได้เทรนโมเดล**

- [x] คัดเลือก base model + tokenizer screen
- [x] แช่แข็ง eval suite **ก่อน** วัด baseline
- [x] สร้าง held-out set + วัด baseline + ประเมิน headroom
- [x] Clean corpus + PII redaction
- [ ] decontamination กับ ThaiExam / M3Exam
- [ ] downstream accuracy (lm-evaluation-harness)
- [ ] **ยังไม่ได้เทรนโมเดลจริง**

---

## ทำไมถึงเชื่อว่าคุ้มที่จะเทรน

วัด **ก่อน** ใช้ GPU แม้แต่ชั่วโมงเดียว บนชุด held-out ที่แช่แข็งไว้ (2,000 docs)

| โมเดล | พารามิเตอร์ | ผ่าน Thai CPT | **Thai BPB** |
|---|---|---|---|
| Sailor2-1B | 0.99B | ✅ 500B tokens | **0.378** |
| **Qwen3-1.7B-Base** *(base ของเรา)* | 1.72B | ❌ | **0.454** |
| Qwen3-0.6B-Base | 0.60B | ❌ | 0.521 |

โมเดลที่**เล็กกว่า 42%** แต่ผ่าน Thai CPT มา ทำได้**ดีกว่า 16.8%**
→ ที่สเกลนี้ ตัวจำกัดคือ **ปริมาณภาษาไทยที่เคยเห็น ไม่ใช่ขนาดโมเดล**

> ⚠️ **ตัวเลขนี้เป็นขอบบน** — Sailor2 อาจเคยเห็นเอกสารในชุดวัดของเรา เพราะทั้งคู่มาจาก
> CommonCrawl การยืนยันต้องใช้ held-out จากข้อความที่เผยแพร่หลัง cutoff ของ Sailor2

---

## Base model: Qwen3-1.7B-Base

| | |
|---|---|
| License | **Apache-2.0** · ไม่ gated · ไม่บังคับชื่อ derivative |
| Architecture | **dense decoder** 28 ชั้น · hidden 2048 · GQA 16Q/8KV · vocab 151,936 |
| Pretraining | 36T tokens · 119 ภาษา |
| Tokenizer | **ไม่ขยาย vocab** ตามแนวทาง Typhoon 2 |

**ข้อแลกเปลี่ยนที่ยอมรับอย่างรู้ตัว:** Qwen มี Thai fertility 1.838 chars/token เทียบกับ
Gemma-4 ที่ 2.833 → **ข้อความไทยชุดเดียวกันกิน compute มากกว่า ~54%**
เรายอมจ่ายเพื่อแลกกับ dense architecture ที่ recipe ทุกอันใช้ได้ตรง ๆ และ tooling ที่สมบูรณ์
รายละเอียดใน [`base_selection/`](base_selection/)

---

## Corpus

**SEA-PILE-v2** subset `th` · rev `77573cc8...` · ODC-By 1.0 + CommonCrawl ToU

| | |
|---|---|
| เอกสารตั้งต้น | **16,428,048** |
| เหลือหลัง clean | **4,567,214 (27.8%)** · 29.5 GB · **~5.85B tokens** |

**ตัดออกด้วยกฎที่บันทึกไว้ทุกข้อ:**

| กฎ | % ของ corpus (bytes) |
|---|---|
| ซ้ำแบบตรงตัว | 15.0% |
| สั้นกว่า 500 ตัวอักษร | 12.4% |
| สแปมพนัน | 10.6% |
| หวย | 1.6% |
| กันไว้เป็นชุดวัด | 1.0% |

**PII redaction** — แก้ 108,237 เอกสาร (2.37%):
Line ID 104,121 · social handle 18,578 · เบอร์โทร 18,127 · บัญชีธนาคาร 4,163 ·
อีเมล 575 · **เลขบัตรประชาชน 420** *(ตรวจ checksum ก่อนลบ)*

---

## โครงสร้าง

```text
├── plans/            แผนวิจัย · แผนสร้างโมเดล · แผน compute/storage
├── configs/          พารามิเตอร์ที่ freeze + data manifest template
├── novelty/          novelty audit · nearest-work matrix · pre-mortem · decision memo
├── sources/          ทะเบียนแหล่งอ้างอิงที่ตรวจแล้ว
├── base_selection/   tokenizer screen · port audit · เหตุผลเลือก base
├── phase0/           สคริปต์ทั้งหมด + ผลลัพธ์
├── data/             manifest เท่านั้น (ข้อมูลจริงไม่ commit)
├── validation/       VALIDATION_OUTPUT_LOG.md
└── commits/          แผนการ commit
```

## ทำซ้ำได้

```bash
pip install -r phase0/requirements.txt

python3 phase0/tokenizer_screen_ext.py     # เทียบ tokenizer
python3 phase0/freeze_eval_suite.py        # แช่แข็ง eval suite
python3 phase0/build_heldout.py --set TH-WEB-HELDOUT --target 2000
python3 phase0/measure_baseline.py --model Qwen/Qwen3-1.7B-Base \
    --heldout th=data/heldout/TH-WEB-HELDOUT.jsonl
python3 phase0/corpus_audit.py 1000000     # audit corpus
python3 phase0/clean_corpus.py             # clean
python3 phase0/apply_pii.py                # PII redaction
```

**ไม่ได้เผยแพร่ตัว corpus** แต่สร้างซ้ำได้เป๊ะจากสคริปต์ + revision ที่ pin ไว้
(กฎเลือกชุดวัดเป็น deterministic hash bucket — [`phase0/heldout_rule.py`](phase0/heldout_rule.py))

---

## ข้อจำกัดที่รู้ตัว

- **ยังไม่ได้เทรนโมเดล** ตัวเลขทั้งหมดเป็น baseline ก่อนเทรน
- **ยังไม่ได้ decontaminate** held-out กับ ThaiExam / M3Exam
- ตัวกรองเป็น **regex ไม่ใช่ classifier** ตัดเกินและตัดไม่หมดทั้งคู่ —
  ตรวจด้วยตาแล้วพบว่าตัวกรองหวยมี false positive
- **ทำ dedup แบบตรงตัวเท่านั้น** ยังไม่ได้ทำ near-duplicate
- PII detector จับ **ชื่อคน ที่อยู่ วันเกิด ไม่ได้**
- **ยังไม่ได้ประเมิน safety** ใด ๆ

---

## กฎที่ยังบังคับใช้

- ทุก local/smoke output ต้องระบุ `scientific_evidence_allowed=false`
- ทุก condition ต้องใช้ base checkpoint, tokenizer, ลำดับข้อมูล และ optimizer เดียวกัน
- freeze metric และ threshold **ก่อน** เปิดผล
- ห้าม claim ว่า "เป็นงานแรกที่ใช้ X กับภาษาไทย" เพียงอย่างเดียว

## งานวิจัยสาย 2 (secondary)

Controlled study ของ Standard Residual vs Delta Block AttnRes vs MHAR ภายใต้ข้อมูลและ
GPU-hours ที่ควบคุม — **ไม่ใส่ในตัว product** เพื่อไม่ให้ความเสี่ยงงานวิจัยกระทบ deliverable
รายละเอียดและผล novelty audit อยู่ใน [`novelty/`](novelty/)

## License และ Attribution

ดู [`LICENSE_COMPLIANCE.md`](LICENSE_COMPLIANCE.md) · ร่าง model card ที่
[`MODEL_CARD_DRAFT.md`](MODEL_CARD_DRAFT.md)

```
Base model : Qwen3-1.7B-Base (Alibaba Cloud), Apache-2.0
Data       : SEA-PILE-v2 (AI Singapore), ODC-By 1.0
             เนื้อหาต้นทางจาก CommonCrawl ภายใต้ CommonCrawl Terms of Use
```
