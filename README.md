# ThaiLLM-1B

**Continued pretraining ของ Qwen3-1.7B ให้เป็นโมเดลภาษาไทยแบบ generalist**

เป้าหมายคือโมเดลที่ใช้ภาษาไทยได้ดีขึ้นอย่างมีนัยสำคัญ **โดยไม่ทิ้งความสามารถภาษาอังกฤษ
การให้เหตุผล และโค้ด** ที่ base model มีอยู่แล้ว

> **กฎของ repo นี้:** ทุกตัวเลขมาพร้อมสคริปต์ที่สร้างมัน, revision hash ของโมเดลและข้อมูลที่ใช้
> และข้อจำกัดที่รู้ตัว — รวมถึง **บั๊กที่เจอระหว่างทาง** ซึ่งบันทึกไว้ครบใน
> [`validation/VALIDATION_OUTPUT_LOG.md`](validation/VALIDATION_OUTPUT_LOG.md)

---

## สถานะ: Data preparation เสร็จ — evaluation gates ยังไม่ครบและ **ยังไม่ได้เทรนโมเดล**

- [x] คัดเลือก base model + tokenizer screen
- [x] แช่แข็ง eval suite **ก่อน** วัด baseline
- [x] สร้าง held-out set + วัด baseline + ประเมิน headroom
- [x] Clean corpus + PII redaction
- [x] Replay corpus อังกฤษ/code/math + EN/CODE held-out
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

| | docs | Qwen tokens |
|---|---|---|
| **ไทย** SEA-PILE-v2 (clean + PII) | 4,567,214 | **5.851B** |
| **อังกฤษ** FineWeb-Edu (secret+PII scanned) | 3,507,052 | **3.513B** |
| **code** github-code-clean (license-filtered, rebalanced) | 690,716 | **1.000B** |
| **math** FineMath-4plus | 412,734 | **0.592B** |
| **รวม** | **9,177,716** | **10.96B** |

**ผ่านการทำความสะอาดครบ:**
exact dedup · length floor · Thai gambling/lottery filter · **PII redaction** (ไทย + สากล) ·
**secret redaction** (AWS/GitHub/private key/JWT ฯลฯ) · **benchmark decontamination** · **near-dedup**

**Removal list** — `data/removal_list.txt` มี 273,703 docs (2.98%) ที่ต้องตัดออกตอน tokenize:
ปนเปื้อน benchmark 3,233 + near-duplicate 270,841

> 🎯 decontamination จับเว็บติวข้อสอบ **ก.พ. ภาษาไทย** ได้ (364 n-gram hits) ซึ่งมีข้อสอบ
> รูปแบบเดียวกับ ThaiExam — ถ้าเทรนแล้วเอาไปสอบ คะแนนจะเป็นการจำ ไม่ใช่ความสามารถ


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

# Phase 0 — คัดเลือก base + ล็อกชุดวัด + baseline
python3 phase0/tokenizer_screen_ext.py
python3 phase0/freeze_eval_suite.py
python3 phase0/build_heldout.py --set TH-WEB-HELDOUT --target 2000
python3 phase0/measure_baseline.py --model Qwen/Qwen3-1.7B-Base \
    --heldout th=data/heldout/TH-WEB-HELDOUT.jsonl

# Phase 1 — ข้อมูลไทย
python3 phase0/corpus_audit.py 1000000
python3 phase0/clean_corpus.py
python3 phase0/apply_pii.py

# Phase 1 — ข้อมูล replay (อังกฤษ/code/math)
python3 phase0/build_replay.py
python3 phase0/apply_scan.py --languages en math --offline
python3 phase0/rebalance_code.py --offline
python3 phase0/verify_replay_v2.py

# Phase 1 — decontamination + near-dedup (ต้องตั้ง PYTHONHASHSEED)
PYTHONHASHSEED=0 python3 phase0/build_benchmark_index.py
PYTHONHASHSEED=0 python3 phase0/decon_and_neardedup.py
```

**ไม่ได้เผยแพร่ตัว corpus** แต่สร้างซ้ำได้เป๊ะจากสคริปต์ + revision ที่ pin ไว้
(กฎเลือกชุดวัดเป็น deterministic hash bucket — [`phase0/heldout_rule.py`](phase0/heldout_rule.py))

## ข้อจำกัดที่รู้ตัว

- **ยังไม่ได้เทรนโมเดล** ตัวเลขทั้งหมดเป็น baseline ก่อนเทรน
- **decontamination ทำแล้ว** แต่จับได้เฉพาะการซ้ำแบบตรงตัว/เกือบตรงตัว — ข้อสอบที่ถูกถอดความหรือแปล **จับไม่ได้**
- ตัวกรองเป็น **regex ไม่ใช่ classifier** ตัดเกินและตัดไม่หมดทั้งคู่ —
  ตรวจด้วยตาแล้วพบว่าตัวกรองหวยมี false positive
- near-dedup ใช้ MinHash 32 permutations เน้นคู่ที่คล้ายกันสูง — คู่ที่คล้ายปานกลางจะหลุด
- LSH เก็บเอกสาร**ตัวแรก**ที่เจอในแต่ละ bucket ตัวไหนรอดจึงขึ้นกับลำดับไฟล์ ไม่ใช่คุณภาพ
- PII/secret detector เป็น regex + context guard ไม่ใช่โมเดล — จับ **ชื่อคน ที่อยู่ วันเกิด ไม่ได้**
- code build ที่รับมามี `[REDACTED_SECRET]` แทน placeholder ใน 656 จาก 4.6M docs (0.014%) — เป็นการ redact เกิน ไม่ใช่ขาด
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
