# งาน A — จัดหาและทำความสะอาดข้อมูล replay (อังกฤษ / code / math)

> พร็อมพ์นี้เขียนให้ agent ทำงานต่อได้เองโดยไม่ต้องถามย้อน
> **repo:** `/Users/prince/Documents/Research for AIAT/research/thai-llm-1b-attnres`

---

## บริบท

โปรเจกต์ทำ continued pretraining ของ **Qwen3-1.7B-Base** ให้เป็นโมเดลภาษาไทยแบบ generalist

**ข้อมูลไทยเสร็จแล้ว** — SEA-PILE-v2 subset `th` ผ่าน clean + PII redaction เหลือ
4,567,214 docs ≈ **5.851B Qwen tokens** อยู่ที่ `data/clean_pii/th/*.jsonl.gz`

**ข้อมูล replay ยังไม่มีเลยแม้แต่ไบต์เดียว** ซึ่งเป็นครึ่งหนึ่งของ mixture และเป็นครึ่งที่กันโมเดลลืม
(Typhoon 2 ใช้อังกฤษ 50% โดยอ้างเหตุผล catastrophic-forgetting mitigation โดยตรง)

### เป้าหมายเชิงตัวเลข

สัดส่วนที่ freeze ไว้ใน `configs/experiment_parameters.json` → `data_mixture_provisional`

| ภาษา | สัดส่วน | ต้องการ @6B | @10B | สถานะ |
|---|---|---|---|---|
| ไทย | 50% | 3.00B | 5.00B | ✅ มี 5.85B |
| **อังกฤษ** | 35% | 2.10B | **3.50B** | ❌ |
| **code** | 10% | 0.60B | **1.00B** | ❌ |
| **math** | 5% | 0.30B | **0.50B** | ❌ |

**ให้ทำเป้า @10B** (เผื่อไว้) นับ token ด้วย **tokenizer ของ Qwen3-1.7B-Base เท่านั้น**
ห้ามใช้ตัวเลข token จาก dataset card ของต้นทาง เพราะนับด้วย tokenizer คนละตัว

---

## ข้อบังคับ (ห้ามฝ่าฝืน)

1. **CPU เท่านั้น** ห้ามใช้ GPU ห้ามโหลด model weights
2. **ห้ามแตะข้อมูลไทย** — `data/clean/`, `data/clean_pii/`, `data/heldout/TH-*` อ่านได้ ห้ามเขียนทับ
3. **ห้าม `git commit` หรือ `git push`** เจ้าของ repo เป็นคนทำเอง
4. **ทุก output ต้องมี `"scientific_evidence_allowed": false`**
5. **ต้องใช้ `phase0/heldout_rule.py` ตัวเดิม** ห้ามเขียนกฎใหม่ ห้ามแก้ไฟล์นั้น
6. **pin revision ของ dataset ทุกตัว** และบันทึกลง manifest
7. ห้ามแก้ `phase0/eval_suite_frozen.json` — มัน freeze แล้ว
8. เขียนสคริปต์ให้ **resume ได้** ถ้าหลุดกลางทาง

---

## รูปแบบที่ต้องทำตาม (มีของจริงให้ดูแล้ว)

อ่านไฟล์เหล่านี้ก่อนเริ่ม แล้วทำตามรูปแบบเดิม **อย่าประดิษฐ์ convention ใหม่**

| ไฟล์ | ใช้เป็นแบบอย่างของ |
|---|---|
| `phase0/heldout_rule.py` | กฎ bucket — **import มาใช้ ห้ามเขียนใหม่** |
| `phase0/build_heldout.py` | วิธีสร้าง held-out + manifest + `SETS` dict |
| `phase0/clean_corpus.py` | โครง pipeline + การนับ drop แยกตามกฎ |
| `phase0/corpus_audit.py` | รูปแบบ audit report |
| `data/clean_th_manifest.json` | โครงสร้าง manifest ที่ต้องได้ |

### กฎ held-out (จาก `heldout_rule.py` — ห้ามแก้)

```python
bucket = int(sha256(NFC(text) + whitespace-collapse)[:8], 16) % 100
is_heldout = (bucket == 0)     # 1% ของ corpus กันไว้วัด ห้ามเทรน
```

ใช้ **กฎเดียวกันนี้กับทุกภาษา** เอกสารที่ตกใน bucket 0 ต้องไม่เข้า training pool ไม่ว่าภาษาอะไร

---

## แหล่งข้อมูล

### 1. อังกฤษ — FineWeb-Edu

- repo: `HuggingFaceFW/fineweb-edu` · license **ODC-By 1.0** (บันทึกแล้วใน `sources/source_registry.csv` id `DS03`)
- มี subset ย่อยเช่น `sample/10BT` ให้เลือกใช้ ไม่ต้องโหลดทั้ง dataset
- เป้า **3.5B Qwen tokens**

### 2. Code — ⚠️ ตรวจ gating ก่อนเป็นอันดับแรก

`bigcode/the-stack-v2` **เป็น gated dataset ต้องยอมรับเงื่อนไขก่อนโหลด**
เครื่องนี้ไม่มี HF token → **ถ้าโหลดไม่ได้ ห้ามเสียเวลา ให้เปลี่ยนไปตัวสำรองทันที**

ตัวสำรองที่ไม่ gated (ตรวจตามลำดับ แล้วบันทึกว่าเลือกอะไรเพราะอะไร):
- `bigcode/starcoderdata`
- `bigcode/the-stack-dedup` (permissive subset)
- `codeparrot/github-code-clean` (กรอง `licenses` field เอาเฉพาะ MIT/Apache-2.0/BSD)

**ต้องเก็บ license ราย record ไว้ใน output** ถ้า dataset ไม่มี field license ให้บันทึกว่าไม่มี
และ **flag ไว้ใน manifest ว่าเป็นความเสี่ยงที่ยังไม่ปิด** ห้ามเดาว่าเป็น permissive

เป้า **1.0B Qwen tokens**

### 3. Math

- `HuggingFaceTB/finemath` หรือ `open-web-math/open-web-math`
- **ตรวจ license จาก dataset card จริง** แล้วบันทึก ห้ามสมมติ
- เป้า **0.5B Qwen tokens**

---

## สิ่งที่ต้องส่งมอบ

### 1. สคริปต์

```
phase0/build_replay.py        # ดึง + clean + นับ token ต่อภาษา (resume ได้)
```

ตัวกรองขั้นต่ำ ปรับตามภาษาได้แต่ต้องบันทึกเหตุผล:
- `is_heldout()` — **ต้องมาก่อนสุดเสมอ**
- ความยาวขั้นต่ำ (ไทยใช้ 500 chars; อังกฤษ/code ปรับได้ตามความเหมาะสม + บันทึกเหตุผล)
- exact dedup ด้วย `doc_hash()` จาก `heldout_rule.py`
- code: กรองเอาเฉพาะ license ที่อนุญาต
- **นับ token จริงด้วย Qwen3-1.7B-Base tokenizer** ไม่ใช่ประมาณจาก bytes

### 2. ข้อมูล (จะถูก gitignore ไม่ commit)

```
data/clean_replay/en/*.jsonl.gz
data/clean_replay/code/*.jsonl.gz
data/clean_replay/math/*.jsonl.gz
```

### 3. Manifest — รูปแบบเดียวกับ `data/clean_th_manifest.json`

```
data/clean_replay_manifest.json
```

ต้องมี: source repo + **revision ที่ pin** + license · จำนวน docs เข้า/ออก ·
**จำนวน token จริงที่นับด้วย Qwen tokenizer** · drop แยกตามกฎ (docs + bytes) ·
`scientific_evidence_allowed: false` · `limitations`

### 4. Held-out sets — ตามที่ freeze ไว้ใน `eval_suite_frozen.json`

| set_id | ภาษา | target_docs |
|---|---|---|
| `EN-HELDOUT` | en | **2000** |
| `CODE-HELDOUT` | code | **1000** |

**วิธีทำ:** เพิ่ม entry ใน `SETS` dict ของ `phase0/build_heldout.py` แล้วรัน
ห้ามเขียนสคริปต์ใหม่ซ้ำซ้อน

ผลลัพธ์: `data/heldout/EN-HELDOUT.jsonl` + `.manifest.json` (และ CODE เช่นกัน)
manifest ต้องมี `set_sha256` เหมือนของไทย

⚠️ ถ้า corpus ต้นทางเรียงตามเวลาหรือตาม domain **ต้องสุ่ม shard แบบกระจาย**
เคยเจอปัญหานี้กับข้อมูลไทยมาแล้ว: ใช้ shard 0–2 ทำให้ได้แต่ข้อมูลปี 2020–2022
และตัวเลข BPB ขยับไป 10% ซึ่งเป็น 5 เท่าของเกณฑ์ตัดสิน 2% — ดูคอมเมนต์ใน `build_heldout.py`

### 5. อัปเดตไฟล์เดิม

- `sources/source_registry.csv` — เพิ่มแถวของ dataset ที่ใช้จริง
  (ใช้ id ที่ยังว่าง เช่น `DS04`, `DS05` — **ตรวจก่อนว่าไม่ซ้ำ** เคยมีปัญหา id ชนกันมาแล้ว)
- `configs/data_manifest.template.json` — กรอกช่องของ `fineweb_edu_en` และ `permissive_code`
- `validation/VALIDATION_OUTPUT_LOG.md` — **เพิ่ม entry ใหม่ท้ายไฟล์ ก่อนหัวข้อ
  `## Required future entries`** รูปแบบ YAML เหมือน entry เดิมทุกประการ

---

## เกณฑ์ว่างานเสร็จ

- [ ] มี ≥ 3.5B / 1.0B / 0.5B Qwen tokens สำหรับ en / code / math
- [ ] ทุก token นับด้วย Qwen3-1.7B-Base tokenizer จริง ไม่ใช่ประมาณจาก bytes
- [ ] `EN-HELDOUT` 2000 docs และ `CODE-HELDOUT` 1000 docs พร้อม `set_sha256`
- [ ] manifest ระบุ revision + license ครบทุกแหล่ง
- [ ] license ของ code ตรวจแล้ว หรือ **flag ชัดเจนว่ายังไม่ปิด**
- [ ] มี entry ใหม่ใน `VALIDATION_OUTPUT_LOG.md`
- [ ] `source_registry.csv` ไม่มี id ซ้ำ (ตรวจด้วยสคริปต์)
- [ ] ไม่มีไฟล์ข้อมูลใหญ่หลุดออกนอก `.gitignore`

---

## ⚠️ บทเรียนจากงานก่อนหน้า — ระวังให้มาก

โปรเจกต์นี้เจอบั๊กหลายตัวที่ **โค้ดรันผ่าน ไม่ error และให้ตัวเลขที่ดูสมเหตุสมผล แต่วัดผิด**
ทุกตัวจับได้เพราะมี control หรือมีคนเช็คด้วยตา ไม่ใช่เพราะโค้ดพัง

| บั๊ก | อาการ |
|---|---|
| `\b` ในภาษาไทย | อักษรไทยเป็น word character → detector อีเมลจับได้ **0 จาก 809 เอกสารที่มี `@`** |
| byte-level BPE vocab | ไม่ถอดรหัสก่อนเช็ค → นับ Thai vocab pieces ได้ **0** ทั้งที่มี 2,571 |
| `tokenizers.encode()` | ใส่ `<bos>` อัตโนมัติ → นับ token เกิน 1 ต่อเอกสาร |
| Luhn อย่างเดียว | เลขสุ่มผ่าน ~1 ใน 10 → จับเลขอ้างอิงเป็นบัตรเครดิต |

**สิ่งที่ต้องทำเพื่อกัน:**
1. หลังเขียน filter ทุกตัว **สุ่มพิมพ์ตัวอย่างที่ถูกตัดออกมาดูด้วยตาอย่างน้อย 20 อัน**
   (มี `phase0/sample_dropped.py` เป็นแบบอย่าง)
2. ถ้าตัวเลขออกมาเป็น **0 หรือสวยเกินไป ให้สงสัยไว้ก่อนว่า detector พัง** แล้วตรวจย้อน
3. เทียบผลกับสิ่งที่คาดไว้จาก dataset card ถ้าต่างกันมากให้หาสาเหตุก่อนเดินต่อ
4. `add_special_tokens=False` เสมอเวลานับ token

---

## รายงานเมื่อเสร็จ

สรุปสั้น ๆ ว่า:
1. ได้ token จริงเท่าไหร่ต่อภาษา (นับด้วย Qwen tokenizer)
2. แต่ละแหล่งเลือกอะไร license อะไร revision อะไร
3. drop ไปเท่าไหร่ แยกตามกฎ
4. **ตัดสินใจอะไรเองบ้าง** และเพราะอะไร
5. **อะไรที่ยังไม่ปิด / ยังเสี่ยง** โดยเฉพาะเรื่อง license ของ code
