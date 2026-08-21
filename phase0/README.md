# Phase 0 — งานที่ทำได้ก่อนมี GPU

เป้าหมายเดียว: **ตอบให้ได้ว่า base ที่เลือกยังมี Thai headroom เหลือพอให้ CPT คุ้มหรือไม่
ก่อนจะใช้งบ GPU แม้แต่ชั่วโมงเดียว**

ทุก output มี `scientific_evidence_allowed: false`

---

## Step A — ต่อยอด tokenizer screen  ✅ รันแล้ว

```bash
python3 tokenizer_screen_ext.py     # -> tokenizer_screen_ext.json
```

ใช้ **sample set แช่แข็งชุดเดียวกับ `base_selection/tokenizer_screen.json`**
(ตรวจ sha256 ราย document ครบ 14/14) ตัวเลขจึงเทียบกันได้โดยตรง

เพิ่มผู้สมัครที่ screen เดิมยังไม่มี: `Qwen3-1.7B-Base`, `gemma-4-E2B`, `Qwen3-4B-Base`
และวัด `Qwen3-0.6B-Base` ซ้ำเป็น **methodology control**

### บั๊กสองจุดที่เจอระหว่างทำ — เก็บไว้เป็นบทเรียน

1. **vocab ของ byte-level BPE ไม่ใช่ตัวอักษรไทยตรง ๆ** — key ในคำศัพท์ของ Qwen เก็บเป็น
   byte-encoded (`'à¸£à¸±'`) ถ้าไล่หาอักขระไทยตรง ๆ จะได้ 0 ชิ้น ต้องถอด byte map ก่อน
2. **`tokenizers.Tokenizer.encode()` ใส่ `<bos>` ให้อัตโนมัติ** — ทำให้ Gemma-4 ถูกนับ token
   เกินไป 1 ตัวต่อเอกสาร (screen เดิมระบุ `add_special_tokens: False`)

> ทั้งสองบั๊ก **สคริปต์รันผ่านและให้ตัวเลขที่ดูสมเหตุสมผล** ตรวจเจอเพราะเทียบกับ control เท่านั้น
> หลังแก้ Gemma-4 ให้ 2.8326 / 1.4524 ซึ่งตรงกับ Gemma-3 เดิม (2.833 / 1.4524) — เป็นการยืนยันว่าแก้ถูก

---

## Step B — แช่แข็ง eval suite  ✅ รันแล้ว

```bash
python3 freeze_eval_suite.py        # -> eval_suite_frozen.json
```

`suite_id: THAILLM-EVAL-FROZEN-V1`
`spec_sha256: 1fae436e05fa99cee8b5b878e72f45c95aa8d51b3c1177e49c1c5a98c565cb19`

**ต้องอ้าง hash นี้ในทุก run record ที่บอกว่าใช้ suite นี้**
อะไรที่ไม่ได้เขียนไว้ในไฟล์นี้ ห้ามนำมาอ้างภายหลังว่าเป็นเมตริกที่วางแผนไว้

---

## Step C — วัด baseline  ✅ pipeline ทดสอบแล้ว

```bash
# smoke test (CPU/MPS) — พิสูจน์ว่า pipeline ถูก ไม่ใช่ผลของโมเดล
python3 measure_baseline.py --model Qwen/Qwen3-0.6B-Base --smoke

# ของจริง (GPU) หลังสร้าง held-out set แล้ว
python3 measure_baseline.py --model Qwen/Qwen3-1.7B-Base \
    --heldout th=../data/heldout/TH-WEB-HELDOUT.jsonl \
    --heldout en=../data/heldout/EN-HELDOUT.jsonl \
    --heldout code=../data/heldout/CODE-HELDOUT.jsonl --device cuda
```

**ใช้ BPB ไม่ใช่ perplexity** เพราะ perplexity ผูกกับ tokenizer จึงเทียบข้าม base ไม่ได้
BPB หารด้วย UTF-8 byte ซึ่งไม่ขึ้นกับ tokenizer

> ⚠️ **BPB เทียบข้ามโมเดลในภาษาเดียวกันได้ แต่เทียบข้ามภาษาไม่ได้**
> ไทยใช้ ~3 bytes/อักขระ อังกฤษ ~1 → BPB ไทยจะต่ำกว่าโดยกลไก
> **ห้ามอ่านว่า "Thai BPB < English BPB แปลว่าโมเดลเก่งไทยกว่า"**

ส่วน downstream accuracy ใช้ lm-evaluation-harness ตาม task list ใน `eval_suite_frozen.json`
และ **ต้องบันทึก commit sha ของ harness** ทุกครั้ง

---

## Step D — สร้าง held-out set  ✅ ไทยเสร็จแล้ว

```bash
python3 heldout_rule.py                                    # ดูกฎ
python3 build_heldout.py --set TH-WEB-HELDOUT --target 2000
```

`TH-WEB-HELDOUT`: **2,000 docs / 9.69 MB**
`set_sha256: 7204c28a0204defec98aa5390829f154f208fe106ce3f4eb345288859afb9cae`

### กฎกันปนเปื้อน — ใช้ **bucket ไม่ใช่ list**

```python
bucket = int(sha256(NFC(text))[:8], 16) % 100   # held out ถ้า bucket == 0
```

**training pipeline ต้อง `import heldout_rule` แล้วใช้ `is_trainable()` ตัวเดียวกัน**

เหตุผลที่ใช้ bucket แทน list ของ id: list ตัดออกได้เฉพาะเอกสารที่เราเปิดอ่านจริง ๆ
shard ที่ไม่ได้สแกนหรือ re-crawl จะรั่วเข้า training ส่วน bucket ตัดทั้ง 1% ทั่วทั้ง corpus
ต้นทุน ~65M tokens ซึ่งไม่มีนัยสำคัญ

### สิ่งที่วัดได้ระหว่างทาง

| ตัวเลข | ความหมาย |
|---|---|
| bucket 0 = **1.034%** จาก 544,716 docs | hash กระจายสม่ำเสมอตามทฤษฎี ✅ |
| **dup 3.2%** ใน bucket | dataset card บอกว่า dedup แล้ว **แต่เป็น within-snapshot เท่านั้น** → **ยืนยันว่าต้องทำ cross-snapshot dedup เอง** |
| **61%** ของ docs สั้นกว่า 500 ตัวอักษร | ถ้าไม่ใส่ length floor ชุด BPB จะเต็มไปด้วยเศษข้อความ |

---


---

## Step E — clean corpus  ✅ รันแล้ว

```bash
python3 clean_corpus.py            # -> data/clean/th/*.jsonl.gz + data/clean_th_manifest.json
python3 sample_dropped.py 60 3     # -> data/dropped_samples/  (ตรวจ filter ด้วยตา)
```

| | |
|---|---|
| เข้า | 16,428,048 docs |
| ออก | **4,567,214 docs (27.8%)** · 29.53 GB · **~5.851B Qwen tokens** |

ตัดออก: ซ้ำ 15.0% · สั้นกว่า 500 ตัวอักษร 12.4% · พนัน 10.6% · หวย 1.6% ·
held-out 1.0% · adult 0.3% *(เป็น % ของ corpus ตาม bytes)*

**ตรวจด้วยตาแล้ว:** ตัวกรองพนันถูกต้อง (huc99.net, betstarth.com, t4over.com เป็นสแปมจริง)
**แต่ตัวกรองหวยมี false positive** (เว็บแฟรนไชส์/กีฬาที่บังเอิญมีคำว่า "หวย")
กระทบแค่ 1.6% จึงยังไม่แก้

## Step F — PII redaction  ✅ รันแล้ว

```bash
python3 pii_filter.py              # ทดสอบ detector
python3 apply_pii.py               # -> data/clean_pii/th/ + data/clean_pii_th_manifest.json
```

**แก้ไข 108,237 เอกสาร (2.37%) · ลบไปเพียง ~1 MB จาก 29.5 GB**
เพราะ**แทนที่ตัวเลข ไม่ทิ้งเอกสาร**

| ประเภท | จำนวน |
|---|---|
| Line ID | 104,121 |
| social handle | 18,578 |
| เบอร์โทร | 18,127 |
| บัญชีธนาคาร | 4,163 |
| อีเมล | 575 |
| **เลขบัตรประชาชน** (ผ่าน checksum) | **420** |

### บั๊กที่เจอใน PII detector — บทเรียนซ้ำรอยเดิม

1. **`\b` ใช้กับภาษาไทยไม่ได้** — อักษรไทยเป็น word character ใน Python
   ไทยติดกับอักษรอังกฤษจึงไม่เกิด word boundary
   → detector อีเมลจับได้ **0 จาก 809 เอกสารที่มี `@`** แก้เป็น lookaround แล้ว
2. **บัตรเครดิต false positive** — เลขสุ่มผ่าน Luhn ราว 1 ใน 10
   จับ `"166617 106017 106617"` เป็นบัตรเครดิต → บังคับต้องมีคำบริบทนำหน้า

> รูปแบบเดียวกับ ROUTE_SCALE_GATE และ byte-level vocab bug:
> **โค้ดรันผ่าน ให้ตัวเลขที่ดูสมเหตุสมผล แต่วัดผิด**

---

## Step G — Replay data อังกฤษ / code / math  ✅ รันแล้ว

```bash
python3 build_replay.py --languages en code math
python3 build_heldout.py --set EN-HELDOUT --target 2000
python3 build_heldout.py --set CODE-HELDOUT --target 1000
python3 verify_replay.py
```

นับ token จริงด้วย tokenizer ของ `Qwen3-1.7B-Base` revision ที่ pin ไว้
และ `add_special_tokens=false`; ไม่โหลด model weights และไม่ใช้ GPU

| pool | แหล่ง | เอกสาร | Qwen tokens | เป้า |
|---|---|---:|---:|---:|
| English | FineWeb-Edu sample/10BT | 3,508,510 | **3,516,598,044** | 3.50B ✅ |
| Code | GitHub Code Clean, MIT/Apache/BSD เท่านั้น | 700,129 | **1,104,617,663** | 1.00B ✅ |
| Math | FineMath-4plus | 412,820 | **592,984,656** | 0.50B ✅ |

Frozen held-out:

- `EN-HELDOUT`: 2,000 docs · `e0a30eae…f807b75`
- `CODE-HELDOUT`: 1,000 docs · `932d5efc…e32cf68`

full verifier อ่านครบ **4,621,459 unique documents**: compressed SHA, document hash,
token totals และ license ผ่านทั้งหมด; held-out leakage = 0; token recount 100 ตัวอย่างต่อภาษา mismatch = 0

ข้อจำกัด: ยังไม่ได้ทำ near-dedup และยังไม่ได้ทำ PII/secrets scan กับ replay pools

---

## ผลรวม Phase 0

| โมเดล | พารามิเตอร์ | Thai CPT | Thai BPB |
|---|---|---|---|
| Sailor2-1B | 0.99B | ✅ 500B tokens | **0.378051** |
| **Qwen3-1.7B-Base** | 1.72B | ❌ | **0.454218** |
| Qwen3-0.6B-Base | 0.60B | ❌ | 0.521386 |

**headroom = 16.8% relative = 8.4 เท่าของเกณฑ์ 2%**
⚠️ เป็น**ขอบบน** — Sailor2 อาจเคยเห็นเอกสารใน held-out เพราะทั้งคู่มาจาก CommonCrawl

## ยังเหลือก่อนจบ Phase 0

- [ ] **decontamination** ชุด held-out กับ ThaiExam / M3Exam ← ห้ามข้าม
- [ ] baseline **gemma-4-E2B** (ต้อง transformers จาก git main)
- [ ] **lm-evaluation-harness** บน frozen task list
- [x] สร้าง EN / CODE held-out
- [ ] สร้าง TH-ENC held-out
- [ ] ตัดสิน **headroom gate**
- [ ] รัน `measure_baseline.py` ของจริงกับ Qwen3-1.7B-Base และ gemma-4-E2B
- [ ] รัน lm-evaluation-harness บน task list ที่ freeze ไว้
- [ ] ตัดสิน **headroom gate** ใน `eval_suite_frozen.json`

## หมายเหตุ Gemma-4 (ตรวจแล้ว)

| ข้อเท็จจริง | สถานะ |
|---|---|
| `transformers 4.57.6` = **รุ่นล่าสุดบน PyPI** และ **ไม่มี `gemma4`** | ✅ ยืนยันแล้ว |
| **transformers git main มี `gemma4`** | ✅ ยืนยันแล้ว |
| `config.json` ของ gemma-4-E2B **ไม่มี `auto_map`** → `trust_remote_code` ช่วยไม่ได้ | ✅ ยืนยันแล้ว |

**คำพูดที่ถูกต้อง:** Gemma 4 รองรับบน main แต่**ยังไม่มีใน release ใด ๆ**
ถ้าจะใช้ต้อง pin git commit ที่ยังไม่ release และตรวจ FSDP / DeepSpeed / gradient checkpointing เอง
→ **เป็นความเสี่ยงที่จัดการได้ ไม่ใช่ความเป็นไปไม่ได้**
