# งาน B — Secret/PII scan + ปรับสัดส่วนภาษา code

> **repo:** `/Users/prince/Documents/Research for AIAT/research/thai-llm-1b-attnres`
> พร็อมพ์นี้เขียนให้ทำงานต่อได้เองโดยไม่ต้องถามย้อน
> **อ่าน `TASK_A_REPLAY_DATA.md` ก่อน** เพื่อเข้าใจว่างานก่อนหน้าทำอะไรไว้

---

## บริบท

งาน A เสร็จแล้วและผ่าน audit อิสระ ได้ replay corpus:

| | docs | Qwen tokens |
|---|---|---|
| en (FineWeb-Edu) | 3,508,510 | 3,516,598,044 |
| code (github-code-clean) | 700,129 | 1,104,617,663 |
| math (FineMath-4plus) | 412,820 | 592,984,656 |

อยู่ที่ `data/clean_replay/{en,code,math}/*.jsonl.gz`
manifest: `data/clean_replay_manifest.json`

**ข้อมูลไทยเสร็จแล้วเช่นกัน** — `data/clean_pii/th/` ≈ 5.851B tokens (ผ่าน PII redaction แล้ว)

งาน B มี **3 ส่วน ทำในรอบเดียวกัน** เพราะต้องอ่าน/เขียนข้อมูลชุดเดียวกันอยู่แล้ว

---

## ข้อบังคับ

1. **ใช้ MPS (GPU ในตัวของ Mac M2) ได้** ถ้าช่วยให้เร็วขึ้น — ห้ามเช่า GPU ห้ามใช้ cloud
2. **ห้ามแตะข้อมูลไทย** — `data/clean/`, `data/clean_pii/` อ่านได้ ห้ามเขียนทับ
3. **ห้าม `git commit` หรือ `git push`**
4. ทุก output ต้องมี `"scientific_evidence_allowed": false`
5. **ห้ามแก้** `phase0/heldout_rule.py` และ `phase0/eval_suite_frozen.json`
6. **ห้ามแตะ held-out sets ที่มีอยู่** (`data/heldout/*`) — ถ้าดึงข้อมูลเพิ่ม ต้องผ่าน `is_heldout()` เหมือนเดิม
7. เขียนสคริปต์ให้ **resume ได้**
8. **เก็บของเดิมไว้** เขียน output ใหม่ไปที่ path ใหม่ ห้ามทับของเดิม

---

## ส่วนที่ 1 — 🔴 Secret scan (สำคัญที่สุด)

repo บน GitHub มี credential ที่ commit ทิ้งไว้จริง **และ key ที่หลุดใช้งานได้ทันที
จึงร้ายแรงกว่าเบอร์โทร**

### ต้องจับอย่างน้อย

| ประเภท | รูปแบบตัวอย่าง |
|---|---|
| AWS access key | `AKIA[0-9A-Z]{16}`, `ASIA...` |
| AWS secret | 40 ตัวอักษร base64 ใกล้คำว่า `aws_secret` |
| GitHub token | `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_` |
| **private key** | `-----BEGIN (RSA\|OPENSSH\|DSA\|EC\|PGP) PRIVATE KEY-----` |
| Google API key | `AIza[0-9A-Za-z_-]{35}` |
| Slack token | `xox[baprs]-` |
| Stripe | `sk_live_`, `rk_live_` |
| OpenAI/Anthropic | `sk-`, `sk-ant-` |
| JWT | `eyJ...` 3 ส่วนคั่นด้วยจุด |
| ค่าใน .env | `PASSWORD=`, `SECRET=`, `TOKEN=`, `API_KEY=` ตามด้วยค่าที่ไม่ใช่ placeholder |
| connection string | `postgres://user:pass@`, `mongodb+srv://`, `mysql://` |

### นโยบาย

- **private key → ลบทั้งเอกสาร** (ทั้งไฟล์คือ key ไม่มีคุณค่าทางภาษา)
- **key/token อื่น → redact เป็น `[REDACTED_SECRET]`** เก็บโครงสร้างโค้ดไว้
- **ต้องกรอง placeholder ออก** — `your_api_key_here`, `xxx`, `<TOKEN>`, `changeme`,
  `AKIAIOSFODNN7EXAMPLE` (ตัวอย่างทางการของ AWS) ไม่ใช่ secret จริง
  **ถ้าไม่กรอง จะได้ false positive มหาศาลจาก tutorial และ template**

---

## ส่วนที่ 2 — PII scan สำหรับ en / code / math

`phase0/pii_filter.py` ที่มีอยู่เป็น **Thai-oriented** (เลขบัตรประชาชนไทย, คำว่า "ธนาคาร/กสิกร",
Line ID) → ใช้กับภาษาอังกฤษแทบไม่ได้

**เขียน detector ชุดใหม่แยก** (อย่าแก้ไฟล์เดิม) จับ:
- email
- เบอร์โทรรูปแบบสากล (US/UK/EU) — **ระวังชนกับเลขเวอร์ชัน เลขบรรทัด timestamp ใน code**
- SSN สหรัฐ `\d{3}-\d{2}-\d{4}` — **ต้องมีคำบริบท** ไม่งั้นชนกับเลขทั่วไป
- บัตรเครดิต — **ต้องผ่าน Luhn และมีคำบริบท** (บทเรียนจากงานก่อน: Luhn อย่างเดียว false positive ~1 ใน 10)
- IP address ที่ไม่ใช่ private range

**นโยบาย: redact ไม่ drop**

---

## ส่วนที่ 3 — ปรับสัดส่วนภาษา code + ดึงข้อมูลเพิ่ม

### ปัญหาที่วัดได้

| ภาษา | ตอนนี้ | ปัญหา |
|---|---|---|
| **HTML** | **17.7% ของ tokens** | มากกว่าภาษาโปรแกรมทุกตัว · ตัวอย่างที่เปิดดูเป็น javadoc-generated boilerplate |
| Markdown | 4.3% | |
| CSS | 3.1% | |
| **รวม markup** | **25.7%** | เจือจางวัตถุประสงค์ของ code slice |
| **Python** | **6.8%** | ต่ำผิดปกติเทียบสถิติการใช้งานจริง |

### ⚠️ เหตุผลของการปรับ — ต้องบันทึกให้ถูก

**ปรับเพราะสัดส่วนปัจจุบันไม่สะท้อนการกระจายของโค้ดในโลกจริง**
(HTML ไม่เคยเป็น 17.7% ของโค้ดในสถิติ GitHub ใด ๆ · Python ติด top-2 เสมอ)
มันเป็น artifact ของวิธีสุ่มของ `github-code-clean`

**❌ ห้ามบันทึกเหตุผลว่า "เพราะ HumanEval เป็น Python"** นั่นคือ metric gaming
ให้อ้างสถิติการใช้งานภาษาโปรแกรมจากแหล่งภายนอก และ**ระบุแหล่งที่อ้าง**

### สัดส่วนเป้าหมาย (± 3 percentage point ยอมรับได้)

| กลุ่ม | เป้า % ของ tokens |
|---|---|
| Python | 15% |
| JavaScript + TypeScript | 15% |
| Java | 12% |
| C + C++ | 12% |
| C# | 4% |
| **หางยาว** (Go, Rust, PHP, Ruby, Shell, SQL, Scala, Perl, Lua, Haskell, Kotlin, Swift, R, …) | **25%** |
| Markdown | 5% |
| HTML | 5% |
| CSS + config (Makefile, Dockerfile, YAML, TOML) | 7% |

**ห้ามให้ภาษาใดเกิน 15%** และ**หางยาวต้องได้ ≥ 20%** เพื่อรักษาความเป็น generalist

### วิธีทำ

`codeparrot/github-code-clean` มี **880 shards** งาน A ใช้ไปแค่ **9 shards**
→ ดึงเพิ่มจาก shard ที่ยังไม่ได้ใช้ **สุ่มแบบกระจายทั่วช่วง 0–879** ไม่ใช่เอาต่อกันเป็นก้อน

- ใช้ **license allowlist เดิม**: `mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`
- ผ่าน `is_heldout()` และ exact dedup เดิม
- **เป้ารวม ≥ 1.0B Qwen tokens** หลังปรับสัดส่วนและหลัง secret redaction
- ถ้าตัด HTML ลงเหลือ 5% จะเหลือราว 0.9B → **ต้องดึงเพิ่มแน่นอน**

---

## สิ่งที่ต้องส่งมอบ

### สคริปต์
```
phase0/secret_scan.py       # detector + redaction (มี __main__ ทดสอบตัวเองได้)
phase0/pii_filter_en.py     # PII สำหรับ en/code/math (แยกจากของไทย)
phase0/rebalance_code.py    # ดึงเพิ่ม + ปรับสัดส่วน
phase0/apply_scan.py        # รัน secret+PII ทับทั้ง replay corpus
```

### ข้อมูล (จะถูก gitignore)
```
data/clean_replay_v2/{en,code,math}/*.jsonl.gz
```

### Manifest
```
data/clean_replay_v2_manifest.json
```
ต้องมี: source + **revision ที่ pin** + license · docs/tokens เข้า-ออก ·
**secret ที่พบแยกตามประเภท** · **PII ที่พบแยกตามประเภท** ·
**การกระจายภาษาก่อน/หลังปรับ (เป็น % ของ tokens)** · `scientific_evidence_allowed: false` · `limitations`

### ตัวอย่างให้ตรวจด้วยตา
```
data/scan_samples/secret_<type>.jsonl     # เอกสารที่โดน redact พร้อมบริบทรอบ ๆ
data/scan_samples/pii_<type>.jsonl
```
**อย่างน้อย 20 ตัวอย่างต่อประเภท** พร้อม **ข้อความรอบ ๆ 100 ตัวอักษร**
(ห้ามใส่ค่า secret จริงลงไฟล์ — ใส่แค่ประเภท ตำแหน่ง และบริบทที่ mask แล้ว)

### อัปเดตไฟล์เดิม
- `sources/source_registry.csv` — เพิ่มแหล่งอ้างอิงสถิติภาษาโปรแกรมที่ใช้ **(ตรวจ id ไม่ซ้ำก่อน — เคยมีปัญหา id ชนกัน)**
- `validation/VALIDATION_OUTPUT_LOG.md` — **เพิ่ม entry ท้ายไฟล์ ก่อน `## Required future entries`**

---

## เกณฑ์ว่างานเสร็จ

- [ ] secret detector ผ่านเทสต์: จับ key ปลอมที่สร้างขึ้นได้ครบ **และไม่จับ placeholder**
- [ ] สแกน replay corpus ครบทุกเอกสาร รายงานจำนวนที่พบแยกประเภท
- [ ] private key ถูกลบทั้งเอกสาร · key อื่น redact แล้ว
- [ ] PII detector สำหรับ en ทำงานจริง (**ทดสอบว่าจับ email ได้จริง** — บทเรียนด้านล่าง)
- [ ] code หลังปรับ: ไม่มีภาษาใดเกิน 15% · หางยาว ≥ 20% · **รวม ≥ 1.0B tokens**
- [ ] เหตุผลการปรับสัดส่วนอ้างสถิติภายนอก **ไม่อ้าง HumanEval**
- [ ] ข้อมูลใหม่ทั้งหมดผ่าน `is_heldout()` และ license allowlist
- [ ] มีตัวอย่างให้ตรวจด้วยตา ≥ 20 ต่อประเภท
- [ ] entry ใหม่ใน VALIDATION_OUTPUT_LOG.md · registry ไม่มี id ซ้ำ

---

## ⚠️ บทเรียนจากงานก่อนหน้า — จะถูกตรวจข้อเหล่านี้

โปรเจกต์นี้เจอบั๊กที่ **รันผ่าน ไม่ error ให้ตัวเลขที่ดูสมเหตุสมผล แต่วัดผิด** มาแล้ว 5 ตัว

| บั๊ก | อาการ |
|---|---|
| `\b` กับภาษาไทย | อักษรไทยเป็น word character → detector อีเมลจับได้ **0 จาก 809 เอกสารที่มี `@`** |
| byte-level BPE vocab | ไม่ถอดรหัสก่อน → นับ Thai vocab pieces ได้ **0** ทั้งที่มี 2,571 |
| `tokenizers.encode()` | ใส่ `<bos>` เอง → นับ token เกิน 1 ต่อเอกสาร |
| Luhn อย่างเดียว | จับ `166617 106017 106617` เป็นบัตรเครดิต |
| `.float()` บน logits เต็ม vocab | memory thrashing 6.6% CPU — ช้าลง 10 เท่าโดยไม่ error |

**กฎที่ต้องทำตาม:**
1. **หลังเขียน detector ทุกตัว ทดสอบกับข้อมูลจริงทันที** ถ้าได้ **0 หรือน้อยผิดปกติ ให้สงสัยว่า detector พัง** ก่อนจะสรุปว่าข้อมูลสะอาด
2. **สุ่มพิมพ์ตัวอย่างที่โดนจับออกมาดูด้วยตา ≥ 20 อัน** ทุกประเภท (`phase0/sample_dropped.py` เป็นแบบอย่าง)
3. **นับ token ด้วย Qwen3-1.7B-Base + `add_special_tokens=False` เสมอ**
4. ตรวจ **cross-check** — เช่นเทียบยอดรวมจาก shard metadata กับ manifest ว่าตรงกัน

---

## รายงานเมื่อเสร็จ

1. secret ที่พบ แยกประเภท กี่รายการ กี่เอกสาร
2. PII ที่พบ แยกประเภท
3. **การกระจายภาษา code ก่อน/หลัง เป็นตาราง %**
4. token รวมสุดท้ายต่อภาษา
5. ดึงข้อมูลเพิ่มจาก shard ไหนบ้าง
6. **ตัดสินใจอะไรเองบ้าง เพราะอะไร**
7. **อะไรที่ยังไม่ปิด / ยังเสี่ยง**
