# ThaiLLM-1B CPT Corpus — v2

> **⚠️ ยังไม่เผยแพร่สาธารณะ** เอกสารนี้ใช้กับ repo แบบ private เท่านั้น
> จนกว่าคำถาม license/PDPA ใน [`LICENSE_COMPLIANCE.md`](LICENSE_COMPLIANCE.md) จะได้ข้อสรุป

ข้อมูลสำหรับ continued pretraining ภาษาไทย ผ่านการทำความสะอาดหลายชั้นและตรวจสอบซ้ำแบบอิสระ

---

## เนื้อหา

| ส่วน | เอกสาร | Qwen3 tokens | ที่มา | license |
|---|---:|---:|---|---|
| **ไทย** | 4,567,214 | **5.851B** | SEA-PILE-v2 subset `th` | ODC-By 1.0 + CommonCrawl ToU |
| **อังกฤษ** | 3,507,052 | **3.513B** | FineWeb-Edu `sample/10BT` | ODC-By 1.0 + CommonCrawl ToU |
| **code** | 690,716 | **1.000B** | github-code-clean *(กรองเหลือ MIT/Apache-2.0/BSD-2/BSD-3)* | ต่อ repository |
| **math** | 412,734 | **0.592B** | FineMath-4plus | ODC-By 1.0 + CommonCrawl ToU |
| **รวม** | **9,177,716** | **10.96B** | | |

**Revision ที่ pin ไว้**
```
SEA-PILE-v2       77573cc84631412a781daa8e6f72cf322d4207f0
FineWeb-Edu       87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
github-code-clean c48d40f9e70f0196f8236901ee35807f7d6c44c0
FineMath          e92b25a616738fe95dc186b64dfb19f9c8525594
```

## รูปแบบ

gzipped JSONL หนึ่งเอกสารต่อบรรทัด **เป็นข้อความ ไม่ใช่ token** จึงใช้กับ tokenizer/โมเดลใดก็ได้

```json
{"text": "...", "doc_sha256": "...", "utf8_bytes": 1234, "qwen_tokens": 456,
 "url": "...", "language": "Python", "license": "mit", "source_repo": "...", "source_revision": "..."}
```

`qwen_tokens` นับด้วย Qwen3-1.7B-Base + `add_special_tokens=False` — **ถ้าใช้ tokenizer อื่นต้องนับใหม่**

---

## 🔴 ต้องอ่านก่อนใช้: `removal_list.txt`

```
data/removal_list.txt   273,703 doc_sha256 (2.98% ของทั้งหมด)
```

**ต้องตัดเอกสารเหล่านี้ออกก่อนเทรน** ประกอบด้วย

| เหตุผล | จำนวน |
|---|---|
| ปนเปื้อน benchmark | 3,233 |
| near-duplicate | 270,841 |

> **ถ้าไม่ตัดออก จะเทรนทับข้อสอบที่จะใช้วัดผล** ตัวอย่างที่จับได้จริงคือเว็บ
> `gorporonline.com` ที่มี **แนวข้อสอบ ก.พ. ภาษาไทย** (364 n-gram hits) ซึ่งเป็นข้อสอบ
> รูปแบบเดียวกับ ThaiExam — คะแนนที่ได้จะเป็นการจำ ไม่ใช่ความสามารถ

**ห้ามใช้ `data/heldout/`** เป็นข้อมูลเทรนเด็ดขาด — เป็นชุดกันไว้วัดผล

---

## สิ่งที่ทำไปแล้ว

| ขั้น | ไทย | อังกฤษ | code | math |
|---|:-:|:-:|:-:|:-:|
| exact dedup | ✅ | ✅ | ✅ | ✅ |
| length floor | ✅ | ✅ | ✅ | ✅ |
| language ID filter | ✅ | ✅ | — | — |
| Thai gambling/lottery filter | ✅ | — | — | — |
| license allowlist | — | — | ✅ | — |
| **PII redaction** | ✅ ไทย | ✅ สากล | ✅ | ✅ |
| **secret redaction** | — | ✅ | ✅ | ✅ |
| **benchmark decontamination** | ✅ | ✅ | ✅ | ✅ |
| **near-duplicate detection** | ✅ | ✅ | ✅ | ✅ |

**PII ที่ลบไป:** เลขบัตรประชาชนไทย 420 (ตรวจ checksum) · เบอร์โทร 110,042 · อีเมล 193,439 ·
บัญชีธนาคาร 4,163 · Line ID 104,121 · public IP 5,003 · SSN 143 · บัตรเครดิต 211

**Secret ที่ลบไป:** private key 75 เอกสาร (ลบทั้งเอกสาร) · Google API key 319 ·
connection string 183 · env secret 137 · JWT 135 · AWS 30 · Slack 4 · Stripe 1

**สัดส่วนภาษาโปรแกรมปรับใหม่** อ้างอิงสถิติการใช้งานจริง (GitHub Octoverse 2025)
**ไม่ได้อ้าง benchmark ใด ๆ** เพื่อไม่ให้เป็น metric gaming:
HTML 17.8% → 5.0% · Python 6.9% → 14.5% · ไม่มีภาษาใดเกิน 15% · หางยาว 26.7%

---

## ⚠️ ข้อจำกัดที่รู้ตัว — อ่านก่อนตัดสินใจใช้

1. **ตัวกรองเป็น regex + context guard ไม่ใช่โมเดล** — จับ **ชื่อคน ที่อยู่ วันเกิด ไม่ได้**
2. **decontamination จับเฉพาะการซ้ำแบบตรงตัว/เกือบตรงตัว** (character 64-gram)
   ข้อสอบที่ถูก**ถอดความหรือแปล จับไม่ได้**
3. **near-dedup ใช้ MinHash 32 permutations** เน้นคู่ที่คล้ายกันสูง — คู่ที่คล้ายปานกลางหลุด
   และ LSH เก็บเอกสาร**ตัวแรก**ที่เจอ ตัวไหนรอดขึ้นกับลำดับไฟล์ ไม่ใช่คุณภาพ
4. **code build มี `[REDACTED_SECRET]` แทน placeholder ใน 656 จาก 4.6M docs (0.014%)**
   เป็นการ redact **เกิน** ไม่ใช่ขาด — ฝั่งที่ปลอดภัยสำหรับตัวกรองความปลอดภัย
5. **ตัวกรองหวยมี false positive** (เว็บแฟรนไชส์/กีฬาที่บังเอิญมีคำว่า "หวย") กระทบ 1.6% ของ corpus ไทย
6. corpus ไทยมาจาก CommonCrawl **ปี 2020–2022+** ไม่รู้เหตุการณ์หลังจากนั้น
7. **provenance ราย repository ของ code ตรวจแค่ field license** ยังไม่ได้ audit ลึกกว่านั้น
8. **ยังไม่ได้ทำ**: quality classifier · toxicity filter · PII แบบ NER

---

## Attribution (ODC-By 1.0 บังคับ)

```
ข้อมูลนี้ดัดแปลงมาจาก:
  SEA-PILE-v2 (AI Singapore) — ODC-By 1.0
  https://huggingface.co/datasets/aisingapore/SEA-PILE-v2
  FineWeb-Edu (HuggingFace) — ODC-By 1.0
  FineMath (HuggingFace) — ODC-By 1.0
  github-code-clean (CodeParrot) — ไฟล์ต้นทางคงลิขสิทธิ์ของ repository เดิม

เนื้อหาต้นทางมาจาก CommonCrawl ภายใต้ CommonCrawl Terms of Use
เนื้อหาแต่ละชิ้นยังเป็นลิขสิทธิ์ของเจ้าของเว็บเดิม
```

## ทำซ้ำได้

สคริปต์ทั้งหมดอยู่ใน `phase0/` — bucket rule เป็น deterministic hash และทุก revision ถูก pin
จึงสร้างข้อมูลชุดเดียวกันซ้ำได้เป๊ะจากต้นทาง

## ตรวจสอบแล้วแบบอิสระ

`phase0/verify_replay_v2.py` คำนวณทุกตัวเลขใหม่จากข้อมูลเอง ไม่เชื่อไฟล์สรุปใด ๆ
→ exact duplicates **0** · held-out leakage **0** · token recount mismatch **0** → **PASS**
