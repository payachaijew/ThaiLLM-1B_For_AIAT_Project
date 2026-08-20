# Implementation Area

**สถานะ:** ยังไม่มี scientific implementation ในไดเรกทอรีนี้

สคริปต์ที่ใช้งานจริงทั้งหมดของ Phase 0/1 อยู่ที่ [`../phase0/`](../phase0/)
ไดเรกทอรีนี้สงวนไว้สำหรับ **Track 2 (AttnRes controlled study)** ซึ่งยังไม่เริ่ม

---

## 🔑 การตัดสินใจสำคัญ: ใช้ implementation ทางการ ไม่เขียนเอง

repo ทางการของ Delta AttnRes เป็น **MIT license**
→ https://github.com/wdlctc/delta-attention-residuals-code

มีให้ครบ: `modeling_qwen3_attnres.py` · DDP 220M–1B · FSDP 7B+ · fine-tuning ·
downstream eval · `WANDB_RUNS.md` ที่บันทึก config ของทุกการทดลองใน paper

> เอกสารรุ่นก่อนของไฟล์นี้ระบุว่า repo ทางการ **"unlicensed"** ซึ่ง**ไม่ถูกต้อง**
> การตรวจเมื่อ 2026-08-18 ยืนยันว่าเป็น MIT
> **เหตุผลที่ต้องเขียน implementation เองจึงหมดไป**

**และเนื่องจาก base ที่เลือกคือ Qwen3-1.7B-Base ซึ่ง repo ทางการมี `modeling_qwen3_attnres.py`
ให้อยู่แล้ว Track 2 จึงแทบไม่ต้องเขียนโค้ดใหม่เลย** — นี่เป็นหนึ่งในเหตุผลที่เลือก base ตัวนี้

---

## ⛔ ห้ามนำ implementation เดิมของทีมมาใช้โดยไม่แก้

harness เก่าอยู่ที่ `../../thai-llm-five-to-two/depth_routing/` (archived, ห้ามแก้ไข)
การตรวจพบข้อบกพร่องระดับ **blocking**:

| ID | ปัญหา |
|---|---|
| **ROUTE_SCALE_GATE** | ตัวแปร `route_scale` ที่ init = 0 (ไม่มีใน paper) ทำให้ gradient ของ router **เป็นศูนย์พอดีตอนเริ่ม** และหลัง 50 steps ยังได้แค่ 0.095% ของแบบที่ทำตาม paper |
| E1 | **MHAR (D2) ไม่มีโค้ดเลย** |
| E2 | `last_routing` ค้าง autograd graph หลัง forward → กิน memory เฉพาะฝั่ง routed |
| E3 | block-size semantics กำกวม — `block_size_layers=4` บน 8 ชั้น ทำให้ชั้น 0–3 ได้ 0 sources |
| E4 | ปิด gradient checkpointing เฉพาะ routed arms → ลำเอียงบนแกน GPU-hour |
| E5 | ยังบังคับกฎ 10% overhead kill ที่โปรเจกต์นี้ยกเลิกไปแล้ว |

**E2 และ E4 ลำเอียงต่อต้านฝั่ง routed ทั้งคู่** เมื่อรวมกับ throughput ของ AttnRes ที่
`arXiv:2607.27230` รายงานไว้ที่ 0.55–0.88× ผลลัพธ์ "routing แพ้" จะตีความไม่ได้

รายละเอียดเต็มใน [`../validation/VALIDATION_OUTPUT_LOG.md`](../validation/VALIDATION_OUTPUT_LOG.md)
entry `VAL-2026-08-18-LOCAL-STATIC-AUDIT`

---

## เงื่อนไขก่อนเริ่ม Track 2

1. ใช้ implementation ทางการ (MIT) **หรือ** ลบ `route_scale` แล้วพิสูจน์ว่า query gradient ≠ 0 ที่ step 1
2. ปิด E2 และ E4 ให้ทั้ง 3 conditions อยู่ใน memory regime เดียวกัน
3. D2/MHAR มีโค้ด + ผ่าน identity conversion + router gradient ≠ 0
4. รักษา `scientific_evidence_allowed=false` ในทุก local/smoke output
5. **ห้ามใส่สถาปัตยกรรมวิจัยลงใน product** — ThaiLLM-1B-Base ใช้ Standard Residual เท่านั้น
