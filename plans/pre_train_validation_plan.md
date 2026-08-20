# Pre-Training Validation Plan

> **เปลี่ยนชื่อจาก `pre_lanta_validation_plan.md` (2026-08-19)** — โครงการเปลี่ยนจากรอ LANTA
> ไปเป็น **เช่า GPU** เพราะ LANTA อนุมัติช้า เนื้อหาของแผนยังใช้ได้ทั้งหมด
> เพียงแต่ "ก่อน LANTA" ให้อ่านเป็น "ก่อนจ่ายค่าเช่า GPU"

The goal before spending any GPU budget is to reject broken implementations and poor base/data
choices cheaply. Local wins are not scientific evidence for the 1B claim.

## Tier A — no pretrained weights

1. Equation-level Standard, Delta Block and MHAR tests.
2. Exact identity conversion on tiny random configurations.
3. Maximum logit and loss error before/after conversion.
4. Non-zero router gradient and one optimizer-step test.
5. Checkpoint save/load and resume determinism.
6. Standard-path recovery.
7. Causal mask, padding, packing and mixed-length tests.
8. Deterministic seed, manifest and hash tests.
9. Data filtering, language ID, deduplication and split-leakage tests.
10. BPB/bootstrap fixture tests with known outputs.

## Tier B — tokenizer and frozen model

1. Download tokenizer-only artifacts for all base candidates.
2. Compare Thai/English/code bytes/token, characters/token and tokens/document.
3. Resolve immutable model revisions and licenses.
4. Run frozen Thai/English/code BPB where local hardware permits.
5. Run a small frozen downstream screen.
6. Check benchmark hashes against the candidate CPT sample.

## Tier C — local/MPS or small-GPU smoke

1. Overfit 32–128 short sequences with S0 and D1.
2. Confirm loss decreases and routed parameters update.
3. Confirm resume reproduces the next-step loss.
4. Confirm functional equivalence immediately after migration.
5. Optionally train a 20–50M toy decoder to expose routing collapse or numerical
   instability.

Every Tier C record must use:

```text
scientific_evidence_allowed=false
```

## One-GPU preflight (rented A100/H100)

- 20 warm-up and 200 measured steps;
- BF16 with frozen sequence length/global batch;
- conditions S0, D1 and D2;
- record step-time distribution, tokens/second, allocated/reserved VRAM,
  compile time, loss stability and failures;
- do not reject solely because overhead exceeds 10%; evaluate whether the
  quality-per-GPU-hour curve can compensate.

## Scientific premise pilot

- Pilot model: Qwen3-0.6B-Base (base screen ปิดแล้ว; main base = Qwen3-1.7B-Base).
- 50M–100M observed tokens per condition.
- S0, D1 and D2.
- One seed for screening; a second seed is mandatory before promotion.
- Primary analysis: Thai BPB versus observed tokens and versus GPU-hours.
- Retention controls: English and code BPB/downstream aggregates.
- Mechanism controls: entropy, max route weight, source usage and clamping.



---

## สถานะ ณ 2026-08-19

| Tier | สถานะ |
|---|---|
| **Tier A** — ทดสอบสมการ/identity/gradient บน tiny model | ✅ ผ่าน 25/25 บน harness เดิม **แต่พบบั๊ก ROUTE_SCALE_GATE** ที่ unit test มองไม่เห็น |
| **Tier B** — tokenizer + frozen model | ✅ เสร็จ — ดู `../base_selection/` และ `../phase0/tokenizer_screen_ext.json` |
| **Tier C** — local smoke | ✅ BPB pipeline ทดสอบแล้วบน MPS |
| **One-GPU preflight** | ⬜ ยังไม่ทำ — ต้องทำบนเครื่องเช่า |
| **Scientific premise pilot** | ⬜ ยังไม่ทำ |

**บทเรียนสำคัญจาก Tier A:** unit test ผ่านครบ 25/25 แต่ยังปล่อยบั๊กที่ทำให้ router
เรียนรู้ไม่ได้ผ่านไปได้ เพราะเทสต์ตรวจ "ฟังก์ชันทำตามที่เขียนไหม" ไม่ได้ตรวจว่า
"สาม condition เทียบกันได้จริงไหม" → **ต้องเพิ่มเทสต์ที่ปกป้องข้อสรุป ไม่ใช่แค่ปกป้องฟังก์ชัน**
