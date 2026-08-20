# ลำดับการ commit

**หลักการ:** 1 commit = **1 หน่วยงานที่สมบูรณ์** = สคริปต์ + ผลลัพธ์
→ `git show <commit>` แล้วเห็นครบว่า *ทำอะไร ได้อะไร* โดยไม่ต้องเปิดไฟล์อื่น

เรียงจาก **รากฐาน → หลักฐาน** เพื่อให้ `git log` อ่านเป็นเรื่องราวของงาน

| # | commit | เนื้อหา |
|---|---|---|
| C01 | scaffolding | `.gitignore` `README.md` `requirements.txt` `COMMIT_PLAN.md` |
| C02 | แผนงาน | `plans/` `src/README.md` |
| C03 | config + template | `configs/` `run_record.template.json` |
| C04 | ทะเบียนแหล่งอ้างอิง | `sources/source_registry.csv` (46 รายการ) |
| C05 | novelty audit | `novelty/` |
| C06 | base-model screen | `base_selection/` |
| **C07** | tokenizer screen ต่อยอด | Gemma-4 2.8326 vs Qwen3 1.838 chars/token |
| **C08** | 🔑 **แช่แข็ง eval suite** | **ต้องมาก่อน C10 เสมอ** |
| C09 | held-out rule + builder | bucket rule + stratified shards |
| **C10** | baseline BPB + headroom | 1.7B 0.454218 · 0.6B 0.521386 · Sailor2 0.378051 |
| C11 | corpus audit | dup 21.0% · gambling 3.25% bytes |
| C12a | clean pipeline | 16.4M → 4.57M docs · 5.851B tokens |
| C12b | PII redaction | 108,237 docs · เลขบัตรประชาชน 420 |
| C13 | เอกสาร + validation log | log ครบทุก entry · license · model card |

---

## 🔑 ทำไม C08 ต้องมาก่อน C10

git history เป็น **หลักฐานทางเวลา** ว่ากำหนด metric **ก่อน** เห็นผล
reviewer ตรวจเองได้จาก timestamp ไม่ต้องเชื่อคำพูดเรา
ถ้า commit baseline ก่อนแล้วค่อย commit eval suite จะดูเหมือนเลือก metric ให้เข้ากับผล

---

## คำสั่ง

```bash
git init && git branch -M main
./commits/make_commits.sh          # หรือรันทีละ commit ตาม commits/*.txt
```

**ตรวจก่อน push**
```bash
git log --oneline                              # ควรได้ 13 commits
git count-objects -vH | grep size-pack         # ควรต่ำกว่า ~5 MB
git ls-files | xargs du -h 2>/dev/null | sort -rh | head -5
```

---

## ⚠️ ข้อควรระวัง

1. **ตั้ง repo เป็น private ก่อน** จนกว่าจะเคลียร์เรื่อง license กับ mentor — ดู `LICENSE_COMPLIANCE.md`
2. `.gitignore` กัน `data/clean/` (6.7 GB) · `data/clean_pii/` · `*.jsonl` · `overnight_logs/` ไว้แล้ว
3. **ไม่ต้องใช้ Git LFS** — ทุกไฟล์ที่ commit เป็น text เล็ก ๆ รวมราว 11 MB
4. ข้อมูลไม่ถูกเผยแพร่ แต่**สร้างซ้ำได้เป๊ะ**จากสคริปต์ + revision ที่ pin ไว้
