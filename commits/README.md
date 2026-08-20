# ลำดับ commit

ไฟล์ `C01..C12_*.txt` แต่ละไฟล์คือ **รายชื่อไฟล์ที่จะเข้า commit นั้น** อ่านและแก้ได้ตามต้องการ

```bash
./commits/make_commits.sh      # สร้าง history ทั้งหมดตามลำดับ
```

## หลักการเรียง

**รากฐาน → หลักฐาน** เพื่อให้ `git log` อ่านเป็นเรื่องราวของงานวิจัย
และ 1 commit = สคริปต์ + ผลลัพธ์ ที่อยู่ด้วยกันได้

| commit | เนื้อหา | ทำไมอยู่ตรงนี้ |
|---|---|---|
| C01 | scaffolding | ต้องมี .gitignore ก่อนแตะอย่างอื่น |
| C02 | แผนงาน | บอกว่าจะทำอะไร ก่อนแสดงว่าทำอะไรไปแล้ว |
| C03 | config + template | พารามิเตอร์ที่ทุกอย่างอ้างถึง |
| C04 | ทะเบียนแหล่งอ้างอิง | หลักฐานว่าตรวจ paper/model/dataset มาแล้ว |
| C05 | novelty audit | เหตุผลว่าทำไมหัวข้อนี้ยังทำได้ |
| C06 | base-model screen | งานคัดเลือก base ของทีม |
| **C07** | tokenizer screen ต่อยอด | Gemma-4 2.83 vs Qwen3 1.84 chars/token |
| **C08** | **แช่แข็ง eval suite** | 🔑 **ต้องมาก่อน C10 เสมอ** |
| C09 | held-out rule + builder | นิยามชุดวัด |
| **C10** | baseline BPB | 1.7B 0.454218 · 0.6B 0.521386 |
| C11 | corpus audit | dup 21% · gambling 3.25% bytes |
| C12a | clean pipeline | 16.4M → 4.57M docs · 5.851B tokens |
| C12b | PII redaction | 108,237 docs · เลขบัตรประชาชน 420 |
| C13 | เอกสาร + validation log | log ครบ · license · model card |

## 🔑 ทำไม C08 ต้องมาก่อน C10

git history เป็น **หลักฐานทางเวลา** ว่าเรากำหนด metric **ก่อน** เห็นผล
reviewer ตรวจเองได้จาก timestamp ไม่ต้องเชื่อคำพูดเรา
ถ้า commit baseline ก่อนแล้วค่อย commit eval suite จะดูเหมือนเลือก metric ให้เข้ากับผล
