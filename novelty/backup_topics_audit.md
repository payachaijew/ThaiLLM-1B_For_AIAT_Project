---
scientific_evidence_allowed: false
audit_date: 2026-08-24
venue_target: iSAI-NLP 2026
project: ThaiLLM-1B
---

# Novelty audit — backup topics for ThaiLLM-1B

## สถานะและขอบเขต

เอกสารนี้เป็น **desk-based novelty audit** ไม่ใช่ผลการทดลองและไม่ใช่คำปรึกษากฎหมาย
ตัวเลขภายในโครงการ (86.6%, 6.7%, 803 เอกสาร, 0.018%, 2.95%, ค่าใช้จ่าย US$8 และบั๊ก 8 รายการ)
ถือเป็น **ข้ออ้างจากผู้ใช้ที่ยังไม่ได้ตรวจซ้ำ** จึงห้ามนำเอกสารนี้ไปอ้างเป็น scientific evidence
(`scientific_evidence_allowed: false`)

วันที่ค้นทุก query: **2026-08-24 (Asia/Bangkok)**. Cutoff ของ audit คือสิ่งที่ค้นพบได้ในวันดังกล่าว
และต้อง rerun ก่อนส่งบทความจริง เพราะงานปี 2026 ยังเพิ่มเร็วมาก

### วิธีนับผลลัพธ์

- Query log ใช้ OpenAlex Works API (`https://api.openalex.org/works?search=<QUERY>&per-page=1`)
  และบันทึก `meta.count` ซึ่งเป็นจำนวนระเบียนงานวิชาการที่ OpenAlex จับคู่ ไม่ใช่จำนวนหน้าเว็บจาก Google
- “ชนเพดาน” หมายถึง API แจ้งว่าจำนวนถูก truncate/จำกัดโดยระบบ ไม่ใช่แค่จำนวนมาก; รอบนี้ไม่มี query ใดแจ้ง truncation
- จำนวนเป็น snapshot ที่เปลี่ยนได้เมื่อ OpenAlex re-index; query string ด้านล่างคือข้อความเต็มก่อน URL encoding
- การคัด nearest work ใช้ web search เพื่อ discovery แล้วเปิดอ่านแหล่งต้นฉบับ ได้แก่ arXiv HTML/abstract,
  ACL Anthology, เอกสารทางการ หรือ Hugging Face dataset/model card; ไม่ตัดสินจาก search snippet
- การไม่พบงานไทย/SEA หลัง 4 query เป็นเพียง “ไม่พบใน audit นี้” ไม่ใช่ข้อพิสูจน์ว่าไม่มีงานดังกล่าว

## 1. Query log

| หัวข้อ | ID | Query string เต็ม | จำนวนผล (OpenAlex works) | ชนเพดานไหม | วันที่ค้น |
|---|---|---|---:|---|---|
| A | A1 | `"replay ratio" "continued pretraining" language model multilingual` | 2 | ไม่ชน | 2026-08-24 |
| A | A2 | `"data mixture" catastrophic forgetting "continued pre-training" language adaptation` | 34 | ไม่ชน | 2026-08-24 |
| A | A3 | `Thai language model continued pretraining English replay mixture` | 117 | ไม่ชน | 2026-08-24 |
| A | A4 | `Southeast Asian LLM continual pretraining catastrophic forgetting replay` | 20 | ไม่ชน | 2026-08-24 |
| B | B1 | `"dataset license audit" "instruction tuning" provenance` | 1 | ไม่ชน | 2026-08-24 |
| B | B2 | `"synthetic data" "model output" license terms instruction dataset` | 872 | ไม่ชน | 2026-08-24 |
| B | B3 | `"Hugging Face" dataset card license provenance instruction tuning` | 188 | ไม่ชน | 2026-08-24 |
| B | B4 | `Thai NLP instruction dataset license legal audit` | 365 | ไม่ชน | 2026-08-24 |
| C | C1 | `"benchmark contamination" Thai language model` | 91 | ไม่ชน | 2026-08-24 |
| C | C2 | `"test set leakage" Southeast Asian NLP benchmark` | 3 | ไม่ชน | 2026-08-24 |
| C | C3 | `"data decontamination" n-gram LLM pretraining` | 30 | ไม่ชน | 2026-08-24 |
| C | C4 | `"character n-gram" benchmark contamination low-resource language` | 163 | ไม่ชน | 2026-08-24 |
| D | D1 | `"checkpoint resume" correctness distributed training nondeterminism` | 2 | ไม่ชน | 2026-08-24 |
| D | D2 | `CUDA nondeterminism resumed training reproducibility checkpoint` | 11 | ไม่ชน | 2026-08-24 |
| D | D3 | `GPU training preflight checklist low resource lab cost` | 2 | ไม่ชน | 2026-08-24 |
| D | D4 | `empirical study distributed training bugs checkpointing reproducibility` | 908 | ไม่ชน | 2026-08-24 |

**อ่านความหนาแน่นอย่างระวัง:** B2 และ D4 ให้ผลกว้างมาก (872 และ 908) เพราะถ้อยคำครอบคลุมหลายสาขา
แต่ไม่ชนเพดานของ API; จึงเป็นสัญญาณวรรณกรรมหนาแน่น ไม่ใช่หลักฐานว่าข้ออ้างเฉพาะของโครงการถูกทำแล้ว

## 2. หัวข้อ A — replay ratio สำหรับ CPT ภาษาไทย

### Nearest work

| ชื่องาน | ปี | ทำอะไร | ทับกับเราตรงไหน | เหลือช่องว่างอะไร | แหล่งต้นฉบับที่เปิดอ่าน |
|---|---:|---|---|---|---|
| Breaking Language Barriers: Cross-Lingual Continual Pre-Training at Scale | 2024 | ทำ CPT English→Chinese; ที่ 1.4B sweep English replay **1, 5, 10, 20, 50, 80%**; วาด loss curves ทั้ง source/target และสรุป 5–30% ช่วยกันลืม | ชนแกนกลางแทบตรงทั้งหมด: สเกล ~1B, ratio sweep, acquisition–retention curves, catastrophic forgetting | ไม่ใช่ไทย; retention หลักเป็น English held-out cross-entropy loss และ downstream multilingual tasks ไม่ใช่ชุด general-capability ไทย/อังกฤษที่ออกแบบสำหรับ Qwen | [ACL Anthology](https://aclanthology.org/2024.emnlp-main.441/) และ [PDF](https://aclanthology.org/2024.emnlp-main.441.pdf) |
| Efficiently Adapting Pretrained Language Models to New Languages / SambaLingo | 2023/2024 | ทดลอง adaptation ภาษาไทยและฮังการี แล้วขยายเป็น 9 ภาษา; สูตร CPT ใช้ English:target = **1:3** (25% English), ที่ 7B และบางภาษา—รวมไทย—70B | มีภาษาไทยโดยตรงและมี replay/mixing เพื่อกัน English regression | ใช้สัดส่วน CPT เดียว ไม่ได้ sweep replay ratio สำหรับไทย; สเกลสูงกว่า ~1B | [arXiv 2311.05741](https://arxiv.org/abs/2311.05741), [ACL SambaLingo](https://aclanthology.org/2024.mrl-1.1/), [Thai model card](https://huggingface.co/sambanovasystems/SambaLingo-Thai-Base) |
| Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models | 2026 | Spectra/Llama 99M, 560M, 1B, 6B; ภาษาต่อเนื่อง English→French→German (เพิ่ม Arabic/Japanese ใน corpus); เทียบ **0, 25, 50%** replay และ gradient alignment; วัด retained validation loss, average forgetting, plasticity และ HellaSwag/PIQA/PubMedQA | ตรงทั้ง replay sweep, scale analysis, retention/plasticity และมี 1B | ไม่มีไทย; task เป็นลำดับหลายภาษาและมีวิธี Reptile/MER เพิ่ม จึงไม่ตอบ optimum เฉพาะ Thai CPT ของ Qwen | [arXiv 2508.01908](https://arxiv.org/abs/2508.01908), [PMLR CoLLAs 2026](https://proceedings.mlr.press/v330/abbes26a.html) |
| Balancing Synthetic Data and Replay for Enhancing Task-Specific Capabilities | 2025 | SmolLM2-1.7B; grid 25 เงื่อนไขจาก token budget 10^7–10^9 และ replay **5, 10, 15, 20, 25%**; acquisition = bAbI accuracy; retention = ค่าเฉลี่ย HellaSwag, ARC-Easy, PIQA, MMLU, CommonsenseQA, WinoGrande, OpenBookQA, MathQA | สเกลและคำถาม optimum ratio ใกล้มาก และนิยาม acquisition/retention แบบ downstream | domain/task adaptation ไม่ใช่ cross-lingual Thai; ไม่วัด Thai capability | [arXiv 2510.11842](https://arxiv.org/abs/2510.11842), [OpenReview PDF](https://openreview.net/pdf/c4988f85d3bb42aa6f35baf00f852f5bc4da68f6.pdf) |
| OpenSeal: Good, Fast, and Cheap Construction of an Open-Source Southeast Asian LLM via Parallel Data | 2026 | CPT OLMo 2 ที่ 1B และ 7B บน 10 ภาษา SEA รวมไทย; ใช้ replay คงที่ **25%** ในทุก setting และเปรียบเทียบชนิด/ลำดับ monolingual vs parallel data | SEA/Thai, 1B, CPT, retention concern และ controlled data-composition experiments | ไม่ sweep replay ratio; เปลี่ยนชนิดและลำดับข้อมูลโดยตรึง replay 25% | [arXiv 2602.02266](https://arxiv.org/abs/2602.02266) |

### คำตอบคำถามบังคับ

- **มี replay-ratio sweep สำหรับภาษาไทยโดยเฉพาะไหม:** ไม่พบ sweep ไทยโดยเฉพาะใน 4 query นี้ แต่พบงานไทยที่ใช้ 25% แบบคงที่ (Csaki/SambaLingo) และงาน SEA รวมไทยที่ตรึง 25% (OpenSeal)
- **ของภาษาอื่นครอบคลุมเท่าไร:** Zheng et al. ครอบคลุม 1–80% ที่ 1.4B; Abbes et al. 0/25/50% ที่ 99M–6B; Spiegelhalter et al. 5–25% ที่ 1.7B
- **วัด retention ด้วยอะไร:** held-out source-language loss, average forgetting/retained loss, และ downstream general benchmarks ตามตาราง; ไม่มี metric เดียวที่เป็นมาตรฐาน

### คำตัดสิน A — **NARROW**

แกนวิธีวิจัย “sweep replay แล้ววาด acquisition–retention curve ที่ ~1B” ถูก Zheng et al. ทำแล้วอย่างตรงตัว
และใหม่กว่านั้นมีทั้ง grid 1.7B และ scale study ถึง 6B. ช่องว่างเหลือเพียง **Thai-specific replication บน base model/corpus/eval ที่เปิดเผย**
ซึ่งเหมาะกับ venue ระดับภูมิภาคได้ แต่ห้ามขายเป็นวิธีใหม่หรือ discovery ว่า replay มี trade-off.
ขอบเขตที่ยังพอป้องกัน novelty ได้คือ pre-register sweep ที่ครอบคลุมอย่างน้อย 0/1/5/10/20/30/50/80%,
รายงาน acquisition–retention Pareto frontier ด้วย Thai held-out loss + Thai downstream และ English held-out loss + English general benchmarks,
แล้วอธิบายว่าภาษาไทย (script/tokenizer/domain) ทำให้ optimum ต่างจาก Chinese/European baselines หรือไม่.

## 3. หัวข้อ B — licence audit ของ instruction data ภาษาไทย

### Nearest work

| ชื่องาน/แหล่ง | ปี | ทำอะไร | ทับกับเราตรงไหน | เหลือช่องว่างอะไร | แหล่งต้นฉบับที่เปิดอ่าน |
|---|---:|---|---|---|---|
| The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI | 2023/2024 | ตรวจ 44 alignment/instruction collections รวม 1,858 datasets; ไล่ lineage, source, license conditions; พบ license omission 69–72% และ label permissive เกินจริง 16–29%; วิเคราะห์ OpenAI-generated data/ToS โดยตรง | ชนข้ออ้าง (3) เต็ม และชนแกน “systematic instruction dataset licence audit”; ชี้ว่าข้อมูล synthetic/low-resource มักถูกจำกัด | ไม่ใช่ inventory ไทยแบบครบชุดและไม่ให้ตัวเลข 86.6/6.7 ของ WangchanThaiInstruct | [arXiv 2310.16787](https://arxiv.org/abs/2310.16787), [Nature Machine Intelligence](https://www.nature.com/articles/s42256-024-00878-8) |
| Legal Considerations in the Use of Synthetic Data for AI Development and Finetuning: The Case of LLMs4EU | 2026 | วิเคราะห์ GDPR, copyright, model licences และ acceptable-use policies ที่จำกัดการเอา output ไปสร้าง/finetune โมเดลอื่นตลอด data value chain | ชนข้ออ้าง (2) เรื่องข้อกำหนด teacher/model ส่งผลต่อสิทธิ์ใช้ synthetic training data | EU case study ไม่ใช่ไทยและไม่คำนวณ usable fraction ต่อ base family | [ACL Anthology](https://aclanthology.org/2026.legal-1.10/) |
| The Mirage of Artificial Intelligence Terms of Use Restrictions | 2024/2025 | วิเคราะห์ enforceability ของข้อกำหนดที่ผูก model weights/outputs และโต้แย้งว่าหลายข้ออาจบังคับใช้ยากเพราะ output/weights อาจไม่อยู่ใต้ copyright แบบที่ผู้ให้บริการอ้าง | หักล้างถ้อยคำแบบเด็ดขาดว่าเงื่อนไข “ติดกับ output เสมอ” | ไม่ใช่ dataset audit; เป็น legal analysis ภายใต้บริบทกฎหมายสหรัฐฯ | [arXiv 2412.07066](https://arxiv.org/abs/2412.07066) |
| WangchanThaiInstruct dataset card | 2023–ปัจจุบัน | การ์ดระบุ 100% human-annotated, มีทั้ง NC และ SA และ **license อยู่ระดับแถวตาม source** | เป็นวัตถุหลักของข้ออ้าง 86.6%/6.7% และยืนยันว่าป้ายระดับ repo ไม่พอ | ตัวเลข 86.6/6.7 ยังเป็นผลคำนวณภายในที่ audit นี้ไม่ได้ reproduce; ต้อง snapshot revision และแจกแจง dedup/config | [Hugging Face dataset card](https://huggingface.co/datasets/airesearch/WangchanThaiInstruct) |
| wangchanx-seed-free-synthetic-instruct-thai-120k + Qwen2-72B licence | 2024–ปัจจุบัน | dataset card ติด MIT แต่ระบุ Qwen2-72B-Instruct เป็น generator; Qwen licence ข้อ 5(b) ห้ามใช้ Materials หรือ output เพื่อ improve LLM อื่น ยกเว้น Tongyi Qianwen/derivatives | เป็นกรณีตัวอย่างตรงของ tag–upstream-term mismatch และทำให้ usable set ขึ้นกับสายโมเดลที่นำไปฝึก | ต้องวินิจฉัยว่า Qwen3-1.7B งานนี้เป็น “derivative works” ตามสัญญาหรือไม่ และ terms มีผลกับผู้รับ dataset ต่อหรือไม่—ไม่ควรตัดสินเอง | [Dataset card](https://huggingface.co/datasets/airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k), [Qwen2-72B-Instruct LICENSE](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/LICENSE) |

### คำตอบคำถามบังคับและการแก้ข้ออ้าง

- **มี systematic audit แล้วไหม:** มีแล้วอย่างชัดเจน—Data Provenance Initiative ตรวจ instruction/alignment collections 1,858 datasets และประเด็น license drift/mislabeling โดยตรง ดังนั้นห้ามอ้างว่าการ audit instruction licences เป็นสิ่งใหม่
- **เงื่อนไข teacher ส่งต่อไปยัง output หรือไม่:** ต้องแยกอย่างน้อยสามชั้น: (i) copyright/licence ของ input/source content, (ii) สัญญาหรือ ToS ระหว่างผู้ generate กับ provider, และ (iii) สิทธิ/ภาระของ downstream recipient ที่ไม่ได้เป็นคู่สัญญาเดิม. งาน DPI เองระบุว่าการผูก third party ที่ไม่ได้ generate ข้อมูลยังเป็นข้อถกเถียง; LLMs4EU ชี้ว่าต้องบริหาร contractual risk ทั้ง value chain; Lemley/Henderson โต้แย้ง enforceability. ดังนั้นประโยค **“เงื่อนไขติดกับ output เสมอ” ใช้ไม่ได้ใน paper**
- **base model มีผลต่อสิทธิ์ใช้ข้อมูลไหม:** กรณี Qwen เป็นหลักฐานเชิงสัญญาว่า *สายโมเดลปลายทาง* อาจทำให้ข้อยกเว้นต่างกัน จึงเป็น contract-conditioned eligibility ที่น่าสนใจ แต่ควรเรียกว่า “base-family-conditioned risk/eligibility under a stated interpretation” ไม่ใช่ข้อสรุปกฎหมาย
- **HF tag ไม่เท่ากับสิทธิจริง:** มี prior art รองรับแข็งมาก; novelty ไม่อยู่ที่ข้อสังเกตนี้

### คำตัดสิน B — **NARROW**

หัวข้อกว้าง “audit licence ของ instruction data” ถูกทำแล้ว และสองในสามข้ออ้างเป็น prior-art territory.
ช่องที่เหลือคือ **Thai-specific, revision-pinned, row-level executable audit** ที่ให้ตัวเลข advertised rows → unique rows → legally/contractually eligible rows
ภายใต้ base-family scenarios หลายแบบ. สำหรับ iSAI-NLP ช่องนี้ยังมีค่าเชิงทรัพยากรภาษาไทย แต่ต้องลดข้ออ้างจากกฎหมายแน่นอนเป็น
scenario analysis, เก็บสำเนา terms พร้อมวันที่/commit, ระบุ jurisdiction และให้ผู้เชี่ยวชาญกฎหมายตรวจ.
หากยืนยันถ้อยคำ “ติดกับ output เสมอ” หรือ “Qwen3 เป็น derivative จึงใช้ได้แน่” โดยไม่มี legal review หัวข้อนี้ควรถูก reject ไม่ใช่ narrow accept.

## 4. หัวข้อ C — benchmark contamination ในเว็บคอร์ปัสไทย

### Nearest work

| ชื่องาน | ปี | ทำอะไร | ทับกับเราตรงไหน | เหลือช่องว่างอะไร | แหล่งต้นฉบับที่เปิดอ่าน |
|---|---:|---|---|---|---|
| Contamination Report for Multilingual Benchmarks | 2024 | ใช้ black-box tests กับ 7 multilingual benchmarks × 7 LLMs; พบสัญญาณ contamination เกือบทุกคู่ | multilingual contamination และ benchmark reliability | วัดการจำของโมเดล ไม่ได้สแกน Thai/SEA web corpus หรือรายงาน document-level prevalence/domain clustering | [arXiv 2410.16186](https://arxiv.org/abs/2410.16186) |
| Benchmark Data Contamination in Underrepresented Languages: A Comprehensive Analysis Using Brazilian Data | 2026 | Brazilian Portuguese, 4 benchmarks; ใช้ TS-Guessing และ 50-character n-gram similarity เพื่อหา leaked items/corpora | low-resource/underrepresented-language contamination และ character-based matching | ไม่ใช่ไทย/SEA; เน้น model/benchmark contamination มากกว่าสแกน 4.57M Thai documents พร้อม source-domain analysis | [ACL Anthology](https://aclanthology.org/2026.lrec-1.374/) |
| Rethinking Benchmark and Contamination for Language Models with Rephrased Samples | 2023/2024 | แสดงว่า exact n-gram/embedding filters พลาด paraphrase/translation; พบ HumanEval overlap 8–18% ใน RedPajama/StarCoder และเสนอ LLM decontaminator | ทับ near-duplicate/semantic contamination และเตือนว่า exact match ต่ำไม่ได้แปลว่าสะอาด | ไม่ใช่ไทยและไม่รายงาน web-document prevalence; protocol near-duplicate ของเราต้องเทียบ calibration กับงานนี้ | [arXiv 2311.04850](https://arxiv.org/abs/2311.04850) |
| Dolma: An Open Corpus of Three Trillion Tokens | 2024 | decontamination ด้วย exact paragraph matching; พิจารณา paragraph ยาวอย่างน้อย 13 Unicode-segmented tokens; รายงาน <0.001% characters และ <0.02% documents ถูกตัดใน final corpus | corpus-side decontamination และอัตราระดับ document ใกล้กับข้ออ้าง 0.018% | English/general corpus; exact paragraph protocol ต่างจาก Thai character 64-gram และไม่มี Thai tutoring-site cluster | [arXiv 2402.00159](https://arxiv.org/abs/2402.00159), [Dolma config](https://github.com/allenai/dolma/blob/main/configs/dolma-v1_6/decontamination/README.md) |
| INFINI-GRAM MINI / Benchmark Contamination Monitoring System | 2025 | exact-match search บน Internet-scale corpora ด้วย FM-index; รายงาน contamination ถึง 74.2% ใน GSM8K | ขนาด index/search และ corpus-to-benchmark monitoring ใกล้มาก | ไม่เน้นไทย/SEA และไม่ได้ตอบ source ecology เช่นเว็บติวสอบราชการ | [ACL/EMNLP PDF](https://aclanthology.org/2025.emnlp-main.1268.pdf) |
| GPT-3 / GPT-4 / Llama 3 technical protocols | 2020–2024 | GPT-3 ใช้ word N-gram โดย N = 5th-percentile length และ clamp 8–13; GPT-4 สุ่ม 3 substrings × 50 characters หลัง normalize; Llama 3 ใช้ token 8-gram overlap score | เป็น baselines ที่ reviewer จะใช้ถามว่า 64-character gram และ threshold ของเราเทียบอย่างไร | ไม่มีมาตรฐานเดียว และไม่ได้ calibrate ภาษาไทยแบบไม่มีตัวแบ่งคำ | [GPT-3 paper](https://arxiv.org/abs/2005.14165), [GPT-4 report](https://cdn.openai.com/papers/gpt-4.pdf), [Llama 3 report](https://arxiv.org/abs/2407.21783) |

### คำตอบคำถามบังคับ

- **มีใครวัดอัตราสำหรับไทยหรือ SEA ไหม:** ไม่พบงานที่รายงาน document-level contamination rate ของ Thai/SEA web corpus ใน 4 query นี้. งาน multilingual วัด black-box model contamination; OpenSeal มี Thai/SEA CPT แต่ไม่ใช่ contamination-rate study. ข้อสรุปที่ปลอดภัยคือ “audit นี้ยังไม่พบ” ไม่ใช่ “ไม่มีใครทำ”
- **มาตรฐาน n-gram/stride ปัจจุบัน:** ไม่มีมาตรฐานเดียว. GPT-3 = word 8–13-gram (ตามความยาว benchmark), GPT-4 = 3 ตัวอย่างย่อย 50 characters ไม่ใช่ sliding stride, Llama 3 = token 8-gram overlap, Dolma = exact paragraph ≥13 Unicode tokens, งาน Brazilian = 50-character n-gram similarity. เมื่อ implementation สร้างทุก contiguous n-gram โดยทั่วไปเทียบได้กับ stride 1 แต่หลาย paper ไม่ระบุ stride หรือใช้ sampling; จึงห้ามเขียนว่า stride ใดเป็น “มาตรฐาน”
- **ตำแหน่งของ 64-character gram:** เป็น design ที่สมเหตุผลสำหรับไทยแต่ยาวกว่า GPT-4 substring baseline; ต้องทำ sensitivity/calibration อย่างน้อย char-32/50/64, stride 1 vs 8, normalization variants และ near-duplicate threshold พร้อม labeled sample เพื่อรายงาน precision/recall

### คำตัดสิน C — **GO**

ช่องว่างชัดที่สุดคือการวัดแบบ corpus-side สำหรับภาษาไทย/SEA พร้อม **prevalence + type + source ecology**.
Method exact/near-duplicate ไม่ใหม่ แต่การรายงาน 4.57M Thai web documents เทียบ benchmark index, แยก verbatim/near-duplicate,
และพบการกระจุกในเว็บติวสอบราชการมี contribution ที่ตรงกับ iSAI-NLP. เงื่อนไข GO คือเปิด benchmark list/version,
normalization, gram unit, stride, threshold, document denominator, confidence intervals, false-positive audit และ domain-label protocol;
ตัวเลข 803/0.018%/2.95% ยังห้ามอ้างเป็นผลจน pipeline และ sample audit ผ่านการตรวจ.

## 5. หัวข้อ D — low-cost preflight ก่อนเช่า GPU

### Nearest work / venue evidence

| ชื่องาน/แหล่ง | ปี | ทำอะไร | ทับกับเราตรงไหน | เหลือช่องว่างอะไร | แหล่งต้นฉบับที่เปิดอ่าน |
|---|---:|---|---|---|---|
| Towards Understanding Bugs in Distributed Training and Inference Frameworks for Large Language Models | 2025 | empirical study บั๊กที่แก้แล้ว 308 รายการใน DeepSpeed, Megatron-LM, Colossal-AI; จัด symptoms/root causes/fix effort; ชี้ distributed-only failure modes และต้นทุน reproduction สูง | ชนข้ออ้างว่าบั๊กจำนวนมากปรากฏเฉพาะ distributed setup และทำให้เสีย GPU cost | ไม่เสนอ US$8 two-GPU preflight protocol และไม่ทดสอบ project pipeline ของแล็บเล็ก | [arXiv 2506.10426](https://arxiv.org/abs/2506.10426) |
| A Comprehensive Study of Bugs in Modern Distributed Deep Learning Systems | 2025 | วิเคราะห์ 849 issues, taxonomy 34 symptoms/28 causes/6 fix patterns; 45.1% symptoms เฉพาะ distributed frameworks | prior art ที่ใหญ่กว่ามากสำหรับ empirical distributed bug study | ไม่ใช่ controlled preflight experiment แต่ทำให้ case study 8 bugs ดูเล็กเกิน paper | [arXiv 2512.20345](https://arxiv.org/abs/2512.20345) |
| PyTorch reproducibility / deterministic algorithms / TorchTitan debugging docs | ปัจจุบัน | เอกสารทางการระบุว่า deterministic guarantees ผูกกับ input + software + hardware เดียวกัน; deterministic mode ยังไม่ทำให้ทุก config/run เหมือนกัน และอาจช้าลง | ชน observation ว่า fixed-seed CUDA run อาจต่างและ bitwise equality เป็น oracle ที่ไม่ปลอดภัย | ช่องที่เหลือคือ **calibrated resume oracle** ซึ่งเทียบ resume divergence กับ twin uninterrupted run-to-run distribution | [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [deterministic algorithms](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html), [TorchTitan debugging](https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md) |
| MLSys 2026 CFP / NeurIPS 2026 STEPS / ML for Systems workshop | 2026 | venues รับ efficient LLM training, testing/debugging/monitoring, training dynamics และ reproducibility/reliability; ต้องการ novelty/quality/impact หรือ benchmark/methodology ที่ generalize | ชี้ venue family ที่ถูกต้องหากขยายงาน | iSAI-NLP main paper ไม่ใช่ natural home หากไม่มี Thai-NLP-specific scientific question | [MLSys CFP](https://mlsys.org/Conferences/2026/CallForPapers), [STEPS](https://science-of-training.github.io/), [ML for Systems](https://mlforsystems.org/) |

### คำตอบคำถามบังคับ

- **paper หรือ blog:** หลักฐานปัจจุบันเป็น 1 project × 1 rental setup × US$8 × 8 defects จึงเป็น engineering postmortem/checklist ที่ดี แต่ยังไม่เป็น empirical paper. ประโยค “fixed tolerance ใช้บน CUDA ไม่ได้” กว้างเกินจริง; tolerance ใช้ได้หาก calibrate กับ run-to-run baseline และล็อก hardware/software. สิ่งที่ใช้ไม่ได้ทั่วไปคือการคาดหวัง bitwise equality หรือเลือก tolerance คงที่โดยไม่มี null distribution
- **ถ้าจะเป็น paper ต้องเพิ่มอะไร:** อย่างน้อยหลาย framework (เช่น FSDP/DeepSpeed/Megatron), หลาย GPU/software stack, หลาย seed, fault-injection matrix, twin uninterrupted controls, precision/recall ของแต่ละ preflight check, เวลา/ค่าใช้จ่าย และ artifact ที่ rerun ได้. คำถามวิจัยควรเป็น “budget-constrained preflight protocol detects what fraction of expensive distributed failures?” ไม่ใช่ “เราเจอ 8 bugs”
- **venue:** เมื่อขยายแล้วเหมาะกับ MLSys/ระบบ, reproducibility/ML-systems workshop, artifact/experience track หรือ short systems paper; สำหรับ iSAI-NLP เหมาะเป็น system/demo/resource note เฉพาะเมื่อผูกกับ reproducible Thai LLM training และมีผลทั่วไปเกิน project เดียว. ในรูปปัจจุบันควรเป็น blog post/technical report

### คำตัดสิน D — **DEAD** (ในฐานะ standalone iSAI-NLP research paper รูปปัจจุบัน)

ข้อเท็จจริงพื้นฐานเรื่อง CUDA nondeterminism และ distributed-only bugs มี prior art/official documentation แล้ว;
ขนาด 8 bugs จาก one-off preflight ไม่พอแยก anecdote ออกจาก protocol ที่ generalize. อย่าฝืนส่งเป็น paper.
รักษาไว้เป็น blog/checklist และ artifact ประกอบหัวข้อหลัก. หากวันหนึ่งทำ multi-framework controlled study ตามเกณฑ์ด้านบน
ให้ถือเป็น **หัวข้อใหม่ที่ต้อง audit ใหม่** ไม่ใช่การ narrow หัวข้อเดิม.

## 6. จัดอันดับช่องว่างจากกว้างสุดไปแคบสุด

1. **C — GO:** ยังไม่พบ corpus-side contamination rate สำหรับเว็บไทย/SEA; source clustering ที่เว็บติวสอบเพิ่ม insight เฉพาะภูมิภาค และ venue fit สูง
2. **B — NARROW:** systematic licence audit และ synthetic-output ToS มี prior art แล้ว แต่ Thai row-level, revision-pinned, base-family scenario audit ยังเป็น resource contribution ที่ใช้งานได้จริง
3. **A — NARROW:** exact scientific shape ถูกทำแล้วที่ 1.4B พร้อม 1–80% sweep และ curves; เหลือ Thai-specific replication/measurement เท่านั้น
4. **D — DEAD:** observation หลัก documented แล้วและ sample เล็กเกิน research paper; เป็น blog/artifact จนกว่าจะออกแบบ study ใหม่

เหตุผลรวม: C มี novelty ที่เกิดจากทั้งภาษาที่ยังขาดการวัดและข้อค้นพบเชิง ecology ของแหล่งรั่ว ไม่ใช่เพียงนำวิธีเดิมมาใช้.
B ยังมีคุณค่าจาก executable Thai inventory แต่ต้องยอมรับว่าข้อสังเกตเชิง licence/provenance ถูกงานใหญ่ทำแล้วและกฎหมายไม่แน่นอน.
A มี venue relevance แต่ methodological novelty เกือบหมดหลังเจอ EMNLP 2024 และงาน 2025–2026.
D ไม่ได้แข่งในคำถาม NLP และหลักฐานปัจจุบันไม่พ้นระดับ postmortem.

## 7. งานที่เกือบพลาด

| งาน | ทำไมเกือบพลาด | Query ที่พาไปเจอ/ขยายผล |
|---|---|---|
| Breaking Language Barriers (EMNLP 2024) | ชื่อไม่มีคำว่า “replay ratio”; replay sweep อยู่ใน Figure 4/Section 5.2 ของ PDF จึงไม่เด่นใน abstract | A2 — `"data mixture" catastrophic forgetting "continued pre-training" language adaptation` |
| Efficiently Adapting… / SambaLingo-Thai | paper workshop ปี 2023 กับ paper MRL 2024 มีชื่อคนละแบบ; Thai model card ระบุ 75:25 ชัดกว่าบทคัดย่อ | A3 — `Thai language model continued pretraining English replay mixture` |
| OpenSeal | ชื่อเน้น parallel data ไม่เน้น replay แต่ภายในตรึง replay 25% และมี Thai ที่สเกล 1B | A4 — `Southeast Asian LLM continual pretraining catastrophic forgetting replay` |
| Legal Considerations… LLMs4EU | อยู่ใน LEGAL/LREC workshop ปี 2026 ไม่ใช่ keyword “dataset licence audit” ตรง ๆ | B2 — `"synthetic data" "model output" license terms instruction dataset` |
| The Mirage of AI Terms of Use Restrictions | เป็น legal scholarship และตั้งคำถาม enforceability จึงไม่โผล่ถ้าค้นแต่ dataset card/Hugging Face | B2 — `"synthetic data" "model output" license terms instruction dataset` |
| Benchmark Data Contamination in Underrepresented Languages (Brazilian Portuguese) | ตีพิมพ์ LREC 2026 และใช้คำ “underrepresented” แทน low-resource; ใกล้วิธี character matching มาก | C4 — `"character n-gram" benchmark contamination low-resource language` |
| INFINI-GRAM MINI | ชื่อระบบไม่บอกว่าเป็น contamination paper; use case อยู่ในเนื้อหา | C3 — `"data decontamination" n-gram LLM pretraining` |
| Towards Understanding Bugs in Distributed Training… | ชื่อรวม training และ inference และไม่ได้ใช้คำ “preflight”; แต่เป็น collision หลักด้าน empirical bug evidence | D4 — `empirical study distributed training bugs checkpointing reproducibility` |

## 8. Decision memo สำหรับการส่ง iSAI-NLP 2026

เลือก **C เป็น backup paper ที่แข็งที่สุด**. B ใช้ได้หากวางเป็น Thai resource/governance audit พร้อม legal-review caveat
และตัดคำอ้างเด็ดขาดเรื่อง output inheritance. A ใช้ได้เฉพาะ paper เชิง replication/measurement ที่อธิบายว่า Thai curve
ต่างจาก prior languages อย่างมีนัยสำคัญ; ไม่ควรอ้างวิธีใหม่. D ไม่ควรเป็น paper หลักในสถานะนี้.

ก่อนเปลี่ยน verdict จาก desk audit เป็น submission decision ต้องมี: (1) rerun query log ใกล้วันส่ง,
(2) independent source screening อย่างน้อยหนึ่งคน, (3) protocol/analysis preregistration,
(4) ผลทดลองจริงพร้อม uncertainty และ (5) สำหรับ B การตรวจโดยผู้เชี่ยวชาญกฎหมาย.
