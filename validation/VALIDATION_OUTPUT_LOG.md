# Validation Output Log

This file records desk, software, engineering and scientific evidence. Smoke
tests must never be presented as model-quality evidence.

## `VAL-2026-08-16-LOCAL-UNIT`

```yaml
run_id: VAL-2026-08-16-LOCAL-UNIT
stage: local_unit
scientific_evidence_allowed: false
condition: strict_identity_team_delta_block_fixture
results:
  tests_passed: 27
  tests_skipped: 2
  skipped_reason: Transformers Qwen3/OLMo integration dependencies reserved for LANTA environment
  toy_model_exact_identity: true
  toy_model_router_gradient_nonzero: true
  checkpoint_roundtrip_pass: true
limitations:
  - No real pretrained weights were used.
  - No Thai training was performed.
  - The local adapter is a team strict-identity variant, not a certified reproduction of official Delta AttnRes.
verdict: promote_to_model_family_preflight_only
```

## `VAL-2026-08-18-NOVELTY-DESK`

```yaml
run_id: VAL-2026-08-18-NOVELTY-DESK
stage: desk_audit
scientific_evidence_allowed: false
claim: Public-search check for an existing matched ThaiLLM-1B Attention Residual study
sources_checked:
  - Attention Residuals
  - Delta Attention Residuals
  - Multi-Head Attention Residuals
  - Low-Rank Attention Residuals
  - WAV multi-resolution routing
  - targeted Thai/ThaiLLM paper and model-card searches
finding: No exact public Thai study found in the searched record.
limitations:
  - Absence from search is not proof of non-existence.
  - Earlier depth-aggregation and Thai CPT work still require a completed claim matrix.
  - First-in-Thai alone is not an eligible contribution.
verdict: pending
next_required:
  - Complete nearest-work claim matrix.
  - Freeze the empirical contribution sentence.
  - Repeat the search immediately before paper freeze.
```

## `VAL-2026-08-18-NOVELTY-MATRIX`

```yaml
run_id: VAL-2026-08-18-NOVELTY-MATRIX
stage: desk_audit
scientific_evidence_allowed: false
claim: Completed nearest-work claim matrix and independent novelty re-audit through 2026-08-18
method:
  channel_a: arXiv API structured queries, sortBy=submittedDate
  channel_b: primary-source verification of every claim-bearing work
  queries_run:
    Q1: 'all:"attention residuals"  -> 60 results (capped)'
    Q2: 'ti:"attention residual"  -> 35 results'
    Q3: 'abs:"depth routing" OR abs:"depthwise routing" OR abs:"residual routing"  -> 19'
    Q4: 'ti:DenseFormer OR ti:DeepCrossAttention OR ti:Hyper-Connections OR abs:hyper-connections -> 60 (capped)'
    T1: 'abs:Thai AND (continued pretraining OR continual pre-training OR language adaptation) -> 8'
    T2: 'ti:Thai AND (ti:language model OR ti:LLM) -> 22'
    T3: 'abs:compute-matched OR abs:GPU-hour OR abs:GPU hours AND abs:continued pretraining -> 40 (capped)'
    T4: '(attention residuals OR hyper-connections) AND (multilingual OR low-resource OR cross-lingual) -> 1'
  unique_arxiv_ids_surfaced: 144
results:
  all_five_registry_attnres_papers_exist: true
  decisive_finding: >
    Query T4 intersecting the entire AttnRes/hyper-connection literature with
    multilingual/low-resource/cross-lingual returns exactly ONE document
    (arXiv:2608.09119 Motif 3), a 314B frontier MoE technical report, not a
    controlled study. The language-adaptation axis of the AttnRes family is
    unoccupied in the searched record.
  retention_column_empty: true   # no work in the matrix measures source-language retention
  contributions_still_open: 1.5 of 4 planned
corrections_made:
  - AR05 title stale: arXiv:2606.06564 v1 "WAV" renamed to "HAARES" in v2 (2026-06-17)
  - Delta AttnRes official repo is MIT-licensed, NOT unlicensed as stated in src/README.md
  - 7 relevant works were absent from source_registry.csv; 2 posted in the last 3 weeks
severe_overlaps_found:
  - arXiv:2606.13168 performs causal routing-ablation probes on Block AttnRes using
    Qwen3-0.6B, our designated pilot model; pre-empts planned contribution 4
  - Official MIT repo ships conversion/training/eval/W&B configs; pre-empts contribution 3
limitations:
  - Absence from search is not proof of non-existence.
  - arXiv API cannot see non-arXiv venues, industry work, or different vocabulary
    ("cross-layer attention", "layer aggregation").
  - Field is producing ~1 relevant paper every 10 days; re-check is mandatory.
artifacts:
  - ../novelty/nearest_work_matrix.csv
  - ../novelty/novelty_audit.md
  - ../novelty/reviewer_pre_mortem.md
  - ../novelty/decision_memo_th.md
verdict: novelty_gate_conditional_pass_narrow
next_required:
  - Re-run Q1-Q4 and T4 within 48 hours of submission.
  - Cite arXiv:2606.06564 by its v2 title (HAARES).
  - Demote mechanism analysis to a supporting result citing arXiv:2606.13168.
```

## `VAL-2026-08-18-LOCAL-STATIC-AUDIT`

```yaml
run_id: VAL-2026-08-18-LOCAL-STATIC-AUDIT
stage: local_unit
scientific_evidence_allowed: false
condition: static_and_dynamic_audit_of_archived_team_router
environment:
  python: 3.9.6
  torch: 2.8.0
  numpy: 2.0.2
  pytest: not_installed
  transformers: not_installed
provenance:
  archived_harness_copied_to_scratchpad_before_execution: true
  archived_directory_modified: false
results:
  configs_valid_json: true          # experiment_parameters, data_manifest.template, run_record.template
  source_registry_well_formed: true # 13 rows -> 30 rows after update, no duplicate ids, no ragged rows
  archived_unit_tests: "25/25 passed (unittest, 0.091s)"
  archived_unit_tests_skipped: "test_hf_integration.py not run - transformers unavailable locally"
  toy_model_exact_identity: true
  max_logit_abs_error: 0.0          # threshold in experiment_parameters.json is 1e-5
  checkpoint_roundtrip_pass: true
equation_fidelity_vs_arxiv_2605_18855:
  query_is_learned_zero_init_vector_not_from_hidden_state: match
  keys_are_norm_of_values: match
  values_are_block_deltas: match
  output_is_additive: match
  init_behaviour: DEVIATION   # paper = bounded perturbation; team = exact identity via extra route_scale
blocking_finding:
  id: ROUTE_SCALE_GATE
  severity: blocking
  analytic_basis: "out = residual + s*mixture(w)  =>  dL/dw = (dL/dout)*s*(dmixture/dw)"
  consequence: router parameter gradients are identically zero at s=0
  measured_at_init:
    query_weight_grad_all_layers: 0.0
    null_logit_grad_all_layers: 0.0
    route_scale_grad: 0.093-0.150
  measured_warmup_trace_surrogate_loss_adamw_lr1e-3:
    step_1:  {route_scale: 0.000000, frac_of_paper_faithful_query_grad: 0.0}
    step_10: {route_scale: 0.008177, frac_of_paper_faithful_query_grad: 0.00047}
    step_50: {route_scale: 0.016297, frac_of_paper_faithful_query_grad: 0.00095}
  caveat: >
    Step counts come from a synthetic fixed-batch surrogate loss and do not transfer
    numerically to real CPT. The mechanism (multiplicative zero gate => vanishing
    router gradient) is analytic and holds at any scale.
  risk: >
    Confounds the central comparison. If D1/D2 underperform S0 we cannot distinguish
    "delta routing does not help Thai CPT" from "the team's extra gate prevented the
    router from learning within the pilot budget".
other_engineering_findings:
  E1_mhar_d2_not_implemented: true      # archive has only IdentityDeltaRouter + HFDepthRouterAdapter
  E2_last_routing_retains_autograd_graph: true   # weights requires_grad=True and grad_fn not None after forward
  E3_block_size_semantics_ambiguous: true        # bs=4 on 8 layers -> layers 0-3 see 0 sources, 4-7 see 1
  E4_gradient_checkpointing_hard_disabled: true  # routed arms only -> biases GPU-hour axis
  E5_archived_10pct_overhead_kill_rule_still_enforced: true  # repudiated by new experiment_parameters.json
bias_direction_warning: >
  E2 and E4 both handicap the ROUTED arms, not the baseline. Combined with the
  0.55-0.88x throughput ratio reported in arXiv:2607.27230, a "routing loses per
  GPU-hour" result would currently be uninterpretable.
limitations:
  - No real pretrained weights were used.
  - No Thai training was performed.
  - Tiny random models only; nothing here supports any claim about 1B models or Thai.
  - These probes can only REJECT an implementation, never support a scientific claim.
verdict: blocked_pending_router_fidelity_fix
next_required:
  - Adopt the MIT-licensed official implementation, or remove route_scale and prove
    query gradient is non-zero at step 1.
  - Fix E2 and E4 so all three conditions share one memory regime.
  - Implement and validate MHAR (D2).
  - Do not spend GPU hours before the above pass.
```

## `BASE-SCREEN-2026-08-18-TOKENIZER-LICENSE-PORT`

```yaml
run_id: BASE-SCREEN-2026-08-18-TOKENIZER-LICENSE-PORT
stage: tier_b_tokenizer_and_desk_audit
scientific_evidence_allowed: false
claim: Base-model screen for ThaiLLM-1B across tokenizer, license, architecture and compute
environment:
  python: 3.13 (isolated venv; system python is 3.9.6)
  tokenizers: 0.22.2
  transformers: not_installed
  local_model_weights_available: false   # HF cache holds only 4KB refs stubs, no snapshots
method:
  metadata: Hugging Face API at pinned commit SHAs; parameter counts read from safetensors metadata
  architecture: transformers v4.57.1 sources read for llama, qwen3, gemma3, olmo2
  tokenizer: tokenizers.Tokenizer.from_file on tokenizer-only artifacts, add_special_tokens=false
  sample_set_id: THAI-EN-CODE-SCREEN-V1
  sample_set_sha256: 6abc31ea71544ca227b329554e6a296b54c0a6a1f8ad9dfdb60283d757e3a963
  sample_set_size: 14 documents / 8197 bytes / authored for this project
candidates_screened: 8   # 5 from the plan plus Sailor2-1B, Falcon3-1B-Base, SmolLM2-1.7B; Granite-3.3-2b and Gemma-SEA-LION-v3-9B size-checked only
results:
  tokenizer_correctness: all 8 lossless round-trip, 0 U+FFFD, 0 <unk>  # nobody eliminated on correctness
  thai_chars_per_token:
    google/gemma-3-1b-pt: 2.833
    meta-llama/Llama-3.2-1B: 2.179
    Qwen/Qwen2.5-1.5B: 1.838
    Qwen/Qwen3-0.6B-Base: 1.838
    allenai/OLMo-2-0425-1B: 1.082
    tiiuae/Falcon3-1B-Base: 0.737
    HuggingFaceTB/SmolLM2-1.7B: 0.583
  vocab_size_does_not_predict_thai_fertility: true
  # Qwen: 151936 vocab / 2570 Thai-bearing pieces -> 1.838 c/t
  # Llama: 128256 vocab / 1391 Thai-bearing pieces -> 2.179 c/t
  qwen2_5_and_qwen3_tokenizer_byte_identical: true   # sha256 match on tokenizer.json
  port_gate: all candidates admit exact identity-preserving Delta Block conversion
  port_effort_engineer_days: {qwen3_0_6b: 4-6, llama_3_2_1b: 6-9, qwen2_5_1_5b: 7-10, olmo2_1b: 8-12, gemma3_1b: 13-19}
  mhar_subspace_divisibility: all candidates pass for H_r in {4,8}
  router_parameter_share: ~0.02 percent of 1.2B -> parameter-matched control satisfied arithmetically
corrections_made:
  - allenai/OLMo-2-0425-1B recorded as 1.0B; verified total is 1,484,916,736 (48 percent error)
  - Qwen/Qwen3-0.6B-Base recorded as 600M; verified total is 596,049,920
  - all five plan candidates had revision=null; immutable SHAs now recorded
blocking_findings:
  - id: GATE_A_GATED_ACCESS
    severity: blocking
    detail: >
      meta-llama/Llama-3.2-1B and google/gemma-3-1b-pt are gated:manual and no Hugging Face
      token exists in this environment. Their config.json and tokenizer.json could not be read
      from the origin repositories (HTTP 401/403). Artifacts were taken from ungated community
      mirrors at pinned revisions (unsloth/Llama-3.2-1B @ 9535bd9b, unsloth/gemma-3-1b-pt @ 34a98bf3).
      Parameter totals match the origin API values exactly, but byte-identity of the tokenizer
      artifacts is UNVERIFIED. All Llama and Gemma numbers in this screen are provisional.
  - id: GATE_B_DERIVATIVE_NAMING
    severity: blocking_for_deliverable_naming
    detail: >
      Llama 3.2 Community License requires derivative model names to begin with "Llama" plus a
      "Built with Llama" notice. Gemma Terms require derivative names to begin with "Gemma".
      Both make the declared deliverable names ThaiLLM-1B-Base and ThaiLLM-1B-Instruct
      non-compliant. Apache-2.0 candidates carry no naming constraint. This is a project
      decision, not an engineering one, and it changes the final answer.
plan_deviation_proposed:
  fallback_changed: google/gemma-3-1b-pt -> Qwen/Qwen2.5-1.5B
  reason: >
    Gemma 3 shares BOTH blocking failure modes with Llama 3.2 (gated:manual and derivative
    naming), so it provides no independent fallback coverage. Qwen2.5-1.5B is Apache-2.0 and
    ungated, and can ship as ThaiLLM-1B-Base. Cost: 1.544B needs mentor size approval.
rejected_with_reasons:
  sail/Sailor2-1B: scored 84/100 but REJECTED - already Thai/SEA continued-pretrained, so Thai
    acquisition cannot be measured from it. Demonstrates that total score cannot override a
    fatal blocker.
  allenai/OLMo-2-0425-1B: English-only per model card; 1.082 Thai chars/token; same corpus costs
    2.62x the tokens and 2.35x the GPU-hours. Retained for architecture replication on English only.
  tiiuae/Falcon3-1B-Base: Thai unsupported; 0.737 Thai chars/token; custom license.
  HuggingFaceTB/SmolLM2-1.7B: English-only; 0.583 Thai chars/token; 2 Thai vocab pieces.
limitations:
  - NO model weights were loaded anywhere. No frozen BPB, no inference smoke, no RAM or latency
    measurement was produced. The local HF cache contains only 4KB refs stubs.
  - All compute figures are analytic, assuming 120 TFLOPS sustained per A100 (MFU ~38 percent).
    Nothing was measured on a GPU. Uncertainty is roughly +/-30 percent.
  - Routed-arm slowdown 0.55-0.88x is taken from AR03, measured on different hardware and model.
  - CODE01 (MIT reference implementation) was NOT cloned or read; the 1-2 day Qwen3 port estimate
    depends on modeling_qwen3_attnres.py being what source_registry.csv says it is.
  - Qwen2DecoderLayer source was NOT re-read; its "low risk" rating is inferred from config.
  - Tokenizer fertility comes from 14 screening documents, not from SEA-PILE-v2. Absolute values
    will shift on real corpus text; the ranking gaps are large enough that the order should hold.
  - ROUTE_SCALE_GATE, E1, E2 and E4 from VAL-2026-08-18-LOCAL-STATIC-AUDIT remain open and are
    independent of base-model choice. Selecting a base does not unblock them.
artifacts:
  - ../base_selection/candidate_matrix.csv
  - ../base_selection/tokenizer_screen.json
  - ../base_selection/architecture_port_audit.md
  - ../base_selection/compute_estimate.md
  - ../base_selection/final_recommendation_th.md
verdict: conditional_recommendation_no_final_lock
next_required:
  - Request gated access from Meta and Google in parallel.
  - Obtain a project decision on GATE_B naming before anything else; it determines the answer.
  - Re-run the tokenizer screen against origin revisions to close GATE_A.
  - Re-measure Thai fertility on >=100MB of real SEA-PILE-v2 text before locking the token budget.
  - Run the one-A100 200-step preflight to replace analytic throughput with measured throughput.
  - Do not lock the final base until GATE_A, GATE_B and the preflight all pass.
```

## `CONFIG-2026-08-18-DATA-MIXTURE`

```yaml
run_id: CONFIG-2026-08-18-DATA-MIXTURE
stage: desk_decision
scientific_evidence_allowed: false
claim: Provisional CPT data mixture revised from 90/5/5 to 50/35/10/5 and a replay-ratio ablation added
change:
  previous: {thai: 0.90, english: 0.05, code: 0.05}
  new:      {thai: 0.50, english: 0.35, code: 0.10, math: 0.05}
rationale: >
  The prior mixture was roughly 10x lower in replay than every published recipe that
  succeeded at target-language continued pretraining. Typhoon 2 (Thai, 1B-70B) used
  50 percent English and cites catastrophic-forgetting mitigation as the reason.
  Racka (Hungarian, academic HPC) used ~56 percent non-target data
  (24 EN + 21 DE + 11 code). EstLLM used English replay plus code, math and
  instruction-like data. At 1-2B capacity the forgetting risk is higher than at the
  scales those papers targeted, so a 90 percent Thai mixture risks a model that gains
  little Thai while losing English and reasoning.
evidence_sources:
  - arXiv:2412.13702  # Typhoon 2
  - arXiv:2601.01244  # Racka
  - arXiv:2603.02041  # EstLLM
  - arXiv:2510.25947  # Revisiting Multilingual Data Mixtures
ablation_added:
  id: REPLAY-ABL-2026
  arms: [MIX50, MIX70, MIX90]
  tokens_per_arm: 1e9 to 2e9
  primary_metric: thai_bpb_on_frozen_heldout
  seeds: [0]
  status: planned
limitations:
  - The new ratio is a literature-derived prior, NOT a measured optimum for Thai at this scale.
  - It stays provisional until REPLAY-ABL-2026 reports.
  - One screening seed only; not evidence for any Track 2 architecture claim.
verdict: provisional_pending_replay_ratio_ablation
```

## `DATA-2026-08-18-SEA-PILE-V2-DESK`

```yaml
run_id: DATA-2026-08-18-SEA-PILE-V2-DESK
stage: desk_audit
scientific_evidence_allowed: false
claim: Desk read of the SEA-PILE-v2 dataset card for Thai CPT budgeting
source: https://huggingface.co/datasets/aisingapore/SEA-PILE-v2
findings:
  total_tokens: 120e9 across 9 SEA languages
  thai_tokens_reported: 6.5e9
  thai_share_of_corpus: 0.0533
  token_counting_tokenizer: Gemma3     # NOT our tokenizer - must be recounted
  license: ODC-By 1.0 plus CommonCrawl terms of use
  provenance_fields: text, dump id, timestamp, url, WARC record id
  processing_already_applied:
    - deduplication within each snapshot
    - heuristic quality filters
    - perplexity scoring (CCNet / Sailor methodology)
  dataset_card_warning: "some documents containing harmful, toxic, or private content may still pass through"
budgeting_implication: >
  Thai token counts are reported under the Gemma3 tokenizer. Under the Qwen tokenizer
  family (measured Thai fertility 1.838 chars/token vs Gemma 2.833) the same Thai text
  yields roughly 1.54x more tokens, i.e. about 10B Qwen tokens. At a 50 percent Thai
  mixture this supports roughly a 20B-token CPT run at about one epoch of Thai.
  Beyond that, supplementary Thai sources are required.
required_next:
  - Recount Thai tokens under the finally selected tokenizer; do not reuse the Gemma3 figure.
  - Cross-snapshot (not only within-snapshot) deduplication.
  - Benchmark decontamination against the frozen Thai eval sets.
  - PII screening; the card explicitly disclaims residual private content.
  - Pin an immutable dataset revision.
  - Audit the ThaiLLM Data Repository per component (site returned HTTP 403 to anonymous access;
    provenance and license remain UNVERIFIED and it must not be ingested as one corpus).
verdict: usable_pending_recount_decontamination_and_pii_screen
```

## `PHASE0-2026-08-18-TOKENIZER-EXT-AND-EVAL-FREEZE`

```yaml
run_id: PHASE0-2026-08-18-TOKENIZER-EXT-AND-EVAL-FREEZE
stage: phase0_no_gpu
scientific_evidence_allowed: false
claim: Extended the frozen tokenizer screen to the live base candidates and froze the evaluation suite
artifacts:
  - ../phase0/tokenizer_screen_ext.py
  - ../phase0/tokenizer_screen_ext.json
  - ../phase0/freeze_eval_suite.py
  - ../phase0/eval_suite_frozen.json
  - ../phase0/measure_baseline.py
  - ../phase0/baseline_Qwen_Qwen3-0.6B-Base_smoke.json

step_a_tokenizer_extension:
  sample_set: THAI-EN-CODE-SCREEN-V1 reused unchanged; per-sample sha256 verified 14/14
  methodology_control:
    model: Qwen/Qwen3-0.6B-Base
    thai_chars_per_token: {original: 1.838, this_run: 1.838, match: true}
    thai_vs_english_token_ratio: {original: 2.8571, this_run: 2.8571, match: true}
    vocab_pieces_containing_thai: {original: 2570, this_run: 2571, match: false}
    control_assessment: >
      Both decision-relevant metrics reproduce to 4 decimal places. The vocab-piece count
      differs by 1 of 2571 (0.04 percent), a counting-convention difference between the raw
      tokenizers vocab (151665 entries) and AutoTokenizer (151669, includes added specials).
      Not chased further; it does not affect any decision.
  results:
    Qwen/Qwen3-1.7B-Base:
      revision: ea980cb0
      gated: false
      thai_chars_per_token: 1.838
      thai_bytes_per_token: 5.0468
      thai_vs_english_token_ratio_parallel_doc: 2.8571
      vocab_pieces_containing_thai: 2571
      roundtrip_lossless: true
    google/gemma-4-E2B:
      revision: d29ff6b4
      gated: false
      license_on_card: apache-2.0     # Gemma 3 required gated access and a Gemma naming prefix; Gemma 4 does not
      thai_chars_per_token: 2.8326
      thai_bytes_per_token: 7.689
      thai_vs_english_token_ratio_parallel_doc: 1.4524
      vocab_pieces_containing_thai: 2177
      roundtrip_lossless: true
      loader_note: >
        transformers 4.57.6 cannot load model_type gemma4 via AutoTokenizer
        (AttributeError); measurement used the raw tokenizer.json. This is measured
        evidence of ecosystem immaturity, not speculation.
    Qwen/Qwen3-4B-Base:
      thai_chars_per_token: 1.838     # byte-identical tokenizer to the other Qwen3 sizes
  cross_validation: >
    gemma-4-E2B reproduces the Gemma 3 figures from the original screen exactly
    (2.8326 vs 2.833 chars/token; 1.4524 vs 1.4524 ratio), independently confirming
    both the measurement fix and that Gemma 4 keeps the Gemma 3 tokenizer behaviour on Thai.
  corpus_cost_projection_sea_pile_v2_th:
    anchor: 6.5B tokens under the Gemma 3 tokenizer per the dataset card
    google/gemma-4-E2B:    {multiplier: 1.000,  tokens: 6.50e9}
    Qwen/Qwen3-1.7B-Base:  {multiplier: 1.541,  tokens: 1.002e10}
    implication: >
      Choosing the Qwen tokenizer costs about 54 percent more compute for the same
      Thai corpus bytes. On a 20B-token budget at a 50 percent Thai mixture that is
      roughly 3.5B tokens of pure tokenizer overhead.

measurement_bugs_found_and_fixed:
  - id: BUG1_bytelevel_vocab_not_decoded
    detail: >
      Byte-level BPE vocab keys are byte-encoded surfaces, so a literal Thai-character
      scan returned 0 Thai pieces for every Qwen tokenizer. Fixed by applying the
      reverse GPT-2 byte-to-unicode map before detection.
  - id: BUG2_raw_tokenizer_added_bos
    detail: >
      tokenizers.Tokenizer.encode() prepends <bos> by default, inflating gemma-4 token
      counts by one per document and breaking the round-trip check. The original screen
      specifies add_special_tokens False. Fixed.
  lesson: >
      Both bugs produced plausible-looking numbers and did not raise an error. They were
      caught only by the methodology control. This is the same failure class as the
      ROUTE_SCALE_GATE finding: code that runs correctly while measuring the wrong thing.

step_b_eval_suite_freeze:
  suite_id: THAILLM-EVAL-FROZEN-V1
  spec_sha256: 1fae436e05fa99cee8b5b878e72f45c95aa8d51b3c1177e49c1c5a98c565cb19
  thai_primary: [thaiexam, m3exam_th, thai_bpb_heldout]
  retention_primary: [mmlu, english_bpb_heldout, code_bpb_heldout]
  rule: >
    Any metric not listed in the frozen spec may not later be presented as a planned
    metric. Changing the spec requires a new suite_id and a logged justification.

step_c_baseline_pipeline:
  status: smoke_only
  model: Qwen/Qwen3-0.6B-Base @ da87bfb6, device mps, bfloat16
  bpb_on_frozen_14doc_sample: {th: 0.482449, en: 0.837414, code: 0.446267}
  interpretation: >
    PIPELINE VERIFICATION ONLY. Fourteen short authored documents say nothing about any
    model's Thai ability. Note also that BPB is not comparable across languages: Thai is
    ~3 UTF-8 bytes per character versus ~1 for English, so Thai BPB is mechanically lower.

open_decision:
  question: Qwen3-1.7B-Base or gemma-4-E2B as the main base
  for_qwen:   [dense standard decoder, mature tooling, official modeling_qwen3_attnres.py enables Track 2 cheaply]
  for_gemma:  [1.54x better Thai tokenization, now apache-2.0 and ungated with no naming constraint]
  against_gemma: [MatFormer plus per-layer embeddings plus audio and vision encoders, transformers 4.57.6 cannot load it, Track 2 incompatible]
  resolution_required: run Step C baselines on both before committing GPU budget

limitations:
  - 14 documents is a screening sample. It supports the RANKING, not corpus-scale fertility.
  - Fertility is a cost metric and does not predict how well a model learns Thai.
  - No Thai training was performed. No downstream accuracy was measured.
verdict: phase0_steps_a_and_b_complete_step_c_pipeline_verified
next_required:
  - Build and hash the four held-out BPB sets; remove them from the training pool.
  - Run real baselines for Qwen3-1.7B-Base and gemma-4-E2B.
  - Run lm-evaluation-harness on the frozen task list and record its commit sha.
  - Decide the headroom gate before any GPU budget is committed.
```

## `PHASE0-2026-08-18-HELDOUT-AND-GEMMA4-STACK`

```yaml
run_id: PHASE0-2026-08-18-HELDOUT-AND-GEMMA4-STACK
stage: phase0_no_gpu
scientific_evidence_allowed: false
claim: Built the frozen Thai held-out BPB set and established the Gemma 4 stack-support facts

heldout_set:
  set_id: TH-WEB-HELDOUT
  source: aisingapore/SEA-PILE-v2
  dataset_revision: 77573cc84631412a781daa8e6f72cf322d4207f0
  files: [th/train-00000-of-00054.parquet, th/train-00001-of-00054.parquet, th/train-00002-of-00054.parquet]
  documents: 2000
  utf8_bytes: 9686782
  set_sha256: 7204c28a0204defec98aa5390829f154f208fe106ce3f4eb345288859afb9cae
  filters: {min_chars: 500, max_chars: 20000, thai_script_min_ratio: 0.5}
  selection_stats: {scanned: 544716, in_bucket: 5633, too_short: 3444, too_long: 2, wrong_script: 9, exact_dup: 178}

exclusion_rule:
  rule_id: HELDOUT-BUCKET-V1
  module: phase0/heldout_rule.py
  definition: bucket = int(sha256(NFC + whitespace-collapsed text)[:8], 16) % 100 ; held out iff bucket == 0
  contract: >
    The training pipeline MUST import heldout_rule.is_trainable and drop every document for
    which it returns False. A bucket rule was chosen over an id list because a list only
    excludes documents the builder happened to materialise; unscanned shards and re-crawls
    would leak into training. The bucket excludes the whole 1 percent slice corpus-wide.
  cost: ~1 percent of the Thai corpus withheld, about 65M tokens. Negligible.

validation_observations:
  bucket_uniformity: >
    5633 of 544716 scanned documents landed in bucket 0 = 1.034 percent against a 1.000 percent
    expectation. The hash partition is behaving uniformly.
  duplicate_rate_within_bucket: >
    178 of 5633 in-bucket documents (3.2 percent) were EXACT duplicates after normalisation,
    despite the dataset card stating deduplication was already applied within each snapshot.
    This is direct evidence that CROSS-snapshot deduplication is still required on the training
    pool, and it is a measured finding rather than an assumption.
  short_document_rate: >
    3444 of 5633 in-bucket documents (61 percent) fell below the 500-character floor, consistent
    with the 335-character median observed in the schema probe. Any BPB set built without a
    length floor would be dominated by fragments.

gemma4_stack_support:
  finding_1: transformers 4.57.6 is the LATEST released version on PyPI and contains no gemma4 module.
  finding_2: transformers git main DOES contain gemma4, gemma4_assistant, gemma4_unified modules.
  finding_3: google/gemma-4-E2B config.json has no auto_map, so trust_remote_code cannot substitute for library support.
  correct_statement: >
    Gemma 4 is supported on transformers main but in NO released version. Using it means pinning
    an unreleased git commit and independently verifying FSDP / DeepSpeed / gradient-checkpointing
    interop. That is a real and manageable risk, NOT an impossibility.
  correction: >
    An earlier framing in this project implied Gemma 4 might be unusable. That was too strong.
    The accurate risk is dependency-pinning and unverified training-stack interop.

limitations:
  - TH-WEB-HELDOUT is drawn from the first 3 of 54 shards. It is a held-out sample, not a stratified sample of the corpus.
  - Token counts are NOT recorded; they depend on the final tokenizer and must be computed after the base is locked.
  - Benchmark decontamination against ThaiExam / M3Exam has NOT been run yet. It is a separate required step.
  - TH-ENC-HELDOUT, EN-HELDOUT and CODE-HELDOUT are not built yet.

verdict: heldout_th_web_frozen
next_required:
  - Run benchmark decontamination against the frozen eval suite.
  - Build TH-ENC, EN and CODE held-out sets.
  - Complete baseline BPB for Qwen3-1.7B-Base and gemma-4-E2B.
```

## `PHASE0-C-2026-08-18-BASELINE-QWEN3-1.7B`

```yaml
run_id: PHASE0-C-BASELINE-Qwen_Qwen3-1.7B-Base
stage: phase0_baseline_bpb
scientific_evidence_allowed: false
claim: Pre-CPT Thai BPB baseline for the proposed main base on the frozen held-out set
model: {repo: Qwen/Qwen3-1.7B-Base, revision: ea980cb0a6c2ae4b936e82123acc929f1cec04c1, parameters: 1.7e9}
eval_suite: {suite_id: THAILLM-EVAL-FROZEN-V1, spec_sha256: 1fae436e05fa99cee8b5b878e72f45c95aa8d51b3c1177e49c1c5a98c565cb19}
heldout_set: {set_id: TH-WEB-HELDOUT, set_sha256: 7204c28a0204defec98aa5390829f154f208fe106ce3f4eb345288859afb9cae}
measurement: {device: mps, dtype: float16, max_length: 2048, scoring: chunked fp32 log_softmax, chunk: 256}
result:
  thai_bits_per_byte: 0.504682
  nats_total: 3388621.184
  utf8_bytes: 9686782
  scored_tokens: 1780141
  documents: 2000

engineering_findings:
  performance_bug:
    symptom: under 100 documents in 8 minutes at 6.6 percent CPU
    cause: >
      A full float32 copy of the logits tensor [1, 2048, 151936] is ~1.2 GB, and log_softmax
      needs another. On a 16 GB unified-memory machine this drove the process into memory
      thrashing rather than compute.
    fix: chunked scoring over sequence positions (256 at a time), bounding the transient to ~155 MB
    effect: from under 100 docs in 8 min to 100 docs in 2 min 48 s; full run completed in ~75 min
    numerical_check: chunked vs full-tensor agreement rel. diff 5.4e-08
  rejected_fix:
    tried: F.cross_entropy instead of explicit float() + log_softmax
    result: no speed gain (0.67 vs 0.70 doc/s) AND fp16-rounded accumulator (rel. error ~4e-4/doc)
    decision: rejected; the hypothesis that the float32 copy dominated *compute* was wrong,
              the real problem was peak memory
  dtype_note: fp16 measured ~1.5x faster than bf16 on MPS (0.79 vs 0.52 doc/s); log-softmax is fp32 regardless

limitations:
  - This is a PRE-CPT baseline only. It is not evidence about any trained model.
  - BPB is comparable across models for the same language and set. It is NOT comparable across
    languages: Thai averages ~3 UTF-8 bytes per character versus ~1 for English.
  - TH-WEB-HELDOUT is drawn from 3 of 54 shards and is web text only.
  - Benchmark decontamination has NOT been run on this set yet.
  - No downstream accuracy measured; lm-evaluation-harness on the frozen task list is still required.
verdict: baseline_recorded_pre_cpt
next_required:
  - Same measurement for gemma-4-E2B (needs transformers from git main; no released version has gemma4).
  - lm-evaluation-harness on ThaiExam and M3Exam for both candidates.
  - Decide the headroom gate.
```

## `PHASE0-2026-08-19-OVERNIGHT-BATCH`

```yaml
run_id: PHASE0-2026-08-19-OVERNIGHT-BATCH
stage: phase0_no_gpu
scientific_evidence_allowed: false
claim: Full Thai corpus acquired, held-out set rebuilt without temporal bias, and two baselines measured
runtime: 05:13 to 08:20 local, all steps completed, no failures

step_1_corpus_download:
  shards: 54 of 54, zero failures
  on_disk: 21 GB
  note: Phase 1 cleaning is now fully unblocked and needs no GPU at all.

step_2_heldout_rebuild:
  set_id: TH-WEB-HELDOUT
  set_sha256: 48aaf8623e7f1a7ece19cbfe28f3eeb5ae4f35f6870213b91ed6ca771651c631
  supersedes_sha256: 7204c28a0204defec98aa5390829f154f208fe106ce3f4eb345288859afb9cae
  shards_used: [0, 13, 27, 40, 53]     # stratified across the 54-shard chronological range
  documents: 2000
  utf8_bytes: 9323384
  stats: {scanned: 543476, in_bucket: 5620, too_short: 3458, too_long: 3, wrong_script: 9, dup: 150}
  defect_fixed: >
    The previous set was built from shards 0-2. The corpus is ordered chronologically by
    CommonCrawl dump, so that set covered only CC-MAIN-2020-45 to 2022-05 -- the oldest
    slice. Evaluating a model trained on the full 2020-2025 range against a 2020-2022
    held-out set is a temporal bias.

step_3_corpus_audit:
  documents_sampled: 1000000
  bytes: 2315 MB
  dumps_covered: [CC-MAIN-2020-45, CC-MAIN-2021-43, CC-MAIN-2021-49, CC-MAIN-2022-05, CC-MAIN-2022-21]
  length_chars: {p10: 116, median: 379, p90: 2369, p99: 6159, max: 238189}
  short_doc_rate_under_500_chars: 0.594
  thai_script_ratio_under_0.5: 0.010
  exact_duplicate_rate: 0.210          # 21.0 percent of documents
  exact_duplicate_bytes_share: 0.117   # 11.7 percent of bytes
  content_flags_byte_share: {gambling: 0.0325, lottery: 0.0170, adult: 0.0029, seo_spam: 0.0011, boilerplate: 0.0015}
  top_domains: [thairath.co.th, pantip.com, news.thaipbs.or.th, mthai.com, sistacafe.com, bloggang.com]
  escalation_vs_200k_sample: >
    Duplicates rose from 11.3 to 21.0 percent of documents and gambling from 1.15 to 3.25
    percent of bytes when the sample went from 200k to 1M documents spanning more dumps.
    The earlier figures understated both. Cross-snapshot deduplication and a Thai gambling
    filter are now quantified requirements, not suggestions.

step_4_5_baselines:
  eval_suite: {suite_id: THAILLM-EVAL-FROZEN-V1, spec_sha256: 1fae436e05fa99cee8b5b878e72f45c95aa8d51b3c1177e49c1c5a98c565cb19}
  heldout_set_sha256: 48aaf8623e7f1a7ece19cbfe28f3eeb5ae4f35f6870213b91ed6ca771651c631
  measurement: {device: mps, dtype: float16, max_length: 2048, scoring: chunked fp32 log_softmax}
  results:
    Qwen/Qwen3-1.7B-Base: {thai_bits_per_byte: 0.454218, documents: 2000}
    Qwen/Qwen3-0.6B-Base: {thai_bits_per_byte: 0.521386, documents: 2000}
  sanity_check:
    expectation: the smaller model must score WORSE (higher BPB) or the measurement is broken
    observed: 0.6B = 0.521386 > 1.7B = 0.454218
    verdict: PASS
    interpretation: >
      A 0.067 BPB gap between 0.6B and 1.7B confirms the metric is sensitive to real model
      capability at this scale, so it can be expected to detect a CPT effect.
  heldout_choice_sensitivity:
    old_biased_set: 0.504682
    new_stratified_set: 0.454218
    delta: 0.050464 (10.0 percent relative)
    significance: >
      Changing which shards the held-out set was drawn from moved the number by 10 percent,
      five times the 2 percent relative improvement threshold set for promoting a variant.
      Had the biased set been kept, held-out composition alone could have dominated any
      CPT effect we later claimed.

limitations:
  - Pre-CPT baselines only. Nothing here is evidence about a trained model.
  - Benchmark decontamination against ThaiExam / M3Exam has still NOT been run.
  - No downstream accuracy measured; lm-evaluation-harness is still required.
  - gemma-4-E2B baseline not measured; it needs transformers from git main in a separate venv.
  - Regex content flags are recall-oriented screens, not classifiers. Treat as a lower bound.
verdict: phase0_measurement_infrastructure_validated
next_required:
  - Decontaminate TH-WEB-HELDOUT against ThaiExam and M3Exam.
  - Baseline gemma-4-E2B to close the base-model decision.
  - lm-evaluation-harness on the frozen task list.
  - Decide the headroom gate.
```

## `PHASE0-2026-08-19-HEADROOM-SAILOR2`

```yaml
run_id: PHASE0-C-BASELINE-sail_Sailor2-1B
stage: phase0_baseline_bpb
scientific_evidence_allowed: false
claim: Empirical headroom estimate for Thai CPT, using a Thai-adapted model of the same family as the reference
question_answered: >
  Before spending any GPU budget: is there measurable room for Thai continued pretraining
  to improve this base, or is it already near what CPT can deliver at this scale?
method: >
  Measure a model that has ALREADY received large-scale Thai CPT on the identical frozen
  held-out set, with the identical tokenizer family. Sailor2-1B is Qwen2.5-0.5B expanded
  to 988M and continued-pretrained on 500B tokens across 15 SEA languages including Thai.
  The gap between it and our un-adapted base is a direct empirical estimate of headroom.
heldout_set_sha256: 48aaf8623e7f1a7ece19cbfe28f3eeb5ae4f35f6870213b91ed6ca771651c631
results:
  sail/Sailor2-1B:      {thai_bpb: 0.378051, parameters: 0.99e9, thai_cpt: "500B tokens"}
  Qwen/Qwen3-1.7B-Base: {thai_bpb: 0.454218, parameters: 1.72e9, thai_cpt: none}
  Qwen/Qwen3-0.6B-Base: {thai_bpb: 0.521386, parameters: 0.60e9, thai_cpt: none}
headroom:
  absolute_bpb_gap: 0.076167
  relative_gap: 0.1677          # 16.8 percent
  promotion_threshold: 0.02     # 2 percent relative, from scientific_thresholds
  headroom_as_multiple_of_threshold: 8.4
finding: >
  A model with 42 percent FEWER parameters (0.99B vs 1.72B) scores 16.8 percent better on
  Thai after Thai-focused CPT. Parameter count is not the binding constraint here; Thai
  exposure is. This is a positive headroom signal and the strongest single piece of
  evidence so far that the planned CPT is worth its GPU budget.
CRITICAL_CAVEAT: >
  Sailor2 was trained on SEA web data that very likely includes CommonCrawl Thai from the
  same dumps our held-out set is drawn from. Part of its advantage may therefore be
  memorisation of documents it has seen, not generalisation. This measurement is an UPPER
  BOUND on achievable headroom, not an unbiased estimate. Confirming it requires a held-out
  set built from text published after Sailor2's training cutoff, which we do not have.
budget_caveat: >
  Sailor2 received 500B tokens. The plan is 6-10B, about 2 percent of that. CPT gains are
  strongly sublinear in tokens, so closing the full gap is not expected. Closing 20-50
  percent of it would still yield a 3.4-8.4 percent BPB improvement, comfortably above the
  2 percent promotion threshold.
limitations:
  - Pre-CPT reference points only. No model has been trained yet.
  - BPB only. No downstream accuracy has been measured for any of the three models.
  - The contamination caveat above is not quantified and is the main threat to this reading.
verdict: headroom_gate_provisional_pass
next_required:
  - Quantify or bound the Sailor2 contamination concern before quoting 16.8 percent in a paper.
  - Run lm-evaluation-harness on all three models for a downstream cross-check.
  - Decontaminate the held-out set against ThaiExam and M3Exam.
```

## `DATA-2026-08-20-REPLAY-EN-CODE-MATH`

```yaml
run_id: DATA-2026-08-20-REPLAY-EN-CODE-MATH
stage: replay_data_preparation
scientific_evidence_allowed: false
claim: >
  CPU-only, revision-pinned English/code/math replay pools meet the 10B-mixture targets,
  use the frozen Qwen3-1.7B tokenizer for exact counts, and are disjoint from the frozen
  EN/CODE held-out sets under HELDOUT-BUCKET-V1.

tokenizer:
  repo: Qwen/Qwen3-1.7B-Base
  revision: ea980cb0a6c2ae4b936e82123acc929f1cec04c1
  tokenizer_json_sha256: c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
  add_special_tokens: false
  model_weights_loaded: false

sources:
  en:
    repo: HuggingFaceFW/fineweb-edu
    revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
    subset: sample/10BT
    license: ODC-By-1.0 plus CommonCrawl Terms of Use
    source_shards: 5
    input_documents: 3655000
    output_documents: 3508510
    qwen_tokens: 3516598044
    target: 3500000000
  code:
    repo: codeparrot/github-code-clean
    revision: c48d40f9e70f0196f8236901ee35807f7d6c44c0
    license: Apache-2.0 dataset packaging; source files retain per-repository licenses
    access_decision: >
      The Stack v2, StarCoderData and The Stack Dedup were gated. The non-gated fallback
      was selected as required by TASK_A_REPLAY_DATA.md.
    retained_row_licenses: [apache-2.0, bsd-2-clause, bsd-3-clause, mit]
    source_shards: 9
    input_documents: 1142319
    output_documents: 700129
    qwen_tokens: 1104617663
    target: 1000000000
  math:
    repo: HuggingFaceTB/finemath
    revision: e92b25a616738fe95dc186b64dfb19f9c8525594
    subset: finemath-4plus
    license: ODC-By-1.0 plus CommonCrawl Terms of Use
    source_shards: 4
    input_documents: 418720
    output_documents: 412820
    qwen_tokens: 592984656
    target: 500000000

dropped_documents:
  en: {heldout_bucket: 36430, too_short: 46707, too_long: 3213, exact_duplicate: 60140}
  code: {heldout_bucket: 11493, too_short: 26783, too_long: 3394, disallowed_license: 400520}
  math: {heldout_bucket: 4172, too_short: 1289, too_long: 429, exact_duplicate: 10}

heldout:
  EN-HELDOUT:
    documents: 2000
    set_sha256: e0a30eae016b9d3f9610503fa0e26d70c9e24adf415d9637445d1b4f2f807b75
    source_files: 5
    quota_per_file: 400
  CODE-HELDOUT:
    documents: 1000
    set_sha256: 932d5efc953d8a83856ae97f642b89d8f28ef1f425eae42a74bac3e07e32cf68
    source_files: 5
    quota_per_file: 200

verification:
  report: validation/replay_verification.json
  unique_documents: 4621459
  full_output_shards_checked: 18
  compressed_sha_mismatches: 0
  document_hash_mismatches: 0
  cross_shard_duplicates: 0
  heldout_leakage: 0
  token_recount: {en: 100/100, code: 100/100, math: 100/100}
  unit_and_fixture_tests: 6/6
  dropped_sample_audit: >
    Deterministic samples were inspected for every observed drop reason: 20 per reason,
    except math exact_duplicate where all 10 observed examples were inspected.

decisions:
  - Length floors: EN 500 chars, code 100 chars, math 200 chars.
  - Maximum lengths: EN/math 100000 chars; code 200000 chars.
  - Exact dedup uses heldout_rule.doc_hash (NFC + whitespace collapse + SHA-256).
  - Source shards are selected in a deterministic spread; held-out sets additionally use
    equal per-file quotas so the first shard cannot fill the set.
  - Processing checkpoints atomically at source-shard boundaries using SQLite state.

limitations:
  - Exact dedup only; no near-duplicate removal.
  - No PII or secret scanner was applied to replay data in this task.
  - Per-row code licenses come from the source dataset metadata and were not independently
    re-verified against every upstream repository.
  - ODC-By applies to FineWeb-Edu/FineMath databases; underlying CommonCrawl content retains
    source rights and CommonCrawl terms.
  - Outputs are data-preparation artifacts, not evidence of model quality.

artifacts:
  - phase0/build_replay.py
  - phase0/build_heldout.py
  - phase0/verify_replay.py
  - phase0/test_replay_pipeline.py
  - data/clean_replay_manifest.json
  - data/heldout/EN-HELDOUT.manifest.json
  - data/heldout/CODE-HELDOUT.manifest.json
  - validation/replay_verification.json

verdict: replay_data_targets_and_isolation_pass
next_required:
  - Run PII/secrets policy decision for replay data before training.
  - Complete benchmark decontamination against the frozen evaluation suite.
  - Tokenize/pack the final capped 50/35/10/5 mixture without exceeding each pool target.
```

## `AUDIT-2026-08-20-REPLAY-INDEPENDENT`

```yaml
run_id: AUDIT-2026-08-20-REPLAY-INDEPENDENT
stage: desk_audit
scientific_evidence_allowed: false
claim: Independent verification of the agent-produced replay corpus, re-measured from the data itself
method: >
  Every headline claim was re-derived from the output files rather than read from the
  agent's report or its own verification artifact.

verified_pass:
  token_counting:
    method: recounted 900 documents (300 per language) with Qwen3-1.7B-Base, add_special_tokens=False
    result: en/code/math all matched the stored qwen_tokens field EXACTLY, zero difference
  heldout_rule_reuse: build_replay.py imports heldout_rule; no reimplementation of the bucket rule
  add_special_tokens_false: confirmed in both build_replay.py and verify_replay.py
  exact_duplicates_in_output: 0 across 750,000 documents scanned (250k per language)
  heldout_leakage: 0 across the same 750,000 documents
  code_license_allowlist:
    method: read the license field of 120,000 output documents
    result: mit 64,550 / apache-2.0 41,437 / bsd-3-clause 11,229 / bsd-2-clause 2,784; ZERO outside the allowlist
    dropped_for_licence: 400,520 documents / 3.53 GB — the filter did real work
  heldout_stratification:
    EN: sample/10BT shards 000,002,004,006,007
    CODE: shards 00000, 00251, 00502, 00753, 00879 of 880 — spread across the full range
    assessment: correct; avoids the temporal-bias failure previously found in the Thai set
  pinned_revisions:
    fineweb-edu: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 (sample/10BT)
    github-code-clean: c48d40f9e70f0196f8236901ee35807f7d6c44c0
    finemath: e92b25a616738fe95dc186b64dfb19f9c8525594 (finemath-4plus)
  per_record_provenance: source_repo, source_revision, source_file, repo_name, path, language, license
  unit_tests: 6/6 pass, re-run independently
  registry_integrity: 48 rows, no duplicate ids
  targets: en 3.517B/3.5B, code 1.105B/1.0B, math 0.593B/0.5B — all met

findings_not_failures:
  - id: CODE_MARKUP_SHARE
    severity: medium
    measured: >
      25.6 percent of code TOKENS are markup rather than source code, on 150,000 documents
      and 238.7M tokens: HTML 17.8, Markdown 4.4, plus CSS/XML/JSON/YAML. HTML alone carries
      more tokens than any real programming language in the sample.
    language_mix_by_tokens: {Java: 14.3, HTML: 17.8, JavaScript: 10.4, C: 9.9, C++: 8.7, Python: 6.8}
    why_it_matters: >
      The purpose of the code slice is to preserve the base model's coding ability. A quarter
      of it being markup dilutes that. The first code document inspected was javadoc-generated
      HTML boilerplate, not authored source.
    recommendation: >
      Consider excluding HTML/Markdown or capping their share before tokenisation. Not a defect
      in the pipeline; a property of github-code-clean that was not specified in the task.
  - id: CODE_GENERATED_FILES
    severity: low
    measured: 2.2 percent of 60,000 code documents self-identify as generated or auto-generated
  - id: MATH_QUALITY_INITIAL_CONCERN_WITHDRAWN
    note: >
      The first math sample inspected read as machine-translated spam. Quantifying it over
      8,000 documents showed only 0.8 percent carry low-quality signals and 6.3 percent lack
      mathematical symbols (some legitimately, e.g. word problems). Samples included kernel
      density estimation, applied-mathematics blogs, RMO practice problems and physics
      exercises. The initial concern was one unlucky draw and does not hold.

open_risks_correctly_self_declared_by_the_agent:
  - near-duplicate deduplication not performed (exact only)
  - benchmark decontamination not performed
  - PII and secrets scanning not performed on the replay corpus
  - per-repository provenance for code not audited beyond the license field

audit_verdict: PASS
notes: >
  The agent's self-report was accurate on every checkable claim. Its declared limitations
  were complete and matched what the data shows. The two findings above are properties of
  the chosen source that the task did not specify, not deviations from instructions.
```

## `DATA-2026-08-21-REPLAY-V2-SCAN-AND-REBALANCE`

```yaml
run_id: DATA-2026-08-21-REPLAY-V2-SCAN-AND-REBALANCE
stage: data_engineering
scientific_evidence_allowed: false
claim: Secret/PII redaction applied to the replay corpus and the code language mix rebalanced
artifacts:
  - ../phase0/secret_scan.py
  - ../phase0/pii_filter_en.py
  - ../phase0/apply_scan.py
  - ../phase0/rebalance_code.py
  - ../phase0/verify_replay_v2.py
  - ../phase0/test_scan_and_rebalance.py
  - ../data/clean_replay_v2_manifest.json
  - replay_v2_verification.json

totals_verified_independently:
  method: >
    phase0/verify_replay_v2.py re-derives every number from data/clean_replay_v2 itself.
    No agent summary file was trusted.
  en:   {documents: 3507052, qwen_tokens: 3513258103}
  code: {documents:  690716, qwen_tokens: 1000075432}
  math: {documents:  412734, qwen_tokens:  592472266}
  replay_total_tokens: 5.106e9
  with_thai: "5.851B Thai + 5.106B replay = 10.96B tokens available against a 6-10B budget"
  exact_duplicates: {en: 0, code: 0, math: 0}
  heldout_leakage:  {en: 0, code: 0, math: 0}
  token_recount_mismatches: {en: 0, code: 0, math: 0}   # 100 documents per language, exact
  verdict: PASS

code_rebalance:
  before: {python: 6.90, javascript_typescript: 14.07, java: 14.76, c_cpp: 18.59,
           csharp: 4.49, long_tail: 15.45, markdown: 4.29, html: 17.84, css_config: 3.62}
  after:  {python: 14.50, javascript_typescript: 14.50, java: 12.00, c_cpp: 12.00,
           csharp: 4.00, long_tail: 26.67, markdown: 5.00, html: 5.00}
  constraints: {max_single_language: "Python 14.499 percent (cap 15)", long_tail: "26.7 percent (floor 20)"}
  rationale_basis: GitHub Octoverse 2025 real-world language usage, recorded in sources/source_registry.csv
  rationale_explicitly_not: >
    HumanEval or any evaluation benchmark. HTML at 17.8 percent of tokens is unsupported by any
    usage survey and was an artefact of github-code-clean sampling. Balancing to the benchmark
    would be metric gaming; balancing to observed real-world usage is not.

secret_and_pii_hits_on_v1_inputs:
  secrets: {aws_access_key: 29, connection_string_password: 183, env_secret: 137,
            google_api_key: 319, jwt: 135, private_key_documents_dropped: 75,
            slack: 4, stripe: 1, aws_secret: 1, github_token: 0, openai: 0, anthropic: 0}
  pii: {email: 192864, phone: 91915, public_ip: 5003, credit_card: 211, us_ssn: 143}

process_note_five_detector_revisions:
  summary: >
    The agent ran the pipeline five times, rejecting and archiving each build after visual
    sampling exposed a NEW false-positive class: sk- matching CSS SpinKit names; version
    numbers read as IPs and ISSN read as SSN; [PASSWORD] and path-valued env variables;
    printf and escaped-angle placeholders. Nothing was deleted; each build was moved under
    data/rejected_task_b_intermediate/.
  assessment: >
    This is correct discipline, not a failure loop. An earlier interim reading by the
    supervising session mistook the archive-and-rerun pattern for a loop and for data loss.
    That reading was WRONG and is corrected here.

known_defect_accepted:
  id: PLACEHOLDER_OVER_REDACTION
  measured: 656 of 4,610,502 documents contain [REDACTED_SECRET] (0.014 percent)
  residual_cases: placeholders inside connection-string templates, e.g. "mongodb://%s:[REDACTED_SECRET]@%s"
  direction: >
    OVER-redaction, not under-redaction. For a security filter this is the safe direction;
    accepting this build exposes no additional real secret.
  decision: accepted; a further full rerun was judged disproportionate to a 0.014 percent cosmetic effect

limitations:
  - Detectors are regex with context guards, not trained models. Names, addresses and dates of birth are NOT detected.
  - Near-duplicate deduplication NOT performed (exact only).
  - Benchmark decontamination NOT performed.
  - Code provenance audited only to the license field, not per repository.
  - Hit counts measured on v1 inputs; nine additional raw code shards scanned during rebalancing were not fully re-aggregated.
verdict: replay_v2_frozen
next_required:
  - Near-duplicate deduplication across all four languages.
  - Benchmark decontamination against the seven tasks in THAILLM-EVAL-FROZEN-V1.
  - Tokenise and push the durable master to a private HF repo before renting any GPU.
```

## `DATA-2026-08-21-DECONTAMINATION-AND-NEAR-DEDUP`

```yaml
run_id: DECON-NEARDEDUP-V1
stage: data_engineering
scientific_evidence_allowed: false
claim: Benchmark decontamination and near-duplicate detection across the full training pool
artifacts:
  - ../phase0/build_benchmark_index.py
  - ../phase0/decon_and_neardedup.py
  - ../data/benchmark_ngrams.npz
  - ../data/benchmark_ngrams_meta.json
  - ../data/decontamination_hits.jsonl
  - ../data/near_duplicate_hits.jsonl
  - ../data/removal_list.txt
  - ../data/decon_neardedup_report.json

method:
  one_pass: >
    Decontamination and near-duplicate detection share the same character n-gram hashes per
    document, so the corpus is read once instead of twice.
  ngram: 64 characters, NFKC + lowercase + whitespace collapse
  hashing: Python str hash with PYTHONHASHSEED=0 (the script refuses to run without it)
  stride: benchmark index built at stride 1, corpus scanned at stride 8
  stride_rationale: >
    Asymmetric striding is deliberate. The index covers every alignment, so a document
    containing a verbatim benchmark span still matches while corpus cost drops eightfold.
  decontamination_threshold: at least 2 distinct benchmark n-grams
  near_duplicates: MinHash with 32 permutations, LSH 8 bands x 4 rows, first occurrence kept

benchmark_index:
  eval_suite: THAILLM-EVAL-FROZEN-V1
  benchmarks_covered: 12 sources (mmlu, hellaswag, arc_challenge, humaneval, belebele th+en,
                                  m3exam_th, thaiexam onet/ic/tgat/tpat1/a_level)
  unique_ngrams: 13697156
  note: All 7 benchmarks named in the frozen suite are covered; ThaiExam ships as 5 configs and all 5 are included.

results:
  documents_scanned: {th: 4567214, en: 3507052, code: 690716, math: 412734, total: 9177716}
  decontamination:
    flagged: {th: 803, en: 1371, code: 826, math: 233, total: 3233}
    rate: 0.0352 percent
    max_ngram_hits_single_document: 455
  near_duplicates:
    flagged: 270841
    rate: 2.95 percent
    note: These are near-duplicates remaining AFTER exact deduplication had already been applied.
  removal_list:
    unique_documents: 273703
    rate: 2.98 percent
    estimated_token_cost: >
      Roughly 0.33B tokens if removal is proportional to document count. NOT measured;
      the removal list stores document hashes only. Even at that estimate about 10.6B tokens
      remain against a 6-10B budget.
  runtime_minutes: 159.4

qualitative_check_documents_actually_caught:
  - "quizlet.com AP Biology Chapter 22-25 Test Questions — 455 n-gram hits"
  - "schoolbag.info AP Statistics — 444 hits"
  - "gorporonline.com Thai civil-service (ก.พ.) exam practice sets — 358-364 hits each"
  assessment: >
    The Thai hits are exam-preparation pages carrying ก.พ. practice questions in the same
    format as ThaiExam. Training on them and then scoring ThaiExam would have measured
    memorisation. This is the failure decontamination exists to prevent, and it was present
    in the corpus.

policy:
  action: FLAG ONLY. Nothing was deleted.
  contract: The tokenisation step MUST drop every doc_sha256 listed in data/removal_list.txt.
  rationale: Keeping the decision as a list makes it reversible and auditable.

limitations:
  - Character 64-grams detect verbatim or near-verbatim overlap only. Paraphrased or
    translated benchmark items are NOT detected.
  - Corpus scanned at stride 8 and capped at 40,000 characters per document.
  - MinHash 32 permutations with 8x4 banding targets high-similarity pairs; moderately
    similar documents are missed.
  - LSH keeps the FIRST document in each bucket, so which member of a near-duplicate cluster
    survives depends on file order rather than on quality.
  - Token cost of the removal list is estimated, not measured.
verdict: decontamination_and_near_dedup_complete
next_required:
  - Tokenise the corpus, applying data/removal_list.txt as a hard exclusion.
  - Push the durable master to a private HF repo before renting any GPU.
  - Re-run decontamination if any new data source is added.
```

## `DATA-2026-08-21-TOKENIZE-V1`

```yaml
run_id: TOKENS-V1
stage: data_engineering
scientific_evidence_allowed: false
claim: Corpus tokenised into per-language pools with the removal list applied
artifacts:
  - ../phase0/tokenize_corpus.py
  - ../phase0/build_training_stream.py
  - ../data/tokens_manifest.json
  - ../data/tokens/{th,en,code,math}.{bin,idx,meta.json}

tokenizer: {repo: Qwen/Qwen3-1.7B-Base, revision: ea980cb0a6c2ae4b936e82123acc929f1cec04c1,
            add_special_tokens: false, eos_token_id: 151643}

results:
  th:   {documents: 4350370, tokens: 5.831e9, share: 0.535}
  en:   {documents: 3492818, tokens: 3.500e9, share: 0.321}
  code: {documents:  649757, tokens: 0.970e9, share: 0.089}
  math: {documents:  410058, tokens: 0.589e9, share: 0.054}
  total_tokens: 10.890e9
  on_disk: 41 GB (uint32)
  dropped_by_removal_list: {th: 215834, en: 14234, code: 40959, math: 2676}

why_per_language_pools: >
  The training mixture is chosen by build_training_stream.py, not baked in here. The
  replay-ratio ablation (50/70/90 percent Thai) therefore rebuilds a stream in minutes
  instead of re-tokenising 10.9B tokens.

track2_contract: >
  S0, D1 and D2 read ONE stream file produced by build_training_stream.py. Identical data
  and identical order across conditions is guaranteed by construction, and the manifest
  records train_bin_sha256 so a reviewer can verify it after the fact rather than trusting
  a statement in the paper.

finding_heldout_guard_fired:
  observed: the redundant is_heldout() guard fired on 1,010 Thai documents during tokenisation
  investigation: >
    Checked directly: the intersection of the 2,000 held-out document hashes with all
    4,567,214 corpus document hashes is EMPTY, and none of the documents the guard rejected
    were in the held-out set. There is no leakage.
  root_cause: >
    The bucket rule hashes document TEXT. PII redaction rewrites text in place, so a
    document's hash changes across the redaction boundary and can land in bucket 0 even
    though its original text did not. The guard is therefore not stable across that
    boundary and produces false positives there.
  impact: 1,010 of 4,350,370 Thai documents (0.02 percent) were excluded unnecessarily. Harmless.
  correct_fix_if_repeated: >
    Compare the stored doc_sha256, which is computed on the pre-redaction text and is stable,
    rather than recomputing is_heldout() from the mutated text. Not applied here because the
    verification shows no leakage and the cost is 0.02 percent of documents.

limitations:
  - Held-out BPB is measured on RAW held-out text while training uses PII-redacted text.
    A mild distribution difference, not contamination.
  - Sequences cross document boundaries; EOS marks each boundary.
  - The final partial sequence of each stream is discarded, not padded.
verdict: token_pools_frozen
next_required:
  - Build the main training stream and the ablation streams.
  - Upload token pools to the private HF repo before renting any GPU.
```

## `SRC-2026-08-21-ROUTER-FIX`

```yaml
run_id: SRC-2026-08-21-ROUTER-FIX
stage: local_unit
scientific_evidence_allowed: false
claim: Corrected router implementation; the six defects found by the 2026-08-18 audit are fixed
artifacts:
  - ../src/routing.py
  - ../src/test_routing.py
supersedes: >
  ../../thai-llm-five-to-two/depth_routing/routing.py — archived and READ-ONLY, not edited.

fixes:
  FIX1_ROUTE_SCALE_GATE:
    was: >
      out = residual + s * mixture(w) with s initialised to zero, so dL/dw is proportional
      to s and identically zero at initialisation. Measured: query gradient still ~0.095
      percent of the paper-faithful magnitude after 50 optimiser steps.
    now: >
      Zero-initialised query only, as in arXiv:2605.18855. That gives a uniform softmax and
      a BOUNDED PERTURBATION at init rather than an exact identity, and non-zero query
      gradients from step 1.
    measured_after_fix:
      D1_delta:    {max_query_grad_step1: 0.737785, layers_receiving_gradient: "6/6"}
      D2_mhar_h4:  {max_query_grad_step1: 0.868846, layers_receiving_gradient: "6/6"}
      before_fix:  {max_query_grad_step1: 0.0, note: "every layer, every step"}
  FIX2_MHAR_IMPLEMENTED:
    was: condition D2 had no code at all in the archive
    now: >
      DeltaRouter takes num_heads; the query is reshaped into H per-subspace heads each with
      its own softmax over the depth history, per arXiv:2607.27230. num_heads=1 reproduces
      plain Delta, so D1 and D2 share one code path and cannot diverge by accident.
  FIX3_AUTOGRAD_GRAPH_RETENTION:
    was: last_routing stored live tensors, pinning activation memory for the routed arms only
    now: diagnostics are detached AND opt-in via collect_routing, default False
  FIX4_GRADIENT_CHECKPOINTING:
    was: >
      the adapter raised if gradient checkpointing was enabled, so S0 could use it and D1/D2
      could not. Different memory regimes per condition make the GPU-hour axis measure the
      wrapper rather than the architecture.
    now: allowed; only genuine re-entrancy is rejected
  FIX5_BLOCK_SEMANTICS:
    was: silent — the first block routes over zero sources, so those routers are dead weight
    now: surfaced as adapter.blocks and adapter.layers_without_sources
  FIX6_TEN_PERCENT_KILL_RULE: not carried over; this project uses compute-normalised curves

tests:
  file: ../src/test_routing.py
  result: 11/11 pass
  the_missing_test: >
    test_router_receives_nonzero_gradient_at_step_one is placed first deliberately. The
    archived suite passed 25/25 while shipping a router that could not learn, because its
    tests asked "does the function do what it says" and none asked "are the three conditions
    actually comparable". This test would have failed on the archived build.

limitations:
  - Tiny random fixtures only. No pretrained weights, no Thai, no throughput measurement.
  - Not a certified reproduction of the official Delta AttnRes implementation.
  - The official MIT implementation at github.com/wdlctc/delta-attention-residuals-code ships
    modeling_qwen3_attnres.py and remains the preferred path for the final runs; this module
    exists so the conditions are runnable and testable now, with the audit fixes explicit.
verdict: router_defects_closed_pending_gpu_preflight
next_required:
  - One-GPU preflight for S0/D1/D2 under identical memory settings.
  - Compare against the official implementation before quoting any D1/D2 result in a paper.
```

## `SRC-2026-08-21-TRAINING-SCRIPTS`

```yaml
run_id: SRC-2026-08-21-TRAINING-SCRIPTS
stage: local_unit
scientific_evidence_allowed: false
claim: CPT training script and one-GPU preflight written and smoke-tested before any GPU rental
artifacts: [../src/train_cpt.py, ../src/preflight.py]

train_cpt:
  data: reads the packed stream from build_training_stream.py; all conditions read ONE file
  resume: >
    Data order is a seeded permutation of sequence indices, so restarting at global_step N
    replays exactly what an uninterrupted run would have seen. Order does not depend on how
    many times the job was killed.
  parity: gradient checkpointing enabled for EVERY condition, so S0 and D1/D2 share one
          memory regime and the GPU-hour axis measures architecture, not the wrapper
  eval: chunked fp32 BPB during training, same method as measure_baseline.py

preflight:
  measures: [tokens_per_s, peak_vram, step_time_spread, checkpoint_resume, S0_vs_D1_vs_D2]
  projects: GPU-hours per billion tokens from MEASURED throughput, replacing the MFU-40 guess
  trains_but_discards: 220 steps; the weights are thrown away

bug_found_by_smoke_test:
  id: ROUTEDLAYER_ATTRIBUTE_PROXY
  symptom: "AttributeError: 'RoutedLayer' object has no attribute 'attention_type'"
  cause: >
    Qwen3 reads per-layer metadata straight off the layer object each forward. The wrapper
    did not proxy unknown attributes, so NO routed condition could run at all.
  fix: RoutedLayer.__getattr__ falls through to the wrapped layer
  note: >
    S0 ran fine; only D1 and D2 failed. Without a local smoke test this would have been
    discovered on a rented GPU.

finding_null_logit_init_is_the_correct_control:
  problem: >
    Removing route_scale fixed the zero-gradient defect but left the routed conditions
    starting from a much worse loss, because a zero-initialised query gives a uniform softmax
    over all sources and therefore a large perturbation to a converged checkpoint.
  measurement: |
    Qwen3-0.6B, one forward, 128 tokens, S0 loss 13.8879
      null_logit_init   loss delta vs S0   max query grad
        route_scale=0         0.0000          0.000000   cannot learn
              0.0            +6.7440          0.589615   paper default, disruptive
              2.0            +0.7739          2.162301   selected
              4.0            -0.0234          0.064080
              8.0            +0.0051          0.002060   approaching the old failure
  why_it_differs_from_route_scale: >
    The null source contributes nothing, so raising its logit moves softmax mass onto "do
    nothing" and shrinks the mixture towards zero. The query gradient decays as e^-c instead
    of being multiplied by zero, so the router keeps learning.
  decision: >
    Default 2.0. The +0.77 starting penalty is recoverable and it gives the LARGEST query
    gradient of any setting tested. Values at or above 4 buy near-identity conversion at the
    cost of a gradient 30-1000x smaller, which risks recreating the route_scale failure in
    milder form: a router that technically can learn but does not, within the token budget.
  status: preregistered choice; must be checked in preflight before the pilot

limitations:
  - Smoke tests used Qwen3-0.6B on MPS with a synthetic random-token stream and 2-3 steps.
    They verify that the code runs and that resume machinery exists. They say nothing about
    model quality, throughput on real hardware, or whether routing helps.
  - Multi-GPU DDP path is written but has NOT been executed; there is no multi-GPU machine here.
  - The null_logit_init table is one forward pass on one model at one sequence length.
verdict: scripts_ready_for_preflight
next_required:
  - Run preflight on a rented GPU; confirm resume passes before committing the main budget.
  - Verify the DDP path on the first multi-GPU session.
```

## Required future entries

- `BASE-SCREEN-*`: tokenizer, frozen BPB, license and port-complexity results
- `DATA-AUDIT-*`: provenance, licenses, filtering, deduplication and leakage
- `LOCAL-SMOKE-*`: overfit/resume/gradient evidence with scientific flag false
- `A100-PREFLIGHT-*`: 200-step throughput, memory and stability evidence
- `PILOT-*`: 50M–100M token S0/D1/D2 results
- `FULL1B-*`: only after pilot promotion gates pass

Use [`run_record.template.json`](run_record.template.json) for machine-readable
records and link each artifact from the corresponding entry here.

---

## A3 — lm-evaluation-harness ติดตั้งและ smoke test

**วันที่:** 2026-08-22 · `scientific_evidence_allowed=false`
**lm-eval:** 0.4.9.1 · **โมเดลทดสอบ:** Qwen3-0.6B-Base `da87bfb` · MPS · fp16 · `--limit 2`

### สิ่งที่ทำ

frozen suite `THAILLM-EVAL-FROZEN-V1` มี 7 งาน + BPB 3 ตัว ตรวจแล้วพบว่า
lm-eval มี built-in อยู่ 5 งาน ขาด **ThaiExam** และ **M3Exam (ไทย)** จึงเขียน task config เอง:

| ไฟล์ | หมายเหตุ |
|---|---|
| `eval/tasks/thaiexam_{onet,tgat,tpat1,a_level}.yaml` | 5 ตัวเลือก (ก–จ) |
| `eval/tasks/thaiexam_ic.yaml` | **4 ตัวเลือก** — subset `ic` ไม่มีคอลัมน์ `e` |
| `eval/tasks/thaiexam.yaml` | group รวม 5 subset |
| `eval/tasks/m3exam_th.yaml` + `utils.py` | map `answer_text` → index |
| `eval/run_eval.py` | อ่าน frozen suite โดยตรง เพื่อไม่ให้ metric drift |

### บั๊กที่เจอและแก้

1. **`jinja2.UndefinedError: 'e' is undefined`** — ตอนแรกใช้ template เดียวกันทั้ง 5 subset
   ตรวจ schema จริงพบว่า `ic` (n=95) มีแค่ `a,b,c,d` และยังมีบางแถวที่ `b`/`c` เป็นค่าว่าง
   แก้โดยแยก YAML ของ `ic` เป็น 4 ตัวเลือก และเพิ่ม `utils.process_thaiexam_{4,5}`
   กรองแถวที่ตัวเลือกว่างหรือ `answer` ไม่อยู่ในช่วง พร้อมพิมพ์จำนวนที่ตัดทิ้ง
   — ถ้าไม่กรอง โมเดลจะถูกวัดจากการเลือกระหว่างสตริงว่าง ซึ่งเป็น noise ที่ถูกรายงานเป็น accuracy

2. **`ValueError: _WARNING` ที่ humaneval** — lm-eval กันไว้เพราะ HumanEval ให้คะแนนด้วยการ
   **รันโค้ดที่โมเดลสร้างขึ้นจริง** แก้โดยเพิ่ม flag `--allow-code-exec` แบบ opt-in
   (ไม่เปิดอัตโนมัติ) ถ้าไม่ใส่ `run_eval.py` จะข้าม humaneval พร้อมแจ้งเหตุผล
   **ให้เปิดเฉพาะบนเครื่อง GPU ที่เช่าและทิ้งได้ ห้ามเปิดบนเครื่องส่วนตัว**

3. **HF dataset cache เสีย** — `Feature type 'List' not found` ที่ hellaswag และ ai2_arc
   แก้โดยลบ `~/.cache/huggingface/datasets/Rowan___hellaswag` และ `allenai___ai2_arc`

### ผล smoke (`--limit 2` — ตัวเลขนี้ห้ามนำไปรายงาน)

| shot | tasks | returncode |
|---|---|---|
| 0 | humaneval | `skipped_needs_allow_code_exec` (ตามที่ออกแบบ) |
| 5 | thaiexam, m3exam_th, belebele_tha_Thai, mmlu | 0 ✅ |
| 10 | hellaswag | 0 ✅ |
| 25 | arc_challenge | 0 ✅ |

BPB 3 ตัวไม่ผ่าน lm-eval โดยตั้งใจ — วัดจาก `src/train_cpt.py` และ
`phase0/measure_baseline.py` และถูกบันทึกไว้ในช่อง `not_run_by_lm_eval` ของรายงาน

**สรุป:** A3 เสร็จ — harness รันได้ครบทุกงานในชุดที่ freeze ไว้ ที่เหลือคือรันเต็ม (ไม่มี `--limit`)
บนเครื่อง GPU ที่เช่า พร้อม `--allow-code-exec` สำหรับ humaneval

---

## A4 — smoke ของ pipeline เทรนจริงบน MPS

**วันที่:** 2026-08-22 · `scientific_evidence_allowed=false`
**โมเดล:** Qwen3-0.6B-Base `da87bfb` (ตัวแทน 1.7B เพราะเครื่องนี้เป็น MPS)
**Stream:** `data/streams/_smoke` — 64 seq × 1024 tokens ตัดมาจาก `main` แล้ว re-chunk
ไม่เคารพขอบเขตเอกสาร **ใช้ทดสอบท่อเท่านั้น ห้ามใช้วัดผล**

### สิ่งที่ยืนยันแล้ว

| รายการ | ผล |
|---|---|
| S0 / D1 / D2 รันจบครบ 3 เงื่อนไข | ✅ |
| gradient checkpointing เปิดเท่ากันทุกเงื่อนไข | ✅ |
| BPB eval ระหว่างเทรน (chunked fp32) | ✅ S0 0.5900 → 0.5904, D1 0.7628 → 0.7281 |
| checkpoint save | ✅ |
| **resume ตรงเป๊ะ** | ✅ เทียบ 395 tensors ระหว่างรันรวดเดียวกับรัน→หยุด→resume **ต่างสูงสุด 0.000e+00** |
| unit tests ของ router | ✅ 11/11 |

### สิ่งที่เจอ และเหตุผล

**1. router ของ layer 0–3 ไม่ขยับเลย (`query`=0.0, `null_logit`=2.0 เป๊ะ)**
ไม่ใช่บั๊ก `route_scale` กลับมา — `block_size=4` ทำให้ layer 0–3 เป็น**บล็อกแรก
ซึ่งไม่มีบล็อกก่อนหน้าให้ route** router พวกนี้จึงตายโดยโครงสร้าง
(`routing.py` FIX 5 บันทึกไว้แล้วในชื่อ `layers_without_sources`)
layer 4 ขึ้นไปขยับจริง: `null_logit` 2.000000 → 2.000041, `|query|max` 4.10e-05 หลัง 4 สเต็ป

**2. `router.norm.weight` ไม่มี gradient ที่ init**
ไม่ใช่บั๊กเช่นกัน — logit คือ `query · norm(x)` ดังนั้น `∂/∂norm.weight` มีตัวคูณเป็น `query`
ซึ่ง paper กำหนดให้เริ่มที่ **0 พอดี** วัดยืนยันแล้ว: grad = 0.000e+00 ที่ init
และ = **1.05e-05 ทันทีที่ query ขยับไป 1e-3** คือมันปลดล็อกตัวเองหลังสเต็ปแรก

**3. D1 กับ D2 ให้ loss เท่ากันเป๊ะที่สเต็ปแรก (3.1348)**
ถูกต้องตามที่ควรเป็น — query ของทุก head เริ่มที่ 0 เหมือนกันหมด logits จึงเท่ากันหมด
ไม่ว่าจะมีกี่ head จำนวน head จะเริ่มมีผลก็ต่อเมื่อ query แยกจากกันแล้ว
(ที่สเต็ป 4 แยกกันจริง: BPB 0.72806 vs 0.72787)

### สิ่งที่เพิ่มเข้าไปจากรอบนี้

**`assert_routers_learn()` ใน `train_cpt.py`** — ทำ forward/backward หนึ่งครั้งก่อนเริ่มเทรน
แล้ว**บังคับ**ว่า routing parameter นอกบล็อกแรกต้องได้ gradient จริง ถ้าไม่ได้ให้ raise ทันที
เหตุผล: บั๊ก `route_scale` เดิมทำให้ gradient เป็นศูนย์**แบบเงียบ** งานเทรนจบสวยงาม
แล้วให้ผลลบที่ดูสะอาดทั้งที่จริงเป็นสิ่งประดิษฐ์ของ wrapper — ความล้มเหลวแบบนี้ต้องถูกตรวจ ไม่ใช่ถูกเชื่อ
ผลจริงทุกรัน D1/D2: `48 receive gradient, 24 gated by the zero-init query, 12 inert in the first block`

แก้ `float(loss)` เป็น `loss.detach().item()` ตัด UserWarning เรื่อง tensor ที่ยัง requires_grad

### ยังไม่ได้ทดสอบ

- **multi-GPU DDP** — เครื่องนี้ไม่มี GPU หลายใบ ต้องทดสอบบนเครื่องเช่าเป็นอย่างแรก
- `peak_vram_gb` ไม่ถูกบันทึกบน MPS (โค้ดอ่านจาก `torch.cuda`) จะมีค่าจริงบนเครื่อง CUDA
- throughput บน MPS (~175 tok/s) **ไม่มีความหมายในการประมาณเวลา** — ตัวเลขที่ใช้จริงต้องมาจาก preflight บน GPU ที่เช่า

### A4 (ต่อ) — บั๊กที่ preflight เจอในตัวมันเอง

รัน `preflight.py` เต็มบน `_smoke` ครบ 4 ส่วน (throughput ×3 เงื่อนไข + resume test)
รอบแรก resume test ขึ้น `max_abs_weight_diff = 3.36e-4` ทั้งที่การทดสอบด้วยมือก่อนหน้าได้ 0.0

**การไล่หาสาเหตุ** — รันเส้นตรงชุดเดิมซ้ำสองรอบก่อน ได้ต่างกัน `0.000e+00`
พิสูจน์ว่า MPS deterministic จึงตัดเรื่อง hardware ออก แล้วตรวจ lr schedule พบว่า:

| step | lr เมื่อขอบฟ้า=30 | lr เมื่อขอบฟ้า=40 |
|---|---|---|
| 5 | 1.949e-05 | 1.972e-05 |
| 15 | 1.201e-05 | 1.528e-05 |
| 29 | 2.057e-06 | 5.472e-06 |

**สาเหตุ:** `--max-steps` ถูกใช้เป็นทั้งจุดหยุด **และ** ตัวหารของ cosine schedule
preflight รันขาแรกด้วย `--max-steps 30` (ขอบฟ้า 30) แล้ว resume ต่อด้วย 40
ส่วนรันเส้นตรงใช้ 40 ตลอด → lr ต่างกันตั้งแต่สเต็ปแรก
**resume ไม่ได้พัง ตัวทดสอบต่างหากที่ระบุผิด**

**ทำไมถึงสำคัญเกินกว่าเรื่องเทสต์:** ถ้า spot instance ถูกเรียกคืนแล้ว resume ด้วย
`--max-steps` ที่ไม่เท่าเดิม cosine schedule จะเปลี่ยนรูปเงียบ ๆ ทุกขาหลังการหยุดจะเทรน
ด้วย lr ที่ไม่ใช่ที่ตั้งใจ และ **loss curve มองไม่เห็นความต่างนี้เลย**

**แก้:**
- แยก `--schedule-steps` ออกจาก `--max-steps` — จุดหยุดกับขอบฟ้า lr เป็นคนละเรื่อง
- เก็บ `schedule_steps` ใน checkpoint และ **raise ตอน resume ถ้าไม่ตรง** พร้อมบอกค่าที่ต้องใส่
- preflight ปักขอบฟ้าเป็นระยะเต็มทั้งสองขา
- resume test เปลี่ยนจากเทียบ loss เป็น **เทียบทุก tensor** — เทียบ loss จับไม่ได้ถ้า resume
  โหลดน้ำหนักถูกแต่เล่นข้อมูลผิดช่วง
- เพิ่ม `--log-every` (preflight ตั้งเป็น 1) เดิม log ทุก 10 สเต็ป ทำให้ `measured_steps`
  ได้แค่ 2 จาก 20 และ `step_time_cv` ที่คำนวณจาก 2 จุดไม่ใช่สถิติ

**ผลหลังแก้:** `resume ok · max_abs_weight_diff 0.0 · measured_steps 6 · step_time_cv 0.021`

### ตัวเลข throughput จาก preflight (MPS — ห้ามใช้ประมาณเวลาจริง)

| เงื่อนไข | tokens/s | เทียบ S0 |
|---|---|---|
| S0 | 164.1 | 1.00 |
| D1 | 99.7 | 0.61 |
| D2 | 137.3 | 0.84 |

อยู่ในช่วง 0.55–0.88× ที่ `arXiv:2607.27230` (MHAR) รายงานไว้ แต่ **ตัวเลขบน MPS
ไม่ใช่หลักฐานว่า GPU จริงจะเป็นแบบนี้** ตัวเลขที่ใช้วางงบต้องมาจาก preflight บนการ์ดที่เช่า
`peak_vram_gb` เป็น null เพราะโค้ดอ่านจาก `torch.cuda` — จะมีค่าจริงบนเครื่อง CUDA

**สถานะ: A1–A4 ครบแล้ว** เหลือ DDP ที่ทดสอบได้เฉพาะบนเครื่องที่มี GPU หลายใบ

---

## A5 — fp32 optimizer และ DDP path ใน preflight

**วันที่:** 2026-08-22 · `scientific_evidence_allowed=false`

### 1. เปลี่ยนจาก bf16 ล้วนเป็น mixed precision

ตรวจ optimizer state ที่ checkpoint เขียนออกมาแล้วพบว่าเทรน **bf16 ทั้งกระบวนการ**:

```
params bfloat16 · grads bfloat16 · exp_avg bfloat16 · exp_avg_sq bfloat16
```

bf16 มี mantissa 8 บิต ที่ lr 2e-5 การอัปเดตแต่ละครั้งมักเล็กกว่า 1 ulp ของน้ำหนักที่มันไปบวก
จึงถูกปัดทิ้งทั้งก้อน — **รันดูปกติทุกอย่าง แต่โมเดลแทบไม่ขยับ** และ `exp_avg_sq` แย่กว่านั้น
เพราะเก็บค่ากำลังสองของ gradient ซึ่งกินช่วงไดนามิกมาก

`--precision` ค่าเริ่มต้นเป็น `mixed` แล้ว: master weights fp32 + autocast เฉพาะ forward
`bf16` ยังเลือกได้ถ้าจำเป็นเรื่องหน่วยความจำ ยืนยันหลังแก้:

```
optimizer: exp_avg float32 · exp_avg_sq float32   params: float32   precision: mixed
```

บน MPS ใช้ autocast fp16 แทน bf16 เพราะ bf16 บน MPS ยังไม่นิ่ง และเคยวัดได้ว่า fp16
เร็วกว่า (0.79 vs 0.52 doc/s)

**ผลต่อการเลือกการ์ด** — ตัวเลขที่ `train_cpt.py` พิมพ์ตอนเริ่มรันคือ weights+grads+optimizer:

| precision | Qwen3-1.7B | การ์ด 24 GB |
|---|---|---|
| bf16 ล้วน | ~13.6 GB | พอ |
| mixed (fp32 master) | ~27.2 GB | ไม่พอเมื่อรวม activations |

### 2. preflight ไม่เคยทดสอบ DDP เลย

`preflight.py` เรียก `train_cpt.py` ด้วย `sys.executable` ตรง ๆ คือ process เดียว
**เส้นทาง DDP ไม่เคยถูกแตะ** ทั้งที่เอกสารของ preflight เองอ้างว่าตรวจสิ่งที่จะพังบนเครื่องเช่า
ถ้าไม่เจอตอนนี้ multi-GPU จะไปพังตอนจ่ายเงินแล้ว

เพิ่ม `--nproc` ถ้ามากกว่า 1 จะ launch ผ่าน `torch.distributed.run` และรายงานบันทึก
`ddp_exercised` ไว้ตรง ๆ เพื่อไม่ให้อ่านรายงานแล้วเข้าใจผิดว่า DDP ผ่านแล้ว

ยืนยันบน MPS: unit tests 11/11 · S0 และ D1 รันจบด้วย `precision=mixed`
DDP จริงยังทดสอบไม่ได้บนเครื่องนี้ (GPU ใบเดียว) — ต้องรัน `--nproc 2` บนเครื่องเช่าเป็นอย่างแรก

---

## A6 — preflight บนฮาร์ดแวร์จริง (2× A100 SXM4 80GB)

**วันที่:** 2026-08-22 · `scientific_evidence_allowed=false`
**เครื่อง:** vast.ai · 2× A100-SXM4-80GB · NVLink · PCIe 4.0 ×16 · $2.348/hr · เช็กเกีย
**สภาพแวดล้อม:** torch 2.11.0+cu128 · transformers 5.15.1 · Qwen3-1.7B-Base `ea980cb`
**ผลดิบ:** `validation/preflight_a100/`

### ผลการวัด

| เงื่อนไข | tokens/s | เทียบ S0 | peak VRAM | step_time p90 | step_time_cv |
|---|---|---|---|---|---|
| S0 | 14,950 | 1.000 | 54.93 GB | 8.81 s | 0.0032 |
| D1 | 12,439 | 0.832 | 72.94 GB | 10.56 s | 0.0019 |
| D2 | 12,292 | 0.822 | 72.95 GB | 10.69 s | 0.0017 |

`ddp_exercised: true` · micro-batch 1 · grad-accum 8 · seq 8192 · mixed precision

routing overhead 0.82–0.83× อยู่ปลายบนของช่วง 0.55–0.88× ที่ `arXiv:2607.27230` รายงาน
D2 (4 heads) แพงกว่า D1 (1 head) เพียง 1% และใช้ VRAM เท่ากัน สอดคล้องกับที่ MHAR
อ้างว่าเพิ่ม parameter เป็นศูนย์ · step_time_cv 0.002–0.003 คือเครื่องนิ่งมาก ไม่มี throttle

### บั๊กที่เจอ — ทั้งหมดเป็นชนิดที่เครื่อง MPS หาไม่เจอ

**1. DDP ล้มทุกสเต็ปในเงื่อนไข routed**

```
RuntimeError: Expected to have finished reduction in the prior iteration ...
parameters that were not used in producing loss
```

router ของบล็อกแรกไม่เคยได้ gradient เพราะไม่มีบล็อกก่อนหน้าให้ route
DDP บังคับว่าทุก parameter ต้องได้ gradient ทุกรอบ reducer จึงรอ bucket ที่ไม่มีวันเต็ม
แก้โดย**ไม่สร้าง router ให้บล็อกแรก** (ยังห่อ layer ไว้ เพราะ block bookkeeping ของมัน
คือสิ่งที่ผลิต sources ให้บล็อกถัดไป) — `find_unused_parameters=True` กลบ error ได้
แต่เพิ่มการไล่กราฟทุกรอบ ซึ่งทำลายจุดประสงค์ของสคริปต์ที่มีไว้วัด throughput
เพิ่มเทสต์ 2 ตัวล็อก invariant ไว้ เพราะรัน GPU ใบเดียวสังเกตไม่เห็น (`a8bd5db`)

**2. เกณฑ์ resume test ผิด — ผมตั้งไว้เป็นค่าคงที่**

รอบแรก resume test FAILED ที่ `3.96e-4` เทียบเกณฑ์ `1e-4`
รันเส้นตรงสองรอบด้วยอาร์กิวเมนต์และ seed เดียวกันเป๊ะ ได้ต่างกัน **`4.10e-4`**
คือ **มากกว่า resume** ต่างกันครบทั้ง 311 tensor

สาเหตุ: CUDA ใช้ atomics ใน backward, cuBLAS เลือก workspace ตอนรัน, NCCL ไม่ fix
ลำดับ all-reduce ความต่างระดับ 1e-4 ที่สเต็ปแรกจึงขยายไปทั่วโมเดลภายใน 40 สเต็ป
ส่วน MPS ที่ผมเขียนเทสต์ตอนแรก deterministic จริงและให้ 0 พอดี **จึงทำให้ตั้งเกณฑ์ผิด**

แก้ให้ preflight **วัดพื้น noise เอง** ด้วยขาที่ 4 ที่เหมือนขาที่ 3 ทุกประการ
แล้วตัดสิน resume เทียบกับพื้นนั้น (`9f030ff`) ผลยืนยันบนเครื่องจริง:

| | |
|---|---|
| resume_vs_straight | **3.9513e-04** |
| nondeterminism_floor | **3.9285e-04** |
| threshold (2× floor) | 7.8569e-04 |
| **ผล** | **ok — resume แม่นเท่าที่ฮาร์ดแวร์นี้จะแม่นได้** |

**ผลที่ตามมาเป็นเงิน: ใช้ spot instance ได้** ซึ่งถูกกว่า On-Demand 2–3 เท่า

**3. ตัวแปรชื่อชนใน preflight** — `ok` ที่เพิ่มใหม่ทับ `ok` ที่ budget projection ใช้อยู่
ทำให้ preflight crash **หลังจาก resume test ผ่านไปแล้ว** และไม่ได้เขียนรายงาน
รันที่สำเร็จจึงดูเหมือนรันที่ล้มเหลว (`e2dcd62`)

**4. ดิสก์เต็ม 100%** — checkpoint ละ **20.6 GB** เพราะ fp32 optimizer state
container disk 82 GB ไม่พอ resume test ต้องมี checkpoint 3 ไฟล์อยู่พร้อมกันเพื่อเทียบคู่
= **62 GB แค่เทสต์เดียว** → **รอบหลักต้องขอดิสก์ ≥ 300 GB ตอนเช่า**

**5. torch อยู่ใน `/venv/main`** ที่ SSH แบบไม่ interactive มองไม่เห็น และ transformers
ไม่ได้ติดตั้งมากับ image ต้องเรียก `/venv/main/bin/python` ตรง ๆ

### ประมาณการที่ผิด และค่าที่ถูก

| | ผมประเมิน | วัดได้จริง |
|---|---|---|
| peak VRAM | 39.5 GB | **54.9 GB (S0) · 72.9 GB (D1/D2)** |
| การ์ดขั้นต่ำ | 48 GB "สบาย" | **48 GB ใช้ไม่ได้ · ต้อง 80 GB** |

ที่ไม่ได้นับคือ logits ของ vocab 151,936 ตัวที่ seq 8192 และ DDP gradient bucket

### งบจากตัวเลขที่วัดได้ (14,950 tok/s)

| token | wall-clock | On-Demand | Spot |
|---|---|---|---|
| 1B | 18.6 ชม. | $44 | $15–22 |
| 3B | 55.7 ชม. | $131 | $44–65 |
| 6B | 111.5 ชม. | $262 | $87–131 |
| 10B | 185.8 ชม. | $436 | $145–218 |

corpus ที่เตรียมไว้มี 10.9B token — เทรนได้ทั้งชุดในงบระดับ $150–220 บน spot

**สถานะ: preflight ผ่านครบทั้ง 4 ข้อ** พร้อมสำหรับ ablation 3×1B และ CPT หลัก
รอเพียงคำตอบจาก mentor เรื่องจำนวน token

### A6 (ต่อ) — การทดลองเพิ่มขณะที่ยังเช่าเครื่องอยู่

**A. หา micro-batch ที่เร็วที่สุด** (S0, token ต่อสเต็ปคงที่ 131,072)

| micro-batch × grad-accum | ผล | VRAM |
|---|---|---|
| 1 × 8 | ✅ **14,949 tok/s** | 54.93 GB |
| 2 × 4 | ❌ OOM | ไปถึง 67.12 GB แล้วตาย |
| 4 × 2 | ❌ OOM ทันที | — |

micro-batch 2 ทำได้ 14,654 tok/s **ช้ากว่า** micro-batch 1 ก่อนจะ OOM
แปลว่าต่อให้ VRAM พอก็ไม่ได้อะไร — **micro-batch 1 คือค่าที่ถูกต้อง เรื่องนี้ปิดแล้ว**

**B. เส้นทาง BPB eval บน CUDA และการตรวจสอบข้าม**

BPB คือตัวชี้วัดหลักของโปรเจกต์แต่ไม่เคยรันบน CUDA กับโมเดล 1.7B จริงมาก่อน
รันด้วย `--max-steps 1 --warmup 100` ซึ่งทำให้ `lr_at(0) = 0` น้ำหนักจึงไม่ขยับเลย
ค่าที่ได้คือ baseline จริงของโมเดลต้นทาง

| eval-docs | bpb_th | bpb_en | bpb_code | เวลา |
|---|---|---|---|---|
| 200 | 0.435155 | 0.688207 | 0.169849 | 2 นาที |
| **2000 (ชุดเต็ม)** | **0.454219** | 0.683817 | 0.157208 | 17 นาที |

**ตรวจสอบข้ามสำเร็จ:** ชุดเต็มให้ `bpb_th = 0.45421855879206163`
เทียบกับ baseline ที่ freeze ไว้ `0.454218` — **ตรงกันถึงทศนิยม 6 ตำแหน่ง**
ทั้งที่วัดคนละเครื่องคนละ dtype (แล็ปท็อป MPS fp16 vs A100 bf16 autocast + fp32 master)
ยืนยันว่า `train_cpt.eval_bpb` กับ `phase0/measure_baseline.py` วัดของสิ่งเดียวกันจริง

**กับดักวิธีวิทยาที่เจอระหว่างทาง:** BPB **ไม่ลู่เข้า** จนกว่าจะให้คะแนนชุดเต็ม
โมเดลเดียวกัน ไฟล์เดียวกัน ได้ 0.435155 ที่ 200 เอกสาร และ 0.454219 ที่ 2000 เอกสาร
ต่างกัน **4.2% ซึ่งเป็นสองเท่าของเกณฑ์ตัดสิน 2%**

ถ้าเอา BPB ระหว่างเทรน (200 เอกสาร) ไปเทียบกับ baseline (2000 เอกสาร)
**เราจะประกาศว่าโมเดลดีขึ้น 4% ทั้งที่ไม่ได้ทำอะไรเลย** ป้องกันโดยให้ทุกเรคคอร์ด eval
บันทึก `eval_docs` และ `decision_grade: false` ไว้ พร้อมระบุใน help ของ `--eval-docs`
ว่าตัวเลขระหว่างเทรนมีไว้ดูเส้นโค้ง ตัวเลขที่ใช้ตัดสินต้องมาจากชุดเต็มเท่านั้น

**เวลาที่ใช้:** ชุดเต็ม 5000 เอกสาร (ไทย 2000 · อังกฤษ 2000 · โค้ด 1000) = 17 นาที
บน A100 หนึ่งใบ วางแผน eval ระหว่างรันหลักได้จากตัวเลขนี้

**C. humaneval — ชิ้นสุดท้ายของ A3 ที่ยังไม่เคยรัน**

เราออกแบบให้ humaneval อยู่หลัง `--allow-code-exec` และรันบนเครื่องที่ทิ้งได้เท่านั้น
เพราะมันให้คะแนนด้วยการ **รันโค้ดที่โมเดลสร้างขึ้นจริง** เครื่องเช่านี้คือเครื่องนั้น

```
|  Tasks  |Version|  Filter   |n-shot|Metric|   |Value|   |Stderr|
|humaneval|      1|create_test|     0|pass@1|↑  |  0.7|±  |0.1051|
```

(20 ข้อ · `--limit` · ไม่ใช่คะแนนที่รายงานได้ · ยืนยันว่าเส้นทางทำงานเท่านั้น)

**สองอย่างที่ต้องแก้และไม่รู้มาก่อน:**
1. lm-eval 0.4.12 ต้องใช้ **ทั้ง** `HF_ALLOW_CODE_EVAL=1` **และ** `--confirm_run_unsafe_code`
   ลำพัง environment variable ยังฟ้อง `ValueError: _WARNING` เหมือนเดิม
   แก้ใน `run_eval.py` ให้ใส่ทั้งสองอย่างเมื่อระบุ `--allow-code-exec`
2. `accelerate` ไม่ได้ติดตั้งมากับ image และ lm-eval ต้องใช้ — ต้องลงเพิ่ม

### รายการติดตั้งสำหรับเครื่องเช่ารอบหน้า

```
/venv/main/bin/pip install "transformers>=4.51" numpy accelerate lm-eval
```

torch อยู่ใน `/venv/main` ซึ่ง SSH แบบไม่ interactive มองไม่เห็น
ต้องเรียก `/venv/main/bin/python` ตรง ๆ ไม่ใช่ `python3`

### สรุปสิ่งที่เครื่องเช่ารอบนี้ให้มา

| หัวข้อ | ผล |
|---|---|
| throughput S0 / D1 / D2 | 14,950 / 12,439 / 12,292 tok/s |
| routing overhead | 0.832× / 0.822× |
| peak VRAM | 54.9 GB / 72.9 GB |
| DDP | ✅ (หลังแก้ router บล็อกแรก) |
| resume | ✅ อยู่ในพื้น noise ของฮาร์ดแวร์ → **ใช้ spot ได้** |
| micro-batch | 1 เท่านั้น 2 ขึ้นไป OOM และไม่เร็วขึ้น |
| BPB บน CUDA | ✅ ตรงกับ baseline ถึงทศนิยม 6 ตำแหน่ง |
| humaneval | ✅ pass@1 0.7 (20 ข้อ) |
| **บั๊กที่จับได้** | **8 ตัว ทุกตัวเป็นชนิดที่เครื่องแล็ปท็อปหาไม่เจอ** |

**ข้อกำหนดเครื่องสำหรับรันหลัก:** VRAM ≥ 80 GB/ใบ · disk ≥ 300 GB · PyTorch 2.7+/cu128

---

## A7 — หมุน checkpoint และ resume อัตโนมัติ สำหรับรันบน spot instance

**วันที่:** 2026-08-23 · `scientific_evidence_allowed=false` · ทดสอบบน MPS + Qwen3-0.6B

### ปัญหาที่แก้

รันหลัก 10B token ≈ **76,000 สเต็ป** ถ้าเซฟทุก 1,000 สเต็ปและไม่ลบอะไรเลย
จะได้ **76 ไฟล์ × 20.6 GB = 1.5 TB** ซึ่งไม่มีดิสก์เช่าตัวไหนรับไหว
เดิม `train_cpt.py` เซฟแล้วไม่เคยลบ

และแม้พิสูจน์แล้วว่า resume แม่นยำ (A6) แต่ยังต้องมีคนสั่ง `--resume <path>` เอง
ซึ่งใช้กับ spot instance ที่ถูกเรียกคืนตอนตีสามไม่ได้

### สิ่งที่เพิ่ม

| ตัวเลือก | ทำอะไร |
|---|---|
| `--resume auto` | หา checkpoint ใหม่สุดใน `--out` เองแล้วต่อจากตรงนั้น |
| `--keep-last N` | เก็บ N ตัวล่าสุด (ค่าเริ่มต้น 2) ที่เหลือลบ |
| `--milestone-every N` | checkpoint ที่ step หารด้วย N ลงตัว ไม่ถูกลบตลอดไป |

`--resume auto` ทำให้**คำสั่งเดิมใช้ได้ทั้งตอนเริ่มครั้งแรกและตอนกลับมาครั้งที่สิบ**
ไม่ต้องมีคนแก้ path

### เขียนแบบ atomic — จุดที่สำคัญที่สุด

เดิม `torch.save` เขียนลงชื่อไฟล์จริงตรง ๆ ถ้า spot instance ถูกเรียกคืน**กลางการเขียน**
จะเหลือ `ckpt_N.pt` ที่ไม่สมบูรณ์ แล้ว `--resume auto` จะหยิบไฟล์นั้นขึ้นมา
**งานจะ crash วนซ้ำทุกครั้งที่บูตจนกว่าจะมีคนไปสังเกตเห็น**

แก้โดยเขียนลง `.pt.partial` ก่อนแล้ว `os.replace` ซึ่ง atomic บนไฟล์ระบบเดียวกัน
ชื่อไฟล์จริงจึงชี้ไปยังไฟล์ที่สมบูรณ์เสมอ และ `find_resume` ยังข้ามไฟล์ที่โหลดไม่ขึ้น
ไปใช้ตัวก่อนหน้าแทน — **เสียงานหนึ่งช่วงดีกว่าเสียทั้งรัน**

### ผลทดสอบ

**1. หมุน checkpoint** — 8 สเต็ป เซฟทุก 2 `--keep-last 2 --milestone-every 4`

```
เหลือ: ckpt_4.pt (milestone) · ckpt_6.pt · ckpt_8.pt      ← ckpt_2 ถูกลบ
[saved] ckpt_6.pt  (ลบเก่า 3.6 GB)
```

**2. resume อัตโนมัติ**

```
[*] --resume auto -> ckpt_8.pt
[*] resumed at step 8  (LR horizon 8)
```

**3. checkpoint เสียจากการถูกฆ่ากลางเซฟ** — ตัด `ckpt_10.pt` ให้เหลือ 500 KB
และวางไฟล์ `ckpt_12.pt.partial` ทิ้งไว้

```
[!] ckpt_10.pt ใช้ไม่ได้ (RuntimeError) — ข้ามไปตัวก่อนหน้า
[*] --resume auto -> ckpt_8.pt
```

ไฟล์ `.partial` ถูกลบทิ้งในรอบ rotation ถัดไป งานเดินต่อโดยไม่ต้องมีคนแตะ

### ค่าที่แนะนำสำหรับรันหลัก

```
--save-every 1000 --keep-last 2 --milestone-every 10000 --resume auto
```

76,000 สเต็ป → milestone 7 ตัว + ตัวล่าสุด 2 ตัว = **9 × 20.6 GB ≈ 186 GB**
พอดีกับดิสก์ 300 GB ที่กำหนดไว้ใน A6

**ยังไม่ได้ทดสอบ:** การถูกเรียกคืนจริงบน spot instance — จำลองด้วยการทำไฟล์เสียได้
แต่การ kill กลางคันจริงต้องทดสอบตอนเช่ารอบหน้า
