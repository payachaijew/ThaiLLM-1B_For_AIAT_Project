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

## Required future entries

- `BASE-SCREEN-*`: tokenizer, frozen BPB, license and port-complexity results
- `DATA-AUDIT-*`: provenance, licenses, filtering, deduplication and leakage
- `LOCAL-SMOKE-*`: overfit/resume/gradient evidence with scientific flag false
- `A100-PREFLIGHT-*`: 200-step throughput, memory and stability evidence
- `PILOT-*`: 50M–100M token S0/D1/D2 results
- `FULL1B-*`: only after pilot promotion gates pass

Use [`run_record.template.json`](run_record.template.json) for machine-readable
records and link each artifact from the corresponding entry here.

