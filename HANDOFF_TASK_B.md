# HANDOFF — Task B: Secret/PII Scan and Code Rebalancing

Stopped at user request on 2026-08-21. No job is running. No files were deleted during this handoff, and no git commit or push was made.

## 1. Per-track status

### Track 1 — Secret detector: DONE

Outputs on disk:

- `phase0/secret_scan.py`
- `phase0/test_scan_and_rebalance.py`

The self-test and unit tests passed after the last detector edit. The detector implements private-key document dropping, redaction of the other requested secret types, masked audit contexts, and placeholder rejection.

### Track 2 — English/international PII detector: DONE

Outputs on disk:

- `phase0/pii_filter_en.py`
- `phase0/test_scan_and_rebalance.py`

The self-test and unit tests passed. The detector includes email, context-guarded international phone, context-guarded US SSN, Luhn-plus-context credit card, and context-guarded public IP detection.

### Track 3 — Full replay scan application: DONE for the release scan

Runner output:

- `phase0/apply_scan.py`
- `data/scan_replay_v2_release_summary.json`
- `data/scan_replay_v2_release_state.sqlite3`
- `data/scan_replay_v2_release_state.sqlite3-shm`
- `data/scan_replay_v2_release_state.sqlite3-wal`

English release outputs:

- `data/clean_replay_v2/en/en-00000.jsonl.gz`
- `data/clean_replay_v2/en/en-00000.jsonl.gz.meta.json`
- `data/clean_replay_v2/en/en-00001.jsonl.gz`
- `data/clean_replay_v2/en/en-00001.jsonl.gz.meta.json`
- `data/clean_replay_v2/en/en-00002.jsonl.gz`
- `data/clean_replay_v2/en/en-00002.jsonl.gz.meta.json`
- `data/clean_replay_v2/en/en-00003.jsonl.gz`
- `data/clean_replay_v2/en/en-00003.jsonl.gz.meta.json`
- `data/clean_replay_v2/en/en-00004.jsonl.gz`
- `data/clean_replay_v2/en/en-00004.jsonl.gz.meta.json`

Math release outputs:

- `data/clean_replay_v2/math/math-00000.jsonl.gz`
- `data/clean_replay_v2/math/math-00000.jsonl.gz.meta.json`
- `data/clean_replay_v2/math/math-00001.jsonl.gz`
- `data/clean_replay_v2/math/math-00001.jsonl.gz.meta.json`
- `data/clean_replay_v2/math/math-00002.jsonl.gz`
- `data/clean_replay_v2/math/math-00002.jsonl.gz.meta.json`
- `data/clean_replay_v2/math/math-00003.jsonl.gz`
- `data/clean_replay_v2/math/math-00003.jsonl.gz.meta.json`

Code release staging outputs:

- `data/scanned_replay_stage_release/code/code-00000.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00000.jsonl.gz.meta.json`
- `data/scanned_replay_stage_release/code/code-00001.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00001.jsonl.gz.meta.json`
- `data/scanned_replay_stage_release/code/code-00002.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00002.jsonl.gz.meta.json`
- `data/scanned_replay_stage_release/code/code-00003.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00003.jsonl.gz.meta.json`
- `data/scanned_replay_stage_release/code/code-00004.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00004.jsonl.gz.meta.json`
- `data/scanned_replay_stage_release/code/code-00005.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00005.jsonl.gz.meta.json`
- `data/scanned_replay_stage_release/code/code-00006.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00006.jsonl.gz.meta.json`
- `data/scanned_replay_stage_release/code/code-00007.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00007.jsonl.gz.meta.json`
- `data/scanned_replay_stage_release/code/code-00008.jsonl.gz`
- `data/scanned_replay_stage_release/code/code-00008.jsonl.gz.meta.json`

Release masked-sample outputs currently on disk:

- `data/scan_samples_release/pii_credit_card.jsonl`
- `data/scan_samples_release/pii_email.jsonl`
- `data/scan_samples_release/pii_phone.jsonl`
- `data/scan_samples_release/pii_public_ip.jsonl`
- `data/scan_samples_release/pii_us_ssn.jsonl`
- `data/scan_samples_release/secret_aws_access_key.jsonl`
- `data/scan_samples_release/secret_aws_secret.jsonl`
- `data/scan_samples_release/secret_connection_string_password.jsonl`
- `data/scan_samples_release/secret_env_secret.jsonl`
- `data/scan_samples_release/secret_google_api_key.jsonl`
- `data/scan_samples_release/secret_jwt.jsonl`
- `data/scan_samples_release/secret_private_key.jsonl`
- `data/scan_samples_release/secret_slack_token.jsonl`
- `data/scan_samples_release/secret_stripe_secret.jsonl`

### Track 4 — Code rebalancing: PARTIAL for the release run

Implementation output:

- `phase0/rebalance_code.py`

Release-run partial outputs on disk:

- `data/rebalance_code_candidates_release/base-code-00000.jsonl.gz`
- `data/rebalance_code_candidates_release/base-code-00001.jsonl.gz`
- `data/rebalance_code_candidates_release/base-code-00002.jsonl.gz`
- `data/rebalance_code_candidates_release/base-code-00003.jsonl.gz`
- `data/rebalance_code_candidates_release/base-code-00004.jsonl.gz`
- `data/rebalance_code_candidates_release/base-code-00005.jsonl.gz`
- `data/rebalance_code_candidates_release/base-code-00006.jsonl.gz`
- `data/rebalance_code_candidates_release/base-code-00007.jsonl.gz`
- `data/rebalance_code_candidates_release/base-code-00008.jsonl.gz`
- `data/rebalance_code_candidates_release/new-00314.jsonl.gz`
- `data/rebalance_code_candidates_release/new-00565.jsonl.gz`
- `data/scan_replay_v2_release_state.sqlite3*` contains the resume state.

At stop time, SQLite contained 11 completed candidates: nine scanned Task-A shards plus new raw shards `00314` and `00565`. `selected_completed` contained zero rows. Therefore there is no release code output under `data/clean_replay_v2/code/`, and no release `data/rebalance_code_summary.json`.

A complete but superseded pre-release rebalance exists at:

- `data/rejected_task_b_intermediate/template_syntax_fp/clean_replay_v2/code/`
- `data/rejected_task_b_intermediate/template_syntax_fp/rebalance_code_summary.json`
- `data/rejected_task_b_intermediate/template_syntax_fp/rebalance_code_candidates_accepted/`
- `data/rejected_task_b_intermediate/template_syntax_fp/scan_replay_v2_accepted_summary.json`

It reached 1,000,075,432 Qwen tokens but was superseded after visual audit found additional placeholder syntax that should not be redacted.

### Track 5 — Manifest, source record, and final validation: PARTIAL

Outputs on disk:

- `phase0/build_replay_v2_manifest.py`
- `sources/source_registry.csv` (contains the GitHub Octoverse 2025 reference added for the code-mix rationale)
- `.gitignore` (contains Task-B generated-data paths)

Not produced for the release run:

- `data/clean_replay_v2_manifest.json` — not present.
- `validation/replay_v2_verification.json` — not present.
- `phase0/verify_replay_v2.py` — not written.
- A final Task-B entry in `validation/VALIDATION_OUTPUT_LOG.md` — not written.
- Required final sample directory `data/scan_samples/` — not populated from the release samples; the release samples remain in `data/scan_samples_release/`.

## 2. Why `rebalance_code.py` restarted

It did not restart because of a training/data failure, an infinite loop, or failure to reach the 1.0B-token target. I relaunched it deliberately after visual audits found false positives in the detectors. Earlier complete runs did reach the 1.0B-token target.

The observed `data/clean_replay_v2/code/` directory with about 598 MB across six files was an in-progress/previous selection output. It was not permanently deleted: the enclosing `data/clean_replay_v2` directory was deliberately moved into a named directory under `data/rejected_task_b_intermediate/` before a fresh run from the immutable Task-A data. This was done so that old redactions could not leak into the next result and so rejected artifacts remained recoverable.

`rebalance_code.py` has been launched five times in total:

1. Completed; rejected and archived under `data/rejected_task_b_intermediate/openai_css_fp/` because the first `sk-` rule matched CSS SpinKit names.
2. Completed; rejected and archived under `data/rejected_task_b_intermediate/ip_ssn_fp/` because visual audit found version/section numbers treated as IPs and `ISSN` context treated as SSN.
3. Completed; rejected and archived under `data/rejected_task_b_intermediate/placeholder_fp/` because visual audit found `[PASSWORD]` and path-valued env variables being redacted.
4. Completed and reached 1,000,075,432 tokens; archived under `data/rejected_task_b_intermediate/template_syntax_fp/` because visual audit found escaped angle, nested-bracket, and printf-style placeholders.
5. Release run; deliberately stopped on the user's instruction. It had completed candidate import for nine base shards and two new shards, had not begun selection, and therefore had not yet attempted to reach 1.0B tokens.

The command was stopped with Ctrl-C. The traceback ends in `pii_filter_en.detect()` while beginning raw shard `00816`; it is a `KeyboardInterrupt`, not a detector or data failure.

## 3. Numbers already measured

### Release full-scan token/document totals

These are measured from `data/scan_replay_v2_release_summary.json`:

| Language | Input docs | Input Qwen tokens | Output docs | Output Qwen tokens |
|---|---:|---:|---:|---:|
| English | 3,508,510 | 3,516,598,044 | 3,507,052 | 3,513,258,103 |
| Code staging | 700,129 | 1,104,617,663 | 699,484 | 1,101,807,553 |
| Math | 412,820 | 592,984,656 | 412,734 | 592,472,266 |

### Release full-scan secret hits

Measured hits on the original English, code, and math replay inputs. These do not include unfinished new-code candidate scanning.

| Secret type | English | Code | Math | Total |
|---|---:|---:|---:|---:|
| AWS access key | 12 | 16 | 1 | 29 |
| AWS secret | 0 | 1 | 0 | 1 |
| Connection-string password | 1 | 182 | 0 | 183 |
| Env secret | 0 | 137 | 0 | 137 |
| Google API key | 0 | 319 | 0 | 319 |
| JWT | 73 | 62 | 0 | 135 |
| Private key document | 7 | 65 | 3 | 75 |
| Slack token | 0 | 4 | 0 | 4 |
| Stripe secret | 0 | 1 | 0 | 1 |
| GitHub token | 0 | 0 | 0 | 0 |
| OpenAI API key | 0 | 0 | 0 | 0 |
| Anthropic API key | 0 | 0 | 0 | 0 |

### Release full-scan PII hits

Measured hits on the original English, code, and math replay inputs. These do not include unfinished new-code candidate scanning.

| PII type | English | Code | Math | Total |
|---|---:|---:|---:|---:|
| Credit card | 11 | 183 | 17 | 211 |
| Email | 71,943 | 103,977 | 16,944 | 192,864 |
| Phone | 84,097 | 5,558 | 2,260 | 91,915 |
| Public IP | 1,354 | 3,496 | 153 | 5,003 |
| US SSN | 34 | 101 | 8 | 143 |

### Code distribution before Task-B rebalance

This is measured from the last complete pre-release run after its base code scan. It is not an estimate, but it belongs to the superseded detector revision rather than the unfinished release run. The release-run before distribution has not been aggregated into a final rebalance summary.

| Group | Tokens | Percent |
|---|---:|---:|
| Python | 76,004,793 | 6.898187% |
| JavaScript + TypeScript | 154,996,368 | 14.067455% |
| Java | 162,590,092 | 14.756661% |
| C + C++ | 204,796,904 | 18.587347% |
| C# | 49,431,346 | 4.486384% |
| Long tail | 170,233,444 | 15.450370% |
| Markdown | 47,265,551 | 4.289817% |
| HTML | 196,588,421 | 17.842346% |
| CSS + config | 39,901,248 | 3.621433% |
| Total | 1,101,808,167 | 100% |

### Code distribution after Task-B rebalance

This is measured from the last complete pre-release run in `data/rejected_task_b_intermediate/template_syntax_fp/rebalance_code_summary.json`. It is not the release result because the detector was subsequently tightened.

| Group | Tokens | Percent |
|---|---:|---:|
| Python | 145,004,856 | 14.499392% |
| JavaScript + TypeScript | 145,049,261 | 14.503832% |
| Java | 120,000,002 | 11.999095% |
| C + C++ | 120,000,647 | 11.999160% |
| C# | 40,003,055 | 4.000004% |
| Long tail | 260,007,278 | 25.998767% |
| Markdown | 50,001,133 | 4.999736% |
| HTML | 50,008,901 | 5.000513% |
| CSS + config | 70,000,299 | 6.999502% |
| Total | 1,000,075,432 | 100% |

That completed pre-release result had 690,716 documents and 3,912,817,614 UTF-8 bytes. The release after-rebalance distribution is not measured because the run was stopped before selection.

### Last complete pre-release scan of nine additional raw code shards

These are measured from the superseded pre-release run, not the unfinished release detector revision:

- Secret hits: AWS access 46, AWS secret 7, connection-string password 221, env secret 156, Google API key 299, JWT 151, OpenAI-like `sk-` 1, private key 68, Slack 8, Stripe 3.
- PII hits: credit card 265, email 106,151, phone 5,629, public IP 2,548, US SSN 54.

The corresponding release-run counts over all nine additional shards are not measured.

## 4. Decisions made that were not explicitly in the task

1. I used named rejected-artifact directories instead of deleting failed/intermediate outputs. Reason: preserve auditability and obey the requirement not to overwrite Task-A data.
2. I reran the whole pipeline after each detector correction instead of patching already-redacted outputs. Reason: redaction is lossy; the original value cannot be reconstructed safely from a redacted record.
3. I set the two 15% target groups to 14.5% internally. Reason: whole-document selection can overshoot a quota, so a 0.5-point margin keeps individual languages below the hard 15% cap while remaining inside the allowed ±3 points.
4. I used the official GitHub Octoverse 2025 page as the external real-world-language rationale and added it to `sources/source_registry.csv`. Reason: Task B prohibited HumanEval-based balancing and required an external source.
5. I added `phase0/test_scan_and_rebalance.py` and `phase0/build_replay_v2_manifest.py`, although those exact filenames were not required. Reason: regression protection and reproducible manifest assembly.
6. I treated syntactically valid signed CDN JWTs as JWTs even when embedded in public image URLs. Reason: decoding showed a valid JWT header, payload, and signature; this was not a regex false positive.
7. I tightened secret detection beyond the first self-test after manual sample review: CSS `sk-*`, bracketed placeholders, path-valued `_FILE/_PATH/_DIR` env variables, escaped angle placeholders, printf placeholders, and common demo/test values. Reason: the task explicitly required placeholder rejection and visual audit exposed real false positives.
8. I retained every available masked example when corpus-wide occurrences were below 20 rather than fabricating examples. Reason: fabrication would violate the audit purpose.

## 5. Exact reproduction commands

Run from `/Users/prince/Documents/Research for AIAT/research/thai-llm-1b-attnres`.

### Finished detector tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase0/secret_scan.py
PYTHONDONTWRITEBYTECODE=1 python3 phase0/pii_filter_en.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest phase0/test_scan_and_rebalance.py
```

### Finished English/math release scan

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase0/apply_scan.py --languages en math --offline --output-root data/clean_replay_v2 --state-db data/scan_replay_v2_release_state.sqlite3 --summary data/scan_replay_v2_release_summary.json --samples data/scan_samples_release
```

### Finished code release scan into staging

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase0/apply_scan.py --languages code --offline --output-root data/scanned_replay_stage_release --state-db data/scan_replay_v2_release_state.sqlite3 --summary data/scan_replay_v2_release_summary.json --samples data/scan_samples_release
```

### Manifest builder implemented but not run successfully for the release

It must only be run after release rebalancing writes `data/rebalance_code_summary.json`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase0/build_replay_v2_manifest.py --scan-summary data/scan_replay_v2_release_summary.json --rebalance-summary data/rebalance_code_summary.json --output data/clean_replay_v2_manifest.json
```

### Command running when stopped

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase0/rebalance_code.py --offline --scanned-code-root data/scanned_replay_stage_release/code --candidate-root data/rebalance_code_candidates_release --output-root data/clean_replay_v2/code --state-db data/scan_replay_v2_release_state.sqlite3 --summary data/rebalance_code_summary.json --samples data/scan_samples_release --max-new-shards 24
```

It was interrupted with Ctrl-C after completing `new-00565.jsonl.gz` and while starting source shard `data/train-00816-of-00880.parquet`. The resume state is on disk; rerunning the exact same command should skip the 11 completed candidates.

### Command used for the last complete but superseded pre-release rebalance

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase0/rebalance_code.py --offline --scanned-code-root data/scanned_replay_stage_accepted/code --candidate-root data/rebalance_code_candidates_accepted --output-root data/clean_replay_v2/code --state-db data/scan_replay_v2_accepted_state.sqlite3 --summary data/rebalance_code_summary.json --samples data/scan_samples_accepted --max-new-shards 24
```

That command's outputs were moved intact under `data/rejected_task_b_intermediate/template_syntax_fp/` after the later placeholder issue was found.

## 6. Anything held only in memory

Nothing required to resume is held only in memory. The release scan summary, scanned shards, 11 completed candidate rows, candidate files, masked samples, and SQLite resume state are all on disk.

The interrupted in-memory batch for raw shard `00816` was not committed and is lost. This is expected and safe: on resume, `00816` will restart from the beginning because it has no `candidate_completed` row. No selected final-code documents existed in memory or on disk because selection had not started.

There was no unwritten release manifest, no unwritten validation result, and no unpublished numeric result beyond what is recorded above.

## 7. Incomplete work and next action

Incomplete:

1. Release code rebalance is incomplete. Only 11 candidates are complete; selection has not started.
2. No release `data/clean_replay_v2/code/*.jsonl.gz` exists.
3. No release `data/rebalance_code_summary.json` exists.
4. The release masked samples have not been copied/renamed from `data/scan_samples_release/` to the required `data/scan_samples/` path.
5. Final visual audit of all release candidate-derived sample types is incomplete. Base EN/code/math samples were generated, but the nine new raw shards were not all scanned.
6. `data/clean_replay_v2_manifest.json` is absent for the release.
7. `phase0/verify_replay_v2.py` and `validation/replay_v2_verification.json` do not exist.
8. The final Task-B entry has not been added to `validation/VALIDATION_OUTPUT_LOG.md`.
9. Final checks (`git diff --check`, JSON validation, duplicate source IDs, full output hash/doc-hash/heldout/dedup/token recount, detector idempotence, sample-count policy) have not been run on a release result.

Next action, if explicitly authorized later:

1. Rerun the exact stopped rebalance command. It should resume from raw shard `00816` and keep the 11 completed candidates.
2. Inspect all final masked samples. If no new false-positive class is found, freeze the detector; do not silently change it after this point.
3. Move/copy the release sample directory to the required `data/scan_samples/` path without deleting the release source until verification passes.
4. Generate `data/clean_replay_v2_manifest.json` with the manifest-builder command above.
5. Implement and run `phase0/verify_replay_v2.py`; verify hashes, row flags, source revisions, licenses, heldout leakage, exact duplicates, code caps, totals, and 100 exact tokenizer recounts per language.
6. Add the final measured Task-B entry before `## Required future entries` in `validation/VALIDATION_OUTPUT_LOG.md`.
7. Run short final repository checks only; do not commit or push unless separately instructed.
