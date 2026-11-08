# Pending trained-Qwen A/B comparison

**2026-11-08 update:** the [new supplied-result review](../../reports/precise-qwen-supplied-results-20261108/README.md)
joins both complete prediction sets to the exact new 4,200 testing pairs and
reproduces the summaries. Prediction files, updated code, training JSON and
adapter/trainer metadata are now available; do not request them again. The
source-verified outcome report keeps actual-run provenance incomplete rather
than manufacturing the missing binary training configuration, resolved base
revisions/environment, raw replies or selection/evaluation-digest evidence.
The strict fully reviewed comparison command below retains those requirements.
The updated supplied parser emits `INVALID` in addition to the older failure
tokens; the importer now preserves it as an explicit invalid-response failure.

**2026-11-07 update:** the user replaced the active benchmark with the new
37,194 training / 4,200 testing pairs and explicitly authorized fresh DINOv2
runs under the [revised proposal](precise_revised_cohort_review_20261107.md).
The offline importer now supports this numeric-group-holdout scope as well as
the original shared cohort. It binds the new JSON and DINO artifacts to the
new bundle; numeric-group/image separation is revalidated before comparison.
Historical Qwen score outputs on the old pair list cannot fill the new gap.
Matching new-cohort fine-tuned outputs and actual run provenance have not been
supplied. The November 1 command below remains for its historical cohort.
For the new cohort, use the dated compatibility verification interpreter and
substitute these already available inputs:

| Argument | New cohort |
| --- | --- |
| `--bundle` | `local/precise/manifests/numeric-group-holdout-20261107/bundle.json` |
| `--config` | `code/precise_matching/configs/dinov2_numeric_group_holdout.json` |
| `--training-json` | `local/precise/collaborator/konain-followup-20261107/slap_subject_disjoint_train_ab.json` |
| `--dino-run-root` | `local/precise/runs/numeric-group-holdout-20261107` |

The filename's subject-disjoint wording does not verify biological identities.
For this scope the report states numeric-prefix group holdout and unconfirmed
biological/capture semantics. There are zero numeric-label disagreements, so
exclusion sensitivity equals the complete testing population. Other evidence,
review, immutability and failure-accounting requirements below remain in force.
No real Qwen outcome, run evidence or review is manufactured by this update.

2026-11-01. DINOv2 experiments and the
[professor report](../../reports/precise-dinov2-shared-20261101/README.md) are
complete and published. The user confirmed that Konain's matching fine-tuned
A/B outputs and actual run/checkpoint provenance are not ready. The full goal's
trained-Qwen comparison therefore remains incomplete. No model experiment,
dataset audit, installation or prior approval is repeated by this follow-up.

`code/compare_precise_qwen_ab.py` prepares the remaining offline comparison path.
It has synthetic validation only; **no real trained-Qwen result or review record
has been created**. The earlier historical 0-100 export remains separate.

## Supported evidence and input format

The supplied `slap_actual_qwen3.py` writes each original class output as:

```text
image_path_a image_path_b parsed_outcome
```

It produces `A`, `B`, `UNKNOWN` or `ERROR`, plus lowercase `a`/`b` in its known
longer-answer parser branch. The importer explicitly maps `A/a` to accept and
`B/b` to reject, independent of ground truth. It counts the original tokens and
preserves source bytes. This mapping is documented; it does not repair raw
responses or imply those longer answers were format-compliant. Raw-response and
parser evidence still need review. Numeric historical scores and guessed words
are rejected by this A/B route.

Both original class files must cover all supplied testing pairs exactly once.
Paths map by basename, with Windows/POSIX separators and reversed endpoint order
handled; duplicate, missing, extra and wrong-class pairs fail. No intersection
join, prediction inferred from a class filename, thumb exclusion, identity
relabeling or score imputation is allowed. `UNKNOWN` and `ERROR` remain explicit
failed attempts, not absent rows.

The actual Qwen run record must contain:

| Field | Required actual evidence |
| --- | --- |
| `model` | `Qwen/Qwen3-VL-8B-Instruct`, the supplied setup |
| `base_revision` | Actual resolved 40-character lowercase Git revision |
| `checkpoint_manifest_sha256` | SHA-256 identifying the actual adapter/checkpoint artifact manifest |
| `training_json_sha256` | Exact supplied training JSON digest, also checked against the DINO bundle |
| `training_pairs_consumed` | Exact consumed count, matching the complete supplied training membership |
| `training_membership_evidence` | Reference to the actual consumed-record manifest/log establishing membership, not just a count or code default |
| `prompt_sha256` | Canonical digest of the actual `system_prompt` and `user_prompt` object, matching the supplied JSON |
| `environment` | Actual nonempty package/runtime mapping |
| `checkpoint_selection_used_test_results` | `false`, supported by actual selection evidence |
| `checkpoint_selection_evidence` | Actual checkpoint-selection rule/history |
| `evaluation_preprocessing` | Actual evaluation image/processor procedure |
| `generation_configuration_evidence` | Actual generation settings/configuration reference |
| `raw_responses_evidence` | Actual saved replies/logs and parsing provenance |

Use `precise_matching.fixed_pairs.canonical_digest` for the prompt object; it
hashes sorted-key JSON with SHA-256. Run settings are not inferred from script
defaults. A count or asserted boolean does not establish consumption/selection.
The evidence references above must be inspected in the actual supplied artifacts.
The tool verifies consistency of reviewed claims, not facts absent from them.

The comparison review JSON binds exact digests under `input_sha256` for
`bundle`, `config`, `training_json`, `genuine_output`, `impostor_output` and
`qwen_run_record`. It includes `comparison_provenance_reviewed`,
`non_test_selection_reviewed` and a nonempty `review_record` pointing to the
actual evidence review. This records source verification for newly supplied
results; it is not a new blanket DINO/model-plan approval or execution request.
No real record is generated before those sources exist.

## Evaluation and provenance contract

The tool reuses each saved DINO arm's full private test/calibration artifacts.
It checks completed outcomes, hashes, frozen selection replay, original one-time
test claims, exact cohort/configuration binding and unchanged training-selected
cutoffs. It loads no model, images or checkpoints and performs no threshold search.

Qwen supplies a single native binary operating point. Report pooled,
left/right/thumb and two-case exclusion-sensitivity TAR/FAR, balanced accuracy,
coverage, statuses, failures and exact denominators. All-attempt primary rates
reject failed attempts; valid-only rates remain a separately labeled population.
Paired DINO-to-Qwen decisions additionally show both-valid and failed coverage
and both-valid supplemental differences. Positive genuine Qwen-minus-DINO
acceptance increases TAR; positive impostor acceptance increases FAR.

The internal 1/0 Qwen indicator is used only for paired accept/reject counts. It
is not a confidence score; no binary Qwen ROC/AUC/EER is reported. Dependence
counts and limitations remain explicit: test images/groups occur in training,
canonical identity semantics are unconfirmed, labels remain supplied, and no
independent-trial interval, statistical significance or population low-FAR claim
is made. The report cannot identify a language-module defect from model differences.

Only aggregate fixed-name fields, counts and model/checkpoint digests enter the
new public report. Run evidence strings, raw outcomes, identifiers and paths stay
in private sources/receipts. Inputs are rehashed after processing, and existing
outputs are never overwritten. This creates a **new** comparison report; the
completed DINO report and original results remain unchanged.

## Future offline command, after actual evidence review

The Qwen files and review below are placeholders, currently unavailable. Do not
run the comparison or invent their contents to fill the gap.

```bash
.venv/bin/python code/compare_precise_qwen_ab.py \
  --bundle local/precise/manifests/shared-pairs-20261101/bundle.json \
  --config code/precise_matching/configs/dinov2_shared_pairs.json \
  --training-json local/precise/collaborator/konain-followup-20261101/slap_verification_binary_ab_13_14_15_balanced.json \
  --genuine-output local/precise/collaborator/FUTURE/genuine_ab.txt \
  --impostor-output local/precise/collaborator/FUTURE/impostor_ab.txt \
  --qwen-run-record local/precise/collaborator/FUTURE/actual_run_record.json \
  --review local/precise/metadata/FUTURE/actual_comparison_review.json \
  --dino-run-root local/precise/runs/shared-pairs-20261101 \
  --private-receipt local/precise/metadata/FUTURE/comparison_receipt.json \
  --output reports/precise-qwen-ab-FUTURE/comparison.json
```

Seven synthetic contracts cover case/truth independence, exact common coverage,
failure denominators, arm-specific frozen cutoffs, binary-only reporting,
incomplete/test-selected run metadata, digest-bound review, immutable sources,
privacy, overwrite refusal and mutation detection. Regular CI test discovery
runs them with no model dependencies or real biometric assets.
