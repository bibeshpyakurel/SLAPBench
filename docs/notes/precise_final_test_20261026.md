# DINOv2 final-test implementation — 2026-10-26

The scheduled 3 a.m. Chicago task found CUDA unavailable and no approved real
split or execution receipts. It therefore implemented and synthetically tested
the missing final-test path in the isolated evaluation worktree. This extends
the [guarded trainer](precise_training_implementation_20261025.md); it is not a
real Precise training or performance result.

## Frozen selection before test pixels

`code/precise_matching/testing.py` and `code/test_precise_matcher.py` accept only
the latest successful, completed training attempt. They check:

- Identical reviewed image/comparison/configuration bytes to training, valid
  subject/capture/exposure metadata, and a separate final-test review receipt.
- The pinned base model, original configuration/weight digests, current recorded
  training source bytes, Python/package versions and dependency-lock digest.
- Completed/frozen selection flags, exact selected/final checkpoint digests and
  sidecars, matching checkpoint/run provenance and complete epoch/scheduler/
  sampler counters. Partial or later failed/incomplete attempts are refused.
- The recorded best epoch under the trainer's development-only selection rule.
  Selected development score rows are replayed through the existing development
  evaluator; stored metrics and the strict-greater-than cutoff must agree.
- A test-only path mapping and paths resolved within the authorized source root.
  Training/development image pixels never enter this test loader.

These checks establish consistency of the reviewed local artifacts, not an
independent attestation that the review assertions or acquisition facts are true.
Checkpoints must come from this project's own reviewed local run; weights-only
loading and digest checks are not permission to consume untrusted checkpoints.
Current code must be committed/reviewed before the real CLI will load a model.

## Private review inputs

Use the existing [image/comparison schemas](precise_matching_tooling_20261017.md)
with exactly the same images, pairs and configuration files used by training.
The new `image_id,path` CSV must cover exactly the final-test comparison endpoints,
once each, with no training/development or extra rows. No image is opened during
metadata preflight.

The separately supplied review JSON requires `protocol_reviewed`,
`split_reviewed`, `capture_exposure_reviewed`, `execution_authorized`,
`selection_reviewed` and `final_test_authorized` to be true, a nonempty
`review_record` referencing actual review, and `input_sha256` containing exactly:

| Key | Exact file bytes bound by SHA-256 |
|---|---|
| `images`, `pairs`, `config` | The original training protocol/configuration inputs |
| `paths` | The new test-only path mapping |
| `selection` | The completed training attempt's `result.json` |
| `run_record` | Its training run's `run_record.json` |
| `checkpoint` | The selected epoch checkpoint |
| `last_checkpoint` | The final completed epoch checkpoint, possibly the same file |
| `development` | The selected epoch's saved development metrics/scores JSON |

The tool does not issue these assertions for real data. They require the actual
protocol/results/split and capture/exposure review described in the central plan.
Original private training inputs, development evidence and checkpoint files are
not rewritten. Input/artifact/source digests are checked again during execution.

## Private outputs and failure policy

After verification and before opening any test pixels, the tool exclusively
creates `final_test_adapted_claim.json` in the training run. A claimed adapted
test cannot be silently repeated with another output path. An interrupted or
failed claim remains visible and requires a separately reviewed recovery scope;
there is no automatic retry, reset, resume or claim-deletion command.

A new separate directory under ignored `local/precise/runs/` holds the frozen
selection, flushed per-image/comparison `events.jsonl`, complete raw scores,
aggregate report and execution/source/environment record. Fatal failures retain
the claim, partial events and a failure record; they do not produce a successful
report. Existing outputs are never overwritten.

**Export update — 2026-10-31:** successful scoring additionally writes private
`scores.csv` with the verified saved development outcomes followed by test
outcomes. It uses the existing `pair_id,status,score` schema and is directly
accepted by the [common-list comparison tool](precise_common_list_comparison_20261031.md).
Original JSON artifacts are retained; no development inference is rerun.

Each required test image is encoded once. Missing, decode/decoded-integrity and
invalid-embedding failures remain explicit comparison outcomes without imputed
scores. Original byte drift and fatal runtime/resource failures stop the attempt.
If either endpoint fails, the comparison remains in all-attempt denominators.
Reports give valid-score discrimination, coverage, failure counts and the
failure-as-reject view. Failed impostors do not prove biometric rejection.

The saved development cutoff is applied unchanged to all test comparisons and
left/right summaries. Test ROC/DET, AUC and EER are descriptive; none choose a
checkpoint or operating cutoff. No confidence interval or population low-FAR
guarantee is added. This entry point evaluates the **adapted arm**; a separately
reviewed frozen-control execution and common-list SDK scores remain distinct
work, not assumed to exist.

The [frozen-control runner](precise_frozen_control_20261031.md) is now implemented
with synthetic validation and separate development/test review gates; its real
execution and the SDK scores remain pending.

## Future reviewed execution

The following paths are placeholders, not generated manifests or approvals:

```bash
local/precise/environments/matcher-candidate-20261025/bin/python code/test_precise_matcher.py \
  --images local/precise/metadata/approved_images.csv \
  --pairs local/precise/manifests/approved_comparisons.csv \
  --paths local/precise/metadata/approved_test_paths.csv \
  --config code/precise_matching/configs/dinov2_first_milestone.json \
  --review local/precise/metadata/approved_final_test_review.json \
  --source-root datasets/Precise \
  --weights local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --training-run local/precise/runs/approved-first-run \
  --selection local/precise/runs/approved-first-run/attempt-0001/result.json \
  --output local/precise/runs/approved-final-test
```

The actual selected result can belong to a later resumed attempt. Optional
`--report reports/<new-path>.json` exports aggregates only; inspect it before
publication. The CLI never downloads a checkpoint, falls back from CUDA to CPU,
changes precision, trains weights or creates a new subject split.

Synthetic tests cover completed selection/history, checkpoint corruption,
development replay, changed inputs, private path/population boundaries,
missing/decode/encoder failures, fatal-attempt preservation, one-time claiming
and a changed test score distribution with an unchanged cutoff. The full CLI
test uses actual synthetic-image decoding and a tiny fake encoder, not Precise
images or downloaded model execution. Existing CPU CI's
`test_precise_training*.py` pattern includes this suite.
