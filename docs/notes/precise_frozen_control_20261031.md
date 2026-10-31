# DINOv2 frozen-control implementation — 2026-10-31

`code/run_precise_frozen.py` and `code/precise_matching/baseline.py` implement
the frozen encoder arm with separate development and final-test phases. This
completes a missing implementation path toward roadmap task 8; it does not
execute a Precise experiment or satisfy the outstanding data/resource reviews.
Only synthetic models and images were used for validation.

## Fixed model and phase boundaries

The CLI loads the selected local DINOv2-small revision through the existing
configuration/weight-digest checks, without downloading anything. Every parameter
must be frozen. It uses evaluation mode, inference without gradients, FP32
weights/embeddings, the shared whole-slap letterbox preprocessing, normalized
CLS representation and continuous cosine score. It never creates an optimizer
or changes the pretrained weights. This uses the same inference precision and
preprocessing as the adapted final-test path; the training configuration's
BF16 field controls adaptation, not baseline inference.

Both phases use the same reviewed image/comparison metadata and matcher config
as the adapted arm. The config retains training fields for a common immutable
experiment input; baseline execution does not use epochs, batches or optimizer
settings. Device and seed remain explicit; unavailable CUDA never triggers an
automatic CPU fallback. A CPU configuration needs its own real review.

Development accepts a path mapping containing exactly the development comparison
endpoints. No training or test path is accepted, and no training/test pixels
enter the loader. It scores each needed image once, saves outcomes privately and
selects the strict `score > cutoff` boundary using only development impostors
at the configured FAR target. Any development failure stops selection for
investigation, preserving raw outcomes/events without a frozen selection.

Test accepts a different mapping containing exactly test endpoints and requires
a completed baseline development directory. It verifies the original
image/comparison/config bytes, pinned model provenance, record/score digests and
completed report, then replays saved development scores through the evaluator.
The cutoff must match the saved selection exactly. Current source bytes,
Python/package versions, dependency lock and source root must match development;
an incompatibility requires separately reviewed work, not an automatic override.
No development pixels are reloaded and no threshold is fitted on test.

## Review inputs

Use the [existing image/comparison CSV contracts](precise_matching_tooling_20261017.md)
and a separate `image_id,path` mapping for each phase. Paths resolve inside the
authorized source root. All usual subject/capture/hash/exposure checks must pass.

Each separately supplied review JSON needs:

- `phase` exactly `development` or `test` and a nonempty `review_record` pointing
  to actual evidence of review.
- `protocol_reviewed`, `split_reviewed`, `capture_exposure_reviewed`,
  `execution_authorized` and `frozen_control_reviewed` set to true by the actual
  reviewer.
- For test, `selection_reviewed` and `final_test_authorized` also true.
- `input_sha256` binding exactly the current `images`, `pairs`, `paths` and
  `config` file bytes. Test additionally binds these four development artifacts:

| Digest key | File in the completed baseline development directory |
|---|---|
| `baseline_record` | `run_record.json` |
| `baseline_selection` | `selection.json` |
| `baseline_development` | `development_scores.json` |
| `baseline_report` | `report.json` |

The code does not create real review assertions or establish acquisition facts.
Receipt and source digests are rechecked during execution. Test image paths are
not accepted in the development receipt, or vice versa. This is an internal
reanalysis of a historically inspected collection, not an untouched-cohort claim.

## Private artifacts and failed attempts

Use new directories under ignored `local/precise/runs/`. Development saves the
run/config/source/environment/input record, flushed per-image/comparison events,
raw development scores and a completed report. A successful selection additionally
binds the development score and run-record digests. Partial completion without
the matching successful report is insufficient for test.

Before test pixels, an exclusive `final_test_frozen_claim.json` is installed in
the baseline development directory. Subsequent invocations refuse that claim,
even with a different output path. A fatal failure retains it, flushed events
and `failure.json`; no successful report or automatic retry is supplied. Keep
failed attempts intact and obtain a separately reviewed recovery scope. Original
development artifacts are not changed by scoring; the claim is a new record.

Missing inputs, decode failures and invalid embeddings remain explicit test
outcomes with empty scores. Original byte drift and fatal resource/runtime
failures abort. Reports retain all attempted comparisons, coverage and the
valid-score view, with the same pooled development cutoff applied to both hands.
Failed impostors counted as rejects are not biometric rejection evidence.
Test ROC/DET/EER remain descriptive. No uncertainty interval, independent-trial
test or population low-FAR guarantee is added.

Development `scores.csv` contains development only. Successful final test
`scores.csv` contains the original saved development outcomes followed by test
outcomes, using the common `pair_id,status,score` contract. The adapted scorer
now exports that same combined schema as well as retaining its original JSON
outputs. These private files feed directly into the
[common-list comparison command](precise_common_list_comparison_20261031.md)
without hand conversion or recomputing development scores. They contain private
pair identifiers and must not be published. Optional `--report` exports aggregate
JSON under `reports/`, still requiring publication review.

## Commands for future reviewed inputs

These paths are placeholders; no actual manifests or approvals were generated.
Run development first, review its result, then supply a separate test receipt.

```bash
local/precise/environments/matcher-candidate-20261025/bin/python code/run_precise_frozen.py development \
  --images local/precise/metadata/approved_images.csv \
  --pairs local/precise/manifests/approved_comparisons.csv \
  --paths local/precise/metadata/approved_development_paths.csv \
  --config code/precise_matching/configs/dinov2_first_milestone.json \
  --review local/precise/metadata/approved_frozen_development_review.json \
  --source-root datasets/Precise \
  --weights local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --output local/precise/runs/approved-frozen-development

local/precise/environments/matcher-candidate-20261025/bin/python code/run_precise_frozen.py test \
  --images local/precise/metadata/approved_images.csv \
  --pairs local/precise/manifests/approved_comparisons.csv \
  --paths local/precise/metadata/approved_test_paths.csv \
  --config code/precise_matching/configs/dinov2_first_milestone.json \
  --review local/precise/metadata/approved_frozen_test_review.json \
  --source-root datasets/Precise \
  --weights local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --baseline-run local/precise/runs/approved-frozen-development \
  --output local/precise/runs/approved-frozen-test
```

CLI success returns 0, failure 1 with a generic message that does not echo private
paths or identifiers. Current matcher/entry-point code must be committed/reviewed
before model loading. No task, monitor, sleep inhibitor, system repair or new
environment is created by this implementation.

## Synthetic validation and remaining gates

Twelve new contracts cover frozen weights/no optimizer, development/test-only
pixels, separate receipt/digest/selection review, path populations/escapes,
changed artifacts or runtime, partial development, full failed-class denominators,
fatal-event/claim preservation, repeat refusal and unchanged thresholds under
changed test scores. The complete CLI test decodes artificial images with a tiny
fake encoder in both phases and passes the combined CSV through the common-list
evaluator. The adapted scorer's existing test now checks its combined CSV too.
Both regular CI and its CPU matcher job discover these contracts.

The current preflight still finds no real split/comparison/run artifacts, and
`nvidia-smi` cannot communicate with the driver. Acquisition/exposure evidence,
protocol/results/split review and administrator GPU repair remain necessary.
The full-resolution resource check, frozen/adapted real runs, common-list SDK
scores, agreed uncertainty and final research evidence remain unfinished.
