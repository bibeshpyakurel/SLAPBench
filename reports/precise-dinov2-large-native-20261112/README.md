# Completed DINOv2-Large evaluation with original-resolution Precise images

2026-11-12. Share [the professor report](professor-report.pdf). Its first page
contains the actual before/after accuracy and confusion matrices; subsequent
pages cover original pixels, training time, per-type results, descriptive ROC,
cohort dependence, execution history and Qwen provenance limits. A
[Markdown version](professor-report.md), [all-method table](operating_points.csv),
[ROC figure](pooled_roc.svg), [aggregate](aggregate.json) and
[independent verification](verification.json) accompany it.

The approved Large/native adaptation, both training calibrations and both
one-time final tests are **complete**. All originals, checkpoints, earlier
studies and failed/interrupted attempt records remain preserved. No further
model run or tuning follows from completion.

## Actual accuracy and operating points

All methods below cover the same 4,200 supplied testing attempts: 2,100 genuine
and 2,100 impostor pairs, including 700 of each class per left/right/two-thumb
type. Adaptation uses all 37,194 supplied training pairs. There were zero failed
native or supplied Qwen testing attempts.

| Method | TP | TN | FP | FN | Accuracy | TAR | FAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen DINOv2-small / 448 | 730 | 2,080 | 20 | 1,370 | 66.90% | 34.76% | 0.95% |
| Adapted DINOv2-small / 448 | 727 | 2,095 | 5 | 1,373 | 67.19% | 34.62% | 0.24% |
| Frozen DINOv2-Large / native | 1,013 | 2,064 | 36 | 1,087 | **73.26%** | 48.24% | 1.71% |
| Adapted DINOv2-Large / native | 1,417 | 2,077 | 23 | 683 | **83.19%** | 67.48% | 1.10% |
| Qwen3-VL-8B base, supplied | 1,927 | 732 | 1,368 | 173 | 63.31% | 91.76% | 65.14% |
| Qwen3-VL-8B fine-tuned, supplied | 2,040 | 1,726 | 374 | 60 | 89.67% | 97.14% | 17.81% |

Positive means a genuine match. Standard accuracy is `(TP + TN) / 4200`:
`3077 / 4200 = 73.2619%` before Large adaptation and
`3494 / 4200 = 83.1905%` afterwards. Balanced accuracy coincides because the
classes are balanced and all outcomes are valid.

Large adaptation increases observed accuracy by 9.93 percentage points and
TAR by 19.24 points while reducing FAR by 0.62 points. Its descriptive valid-score
AUC increases from 0.8703 to 0.9670; interpolated EER decreases from 21.52% to
8.67%. These are finite-cohort measurements, not significance or population
confidence claims. The native operating policy still rejects 683 of 2,100
genuine attempts after adaptation.

Each DINO arm uses its own training-impostor cutoff at nominal 1% training FAR.
The native cutoffs are `0.8940030932426453` frozen and `0.9026179909706116` adapted,
accepting only scores strictly greater than the boundary. Each calibration has
185 false accepts among 18,597 training impostors. Both selections were sealed
before either final test; training calibration is resubstitution, not an
independent development set or a 1% test-FAR guarantee.

The adapted Large/native accuracy is 16.00 percentage points above the completed
small/448 result on these same attempts, but its test FAR is also higher
(1.10% versus 0.24%). Model architecture, resolution and numerical execution
change jointly; this does not isolate a resizing or model-size effect.

Qwen's 89.67% supplied fine-tuned accuracy accompanies a 17.81% FAR, versus
1.10% for adapted native DINO. Their native operating policies are **not matched
test FAR**. Qwen has binary A/B answers, not a continuous score ROC/AUC/EER.
Its exact-pair arithmetic and membership are source-verified; missing actual
configuration/base-revision/runtime/raw-response/selection and execution-time
adapter-digest evidence remain explicit. No duplicate Qwen inference or
fine-tuning was performed here, and no causal language/vision-module claim is made.

## Original pixels and measured training time

The pinned model is `facebook/dinov2-large`, revision
`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`: 304,368,640 parameters, with only
blocks 22/23 and final norm adapted (25,198,592 parameters). Every original
1600 x 1500 grayscale raster is preserved on a centered white 1610 x 1512
canvas: 5 pixels per horizontal side and 6 per vertical side. RGB replication
and normalization follow. There is no resizing, cropping or stretching; files
remain unchanged. Positional embeddings are interpolated to the native patch
grid. Earlier 448 processing occurred on load, not by replacing the originals.

The fixed schedule is three full epochs / 3,489 optimizer updates, with final
epoch selection decided in advance. A segmentation fault interrupted the
original third epoch. One specifically approved resume restored the verified
epoch-2 model/optimizer/scheduler/RNG state and replayed the full third epoch.
The resumed attempt took **8h 29m 23s**. Observed attempt intervals total at
least **29h 49m 30s**, including discarded partial work; full start-to-final wall
time was **60h 24m 40s**, including interruption downtime. Calibration/testing
are additional. These are pipeline times, not GPU kernel measurements.

Offline final-checkpoint verification finds all 401 frozen tensors unchanged
and all 38 selected trainable tensors changed. The retained trajectory contains
111,582 pair presentations; at least 130,654 were observed across both attempts.
Checkpoint SHA-256:
`c24099fa41f38a1008b91454597e054013812a885f8d2433ab1598c23ed34c15`.

## Cohort and execution limits

Training/testing share zero exact pairs, images, numeric-prefix groups, raw-file
hashes and decoded-image hashes. Training has 221 numeric groups / 5,721 images;
testing has 55 groups / 1,057 images. All 61 leading-zero alias groups are in
training. All supplied labels and thumbs remain intact; the new cohort has
zero numeric-prefix label disagreements, so exclusion sensitivity equals the
full result. This is **numeric-group holdout, not confirmed biological
subject-disjoint or capture-independent evaluation**. Images/groups recur
within phases, and prior research results for this dataset were already known.
No independent-pair confidence intervals or population low-FAR claim is made.

A later reboot temporarily removed CUDA availability. The failed frozen
preflight read no model/image and produced no scores or test claim. After the
user restored CUDA on kernel 7.0.0-43, they specifically approved a scoring bridge
for that kernel patch and proc-to-nvidia-smi driver discovery metadata change.
Training used kernel 7.0.0-42; the same GPU, driver 595.99.02, Python/packages/
interpreter bytes, model/processing functions and numerical settings were retained.
Original training records were never relabeled. This is not runtime-identity or
bitwise-equivalence evidence.

Training's protected execution source is
`06d12c249ec7f5db53124f673b2c6947c4b144b7`; scoring source is
`12a6df421ac2b12699668480ae741baa50475ac4`. The canonical private
`frozen-calibration` symlink resolves to the sole successful
`frozen-calibration-retry-20261111` directory. Resolved-path one-time claims
remain intact; no score/selection copy or additional test was created.

VeriFinger fused/summed curves retain all source zeros. Positive-component-sum
selection and unknown template-extraction status limit full-pool claims; no
operational SDK threshold is chosen from test scores. DINO is a visual control,
not itself an LLM result. Konain owns Qwen tuning. Extra ablations, new pairs,
segmentation/minutiae, manuscript changes and further experiments are outside
this completed contribution.

## Reproduce the saved-outcome report without model execution

Private images, checkpoints and supplied originals stay under the documented
local paths. Do not rerun any completed phase merely to reproduce the report.
The exporter refuses overwrite and replays source/input identities, the exact
reviewed recovery/runtime transitions, both calibrations and one-time test seals.

```bash
local/precise/environments/verification-compat-20261107/bin/python code/report_precise_native_large.py \
  --bundle local/precise/manifests/numeric-group-holdout-20261107/bundle.json \
  --config code/precise_matching/configs/dinov2_large_native.json \
  --review local/precise/metadata/native-large-execution-20261108/execution_review.json \
  --run-root local/precise/runs/large-native-20261108 \
  --checkpoint-verification local/precise/runs/large-native-20261108/checkpoint-verification.json \
  --sdk-intake local/precise/collaborator/konain-followup-20261107 \
  --cohort-audit reports/precise-revised-cohort-20261107/cohort_audit.json \
  --small-aggregate reports/precise-dinov2-numeric-holdout-20261107/aggregate.json \
  --qwen-intake local/precise/collaborator/konain-results-20261108 \
  --qwen-aggregate reports/precise-qwen-supplied-results-20261108/aggregate.json \
  --resume-review local/precise/metadata/native-large-resume-20261110/resume_review.json \
  --runtime-review local/precise/metadata/native-runtime-transition-20261111/runtime_review.json \
  --study-date 2026-11-12 \
  --output reports/precise-dinov2-large-native-NEWDATE/aggregate.json \
  --private-receipt local/precise/metadata/native-large-final-report-NEWDATE/reproduction.json
```

`code/verify_precise_fixed_metrics.py` independently reproduces both native
cutoffs, confusion counts and ROC/AUC/EER with NumPy/scikit-learn. The public
verification summary strips private source paths while retaining their digests.
`code/render_precise_native_report.py` renders the aggregate into the PDF,
Markdown, CSV and SVG. All seven actual final PDF pages were visually reviewed.
Artifact hashes are in [artifact_manifest.json](artifact_manifest.json).
