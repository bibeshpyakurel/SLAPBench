# Precise DINOv2: fresh numeric-group-holdout results

Completed 2026-11-07 after the user explicitly approved replacement with the
new supplied lists and fresh training, calibration and testing under the
[reviewed protocol](../../docs/notes/precise_revised_cohort_review_20261107.md).
All five real phases completed once: fresh adaptation, two training-only
calibrations and two final tests. The prior November 1 study and its private
checkpoint, sources and scores remain unchanged.

The [four-page professor report](professor-report.pdf),
[editable report](professor-report.md), [aggregate](aggregate.json),
[metric table](dino_metrics.csv) and [descriptive ROC figure](pooled_roc.svg)
are ready for review. Every final PDF page was rendered and visually inspected.
The report has not been sent to the professor.

## Main findings

Both arms scored all 4,200 supplied testing attempts, 2,100 genuine and 2,100
impostor, with zero DINO failures. Each slap type has 700 attempts per class.
Both cutoffs were sealed from training impostors before either final test;
acceptance means strictly greater than the corresponding cutoff.

| Arm | Training cutoff | Genuine accepts | False accepts | Test TAR | Test FAR | Balanced accuracy | Descriptive AUC | Interpolated EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen DINOv2 | 0.8948034048080444 | 730 / 2,100 | 20 / 2,100 | 34.76% | 0.95% | 66.90% | 0.8495 | 22.67% |
| Adapted DINOv2 | 0.9250705242156982 | 727 / 2,100 | 5 / 2,100 | 34.62% | 0.24% | 67.19% | 0.9416 | 10.95% |

Adaptation improves descriptive score ordering, but **does not increase TAR at
the predeclared training-calibrated operating points**: TAR changes by -0.14
percentage points, and FAR by -0.71 points. The adapted arm rejects 1,373 of
2,100 genuine attempts. These are two arm-specific calibrated operating points,
not a matched-test-FAR or statistical-superiority claim. No test-based cutoff,
checkpoint, configuration change or additional run was selected afterward.

| Arm / type | Genuine accepts / 700 | False accepts / 700 | TAR | FAR |
| --- | --- | --- | --- | --- |
| Frozen / left | 277 | 8 | 39.57% | 1.14% |
| Frozen / right | 253 | 7 | 36.14% | 1.00% |
| Frozen / thumbs | 200 | 5 | 28.57% | 0.71% |
| Adapted / left | 288 | 2 | 41.14% | 0.29% |
| Adapted / right | 235 | 2 | 33.57% | 0.29% |
| Adapted / thumbs | 204 | 1 | 29.14% | 0.14% |

The same pooled arm-specific cutoff applies to every type. Paired counts show
357 genuine attempts accepted only by frozen and 354 only by adapted; for
impostors the corresponding counts are 19 and four. These are descriptive
paired changes, not significance tests.

## Population, labels and limits

All 37,194 supplied training pairs (18,597 per class) were used exactly once in
each of three epochs: 111,582 pair presentations and 3,489 optimizer updates.
Left hands, right hands, two thumbs, supplied labels and raw SDK-zero rows are
retained. No pair regeneration, segmentation, augmentation, extra cross-pair
generation or new quality filter was introduced.

| Property | Training | Testing |
| --- | --- | --- |
| Numeric filename-prefix groups | 221 | 55 |
| Raw prefix strings | 282 | 55 |
| Images | 5,721 | 1,057 |
| Images reused in multiple pairs | 5,721 | 1,057 |
| Maximum pairs using one image | 28 | 16 |

Cross-phase exact-pair, image and numeric-group overlap are zero. Existing
raw-file and decoded-grayscale image hashes also have zero cross-phase overlap.
This supports **numeric-group holdout**, not verified biological subject or
capture independence. All 61 inventoried leading-zero alias groups occur in
training. Zero supplied labels disagree with numeric-prefix equality in either
phase, so the ambiguity-exclusion view equals the complete test population;
no identity truth was corrected or independently confirmed.

The older adapted checkpoint saw every new testing image/group, including
1,918 exact new testing pairs. This adaptation instead starts fresh from the
pinned pretrained base. Prior research findings on the dataset are already
known; this is not an untouched-dataset or independent-replication claim. Only
189 of the old 4,200 testing pairs occur in the new list. Old/new headline
scores describe different populations and must not be treated as a controlled
improvement comparison.

Both arms accept 185 of 18,597 training impostors under the nominal 1% policy.
This is resubstitution calibration, not independent development selection or
a guarantee of 1% test FAR. Rows reuse images/groups; report exact finite-cohort
counts without independent-pair intervals, significance or population low-FAR
bounds. One pooled false accept is approximately 0.048 percentage points;
per type it is about 0.143 points. These are empirical resolutions, not bounds.
One fixed-seed adaptation does not estimate training-seed variability.

## Available comparison evidence

Supplied new SDK scores join to the exact same 4,200 pairs, without an
intersection-only population. No new SDK extraction or scoring was performed.

| Continuous-score export | Parsed / attempted | Descriptive AUC | Interpolated EER |
| --- | --- | --- | --- |
| VeriFinger fused | 4,200 / 4,200 | 0.9809 | 3.57% |
| VeriFinger per-finger sum | 4,200 / 4,200 | 0.9801 | 3.57% |

All seven genuine and 142 impostor fused-zero scores remain zero. Per-finger
sums are positive by source selection. Template-extraction status is unresolved;
parsed score coverage is not extraction success or a full upstream-pool result.
Adapted DINO's descriptive EER remains above these supplied SDK EERs. There is
no verified non-test SDK calibration here, so no test-selected SDK operating
point is reported.

Konain owns Qwen3-VL fine-tuning. Matching A/B outputs and actual training/run/
checkpoint provenance for the new cohort have not been supplied. Historical
0-100 Qwen exports cover different testing pairs and are excluded. No claim of
superiority to trained Qwen or a language-module defect follows from these
DINO results. The [offline comparison path](../../docs/notes/precise_qwen_ab_comparison_20261101.md)
is ready for reviewed matching evidence; no duplicate VLM fine-tuning is needed.

## Execution and verification

Actual training/calibration/test source commit:
`b5a6a128b7595afaa5b4dbac4d73cf09cd65788e` on `dev`.
Model: `facebook/dinov2-small`, pinned revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`.
Final adapted checkpoint SHA-256:
`106a860fbba8e0ac28163a34faf0510df309e80ff28a39e375ccb945376f6b09`.
Exact source/configuration/review bindings, model identity and runtime versions
are in the aggregate and private digest-bound receipts.

Offline verification confirms all 185 frozen tensors (18,505,344 parameters)
unchanged and all 38 selected trainable tensors (3,551,232 parameters) changed.
Mean pair losses were 0.1456442, 0.1024788 and 0.0911037; these are training
losses, not accuracy. The three epochs and scheduler state replay completely.
No training resume, automatic retry or batch/precision/device fallback occurred.
Each cutoff has one saved test claim, and both tests have sealed completions.

The aggregate exporter replays calibration and saved test summaries, validates
fresh initialization and input bindings, and excludes the nonmatching Qwen arm.
A separate NumPy/scikit-learn implementation reproduces pooled and per-type
DINO confusion counts, coverage, cutoff replay, AUC and interpolated EER to
numerical precision. All 196 repository tests, correctness lint, Python syntax,
15 legacy metric checks, generated-table and publication/content checks pass.
Private images, identifiers, original filenames, pair rows, checkpoints, raw
scores and approval/source receipts remain under ignored `local/precise/`.

## Reproduce the aggregate offline

See [reproducibility instructions](../../docs/REPRODUCIBILITY.md) for current
Python 3.12 compatibility interpreters, original asset paths and exact exporter
arguments. Use `code/report_precise_group_holdout.py` with this study's bundle,
config, review, new source audit, new SDK input directory, checkpoint-verification
receipt and run root. It reads saved outcomes, not images or models. Render a
new output directory with `code/render_precise_shared_report.py` using the
bundled ReportLab Python, then visually review every page. Existing artifacts
are never overwritten by these tools.

Independently check DINO metrics without another experiment:

```bash
local/precise/environments/verification-compat-20261107/bin/python code/verify_precise_fixed_metrics.py \
  --bundle local/precise/manifests/numeric-group-holdout-20261107/bundle.json \
  --run-root local/precise/runs/numeric-group-holdout-20261107 \
  --aggregate reports/precise-dinov2-numeric-holdout-20261107/aggregate.json \
  --output local/precise/metadata/revised-execution-NEWDATE/independent-metric-check.json
```
