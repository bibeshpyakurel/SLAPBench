# Revised cohort: scope and run proposal for review

2026-11-07. **Reviewed and approved by the user in the resumed goal chat:**
“Yes—use the new lists and run fresh DINOv2 training, calibration and testing.”
This explicitly replaces the active benchmark and authorizes the fresh runs
below. The exact proposal bytes at approval and the human response are retained
privately; no new professor statement, biological identity or capture fact is
inferred. The original goal specified
39,290 balanced training pairs and the completed November 1 study. The shared
folder now contains a different 37,194-pair training cohort and a changed
4,200-pair test cohort. The [read-only intake](../../reports/precise-revised-cohort-20261107/README.md)
records exact counts, source digests, group separation and prior model exposure.

## Decision needed

The user selected the new supplied cohort and authorized fresh training,
calibration and testing. This was a concrete cohort/execution decision under [AGENTS.md](../../AGENTS.md),
which says additional training/test runs require separate authorization. It is
not another request for broad DINOv2 architecture approval, the dataframe,
original lists, prompts or scripts already supplied.

The following is the reviewed complete run policy. Existing input files and results remain unchanged, and the
old report stays a historical result of its original benchmark. New outputs
must use separate dated private/public paths and a new digest-bound review.

## Population and permitted claims

Use exactly the supplied 37,194 training pairs (18,597 per class) and 4,200 new
testing pairs (2,100 per class), including all left/right/thumb attempts and raw
fused-zero rows. Preserve every supplied label. The new JSON and training lists
match exactly; no numeric-identity label conflicts occur in either phase.
Do not regenerate pairs, split off a new development subset, relabel aliases,
crop images or apply an additional quality/SDK filter.

The new files separate 221 training from 55 testing numeric prefix groups and
share no images/pairs. Describe this as the supplied numeric-group holdout.
Biological identity and acquisition independence are still unconfirmed. Source
selection forces all 61 padding-alias groups into training, so unseen-group
claims do not extend to held-out alias groups or an unselected population.

All new testing groups/images, and 1,918 exact new testing pairs, were previously
used by the completed adapted DINO model. That checkpoint cannot be resumed or
reused for an unseen-group claim. New task adaptation must start from the same
pinned pretrained DINOv2-small base. Historical task results are already known;
this is not a claim of an entirely untouched dataset or independent replication.

## Proposed methods and frozen selection

Retain the already reviewed DINO method without a test-based search:
`facebook/dinov2-small` at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`, whole-image grayscale/RGB letterboxing
at 448 x 448, normalized FP32 class-token embeddings and cosine similarity.

Frozen arm: unchanged pretrained base. Adapted arm: only the final two blocks
and final layer norm, initialized afresh from that base. Use the same predeclared
schedule/settings as the approved November 1 protocol: three full epochs, eight
explicit pairs per microbatch, accumulation four with exact tail weighting,
AdamW LR 1e-5/minimum 1e-6, matrix weight decay .01, margin .2, clipping at 1,
100-update warmup then cosine decay, seed 2026 and CUDA/BF16. Every new training
pair appears once per epoch; no within-batch cross-pairs are added. This gives
3,489 updates over three epochs. Preserve epoch checkpoints and failures.

Select the final epoch checkpoint in advance. Each arm then freezes its pooled
cutoff from all 18,597 new training impostors at nominal 1% training FAR: a strict
`score > cutoff` boundary with an integer budget of 185 false accepts, rejecting
boundary ties. This remains training calibration, not independent development
selection or a promised test FAR. No scores from the 4,200 new test pairs may
choose settings, checkpoint or cutoff. Do not reuse the old cutoffs on the newly
adapted model or turn old/new test performance into a hyperparameter search.

Existing dependencies, weights and GPU setup are reused. Scope-aware input,
review and report validation must be implemented/synthetically checked and
committed before any newly authorized real run. The existing shared-cohort scope
must not be mislabeled as a new numeric-group holdout just to bypass a validator.
No automatic batch/precision/device fallback or repeated test attempt.

## Evaluation, comparison and uncertainty

Report every attempted pair and every failure, pooled and left/right/thumb
TAR/FAR, confusion counts, balanced accuracy, coverage and valid-score descriptive
ROC/DET/AUC/EER, all at the unchanged arm-specific pooled training cutoff.
Testing has 700 genuine and 700 impostor pairs per type. Failures remain rejects
in primary all-attempt rates and separately visible in valid-only rates.

Pairs reuse images and groups within phases. Retain the existing descriptive
uncertainty policy: exact denominators/reuse counts, no independent-pair
confidence interval, significance or population low-FAR guarantee. One pooled
false accept is 1/2,100 (about .048 percentage points); one per type is 1/700
(about .143 points). Fifty-five numeric groups are not 4,200 independent trials.
There are no flagged test-label cases to exclude under the previous sensitivity
definition; report that fact rather than inventing corrected ground truth.

Supplied new SDK values may support descriptive common-pair comparisons, with
raw zeros and unknown extraction status explicit. A thresholded SDK comparison
needs its own reviewed non-test selection rule. Konain still owns Qwen3-VL
fine-tuning. Actual Qwen outputs must match this new cohort and its prompt,
training membership, model/checkpoint, preprocessing, generation, parsing and
selection provenance before a trained-model comparison is claimed. The shared
folder currently has new pair inputs, not those model outputs. Historical Qwen
results on the old 4,200 pairs are not replacements for new-cohort outcomes.

Publish a new professor report and reviewed aggregate artifacts after successful
authorized execution and provenance/content checks, then commit/push to
`origin/dev` with matching remote SHA. Keep every original, previous report,
private asset and frozen prior decision intact. No VLM training, paid call,
manuscript change or additional study is included by this proposal.
