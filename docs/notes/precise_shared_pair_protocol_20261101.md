# Concrete shared-pair DINOv2 protocol for review

2026-11-01. Status: **the user reviewed and approved this exact protocol and its
training, calibration and test execution in the goal chat on 2026-11-01**.
Their response was: “I have reviewed and approve this exact protocol and its
training, calibration and test execution.” This records human review; it does
not attribute a new review to the professor or establish canonical identities.
The professor's broad DINOv2 approval was already recorded. It supersedes only the
subject-batch settings for this shared benchmark; the earlier subject-disjoint
route remains available separately.

## Population and labels

Use exactly Konain's supplied 39,290 training and 4,200 testing pairs, including
right, left and two thumbs, with their original A/B targets. Preserve fused-zero
rows and original pair membership. No segmentation, regeneration, exclusion,
capture assumption or identity relabeling enters training. The source hashes and
counts are in the [existing aggregate audit](../../reports/precise-shared-cohort-20261101/cohort_audit.json).

The primary target is agreement with supplied benchmark labels. Numeric IDs
disagree with 24 training and two testing impostor labels. Proposed treatment:
retain all 26 as supplied, flag them privately, and publish only counts. Report
the same frozen decisions with the two flagged testing rows excluded as a
secondary label-ambiguity sensitivity analysis, explicitly with its changed
denominator. Do not call either interpretation corrected identity truth. Training
is not repeated with changed labels. Canonical identity confirmation remains
necessary for biological identity claims, not for this explicitly limited task.

Testing excludes exact training pairs but reuses all testing images and numeric
identity groups. Results concern held-out pairs of seen images/identities,
selected using positive per-finger sums in training. They cannot establish
unseen-person, independent-acquisition or unselected-population performance.

## Fixed training and selection

The [candidate configuration](../../code/precise_matching/configs/dinov2_shared_pairs.json)
uses the existing pinned DINOv2-small revision and whole-image 448-pixel
grayscale/RGB letterbox preprocessing. Frozen: unchanged pretrained encoder.
Adapted: final two blocks and final layer norm only. Every supplied pair is
visited exactly once per epoch, with deterministic epoch shuffles, for **three
full epochs**. No development pairs are removed from the supplied JSON.

Eight explicit pairs (up to 16 images) form a microbatch; no cross-pairs are
manufactured. The pair loss is the mean of `1 - cosine` for genuine targets and
`max(cosine - 0.2, 0)` for impostor targets. Equal global class counts give equal
class weight over each epoch. AdamW: learning rate 1e-5, minimum 1e-6, weight
decay .01 on matrices only, accumulation four, gradient clipping at 1, linear
100-update warmup then cosine decay, seed 2026, CUDA/BF16. Incomplete batches and
accumulation windows retain every pair, weighted by their actual pair counts.

Select the final epoch checkpoint **in advance**, without consulting training
accuracy, testing scores or descriptive test curves. Epoch checkpoints support
recovery, not a retrospective best-epoch search. A failed run retains its record;
no automatic hyperparameter/batch/device/precision fallback or test-based retry.
Resume only at verified unfinished epoch boundaries with identical provenance.

After the model is fixed, each DINO arm computes training scores and selects its
own strict `score > cutoff` boundary from the 19,645 training impostors at a
nominal **1% training FAR** (integer budget 196, boundary ties rejected). Freeze
the cutoff and checkpoint digest before reading test pixels/scores. This is
**training calibration**, with resubstitution optimism, not a development estimate
or promised 1% test/population FAR. Training is unchanged. No common numeric
cutoff is imposed across score scales. No operating point is chosen from test
ROC/EER. The prior draft's 0.1% development-FAR policy is not used here.

## Evaluation, uncertainty and comparison

Preserve one private outcome per attempt, including image/encoder failures.
Training or calibration failures stop the run for investigation; evaluation
records explicit failed attempts without imputing scores. Report pooled and
right/left/thumb coverage, TAR/FAR, balanced accuracy, confusion counts and
valid-score descriptive AUC/ROC/DET/EER. Apply the same pooled arm cutoff to
every slap type and sensitivity subset. Report both all-attempt failure-as-reject
and valid-only denominators; failed impostors are not biometric rejection evidence.

Primary results are finite-cohort descriptive measurements. Reused images and
identities make independent-pair binomial intervals and significance tests
inappropriate. Proposed uncertainty handling for this milestone: publish exact
denominators, dependency/overlap counts, label-ambiguity sensitivity and empirical
FAR resolution (one pooled false accept is 1/2,100, about 0.048 percentage
points); **no population confidence interval, significance or low-FAR guarantee**.
Subject/dyadic resampling needs a reviewed identity mapping and statistical
design and is explicitly deferred. This is a limitation, not an assumption that
the observations are independent.

VeriFinger fused and per-finger-sum test scores may be reported descriptively on
the same 4,200 attempts; raw zero values stay zero and extraction status remains
unknown. A thresholded SDK arm requires a reviewed, non-test selection policy
and appropriate training/calibration provenance. Do not optimize its cutoff on
testing to create a favorable operating-point comparison.

Konain owns Qwen3-VL fine-tuning. A trained-Qwen comparison requires exact output
pair coverage, matching training membership, actual configuration/environment,
model/adapter identity, prompt/preprocessing and checkpoint-selection evidence.
Older supplied Qwen output files are historical until that provenance is verified.
Binary outputs support one operating point, not continuous-score ROC/EER. Missing
evidence is reported explicitly; no local Qwen training is added by assumption.

## Required review and execution boundary

Before real model execution a human must review this concrete protocol, the
cohort/label policy, intended results/limitations and execution scope. This
requirement comes from [AGENTS.md](../../AGENTS.md) and the
[latest handoff](precise_next_goal_handoff_20261101.md), which says to retain
applicable results/protocol review before training. Approval of the broad plan
does not by itself approve these newly specified settings.

The user supplied that concrete review in this goal chat as recorded above.
Implementation and synthetic checks precede real execution.
Private preparation records exact source/configuration/image-inventory hashes;
it does not read image pixels or manufacture an approval receipt. A separate
digest-bound human review record must authorize training, calibration and test
execution. No assertion of subject separation or independent captures is required
or permitted for this shared-cohort route. Code is committed and reviewed before
real execution; private manifests/checkpoints/raw outputs remain under ignored
`local/precise/`. Reviewed aggregates and the final professor report go to new
dated report paths and `origin/dev`. The end-to-end goal remains incomplete until
the reviewed runs, comparison evidence and report exist.
