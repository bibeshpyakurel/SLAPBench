# DINOv2-Large with original-resolution images: concrete run proposal

2026-11-08. The user supplied the professor's request: “Please run a bigger
DinoV2 model if there is one available. Also, use the original image with high
resolution.” This establishes the new scientific direction. The prior small/448
runs remain complete, with all original scores, inputs and decisions preserved.

**Reviewed and explicitly approved by the user on 2026-11-08:** “I approve this exact proposal and its training, calibration and test execution.” The exact proposal bytes at approval and response are retained privately with the new input/configuration digests. No new identity or capture facts are inferred.
[AGENTS.md](../../AGENTS.md) requires retaining results/split review before new
fine-tuning, and states that completion of the previous study does not authorize
further experiments. Its shared-pair route requires actual, digest-bound
protocol/results/cohort review. The user-approved November 7 record binds the
small-model configuration, not this new model/pixel/numerical profile. The direct user approval above supplies the concrete execution review; it is
separate from the earlier quoted direction and does not invent supervisor or
canonical-identity attestations.

## Exact candidate and input policy

Use `facebook/dinov2-large` (304,368,640 parameters), pinned revision
`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`. The downloaded official safetensors
bytes have SHA-256
`399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23`.
Adapt only blocks 22/23 and final layer norm (25,198,592 parameters), starting
fresh from this pinned base. The frozen control uses the same unchanged base.
No task checkpoint from either completed DINO study initializes this run.

Use the **same supplied 37,194 training / 4,200 testing pairs**, labels and all
left/right/thumb attempts from the reviewed November 7 bundle. The bundle is
unchanged: `local/precise/manifests/numeric-group-holdout-20261107/bundle.json`.
No extra pairs, segmentation, development holdout or source-quality filter is
introduced. Training/testing remain separated by numeric filename-prefix groups,
not independently confirmed biological identities or acquisitions. Existing
research findings and test results are already known; no untouched-test or
independent-replication claim is made.

All 6,778 benchmark images have inventoried dimensions 1600 x 1500. Preserve
those grayscale pixels exactly, centered on a white 1610 x 1512 canvas (5 pixels
per horizontal side, 6 per vertical side). Both dimensions are multiples of 14.
Replicate to RGB and use the same image normalization as before. There is **no
image resampling, cropping, stretching or loss of original pixel resolution**.
Unexpected dimensions or changed raw/decoded hashes fail instead of silently
changing the protocol. Original files remain read-only.

## Frozen numerical and training policy

- CUDA/BF16 forward passes for training **and scoring**; normalized CLS vectors
  and cosine arithmetic remain FP32. Unlike the earlier small-model FP32
  inference, BF16 inference is explicit here because the selected Flash backend
  requires BF16/FP16. This protocol difference is recorded in provenance.
- SDPA with Flash attention as the only permitted backend. No implicit math/
  eager/CPU or reduced-resolution fallback. Nonreentrant gradient checkpointing
  applies during adaptation. Deterministic algorithms remain enabled.
- One explicit pair (two images) per microbatch, accumulation 32, preserving an
  effective 32-pair window and exact partial-window weights. Every supplied
  training pair occurs once per epoch; the final window contains 10 pairs.
- Three full epochs, 111,582 pair presentations and **3,489 optimizer updates**.
  Seed 2026; margin .2; AdamW LR 1e-5 to 1e-6 with 100-update warmup/cosine decay;
  matrix-only weight decay .01; clipping at 1; no augmentation or pair generation.
- Select the **final epoch in advance**. Keep epoch checkpoints and failure
  records. No test-based checkpoint/hyperparameter selection.

Exact optimizer settings are in
`code/precise_matching/configs/dinov2_large_native.json`. The entry point
`code/run_precise_native_large.py` requires the corresponding new review profile,
input/config digests, pinned weight digest and native-pixel review. Source must
be checked, committed and published before real execution. Hold that execution
source identity fixed across training, both calibrations and both final tests.
The native entry records the last commit affecting its protected runtime paths;
unrelated documentation commits do not change that execution-source identity.

## Calibration, evaluation and comparisons

After training is sealed, calibrate each arm on the 18,597 supplied training
impostors: nominal 1% training FAR, integer budget 185, accept only scores strictly
above the selected boundary and reject ties. Freeze both selections before the
one-time final tests. This is resubstitution calibration, not independent
validation or a promised 1% test FAR. The existing one-time test-claim and
immutable-output contracts remain in force. No automatic retry after failure.

Report all 4,200 attempts, TP/TN/FP/FN, standard/balanced accuracy, TAR/FAR,
failures/coverage, pooled and left/right/thumb results, and descriptive continuous
ROC/AUC/EER at unchanged cutoffs. Preserve the zero-label-conflict finding;
exclusion sensitivity therefore equals the full population. Report exact reuse
counts; do not use independent-pair confidence intervals or population low-FAR
claims. No model performance improvement is guaranteed by larger inputs/model.

Compare the new arms descriptively with the completed small/448 arms and the
[newly supplied Qwen decisions](../../reports/precise-qwen-supplied-results-20261108/README.md).
Larger architecture, native resolution and efficient numerical execution change
together. The new-vs-old comparison therefore cannot attribute any difference
solely to resizing or model size. Extra ablation arms are outside this proposal.
Qwen has native binary A/B decisions; DINO has training-calibrated cutoffs.
They are not a matched-test-FAR comparison, and Qwen run-provenance gaps remain
explicit until actual missing evidence is supplied.

## Capacity, expected duration and output boundaries

The [exact-policy synthetic check](../../reports/precise-native-large-capacity-20261108/README.md)
passes on the current RTX 4080 SUPER, approximately 16 GiB VRAM: about 3.01 GiB peak
allocated and 3.26 GiB reserved, with more than 11 GiB sampled free. No fingerprint
pixel or real pair was read for that check. The measured synthetic rate implies
about 24.6 hours of model computation for three epochs; real pipeline overhead and
calibration/testing are additional. Short-run fit is not long-run stability or
a guaranteed end-to-end completion time. No driver repair/installation is needed.

Private outputs will be at a new dated
`local/precise/runs/large-native-YYYYMMDD/`, with the actual human review/source
receipts under a separate ignored metadata directory. Export a new dated report
and reviewed aggregate tables/plots after successful authorized execution;
verify checkpoints/metrics, inspect every final report page, commit/push to
origin/dev and verify the remote SHA. No prior result/report is overwritten.

The specific decision requested is approval of **this exact Large/native
training, both training calibrations and both one-time tests**, including the
approximately one-day-plus runtime estimate, fixed final checkpoint, unchanged
supplied cohort and descriptive comparison limits.
