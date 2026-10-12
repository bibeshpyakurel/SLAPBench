# Precise evaluation protocol sketch — principles approved, details pending

2026-10-10. No subject assignments, split percentages, matching pairs, selected
model/loss, preprocessing parameters, thresholds or experiments are created by
this document. It completes the planning task while leaving decisions explicit.

**Current status — 2026-10-11:** The
[central research plan](precise_first_milestone_proposal_20261011.md) records
professor approval of the image-only milestone on whole four-finger slaps and
subject-disjoint, development-selected evaluation principles. This sketch
supplies the detailed safeguards. Data provenance, exposure eligibility,
concrete split/model/metric parameters and execution authorization remain
unresolved. The supplied VeriFinger testing scores are now available and
audited; their thresholds are retrospective rather than development-fixed.

## Evidence available now

The [local inspection report](../../reports/precise-inspection-20261010/README.md)
establishes header dimensions/mode and sampled readability, not acquisition
independence or physical PPI. Konain's message confirms that his baseline uses
full slaps with SDK fused scores and per-finger scores whose sum is also
evaluated. Filename subject/slap-type interpretation agrees with our parser.
Middle-token/session semantics, capture independence and physical PPI remain
unknown. The supplied exports confirm matching filename-derived slap types and
finger counts, including thumb comparisons; anatomical finger-position mapping
and SDK configuration are not established by those checks.

The [2026-10-11 score audit](../../reports/precise-verifinger-audit-20261011/README.md)
validates all 4,200 supplied testing records, their per-finger sums/counts and
filename-derived genuine/impostor labels. It reproduces thresholds 48 (SDK
fusion) and 79 (sum), 2,057/2,100 genuine accepts (97.9524% TAR), and zero
accepts among 2,100 impostors. The text names 0.1% FAR; the screenshot requests
0.01%. The audit reports both requested targets with actual acceptance counts.
Thresholds are selected descriptively from evaluation impostors; development
threshold provenance remains unknown.

The pooled exports include thumbs. Their four-finger-only slice has 1,440
genuine and 1,427 impostor comparisons and gives 1,412/1,440 accepts (98.0556%
TAR) at zero observed false accepts for both methods with slice-specific
thresholds. Only 60 genuine and zero impostor pairs overlap the earlier primary
VLM manifest. Neither aggregate is a common-pair comparison of the methods.
SDK testing records cover 337 filename subjects of the 338 in the local
inventory; a sufficiently sized untouched cohort has not been established.
Training/development exposure remains a separate missing record.

“Skipped lines: 0” is a parsing observation, not proof that extraction failures
or missing fingers were absent before score export.

## Assign subjects before constructing comparisons

```mermaid
flowchart TD
    A[Authorized Precise subjects and parent captures] --> H[Record prior training and evaluation exposure]
    H --> R[Review eligibility and capture provenance]
    R --> TR[Training subjects: learn weights]
    R --> DV[Development subjects: choose settings and thresholds]
    R --> TE[Final-test subjects: frozen evaluation]
    TR --> P1[Construct training comparisons within this partition]
    DV --> P2[Construct development comparisons within this partition]
    TE --> P3[Construct test comparisons within this partition]
```

Let the subject sets be `S_train`, `S_dev` and `S_test`. Require every pairwise
intersection to be empty. Keep both hands, thumbs, sessions, sensors, original
captures, alternate encodings and all crops/augmentations from one subject in
the same partition. Create no cross-partition genuine or impostor pairs.
Check exact and decoded duplicates across partition boundaries; review near
duplicates and shared parent captures without automatically changing labels.
Any cross-subject duplicate/annotation conflict requires a documented resolution
before freezing the split.

| Partition | Allowed purpose | Frozen before final test |
|---|---|---|
| Training | Learn encoder/head/fusion parameters, with approved augmentation and supervision | Trained checkpoint and training provenance |
| Development | Select model/checkpoint, input resolution, prompts if used, fusion and thresholds | Selected configuration and operating thresholds |
| Final test | Apply the fixed system and report all predefined comparisons | No tuning or repeated checkpoint selection on test scores |

Sizes, seed and subject lists remain **TBD**. Decide them after learning which
subjects Konain already used. A cohort already used to choose models or inspect
results should not be described as untouched. Existing collaborator pairs can
remain a retrospective/exploratory reference while a separate eligible cohort
supports a prospective protocol, if the available population permits it.

Excluding `(A1,A2)` from training does not prevent training on `(A1,A3)`;
the same image and subject remain exposed. Even entirely different images of
the same subject do not satisfy this proposed subject-disjoint protocol. This
identifies a protocol question, not a confirmed defect in Konain's work.

## Comparison unit and matching lists

First-milestone comparison unit: two whole four-finger slaps of the same hand
type, following the central plan. This supports comparison to the collaborator's
reported SDK interface. Per-finger processing and minutiae belong to later
extensions with separate decisions. If segmentation is approved, preserve the
parent capture and finger-position mapping; do not let crops cross the subject boundary. Thumb
slaps should be a separate, explicitly scoped analysis or excluded by an agreed
rule; do not mix their two-finger scores with four-finger scores silently.

Genuine comparisons require independent impressions of the same subject and
slap type. Primary impostors use different subjects at matching hand/finger
positions. Sampling rules, trials per subject, seed and trial counts remain
TBD. Do not infer capture independence from distinct filenames or pixel hashes.
The visual review shows variation but does not establish separate sessions.

Run each agreed system on exactly the same external comparison list. Proposed
arms for the first milestone are the image-only learned matcher, a frozen version of
the encoder, SDK fusion and SDK per-finger sum where access permits. The final
arm list remains to be specified; image-plus-minutiae is a later extension,
and Bozorth3/prompted VLM inclusion remains open. Mark method-specific
preprocessing, unavailable fingers and extraction errors rather than dropping
their comparisons silently. SDK and sum scores have different scales: each
needs its own development-selected threshold.

## Training and development choices

Specify whether the encoder comes from a VLM or a dedicated fingerprint model,
which components are trainable, the supervised target and the loss. Ground-truth
pair/identity supervision and VeriFinger-score distillation are distinct options.
Teacher-score generation must respect the partition boundary and recorded
training exposure. Do not use final-test scores as training targets or as
inputs to a supposedly independent learned matcher. Select checkpoints using
development verification metrics, not final-test EER at each checkpoint.

Document original pixel dimensions and PPI provenance, input resizing/padding,
contrast/orientation normalization, augmentation, minutiae conventions and any
quality decisions. PPI is not established by the image headers inspected here.
Keep common inputs where feasible and document method-required differences.
A white-border/intensity statistic is not an exclusion criterion or validated
quality metric. Freeze any quality/exclusion and missing-finger policy before
evaluating final test.

## Thresholds, metrics and uncertainty

Agree whether 0.1% or 0.01% FAR is the target. Choose thresholds on development
impostors under a stated comparison rule (`score >= threshold` or another
documented SDK rule). Freeze them, then report actual test false accepts,
impostor denominator, genuine accepts and genuine denominator. Do not force
the observed test FAR to equal the target by retuning on final-test scores.
Test ROC/DET and EER are descriptive curves/summaries; they must not feed back
into final model or threshold selection.

For 2,100 impostors, one error is 0.047619%, two errors are 0.095238%, and a
0.01% empirical target permits no errors. A fixed-threshold independent-trial
zero-error calculation gives a one-sided 95% bound of about 0.14255%, illustrating
why zero observed errors does not substantiate an underlying FAR below 0.01%.
Repeated subjects/images induce dependence. Choose an uncertainty method that
accounts for that dependence, including both endpoints of impostor comparisons;
ordinary independent-pair bootstrapping is not automatically valid. The method
and achievable FAR precision need statistical review before final claims.
The [primer](fingerprint_matching_primer_20261010.md) shows the zero-error
calculation and its assumptions. General biometric uncertainty background:
[NIST IR 7740](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7740.pdf).
The dependence-aware method for this particular dataset remains to be agreed.

Report ROC, DET, EER, score distributions, TAR at supported FARs, actual
fixed-threshold test FAR/TAR, failures, trial/subject counts and uncertainty.
For missing scores, propose two views for review: valid-score discrimination
with explicit coverage, and all-attempt outcomes using a predefined failure
policy. Keep extraction/decode/refusal errors distinct from scored non-matches.
A failure-as-reject policy reduces genuine acceptance but must not be hidden
as stronger biometric impostor rejection. Do not silently impute scores.

## Records to prepare before an authorized experiment

| Record | Fields to agree | Storage |
|---|---|---|
| Subject exposure/split record | Subject, prior training/development/test use, parent captures, partition, reviewed seed/version | Private `local/precise/metadata/` and `splits/` |
| Comparison manifest | Opaque pair ID, image paths, subject/slap/finger labels, parent capture, partition, genuine/impostor label | Private `local/precise/manifests/` initially |
| Score/error table | Pair ID, method/version, partition, SDK fused score, corresponding per-finger scores, sum, model score, status/error, threshold and decision | Raw local dated run; reviewed exports only |
| Run record | Data/source version, manifest digest, code commit, checkpoint/revision, environment, preprocessing, supervision/loss, seeds, hardware, threshold/fusion policy, exclusions | Local run plus reviewed public summary |

These are schema sketches, not generated manifests or selected splits. Raw
images/crops, minutiae, identifying metadata and weights remain local. Publish
only reviewed aggregate findings, code and appropriate reproducibility artifacts.

## Readiness decisions still pending

1. Professor/implementation review: encoder relationship to an LLM/VLM,
   concrete baseline arms/model/loss and separate execution authorization;
   whole-slap first input and staged milestone direction are recorded in the central plan.
2. Collaborator: training/development pair mappings and exposure records, source
   captures/subjects used in every stage, adapted targets/script and SDK settings.
   Testing-score mappings are already audited; do not request them as missing.
3. Data provenance: acquisition PPI, session/middle-token meanings, independent
   impressions, finger-position correspondence, data terms and thumb scope.
4. Joint review: actual subject sets/sizes, sampling policy, low-FAR target,
   development threshold selection, quality/failure policy and uncertainty method.

The earlier guide remains in Git history at `54bf2f8`; its §§4,8,9,11,16 are
methodological background. This sketch does not restore the removed guide or
replace supervisor review with an assumed decision.
