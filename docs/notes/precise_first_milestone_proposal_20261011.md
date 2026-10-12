# Precise first milestone — proposal for professor review

2026-10-11. This records the user's interpretation and proposed sequence.
It does not establish professor approval, assign subjects, select a model/loss,
or authorize dataset preparation or training. The confirmed direction remains
Precise; the [preparation note](precise_preparation_20261004.md) records the
earlier feedback and the [protocol sketch](precise_protocol_sketch_20261010.md)
contains the detailed evaluation safeguards.

## Proposed contribution and sequence

Learn fingerprint similarity directly: a shared vision encoder maps each input
to a numerical representation, and a verification objective makes independent
matching impressions similar while separating nonmatching impressions. At the
proposed first stage, each input is a whole four-finger slap. Return a continuous
similarity score; apply a development-selected threshold only when a match
decision is needed. A similarity score is not automatically a calibrated
probability.

1. **Image-only baseline:** Adapt a pretrained vision encoder for verification.
   Select its trainable components and verification loss during protocol review.
   Ground-truth matching supervision is the proposed starting point;
   VeriFinger-score distillation is a separate design choice, not implied by
   Konain's training method. A frozen version of the same encoder is a useful
   proposed control for measuring what adaptation adds.
2. **Controlled evaluation:** Establish subject-disjoint training, development
   and test partitions before constructing any comparisons or learning weights.
   Choose checkpoints, preprocessing settings and thresholds on development;
   evaluate the frozen system on test. This protocol is a prerequisite for
   stage 1 execution, not a step to postpone until after training.
3. **Minutiae extension:** After the baseline is reproducible, add local geometry
   and learned correspondences and measure their incremental benefit on the
   same evaluation protocol. Extraction, descriptor conventions, correspondence
   supervision and fusion would require their own design and authorization.
4. **Optional explanation layer:** Consider an LLM for interpreting matcher
   evidence after matching is reliable. It is outside the proposed first
   milestone; generated explanations would need separate validation.

The professor must confirm whether a VLM-derived vision encoder satisfies the
project's intended contribution or a different encoder is acceptable. Language
generation is not required to compute the proposed embedding similarity.
Architecture can be designed independently of Konain's fine-tuning script;
reproducing his system still requires his actual targets, code and exposure
records.

## What the cited work supports

[DeepPrint: Learning a Fixed-Length Fingerprint Representation](https://arxiv.org/abs/1909.09901)
provides precedent for learning compact fingerprint embeddings. Its architecture
incorporates alignment and minutiae detection. It therefore motivates learned
representations without validating this proposed generic image-only encoder,
whole-slap input, verification loss or achievable Precise performance.

[SuperGlue: Learning Feature Matching with Graph Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html)
learns correspondences between local features, including rejection of unmatched
points, using attention and differentiable optimal transport. It was evaluated
for geometric image matching and pose estimation. Using that idea for fingerprint
minutiae is an architectural proposal requiring adaptation and evidence; it is
not an established Precise matcher or an implementation selected here.

## Input unit and comparison evidence

Propose whole four-finger slaps first because those images are already available
and the external VeriFinger result is a slap comparison. Right and left slaps
should be compared within the same hand type. Two-thumb slaps should be excluded
from this first milestone by an agreed rule and retained as historical assets.
Individual-finger processing would add segmentation, finger-position mapping
and correspondence/fusion decisions; none are needed to draft the first proposal.

The [VeriFinger score audit](../../reports/precise-verifinger-audit-20261011/README.md)
already recomputes the supplied benchmark exports. Its four-finger slice has
1,440 genuine and 1,427 impostor comparisons; both fused and summed scores accept
1,412 genuine comparisons (98.0556% TAR) at zero observed false accepts using
slice-specific, evaluation-selected thresholds. This is retrospective evidence,
not a development-fixed operating point or a target the new model is guaranteed
to reach. The pooled 2,100+2,100 comparisons also contain thumb slaps.

Existing VLM results overlap only 60 genuine and zero impostor pairs with the
supplied SDK testing population. A prospective method comparison needs the
same eligible subjects and exactly the same comparison list for every method;
the old VLM/SDK aggregates cannot establish that comparison. SDK access/settings
and scores on the future approved list remain practical dependencies.

## Data protocol to approve before execution

| Decision | Proposed rule | Still needed |
|---|---|---|
| Identity and capture provenance | Genuine pairs use independent captures of the same subject and hand type | Confirm filename semantics, capture/session relationships and annotation issues |
| Subject separation | Keep all hands, sessions, captures and derivatives of a subject in one partition; no cross-partition pairs | Exposure history, eligible subjects, feasible sizes, seed and split review |
| Duplicate handling | Review exact/decoded/near duplicates and shared parent captures before freezing partitions | Agreed conflict-resolution and exclusion rules; distinct filenames do not prove independent captures |
| Learning | Adapt the image encoder using an agreed verification objective | Encoder/revision, trainable components, loss, preprocessing, augmentation and local compute budget |
| Comparison | Use identical four-finger slap pairs for learned/frozen controls and available SDK methods | Method scope, licensed SDK access/settings, sampling policy and failure reporting |
| Model and threshold selection | Use development only, then freeze checkpoint/configuration/thresholds | Primary development metric, FAR target, score acceptance rule and selection policy |
| Final reporting | Report continuous-score ROC/DET/EER and actual fixed-threshold test TAR/FAR, counts, coverage, failures and uncertainty | Supported FAR resolution and uncertainty accounting for reused subjects/images |

Historical evaluation exposure must be recorded separately from training
exposure. The supplied SDK testing records cover 337 filename subjects, while
the local inventory has 338. No sufficiently sized untouched subject cohort
has been established. Subject-disjoint future training alone cannot make an
already inspected cohort an untouched prospective test; agree feasible claim
scope and evaluation exposure with the professor before choosing a split.

Do not select test thresholds to reproduce an observed FAR. A low-FAR target
needs adequate impostor evidence and dependence-aware uncertainty; zero observed
false accepts alone does not establish the requested population FAR. No numeric
success criterion is selected here. First-milestone completion should mean a
reproducible baseline and a valid comparison, including an honest negative
result, rather than an assumed improvement over VeriFinger.

## Priority questions for Konain

Dataset interpretation is the immediate priority:

1. What do the middle filename tokens mean, and which tokens identify capture,
   impression, session, acquisition device or processing version?
2. Do genuine pairs represent independently acquired impressions, or can they
   be alternate encodings/processed versions of the same capture? What evidence
   links each image to its parent acquisition?
3. What acquisition/session, sensor, physical resolution and annotation metadata
   exists, and are there known duplicates or labeling exceptions?

Training/development pair mappings and subject/image exposure records remain
necessary to assess reuse and evaluation eligibility. His adapted training
script, example targets and SDK configuration remain useful for understanding
or reproducing his work, but are not a prerequisite for independently drafting
our architecture. Keep identifying mappings and source filenames local.

## Professor review and next authorized task

Ask the professor to approve or revise (a) an image-only learned matcher as the
first milestone, (b) whole four-finger slaps as the first input unit, (c) the
encoder's required relationship to an LLM/VLM, and (d) the subject/capture,
development-selection and final-test protocol above. Confirm baseline arms,
FAR/uncertainty requirements and the treatment of historical exposure.

After metadata clarification, prepare the concrete split/protocol for review
under an applicable task authorization. Retain the results/split review before
fine-tuning unless the supervisor explicitly revises it. Training, split/pair
creation, segmentation, minutiae extraction, model downloads and experiments
still require their applicable decisions and separate task authorization.
Future private artifacts belong under ignored `local/precise/`; original
datasets, scores and fixed manifests are preserved.

This document supplies discussion material only. No message has been sent to
the professor or Konain.
