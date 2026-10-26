# Precise research plan — selected matcher implemented, data review pending

2026-10-11. The user subsequently confirmed: **“professor approved this.”**
This updates the original proposal to an approved research direction and
milestone sequence: image-only learning on whole four-finger slaps with
subject-disjoint evaluation and development-selected checkpoints/thresholds,
followed by a minutiae extension and optional explanations. The protocol
principles are approved; unspecified implementation parameters and unresolved
data facts below remain open. Approval does not supply subject assignments,
capture metadata, a selected model/loss or a separate execution task.

The [preparation note](precise_preparation_20261004.md) preserves earlier
feedback and the [protocol sketch](precise_protocol_sketch_20261010.md)
contains the detailed evaluation safeguards. The original proposal filename
is retained so existing links remain valid.

**Implementation update — 2026-10-17:** The user requested progress toward the
plan. The offline metadata validator and development/test score evaluator are
now implemented and tested with synthetic inputs in `code/precise_matching/`.
See the [tooling note](precise_matching_tooling_20261017.md) for contracts,
commands and the next collaborator request. This adds validation/summary tooling;
it does not create a real split, select an encoder/loss or run an experiment.

**Collaborator update — 2026-10-24:** the two supplied training lists and the
existing testing files are now [organized and audited](../../reports/precise-training-intake-20261024/README.md).
The 101,695 training pairs exclude all 4,200 exact testing pairs, but include
every testing image basename and subject elsewhere. This is not the approved
subject-disjoint protocol. A [pinned FPBench checkout](fpbench_reference_20261024.md)
is preserved locally; Konain reports Qwen3 binary fine-tuning, but the reference
does not contain his Precise-specific conversion or development procedure.
No local training, subject allocation or target conversion was performed.

## Central plan and completed preparation

**Selected matcher implementation — 2026-10-25:** the user's later decision
selected the proposed DINOv2 matcher first and requested training/testing.
The scheduled resumption recovered the interrupted local implementation and
added a guarded training entry point. The
[implementation guide](precise_training_implementation_20261025.md) documents
whole-image loading, shared frozen/adapted encoding, verification loss,
development-only selection, immutable checkpoints and exact synthetic resume.
A small pretrained CPU runtime probe passes. Capture/exposure facts, concrete
protocol/split review and usable GPU resources remain outstanding; no real
Precise training or new subject allocation has occurred.

**Final-test implementation — 2026-10-26:** the scheduled evaluation work added
[frozen-selection verification and final-test scoring](precise_final_test_20261026.md).
It checks completed training provenance/history and replays selected development
evidence before test pixels, applies the saved cutoff unchanged and preserves
failure outcomes with an exclusive adapted-test claim. The
[readiness check](precise_execution_readiness_20261026.md) still finds no usable
CUDA or reviewed real split/receipts. This is synthetic implementation evidence,
not new fingerprint performance or resolution of capture/exposure questions.

**Roadmap update — 2026-10-25:** the user supplied the ten-task execution roadmap
and confirmed that no further acquisition/exposure clarification is available.
The [full-image readiness audit](../../reports/precise-readiness-20261025/README.md)
extends sampled decoding to the complete local JPEG collection. A
[concrete review proposal](precise_milestone_review_20261025.md) specifies candidate
allocation counts, sampling/selection rules and a pinned DINOv2-small design.
These choices require review; they are not an approved split/model configuration.
GPU inspection again finds the NVIDIA driver unavailable. Acquisition truth,
actual exposure and the later execution gates remain open.

Use this document as the central research plan. Supporting notes explain the
background or protocol in more detail; dated reports preserve the evidence at
the time of each preparation task.

| Supporting record | Completed evidence or role | Remaining limits |
|---|---|---|
| [Professor direction and preparation history](precise_preparation_20261004.md) | Precise-only direction and dated follow-ups | Does not select concrete model, loss or split parameters |
| [Read-only dataset inventory](../../reports/precise-readiness-20261004/README.md) | 7,358 JPEGs, 338 filename subjects, 4,907 four-finger slaps; no exact byte duplicates | Filename labels and distinct bytes do not prove independent acquisitions |
| [Image inspection](../../reports/precise-inspection-20261010/README.md) | All headers are 1600×1500 grayscale; all 35 sampled images decoded | Physical PPI, sessions and whole-dataset decoded/near-duplicate status remain unknown |
| [Collaborator-file discovery](../../reports/precise-collaborator-discovery-20261011/README.md) | Earlier scoped local search and Qwen3 export provenance | Its testing-score request was superseded by Drive intake; its missing-training-list finding was superseded on 2026-10-24 |
| [VeriFinger testing-score audit](../../reports/precise-verifinger-audit-20261011/README.md) | All 4,200 supplied testing records validate; score calculations and comparison overlap checked | Retrospective thresholds, mixed thumb/slap population; no training exposure or SDK configuration verification |
| [Training-list intake and overlap audit](../../reports/precise-training-intake-20261024/README.md) | 23,103 genuine and 78,592 impostor training records; exact testing pairs excluded; image/subject overlap quantified | Right-only impostor training; incomplete component fields in 3,455 genuine rows; actual consumption, binary targets and development selection unknown |
| [Full-image readiness and compute audit](../../reports/precise-readiness-20261025/README.md) | Whole-collection decoding, duplicate screening and private per-image capture/exposure records | Acquisition/exposure review and GPU readiness remain unresolved; no split or model execution |
| [Concrete milestone review proposal](precise_milestone_review_20261025.md) | Proposed allocation/sampling, FAR/selection/failure policy, pinned encoder/loss and implementation design | Requires review; no real assignment, training environment or experiment is certified |
| [Pinned FPBench reference](fpbench_reference_20261024.md) | Full upstream checkout and documented script behavior | Reference code is not Konain's private Precise binary adaptation or our embedding matcher |
| [Evaluation protocol sketch](precise_protocol_sketch_20261010.md) | Detailed subject/capture separation, development selection, failures and uncertainty safeguards | Concrete allocation, operating points and uncertainty method require review |
| [Research brief](precise_research_brief_20261010.md) and [matching primer](fingerprint_matching_primer_20261010.md) | Discussion background, pipeline and metric explanations | Supporting material; current sequence and decision status are recorded here |
| [Workspace layout](../WORKSPACE_LAYOUT.md) and [reproduction notes](../REPRODUCIBILITY.md) | Private storage, reviewed public exports and offline-audit reproduction | Data, collaborator records and weights are not reconstructible from public Git alone |

The architecture sequence below depends on the data/evaluation decisions in
this plan. Completion of the audits does not resolve capture independence or
authorize split creation, model implementation or experiments. Keep original reports
and scores unchanged; export any later derived analyses to new dated paths.

## Approved contribution and sequence

Learn fingerprint similarity directly: a shared vision encoder maps each input
to a numerical representation, and a verification objective makes independent
matching impressions similar while separating nonmatching impressions. At the
approved first stage, each input is a whole four-finger slap. Return a continuous
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

The original direction approval alone did not select an encoder. The user later
selected the proposed DINOv2-first image matcher, with its pinned checkpoint and
implementation documented in the 2026-10-25 update. Language generation is not
required to compute its embedding similarity.
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

Use whole four-finger slaps first because those images are already available
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

## Approved protocol principles and details to resolve before execution

| Decision | Protocol principle | Still needed |
|---|---|---|
| Identity and capture provenance | Genuine pairs use independent captures of the same subject and hand type | Confirm filename semantics, capture/session relationships and annotation issues |
| Subject separation | Keep all hands, sessions, captures and derivatives of a subject in one partition; no cross-partition pairs | Exposure history, eligible subjects, feasible sizes, seed and split review |
| Duplicate handling | Review exact/decoded/near duplicates and shared parent captures before freezing partitions | Agreed conflict-resolution and exclusion rules; distinct filenames do not prove independent captures |
| Learning | Adapt the selected DINOv2 encoder with the implemented verification objective | Review implementation/resource evidence and exact real-run settings; capture/protocol/split prerequisites remain |
| Comparison | Use identical four-finger slap pairs for learned/frozen controls and available SDK methods | Method scope, licensed SDK access/settings, sampling policy and failure reporting |
| Model and threshold selection | Use development only, then freeze checkpoint/configuration/thresholds | Primary development metric, FAR target, score acceptance rule and selection policy |
| Final reporting | Report continuous-score ROC/DET/EER and actual fixed-threshold test TAR/FAR, counts, coverage, failures and uncertainty | Supported FAR resolution and uncertainty accounting for reused subjects/images |

Historical evaluation exposure must be recorded separately from training
exposure. The supplied SDK testing records cover 337 filename subjects, while
the local inventory has 338. The 2026-10-24 supplied training lists cover all
338 subjects and all 7,358 image basenames, including every testing image and
subject. This is list membership, not proof of actual model consumption.
No sufficiently sized untouched subject cohort has been established.
Subject-disjoint future training alone cannot make an
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

Training pair mappings are now available and audited; do not ask for the same
two lists again. A separate development allocation and actual consumption records
remain necessary for reproduction. Clarify the right-only impostor population
and genuine zero-score rows without components. His adapted binary training
script/example targets and SDK configuration remain useful for understanding
or reproducing his work; the public FPBench reference does not supply those
Precise-specific choices. These are not prerequisites for independently drafting
our architecture. Keep identifying mappings and source filenames local.

## Implementation and storage

A dedicated `code/precise_matching/` package now contains offline protocol and
evaluation tooling plus the selected matcher implementation. Historical
experiment scripts keep their original responsibilities and paths:

| Module | Responsibility and status |
|---|---|
| `data.py` | Implemented: validate supplied private metadata/comparisons, capture assertions and partition consistency; no image loading or split creation |
| `images.py`, `encoders.py` | Implemented: reviewed whole-image loading/preprocessing, pinned local DINOv2 and frozen/adapted normalized embeddings |
| `losses.py`, `training.py` | Implemented: cosine verification objective, training-only sampler, optimization/resume and development selection; synthetic validation |
| `testing.py`, `code/test_precise_matcher.py` | Implemented: completed-selection/history/development replay, test-only image paths, unchanged cutoff, immutable private outcomes and adapted-test claim; synthetic validation |
| `evaluation.py` | Implemented: supplied continuous scores, development-only cutoff, unchanged test cutoff, coverage/failures and aggregate metrics; no inference |
| `__main__.py` | Implemented: offline `validate` / `evaluate` commands with aggregate-only output and overwrite refusal |

The 2026-10-11 cleanup documented the package boundary; the 2026-10-17 task
implemented the offline portions using only the standard library. The later
2026-10-25 user-selected DINOv2 implementation adds image/encoder/loss/training
modules and synthetic runtime validation. Preserve existing
`code/precise_pairs.py`, `code/embed_verify.py`, `code/run_ridgebase_prompted.py`
and other historical entry points at their current paths.

Keep datasets, collaborator mappings, future splits/manifests, crops, minutiae,
checkpoints and raw runs local using the [workspace layout](../WORKSPACE_LAYOUT.md).
Publish reviewed code, documentation and aggregate reports. Existing published
manifests/results and SD302b, RidgeBase and paper assets remain intact; their
historical publication does not authorize exporting new identifying records.

## Implementation decisions and next task

The first milestone and general protocol no longer await initial professor
approval. Resolve dataset interpretation and exposure eligibility first, then
review the concrete subject allocation, comparison sampling, baseline arms,
development-selection metric, FAR target, failure policy and uncertainty method.
The selected DINOv2 implementation supplies the encoder/loss/preprocessing design,
frozen control and ground-truth supervision. Review its resource evidence and
concrete execution settings rather than treating implementation as answers to
the open data questions; teacher-score distillation is a separate extension.

After metadata clarification, prepare the concrete split/protocol for review
under an applicable task authorization. Retain the results/split review before
fine-tuning unless the supervisor explicitly revises it. The user requested
selected-matcher training/testing, but its capture/protocol/split review and
compute prerequisites remain incomplete. Segmentation, minutiae extraction and
other experiments still require their own decisions and task authorization.
Future private artifacts belong under ignored `local/precise/`; original
datasets, scores and fixed manifests are preserved.

This document records approval supplied by the user and the remaining
implementation work. No message has been sent to the professor or Konain.
