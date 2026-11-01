# DINOv2 within the LLM slap-matching study — plan for professor review

2026-11-01. The user supplied a new professor reply describing the dataframe
columns and requesting a DINOv2 plan **before experiments**. The professor says
DINOv2 may be a candidate while reaffirming the primary question: whether LLMs
can verify slap fingerprints, what limits them and how to improve reliability.
This document is a proposal for that review, not approval or an experiment.
The earlier user selection, implemented matcher and synthetic checks remain
preparation evidence. No real Precise fine-tuning has started.

## Research question and role of DINOv2

Can a multimodal LLM verify independent same-hand slap impressions of subjects
excluded from training? If performance is poor, which failures arise at image
input, visual representation, language output or decision calibration, and can
a controlled intervention improve them?

DINOv2 is a self-supervised **vision encoder**, not an LLM. Its
[official model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
and [paper](https://arxiv.org/abs/2304.07193) describe visual representations.
Use it as a visual control to establish how a general image encoder responds
to fingerprint-specific supervision. Its success would not establish LLM
matching ability, and its failure would not establish that LLMs cannot match.
Adding an LLM explanation of a DINOv2 decision also would not demonstrate that
the LLM performed or improved verification.

The proposed study therefore retains a direct multimodal-LLM arm. A text-only
LLM cannot be evaluated on raw slap images without a defined visual interface.
Propose one already-used family, `Qwen/Qwen3-VL-8B-Instruct`, as the initial
direct arm; its [official card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
describes an image/text generative model. Pin its exact checkpoint, processor,
prompt, quantization, decoding, environment and score extraction before any
authorized new run. This is a proposed arm, not a newly approved VLM experiment.

## How to use the dataframe

After the schema description, the user supplied a local GitLab folder containing
`CompleteWithRotated.P`, a pandas dataframe rather than an Excel workbook.
The [read-only intake](../../reports/precise-gitlab-dataframe-20261101/README.md)
verifies 26 matching columns and 24,530 Precise finger-annotation rows covering
all 7,358 local images. Field meanings below are reported by the professor;
structural consistency does not independently verify identity or acquisitions.

| Reported fields | Intended use and intake evidence | Boundary |
|---|---|---|
| `Dataset` | Restrict this study to Precise | Other datasets, including children, remain outside this milestone; verify actual value spellings |
| `SubjectID` | Check identity mapping and keep every image/derivative of a subject in one partition | Check consistency with existing filenames; never use identity as a model input |
| `Finger` | Reported codes 2–5 identify right-hand fingers, 7–10 left-hand fingers, 11 right thumb and 12 left thumb; complete finger sets agree with filename hand labels | Rows annotate fingers; do not count them as separate slap acquisitions |
| `Slap` | Join annotation rows to original slap images and group their finger annotations | Verify path/name mapping, uniqueness, repeated rows and whether augmented images have their own names |
| `augmented_angle`, rotated box sizes/centers/corners | Identify possible processing/rotation variants and their parent relationships | A rotation or another box is not a new independent impression; zero angle alone does not prove a native capture |
| Original box coordinates | Annotation geometry, potentially useful in later work | No crop, segmentation or per-finger model input is needed for this whole-slap stage |
| `difficult`, `dirty`, `moisture`, `overlap`, `contrast` | Check annotation coverage and definitions; propose development error strata if usable | `difficult` reportedly flags nfseg failure; it is not automatically an encoder failure or exclusion rule |

The original dataframe and its digest are preserved, with private provenance
and image mappings. Multiple finger annotations refer to each slap image.
There are 277 numeric SubjectID values versus 338 raw filename-prefix strings:
61 numeric groups have two leading-zero padding variants; all images agree
numerically with the dataframe. Confirm whether these are the same participants
before canonicalizing identities or defining a split. Preserve both labels.
Filter thumbs from this four-finger study without removing their source assets.
Keep identifiers, filenames and image-level labels out of hosted model inputs
and public reports. No segmentation, image modification or quality exclusion
was run here; annotation intake is separate from training preparation.

The listed columns do **not yet establish** a capture/session identifier,
parent-acquisition mapping, sensor, physical PPI or prior training/development
consumption. The earlier promise of resolution information is not a verified
PPI field in this schema. We still need a reliable way to distinguish repeated
independent captures from augmented/processed copies. That mapping may be in
another table or acquisition documentation; a filename-shaped timestamp is
insufficient evidence by itself.

## Proposed experiment sequence

1. **Review and metadata clarification.** Obtain professor feedback on this plan
   and confirm identity padding aliases. Resolve parent captures, remaining
   identity/hand issues and exposure/claim scope. The intake is not split approval.
2. **Freeze a shared protocol.** Use whole four-finger left/right slaps,
   same-hand comparisons and subject-disjoint train/development/test sets.
   Keep all fingers, hands and derivatives of each subject together. The
   [concrete protocol draft](precise_milestone_review_20261025.md) proposes
   seed `20261025`, approximately 60/20/20 allocation, reviewed independent-capture
   sampling and a primary development FAR target of 0.1%. The old 202/68/68
   scenario assumed 338 distinct subjects, now unestablished. If 277 numeric
   identities were confirmed and eligible, the existing rounding rule gives
   166/55/56, below the draft's 60-subject evaluation review condition. Revisit
   sizes/statistical support rather than allocating automatically. Review the
   eligible population and actual split before fine-tuning. Historical collection exposure permits an internal
   reanalysis claim, not an untouched external-validation claim.
3. **Compare the proposed arms below.** Construct one immutable attempted
   development/test comparison list for all methods after the preceding review.
   Preserve raw outcomes and failures. Historical scores are contextual unless
   their exact attempted population and protocol match.
4. **Diagnose on development only.** Predeclare a small set of within-VLM
   controls and their compute settings. Investigate why the direct arm fails
   before choosing an intervention. Freeze the selected method/settings/cutoff
   before final-test scoring; do not return to development after inspecting
   final-test errors to improve the reported confirmatory system.
5. **Review one targeted LLM improvement.** The initial candidate is a
   supervised adapter for the same VLM using identity-derived binary targets
   on training subjects, evaluated against its unchanged pretrained baseline.
   Choose whether to proceed based on development diagnostics and a separate
   concrete architecture/loss/resource proposal. Do not assume full VLM tuning
   fits because DINOv2 fits. An unsuccessful controlled intervention is still
   evidence; no improvement is promised.

| Proposed arm | What it tests | Current state |
|---|---|---|
| Pretrained Qwen3-VL, directly given two whole slaps and a fixed verification prompt | Direct multimodal-LLM verification and response/score reliability | Historical outputs exist on a different population; new common-list execution needs professor plan/protocol review and a pinned configuration/resource check |
| Frozen DINOv2-small | General visual representation without fingerprint adaptation | Implemented and synthetically tested; no real-data run |
| Adapted DINOv2-small | Whether task-specific visual supervision improves the same encoder | Implemented and synthetically tested; no real-data run |
| VeriFinger fusion and per-finger sum, if accessible on the new lists | Specialized matcher reference on identical attempts | Historical scores available; licensed SDK/access/configuration and new-list coverage remain conditional |

## Exact proposed DINOv2 use

Use `facebook/dinov2-small` at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`. Decode a whole slap to grayscale,
replicate channels, resize preserving aspect ratio and white-pad to 448×448;
no segmentation, augmentation or SDK-score filtering. Compare FP32
L2-normalized 384-dimensional class embeddings by cosine similarity.

Freeze the entire encoder for the control. For adaptation, train only its final
two transformer blocks and final layer norm with verified subject/hand/capture
supervision. Use same-hand microbatches of eight subjects with two independent
captures each. The proposed loss is the equally weighted mean positive cosine
distance plus mean negative cosine above margin 0.2. AdamW uses LR `1e-5`,
matrix weight decay `0.01`, BF16 and accumulation four, for a proposed 30 epochs
of 200 microbatches; development selects checkpoint and cutoff. Full settings,
failure handling and selection rules remain in the concrete draft for review.

The [GPU capacity check](../../reports/precise-gpu-runtime-20261101/README.md)
passes this synthetic workload on the RTX 4080 SUPER, peaking at 2.67 GiB
allocated / 3.37 GiB reserved. This establishes short-run compute fit only.
Synthetic tensors do not supply fingerprint results or experiment approval.

## Diagnosing LLM limitations without assuming the cause

The [saved-response audit](../../reports/response-audit-20260919/README.md)
already separates confirmed parser/output failures from model answers that
accept many impostors. It does not prove a visual or linguistic cause. Use
those existing findings without rerunning historical experiments.

| Hypothesis | Proposed development-only control | What it could establish |
|---|---|---|
| Input processing loses useful fingerprint detail | Within the same pinned VLM, compare the recorded 448-pixel route with one reviewed higher-detail/native-processor setting; log actual tensor/token dimensions and memory | Sensitivity to that processing choice; no general claim about ridge perception |
| Output format or decision bias harms verification | Validate image/template wiring, retain raw replies and stop reasons, use strict label-independent parsing; compare one fixed structured-answer prompt and one predeclared score route | Whether format/decision changes affect coverage and discrimination; output repair alone is not improved visual matching |
| The pretrained model lacks task-specific supervision | If reviewed and feasible, compare pretrained versus one supervised VLM adapter with shared train/development/test boundaries and prompt/score definition | Adaptation effect for that particular VLM and protocol |

Do not select a large prompt/model search on the small development cohort.
For each approved control, record what changed and keep other settings fixed
where possible. Define a continuous VLM score before execution (a generated
score with strict range/format checks, or a separately specified answer-logit
score); do not call either a calibrated probability. A binary-only output has
one operating point and cannot supply a meaningful full continuous-score ROC.
Any alternative score route is a declared method change, selected on development.

A DINOv2-versus-VLM difference confounds architecture, pretraining and scoring;
it does not isolate a language-module defect. A strong DINOv2 result would
motivate the targeted VLM investigation, not replace it. A later DINOv2/LLM
hybrid would require an explicit interface and an ablation against DINOv2 alone
to establish whether language contributes. Minutiae and generated explanations
remain separately reviewed extensions, not necessary props for a positive claim.

## Reporting and requested review

Report fixed-development-cutoff test TAR/FAR, valid-score ROC/DET/EER where
continuous scores exist, attempted/valid coverage and explicit failure counts,
with pooled and left/right results under the same pooled cutoff. Use identical
attempted pairs across methods. Never drop an SDK extraction failure or a VLM
malformed reply silently. Review subject-dependent uncertainty and low-FAR
support before making reliability claims; accuracy alone is insufficient.

Please review DINOv2's role as the frozen/adapted visual control, inclusion of
the direct VLM arm, the proposed diagnosis/intervention sequence and the
concrete subject-disjoint protocol. Please confirm the intended canonical
identity mapping for the padding variants and identify the parent-capture/session
mapping if it is stored separately. These are the next data inputs; the
annotation table does not approve a split or run.

No message has been sent on the user's behalf. This plan contains no new model
performance result and does not change original data, scores or the manuscript.
