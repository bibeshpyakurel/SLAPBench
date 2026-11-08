# Precise DINOv2 shared-benchmark handoff

**2026-11-08 latest intake/direction:** Konain's full before/after new-cohort
Qwen prediction files and checkpoint metadata have now arrived. Their
[exact-pair outcome review](../../reports/precise-qwen-supplied-results-20261108/README.md)
reproduces 63.31%/89.67% accuracy and retains the remaining actual-run provenance
gaps. The professor requested a larger DINOv2 model and original-resolution
images. A [concrete Large/native proposal](precise_large_native_protocol_20261108.md)
and [synthetic capacity evidence](../../reports/precise-native-large-capacity-20261108/README.md)
are ready; the user explicitly approved the exact new execution proposal. Do not repeat the completed
small-model runs or request the Qwen prediction files already supplied.

**2026-11-07 latest completion:** the user explicitly replaced the active cohort
with the newly supplied 37,194 training / 4,200 testing pairs and approved fresh
training, calibration and testing. Those runs and the
[new professor report](../../reports/precise-dinov2-numeric-holdout-20261107/README.md)
are complete. The new claim is numeric-group holdout with unconfirmed biological
identity/capture semantics. Matching fine-tuned Qwen outputs for the new cohort
remain unsupplied. Preserve both completed studies; the older prompt below is
historical and must not trigger another run of either benchmark.

**Later completion update (2026-11-01):** the subsequent end-to-end goal received
the user's concrete protocol/results/cohort review and execution authorization.
The approved fixed-pair DINOv2 training, calibration and final tests are now
complete. Start with the [completed results and professor report](../../reports/precise-dinov2-shared-20261101/README.md)
and [reviewed protocol](precise_shared_pair_protocol_20261101.md). The user
confirmed matching fine-tuned Qwen outputs are not ready. The original intake
handoff below is retained as history, not an instruction to repeat completed work.

Updated 2026-11-01 after the user's latest professor/Konain conversation and
authenticated Drive verification. This is the starting document for the next
chat/worktree. It supersedes the pending-plan status and division of work in
the earlier DINOv2/LLM proposal. No new experiment was run for this handoff.

## Latest direction and ownership

The professor explicitly approved DINOv2 and the proposed steps, asked Bibesh
to focus on the DINOv2 part, and asked Konain to share the exact training pairs
so the models use the same data. Konain handles Qwen3-VL evaluation/fine-tuning.
The research project still asks whether multimodal LLMs can match slaps;
Bibesh's frozen/adapted DINOv2 results supply the visual comparison.

Konain supplied a concrete balanced cohort, the A/B verification prompt,
evaluation code and fine-tuning code. His suggestion to reproduce Qwen locally
is a collaborator request to consider, not evidence that it has run here or a
requirement to independently duplicate his entire fine-tuning program.
The current user request is to inspect this material and prepare a handoff.

The immediate shared benchmark contains left/right four-finger slaps **and
two-thumb slaps**. Using exactly the supplied pairs therefore includes thumbs.
Do not silently apply the earlier thumb-exclusion rule or substitute a new
subject split and call it the same benchmark. The earlier subject-disjoint,
verified-independent-capture study remains a separate extension if requested.
Its capture requirements must not be presented as prerequisites for every
descriptive analysis or reproduction of the supplied fixed-pair benchmark.

## Sources now available

All three new folders were successfully listed through the connected Drive
account. Six relevant local files were compared byte-for-byte with fresh raw
Drive downloads, and all matched. No access question is pending for these files.

| Source | Contents and status |
|---|---|
| [Evaluation code](https://drive.google.com/drive/folders/11CFJ2FMmBe3Vh-ulSHUb8ClXN2oP0kUp) | `slap_actual_qwen3.py`, read as source, not imported or executed |
| [Testing/data folder](https://drive.google.com/drive/folders/1BveHqYTBQPqP9kQB0aPoeHNsNJXNsEzQ) | Two 2,100-row testing lists, older Qwen script/results, image archive, and nested source folders |
| [Fine-tuning code and JSON](https://drive.google.com/drive/folders/1k32Ez1BuZfbgBDiLC5rp6dWfMVTX2UG-) | `ft_qwen3_slap_binary_ab.py` and `slap_verification_binary_ab_13_14_15_balanced.json` |
| [Remaining-pairs pool](https://drive.google.com/drive/folders/1zKHNxcDheo55QXK6F5nRR2kMl6NfQO3E) | Now accessible; genuine source verified locally; 1,617,445,832-byte full impostor file listed but not downloaded/recounted in this task |
| Local GitLab dataframe | `Gitlab_SlapBench/dataset/dataframe/CompleteWithRotated.P`; already reconciled to all 7,358 source images |
| [FPBench](https://github.com/Ektagavas/FPBench) | Existing local reference at commit `ed27ae669947bfd157522d537a9f9c77b4bf3189`; use Konain's supplied binary-task code for his specific setup |

Private originals are at
`local/precise/collaborator/konain-followup-20261101/`, including
`context/genuine_filtered_minus2100.txt`. Source receipts are there; fresh
verification is recorded in
`local/precise/metadata/shared-cohort-20261101/drive_recheck_receipt.json`.
The full image ZIP was listed (5,679,745,916 bytes), not downloaded again;
the required images already resolve locally. Older result files in the folder
are not automatically outputs of the newly shared script or adapter.

## What the actual cohort establishes

The [aggregate audit](../../reports/precise-shared-cohort-20261101/README.md)
was reproduced exactly from the preserved originals; it is not a model result.

| Property | Training | Testing |
|---|---:|---:|
| Total pairs | 39,290 | 4,200 |
| Genuine / impostor | 19,645 / 19,645 | 2,100 / 2,100 |
| Right-hand genuine / impostor | 6,890 / 6,890 | 749 / 756 |
| Left-hand genuine / impostor | 6,591 / 6,591 | 691 / 671 |
| Two-thumb genuine / impostor | 6,164 / 6,164 | 660 / 673 |
| Missing image references | 0 | 0 |
| Duplicate unordered pairs | 0 | 0 |
| Numeric-identity label disagreements | 24 | 2 |

“Balanced” means equal genuine/impostor counts **within each slap type**;
the three types do not have equal totals. “Same 2100 pairs” refers to each of
two class files, hence 4,200 testing attempts. Both testing files are byte
identical to the prior local intake.

There is no exact training/testing pair overlap. However, all 4,869 testing
images and all 276 testing numeric identity groups also occur in training.
This is a held-out-pair benchmark with seen images/identities, not evidence of
generalization to unseen people. Do not create false subject-disjoint review
receipts to make it pass the older runner.

The JSON defines nonzero as **per-finger sum > 0**, not fused score > 0.
The complete positive-sum genuine source matches its 19,645 genuine pairs,
including source lines and scores. It retains 51 genuine and 1,213 impostor
pairs with zero fused scores. Preserve them; do not silently filter them out.
The older 19,648/78,592 description is historical, superseded for this supplied
cohort. Selection metadata reports seed 2026 and random impostor sampling
within each suffix; it is not a receipt of an actual completed training run.

The 24 training and 2 testing disagreements are provided impostor labels whose
filename prefixes differ but have the same numeric identity. Earlier dataframe
reconciliation established numeric agreement for every filename and 61 padding
alias groups. Preserve original labels and private review cases; confirm the
intended treatment before claiming corrected identity ground truth. No raw
identifiers, image filenames or case lists belong in public documents/chat.

## What the supplied Qwen code does

Evaluation uses `Qwen/Qwen3-VL-8B-Instruct`, its native processor, original PIL
images without custom resizing, BF16 on CUDA, automatic device mapping, greedy
generation and at most five new tokens. It asks A=same person or B=different
people using the prompt supplied in the conversation and JSON. The processor
still performs its own image processing. Model and package revisions are not
pinned in this script. It does not load a fine-tuned adapter.

The fallback parser returns lowercase matches for longer answers, while exact
single-letter responses become uppercase. Output files preserve parsed labels;
raw answers are printed to stdout. A controlled port should retain raw responses,
record parsing behavior, remap source paths locally and refuse overwrite.
Do not run the original at import time: it loads the model at module scope.
Binary answers support an operating point, not a continuous-score ROC/EER.

Fine-tuning uses Unsloth four-bit loading, rank/alpha 16 LoRA, assistant A/B
response-only supervision, and epoch checkpoints. Code defaults include three
epochs, learning rate 1e-4, batch size one, accumulation four and seed 3407.
Vision/language/attention flags and class limits are command-line choices;
defaults and docstring examples do **not** establish Konain's actual invocation.
The script has no development dataset or checkpoint-selection procedure.
Obtain his saved run configuration/environment/checkpoint selection and matching
outputs when needed for a trained-Qwen comparison; do not ask him again for
the training JSON, testing files, prompt or source code already provided.

## Local readiness and implementation gap

The DINOv2-small weights, preprocessing, cosine score, frozen/adapted encoders,
loss, training/checkpoint/resume, development selection, final-test validation
and descriptive comparison code already exist. The pinned revision is
`ed25f3a31f01632728cabb09d1542f84ab7b0056`.
The earlier synthetic RTX 4080 SUPER check passed at 448 pixels, batch 16,
BF16 and accumulation four (2.67 GiB peak allocated, 3.37 GiB reserved).
Driver 595.99.02 and CUDA were working at that check. Recheck availability when
a real job starts; do not reinstall dependencies or repair the driver by default.

The existing trainer creates all within-batch pairs using subject/capture
metadata and accepts only left/right four-finger hands. It cannot implement
“use exactly Konain's training pairs” unchanged. Add an explicit fixed-pair
dataset/loss/training route with supplied A/B targets and all three slap types;
retain the earlier strict subject-disjoint route. Do not infer labels from
numeric IDs or manufacture extra pairs in the shared-cohort route.

Before real training, settle a concrete checkpoint/threshold rule that never
selects on the 4,200 testing pairs. Options have different consequences: holding
development pairs out of the supplied training JSON changes training membership;
a fixed training schedule/final checkpoint preserves it but still needs a
declared decision-threshold rule. Record the choice and claim scope in the
results/protocol review. This is a remaining design decision, not missing data
that Konain must necessarily produce for us.

The DINO environment is
`local/precise/environments/matcher-candidate-20261025/bin/python`.
Verification uses `.venv/bin/python`. Qwen3-VL-8B was absent at its configured
local path in the last readiness check; the other downloaded VLMs are not
automatic substitutes. Running Konain's Qwen scripts would need its own pinned
environment/model and capacity check. That work is optional for Bibesh unless
the user explicitly includes local Qwen reproduction in the next goal.

## Next worktree and completion sequence

1. Start from the latest verified `origin/dev`, not default `main`. Read this
   handoff, AGENTS.md and CONTRIBUTING.md before acting. Preserve the current
   primary checkout; do not force a second `dev` checkout or bypass branch hooks.
   If a managed worktree is detached or on `codex/*`, reconcile its workflow with
   this repository's dev-only commit/push rule before publishing.
2. Git does not carry ignored assets. Resolve the primary checkout's absolute
   dataset, collaborator, dataframe, environment and weight paths explicitly.
   Keep original inputs read-only; place new manifests/checkpoints/runs in new
   dated ignored local paths. Never move the originals or duplicate model/image
   archives merely to initialize a worktree. Check path-based validator guards
   before assuming a symlink will work. A different computer needs authorized
   asset access separately; a Git checkout alone is insufficient.
3. Carry forward the existing audits and six verified input hashes. Resolve
   only the 26 label cases and the concrete selection/threshold policy needed
   for the planned claims. Record approval already given for DINOv2; do not
   ask the professor to approve the same broad plan again.
4. Implement and synthetically test the fixed-pair DINO route, run provenance,
   immutable outputs and fail-safe resume. Prepare a concrete reviewed run
   configuration, including thumbs, learning schedule and failure accounting.
5. Run frozen and adapted DINO on the same declared testing attempts after
   applicable results/protocol review and execution authorization. Preserve
   all failures and outputs; never tune on testing results.
6. Compare with supplied/reproduced Qwen outcomes only after verifying exact
   pairs, prompt, base/adapter identity, preprocessing and checkpoint. Compare
   VeriFinger fused/summed scores on the same attempts where available, with
   extraction/zero-score limitations explicit. Report pooled and slap-type
   metrics, coverage and the image/identity overlap limitation.
7. Agree and implement uncertainty accounting for reused images/subjects;
   report low-FAR limits honestly. Produce reviewed tables/plots, a professor
   report and reproducibility instructions. Manuscript editing and any separate
   subject-disjoint/capture study remain explicit extensions.
8. Review publishable changes, test, commit and push to origin/dev; verify
   remote SHA. The milestone is complete only after the defined real runs and
   report exist. No accuracy improvement or runtime fit of Qwen is guaranteed.

## Suggested next-chat goal prompt

```text
/goal Complete my Precise DINOv2 contribution end to end under the professor's
latest approved direction. Read docs/notes/precise_next_goal_handoff_20261101.md,
AGENTS.md, docs/CONTRIBUTING.md and the linked aggregate reports first.
Use Konain's supplied 39,290 balanced training pairs and 4,200 testing pairs
as the shared benchmark, including left hands, right hands and two thumbs.
Konain owns Qwen3-VL fine-tuning; my primary work is frozen/adapted DINOv2 and
an evidence-based comparison with his matching results. Do not duplicate his
VLM fine-tuning as an assumed requirement.
Continue from the verified inputs and implementation, inspect all supplied
evidence before asking questions, resolve the documented label and selection
issues, implement the fixed-pair route, and carry out the authorized reviewed
experiments, evaluation and professor-ready report. Keep the existing review
requirements; do not invent approvals, infer capture independence or claim
subject-disjoint testing for this shared cohort. Progress on independent
implementation while a genuinely necessary decision is pending.
Preserve all originals and private assets; publish reviewed code, aggregate
results and documentation to origin/dev with verified commits. Do not repeat
completed audits/installations or silently expand into another study.
```

The user can paste this after opening the next chat/worktree. No new chat,
worktree, automation, experiment or professor message was created by this
handoff task.
