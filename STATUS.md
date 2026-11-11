# Research status and audit log

## 2026-09-19 — approved documentation and offline-validation audit

The user approved the proposed documentation/validation plan. No experiment, API
call, model download or training run was performed. This is an evidence audit, not
a certification. Future entries must be appended rather than replacing this log.

### Scope and gap table

The following is an observed-capability table.

| Capability | Status | Evidence / limitation |
|---|---|---|
| Dataset inventory | Done for mounted data | REPO_MAP.md; three datasets under `datasets/` |
| Pair manifests | Present; validation partial | `results/{precise,ridgebase}/pairs_*_eval.csv`; coverage report below |
| Image-only VLM results | Partial | `results/*/*/latest/`; missing scores, invalid answers and unmatched IDs |
| Embedding comparisons | Present | `results/{precise,ridgebase}/*/embed_*_eval.csv`; not a conventional matcher baseline |
| Conventional matcher baseline | Missing evidence | No NBIS commands on PATH or `.xyt` under datasets; manuscript discusses baseline as future work |
| Subject split | Partial | RidgeBase Task2 Train/Test verified; no established development split for all experiments |
| Model/checkpoint provenance | Unclear | `models` target unavailable; repository loader definitions alone cannot establish trained weights |
| Persistent audit context | Done | REPO_MAP.md, STATUS.md, AGENTS.md and dated coverage report |

### Pair coverage and response integrity

Evidence: [coverage.json](reports/audit-20260919/coverage.json). Forty saved runs
were read: five models × embedding/three prompts × two datasets. Coverage is based
on exact pair-ID membership and duplicate counts, not just row counts. It does not
certify image contents or semantic correctness of parsed answers.

- Precise manifest: 676 genuine, 676 primary impostor, 338 diagnostic comparisons.
- RidgeBase manifest: 592 genuine, 592 primary impostor, 300 diagnostic comparisons.
- SD302b exhaustive manifest: 176 genuine and 7,656 impostors (`results/sd302b/task8_pairs_all.csv`).
- Precise Qwen3 similarity has 1,689 rows. Category totals include 676 genuine and
  676 primary impostors, but exact joining finds one unmatched stored ID, one absent
  genuine ID, and one absent diagnostic ID. The unmatched row has one unused
  manifest candidate under label/category/subjects/hands matching. No repair was made.
  **Correction to the initial review:** category counts alone did not establish
  complete ID-level coverage. Both 676-line text exports exist, but their full
  provenance/equivalence is not certified by this audit.
- RidgeBase Gemma zero-shot has one unmatched ID and one missing diagnostic ID;
  the same metadata matching does not identify a unique unused candidate.
- Precise Qwen2.5 zero-shot and RidgeBase Qwen3 zero-shot each miss a primary impostor.
- Pixtral similarity scores are blank for 688/1,690 Precise rows and 604/1,484
  RidgeBase rows. These are parser/missing-score findings, not automatically refusals.
- Pixtral INVALID binary responses: Precise 119 zero-shot / 216 task-description;
  RidgeBase 15 / 17. No literal `__ERROR__` rows were found in the 40 inspected runs.
- No duplicate IDs were found. Exact ID anomalies and missing scores require review
  before claims that all methods were evaluated on identical effective pairs.

### Seven integrity questions

1. **Continuous scores:** Similarity prompts store numeric scores, and embeddings
   store cosine values. Binary prompts retain decisions and raw text, not continuous
   confidence (`code/run_ridgebase_prompted.py`). Continuous-score ROC cannot be
   recovered from these binary decisions alone; no rerun is authorized here.
2. **Manifest schema/counts:** See above and coverage.json. Pair IDs, labels,
   subjects, hands and paths exist.
3. **Subject disjointness:** Using the existing filename parser in
   `code/ridgebase_pairs_full.py`, Task2 contactless Train has 63 subjects and Test
   has 25, with zero overlap. This covers parser-recognized PNG images, not all
   tasks/modalities, and does not establish a separate development set. Precise
   and SD302b development/test disjointness remains unverified.
4. **Same pairs everywhere:** Intended shared manifests exist, but effective
   evaluated subsets differ through missing IDs and scores. A conventional matcher
   output is absent from the evidence; baseline/VLM pair equality is unverified.
5. **Refusals/errors:** Raw text is saved; exceptions use `__ERROR__`, crash skips
   may be sidecars. There is no explicit refusal category. `parse_answer` accepts
   A/B anywhere inside words, and `parse_score_response` accepts the first integer
   and clamps it. These can misinterpret prose/errors. Existing scores were not
   reparsed or overwritten. The summary drops missing continuous scores and counts
   INVALID binary responses in denominators; it pools diagnostic and primary
   impostors for binary FAR. These limitations are now printed on new summaries.
6. **Provenance:** CSVs retain model keys, prompt strategy, timestamps, raw text and
   latency; builders use seeds in source. Exact hosted model revisions, prompt hashes,
   full software versions and preprocessing configurations are not complete per-run
   provenance records. Hosted call sites inspected send prompt text and encoded
   images; no runtime payload capture was performed to certify every call path.
7. **Low-FAR claims:** Inspected paper passages report FAR=0.1%, not 0.01%.
   The 592 and 676 primary-impostor sets have one-error resolutions of about 0.169%
   and 0.148%. Interpolation is not evidence of reliable measurement below these
   increments. With zero errors, the approximate 95% rule-of-three upper bounds
   would be 0.507% and 0.444%; for 7,656 trials about 0.0392%. These assume
   independent trials, while repeated subjects/images further limit inference.
   No claim of validated FAR=0.01% is supported by these counts alone.

### Changes and next steps

Implemented: factual map/context, exact-ID coverage report, corrected README
cross-resolution explanation, Pixtral inclusion, dataset-specific summary titles,
explicit provisional-summary caveats, and refusal to overwrite existing summaries.
New summaries in `reports/audit-20260919/{precise,ridgebase}/` are descriptive
recomputations from stored scores, not corrected benchmark results. Historical
`results/*/SUMMARY.md` and all source result files remain untouched.

Ranked follow-up (no experiments authorized by this entry):
1. Review unmatched IDs and export provenance; specify a versioned repair protocol.
2. Validate parsing against saved raw text and define refusal/error denominators;
   then produce separately versioned analysis with coverage alongside metrics.

Validation completed: repository lint (`ruff --select F,E9,B`), Python syntax,
known-score and empty-class metric fixtures, dataset-title/Pixtral checks,
byte-for-byte deterministic Precise summary regeneration in a temporary directory,
and overwrite-refusal checks all passed. The legacy verifier reproduced all 30
metric files. Git diff confirmed no tracked changes under `results/`, `datasets/`
or `manuscript/`. These checks do not resolve the research findings above.

## 2026-09-19 — repository layout consolidation (user-approved)

The user approved a restructure for a cleaner layout. No experiment, API call,
model run, or change to any result, manifest, or paper content was made. Paths
cited in earlier entries of this log are historical; REPO_MAP.md has current ones.

- Moved: `manuscript/`, `eccv2026_submission/`, `ECCV_2026_Paper_Template/` →
  `paper/{manuscript,eccv2026_submission,eccv2026_template}/`; root analysis scripts
  → `code/paper/`; `CONTRIBUTING.md`, `SECURITY.md` → `docs/`; planning notes and
  `plan /SlapBench_Plan.txt` → `docs/notes/`; `results_current/`,
  `results_matched/`, `results_matched_reversed/` →
  `results/sd302b/{current,matched,matched_reversed}/`. All via `git mv`.
- Removed duplicates: flat `results/<model>/` and `results/task8_pairs*.csv`
  (all 32 files byte-identical at tag `v0.1.0`; SD302b result CSVs/JSONs are also
  byte-identical under `results/sd302b/`), and `github/` (41 of 44 files identical
  at `v0.1.0`; an earlier README, `run_verification.py` and
  `generate_results_table.py` remain in history at `d780484`). Note: the deleted
  flat `task8_pairs.csv` differed from `results/sd302b/task8_pairs.csv` only in
  its image-path prefix (`dataset/` vs `datasets/`).
- Code: path constants updated (figures → `paper/manuscript/figures/`; moved
  scripts resolve the repository root three levels up); `verify_metrics.py` reads
  `results/sd302b/task8_pairs_all.csv`. Behaviour-preserving lint fixes in
  `code/paper/` (now covered by `ruff check code/`): an unused import removed,
  loop variables bound explicitly in closures, `zip(..., strict=False)`.
  Stale usage examples (`results_ridgebase/`, flat `results/`) corrected.

Validation: `verify_metrics.py` passed (manifest 7,832 = 176 + 7,656; all 15
distinct metric files reproduce; the earlier count of 30 included the flat
duplicates). The regenerated results table is byte-identical to the pre-change
one. `ruff check code/ --select F,E9,B`, Python syntax parse of `code/**`, and
`git diff --check` passed; each moved script's root and output paths resolve.
Figure scripts and inference were not executed (they need datasets/GPUs).

## 2026-09-19 — export and parsing audit

The user authorized validation of the Qwen3 exports and response parsing.

Evidence: `reports/response-audit-20260919/{README.md,audit.json,parsed/}` and
`code/audit_saved_responses.py`. All 49 source inputs have before/after SHA-256
checks; no source result was altered. No model or paid service was run.

Both Qwen3 files pass 676/676 filename-pair, score, order and raw-response checks.
The unmatched original ID contains 612 leading NUL bytes before an exact manifest
ID; derived records normalize that prefix with all identity metadata checked.
The missing diagnostic comparison is outside the exported populations. This is
file-consistency validation, not verification that image identity/acquisition or
model decisions are correct.

Across 45 CSVs / 165,087 rows, conservative parsing establishes 30 unambiguous
Pixtral A→B corrections (14 primary impostor, 7 diagnostic, 9 genuine). Missing
Pixtral scores (688 Precise, 604 RidgeBase) are fully consistent with its saved
malformed/empty/uncertain responses and the historical regex. There are no clear
scores to recover from those missing rows under the audit grammar. Other numeric
prose previously assigned scores is withheld, yielding only 145/1,690 and 105/1,484
clear Pixtral score responses. These subsets are unsuitable for unqualified
full-population comparisons. Detailed counts, old/new interpretations and raw-text
hashes are preserved beside the originals.

Qwen3 task-description primary-impostor acceptances remain 671/676 (Precise),
586/592 (RidgeBase), 7,571/7,656 (SD302b) with complete binary parsing coverage.
Thus parser errors explain some Pixtral false accepts, but not the general
same-person bias. No causal explanation of this bias is established.

Validation for this entry: six regression tests pass (including corrupted-ID
normalization, contradictory/error responses, detection of an altered export score,
and preserving existing output directories); CI now runs them. All 15 legacy
metric files reproduce, correctness lint passes, and source hash checks confirm
unchanged originals. The new parsed files are an explicitly post-hoc conservative
analysis; production inference parsers have not been changed. Any formal new run
needs a reviewed parser before execution.
Full offline regeneration in a temporary output directory exactly reproduced the
machine-readable report and all 45 derived CSVs. Staged whitespace checks passed.

## 2026-09-20 — Repository portability and artifact publication

The working branch now tracks the reviewed text run logs and two Pixtral
interrupted-run backups so a clone retains the research trail. These are
historical artifacts, not new validated comparisons. A content scan found no
common credential patterns or email addresses in those seven files; the logs
do contain local workstation paths. Datasets, model weights, credentials,
environments and caches remain excluded. See `docs/REPRODUCIBILITY.md` for
local asset locations, model IDs and the limits of clone-only reproduction.
The seven files' byte sizes and SHA-256 hashes are recorded in
`reports/artifact-inventory-20260920.csv`.

Model setup and SD302b inference now share local weight directory names and
support `SLAPBENCH_MODELS_DIR`. The setup utility also covers Gemma and caches
Pixtral by the repository ID used at inference. The local `models` symlink
currently points to an unmounted `/media/bibesh/DATA/models`, so weight contents
and revisions were not verifiable here. No dataset images, model weights, paid
calls, downloads or GPU inference were involved in this publication update.

### Publication boundary recheck

A fresh index/ignored-file audit found no uncommitted code, results or findings.
The local-only files outside the dataset tree are the Python environment,
model symlink, credentials, caches, third-party paper full texts and a LaTeX
font log. These are excluded deliberately; bibliography, code and research
notes are tracked. The local automation settings remain ignored, and a check
found no local archive credential in tracked files. A new index check
now rejects dataset/model/environment paths, credential files and files over
50 MiB before commit and in CI. Content review remains necessary for raw
fingerprint images and secrets embedded in otherwise publishable files.
Existing tracked pair manifests and scores contain subject identifiers and image
filenames, and manuscript figures include fingerprint examples; that publication
boundary needs a data-rights review before a new public release.

## 2026-09-27 — proposed verification materials removed

At the user's direction, the professor's proposed fingerprint/VLM verification
guide, the draft evaluation protocol built on it, and the single-finger feasibility
inventory made for that protocol were removed. Earlier entries were edited to drop
references to them; their findings about the existing benchmark are unchanged. The
removed files remain in Git history at `54bf2f8`. The SLAPBench benchmark code,
results, paper and the earlier four-finger plan (`docs/notes/SlapBench_Plan.txt`)
are retained. No experiment, model call or result change was involved.

## 2026-10-04 — Precise direction and offline preparation

The user supplied updated professor feedback choosing Precise for the next
study and requested background repository preparation while research decisions
are discussed. README, AGENTS.md, REPO_MAP.md and setup documentation now
identify Precise as the current direction. SD302b/RidgeBase datasets, code,
manifests, original scores and paper materials remain at their existing paths.
No historical claim or result was rewritten.

Synchronizing clean `dev` fast-forwarded from `cd61246` to `97ad07b`, including
the 2026-09-27 removal of the previous proposed guide/protocol. That cleanup is
preserved. A new dated note, `docs/notes/precise_preparation_20261004.md`, records
the confirmed dataset decision, questions for the professor/collaborator and
preparation that can proceed without fixing an architecture. The earlier guide
was read before synchronization; its §§4,8,9,11,16 inform the discussion, with
its historical location recorded in the note. No fine-tuning approval or
completed controlled baseline milestone is established by the new feedback.

New read-only tooling, `code/audit_precise_dataset.py`, inventories filename
labels and SHA-256 byte duplicates without model/image-decoding dependencies.
The aggregate report at `reports/precise-readiness-20261004/inventory.json`
contains no subject identifiers, filenames, per-image hashes or absolute paths.
It observes 7,358 JPEGs, 338 filename-labeled subjects, 4,907 four-finger slap
files (2,452 right / 2,455 left), and 2,451 thumb files. All 338 subjects have
both hands; all 676 slap subject/hand groups have multiple distinct byte
contents. No exact byte duplicates were found. Nine files use three filename
tokens; 7,349 use four. Middle tokens are not assumed to denote sessions.
Byte differences do not establish independent captures or prevent decoded-pixel
or near duplicates. Per-finger labels, acquisition/session metadata, image
resolution/quality and prior training/evaluation exposure remain unverified.

No image movement, segmentation, pair generation, split selection, NBIS build,
model download, GPU job, paid call or fine-tuning was performed. NBIS tools were
not found on PATH; VeriFinger access and the reported 95% result still need
protocol artifacts. Preparation does not imply authorization to run experiments.

Validation: all 17 repository unit tests pass, including four new inventory
contracts for aggregate privacy, byte-duplicate/label conflicts, preserving
existing outputs and redacting source paths on read failures. Correctness lint,
changed-Python syntax and whitespace checks pass. All 15 legacy metric files
reproduce; results-table regeneration in a temporary directory is byte-identical
to the historical table. Those checks do not validate new model performance or
resolve acquisition provenance. No historical result file was regenerated in place.

## 2026-10-04 — Local workspace organization and training-output exclusions

The user clarified that this directory is the primary workspace, future training
will execute locally, and the public repository should receive only appropriate
reviewed artifacts. At the user's request, empty preparation directories were
created under ignored `local/precise/`: `metadata/`, `splits/`, `manifests/`,
`crops/`, `minutiae/`, `checkpoints/` and `runs/`, plus a local README pointing to
`docs/WORKSPACE_LAYOUT.md`. No subjects were partitioned and no model, split,
manifest, crop or training output was generated. The existing runners continue
using their documented paths; these folders are storage preparation only.

The only relocated file was an ignored historical LaTeX font log:
`manuscript/missfont.log` to `local/legacy-build/manuscript/missfont.log`.
SHA-256 equality was checked before/after the rename. The now-empty root
`manuscript/` directory was removed. Original datasets, weights/symlink, scores,
manifests, run logs and paper sources were preserved. No duplicate `github/`
tree was created; `.github/` remains the CI/configuration directory.

`.gitignore` and the index publication guard now exclude `local/` and common
root-level training-output folders: `checkpoints/`, `runs/`, `wandb/`, `mlruns/`,
`lightning_logs/` and `tensorboard/`. These exclusions cover small text/config
files as well as weights, so default training outputs cannot be accidentally
published through these paths. Existing reviewed textual results and paper
figures remain publishable under the existing manual review policy. This is
not a content audit or removal of previously published subject information.

Workspace documentation, README, AGENTS.md, REPO_MAP.md and reproducibility notes
now record the local folder layout, paths to preserve and reviewed exports to
dated reports. Tests validate both Git ignore behavior and index rejection of
private workspace/training paths. All 18 unit tests and correctness lint pass.
No training, inference, paid call, model download or dataset preparation ran.

The index path/size guard, Git ignore checks, changed-Python syntax and whitespace
checks pass. All 15 historical metric files still reproduce. Existing tracked
raster images were checked by path and all remain under `paper/`; this is not
a new image-content or data-rights audit. An unrelated concurrent edit to
`code/model_registry.py` appeared during verification and was left untouched and
excluded from this task's staging.

## 2026-10-04 — authorized local-model downloads and setup

The user explicitly authorized downloading Qwen3.5-9B, Gemma 4 12B IT and
InternVL3.5-8B and preparing them for this project. All three official snapshots
were downloaded at pinned revisions to
`/home/bibesh/.local/share/slapbench/models/`: 60,353,970,909 bytes across 49 files.
Every snapshot file passed remote-metadata size and digest verification. Model
weights and machine-local configuration remain excluded from Git. The existing
external-drive `models` symlink and historical environment are preserved.

Evidence: `reports/model-setup-20261004/{snapshots,preflight,environment}.json`.
The new setup utility selects only the authorized additions, verifies immutable
snapshot digests and refuses to overwrite reports. Separate local storage is
recorded in an ignored config; the old default download set remains unchanged.
The runners now recognize `qwen35`, `gemma4` and `internvl35` through local-only
4-bit loaders. Qwen/Gemma use a separate Transformers 5.18.0 environment;
InternVL's original custom model code was inspected and checked with 4.57.6.

Offline preflight loaded configurations/tokenizers and constructed all three
architectures on the meta device without checkpoint weights. Qwen and Gemma
accepted two synthetic 448px images through their native processors. These
checks establish setup compatibility, not model predictions or fingerprint
performance. NVIDIA device nodes are absent and CUDA is unavailable, so GPU
weight loading, generation, memory headroom and runtime quantization remain
unverified. No driver changes or fingerprint experiments were performed.

Validation: all 26 offline unit tests pass, correctness lint and Python syntax
checks pass, dependency checks pass for the new environment, and all 15
historical metric files still reproduce. The results table regenerates. No
original results, manifests, datasets or paper sources were changed. Formal
experiments still require the pending protocol/parser review and separate run
authorization.

## 2026-10-10 — four Precise background tasks completed

The user authorized a research question, an educational pipeline explanation,
a small read-only image inspection and a subject-disjoint protocol sketch after
sharing Konain's update. The discussion brief is
`docs/notes/precise_research_brief_20261010.md`; it links the worked-example primer,
protocol sketch and aggregate report. Model contribution, whole-slap versus
per-finger processing, split sizes/subjects, capture provenance and first
experiment remain joint decisions rather than implemented choices.

Konain reports full-slap VeriFinger fusion and per-finger sums. His supplied
summary shows 2,057/2,100 genuine accepts and 0/2,100 impostor accepts for both
score types, while his message names 0.1% FAR and the screenshot names 0.01%.
These are collaborator observations, not independently reproduced results.
Raw comparisons, threshold selection and subject/image exposure are still
unverified. The notes distinguish exact-pair exclusion from subject separation
and explain empirical FAR resolution and the assumptions of zero-error bounds.

New `code/inspect_precise_images.py` inventories all 7,358 JPEG headers and fully
decodes a deterministic 35-image inspection sample. All headers opened as
1600×1500 grayscale, with no DPI or recognized EXIF orientation tags. All sample
images decoded, their before/after source-byte hashes agree, and no exact
canonical grayscale duplicates occur within that sample. Native acquisition
PPI, independent impressions, session meanings and dataset-wide decoded/near
duplicates remain unverified. Visual review of all 35 thumbnails and three
native-size images records placement, contrast, ridge-continuity and contact-area
variation without making biometric-quality or identity judgments.

Public aggregates and the method are at `reports/precise-inspection-20261010/`.
Identifying records, contact sheets, review copies and notes stay under ignored
`local/precise/metadata/inspection-20261010/`. No source images, existing scores,
manifests or manuscript assets were changed. No actual subject split, new
matching pairs, segmentation, minutiae extraction, training or matcher inference
was performed. README, repository map, workspace layout and reproducibility
instructions now link the completed preparation.

Validation: all 32 offline unit tests pass, including six synthetic-image tests
covering source preservation, private output, failure counts, deterministic
sampling, output boundaries and overwrite refusal. Correctness lint, changed
Python syntax, whitespace, documentation links, calculations and publication
content checks pass. The current inspection script reproduces both the saved
aggregate and serialized private records exactly. All 15 legacy metric files
still reproduce; the results table regenerates identically in a temporary
directory. These checks do not validate Konain's unavailable raw scores or
establish learned-matcher performance.

## 2026-10-11 — collaborator update and local document discovery

Konain's update confirms VeriFinger-score supervision and says his training
pair files will be uploaded. The user supplied two Downloads score files and
then authorized searching documents on this computer. The supplied files and
matching Desktop copies are byte-identical to the two existing 676-record
Precise/Qwen3 exports. Six copies were checked against the repository sources;
all match and remain byte-identical after inspection. These records do not
identify Konain's reported 2,100 genuine/2,100 impostor VeriFinger population.

Read-only discovery inspected 637 document/archive files under the user home
and accessible media/mount locations, including 24 PDFs with extractable text,
two Word documents and two archives. A further 259 log/configuration/backup/
extensionless candidates yielded 244 readable text files; six credential/history
and nine binary candidates were excluded. Structure checks reviewed 495 text/data
files. No reported SDK testing score table, collaborator training pair list or
adapted fine-tuning script was located in this scope. A local Word copy of the
student guide was found and left local; it is methodological material, not the
missing pair/score data. Document instructions were not treated as task commands.

Two media locations were permission-restricted. Runtime environments, caches,
model/checkpoint trees and program/security state were excluded; symlinks were
not followed. Other user accounts, unmounted storage, remote shares and image
OCR were outside this search. The negative result is not proof that files were
never shared or are absent from all storage. Detailed path/content inventories
stay under ignored `local/precise/metadata/discovery-20261011/`; only aggregates
and reviewed findings are exported at
`reports/precise-collaborator-discovery-20261011/`.

The linked FPBench Qwen3 reference was inspected at revision
`ed27ae669947bfd157522d537a9f9c77b4bf3189`. It uses LoRA supervised answer-text
training with a pattern-classification default. It does not establish Konain's
actual Precise target encoding, split, loss adaptations or threshold procedure.
The report includes a ready-to-send request for the testing files and training
pair mappings; no external message was sent. Training/testing overlap and SDK
metrics cannot yet be audited from the unavailable records. No original data,
results, manifests or models were changed, and no new experiments were run.

A direct recursive check of `datasets/Precise/`, requested during discovery,
finds exactly 7,358 JPEGs and no non-image files or subdirectories. Filename
labels identify 338 subjects and 2,452 right/2,455 left/2,451 thumb slaps. The
folder contains no SDK scores, subject partitions, pair lists or separate
acquisition metadata. This confirms image availability without filling the
collaborator-record gap. Aggregate/private consistency, document-link,
publication-content and whitespace checks pass; all 15 legacy metric files
still reproduce. No executable repository code was changed in this task.

## 2026-10-11 — Drive intake and VeriFinger testing-score audit

The user's two Drive links provide the reported 2,100 genuine and 2,100
impostor testing comparisons, including fused scores, per-finger scores, sums
and finger counts. Original downloads, metadata, hashes and identifying audit
records remain under ignored `local/precise/metadata/verifinger-intake-20261011/`.
The new standard-library utility `code/audit_verifinger_scores.py` exports only
reviewed aggregates and discrete ROC/DET coordinates to
`reports/precise-verifinger-audit-20261011/` and refuses existing output paths.

All 4,200 rows parse with consistent sums/counts and filename-derived classes.
All referenced basenames exist locally; no self-pairs, duplicate unordered pairs
or cross-class pair overlaps were found. The screenshot is reproduced for both
methods: 2,057/2,100 genuine accepts (97.9524% TAR), zero impostor accepts,
thresholds 48 fused and 79 summed. Those thresholds are evaluation-selected
descriptive points. At target 0.1% FAR, thresholds 39/72 yield the same genuine
accept count and 1/2 impostor accepts respectively. Neither observation
establishes prospective low-FAR performance.

Testing covers 337 filename subjects and 4,869 image basenames, and mixes
four-finger and two-thumb slaps. The four-finger slice contains 1,440 genuine
and 1,427 impostor comparisons; both methods accept 1,412 genuine comparisons
(98.0556%) at zero observed false accepts using slice-specific thresholds.
Only 60 genuine and zero impostor pairs overlap our existing primary 676+676
evaluation, so prior VLM scores do not support a full common-pair comparison.
The report includes a revised collaborator-request draft; no message was sent.

Training pair mappings, adapted targets/script, SDK settings, acquisition
independence and threshold-development provenance remain unavailable. Exact
test-pair exclusion alone does not establish image- or subject-disjointness.
No dataset, original result, fixed manifest or model was altered; no new pairs,
splits, segmentation, matcher inference, paid calls or training were performed.

Validation: all 40 offline unit tests pass, including eight synthetic score-audit
tests. Saved aggregate/private records reproduce exactly; all ten ROC/AUC
calculations agree with scikit-learn, and all 30 operating-point counts are
independently verified. Raw source hashes remain unchanged. Correctness lint,
Python syntax, whitespace and publication checks pass; all 15 legacy metric
files still reproduce. The report documents the audit's limits and private-input
reproduction requirements.

## 2026-10-11 — proposed learned-matcher first milestone

The user's interpretation and proposed sequence are recorded in
`docs/notes/precise_first_milestone_proposal_20261011.md`: adapt a vision encoder
to learn continuous fingerprint similarity on whole four-finger slaps, establish
subject-disjoint evaluation before execution, then assess minutiae and optional
matcher explanations. The note distinguishes DeepPrint's domain-informed
representation from the proposed image-only baseline and SuperGlue's general
local-feature correspondence from an unvalidated fingerprint adaptation.
The two earlier DeepPrint links now point to the correct supplied paper.

Dataset interpretation is the immediate collaborator priority: middle filename
tokens, independently acquired genuine impressions and acquisition/session
metadata. Training/exposure mappings remain important for protocol eligibility;
our architecture can be designed independently of the collaborator's script.
The note links the completed VeriFinger audit, separates four-finger and thumb
populations and records that a sufficiently sized untouched subject cohort has
not been established. It supplies explicit professor-review decisions without
claiming milestone approval or selecting a model, loss, split or threshold.
README and earlier preparation/protocol/brief notes link the latest proposal.

Validation: documentation links, Markdown fences, saved aggregate figures,
whitespace and staged publication scope/size/content review pass. Changes are
documentation only; no executable code, original results, manifests, datasets
or manuscript assets were modified. No split/pair creation, segmentation,
minutiae extraction, experiment, training, download or external message occurred.
Results/split review before fine-tuning and separate task authorization remain
in effect pending supervisor decisions.

## 2026-10-11 — professor approval recorded

The user confirmed “professor approved this” following the first-milestone
proposal. The approved direction is an image-only learned similarity matcher
on whole four-finger Precise slaps, with subject-disjoint evaluation and
development-selected checkpoints/thresholds, followed by minutiae and optional
matcher explanations. The milestone note now records approval, and AGENTS.md,
README, REPO_MAP.md and the earlier preparation/brief/protocol notes distinguish
this current state from their historical pending-approval statements.

Approval establishes the direction and protocol principles, without supplying
missing acquisition/session facts or selecting unspecified implementation
parameters. Dataset interpretation, capture independence, training/evaluation
exposure eligibility, concrete split, encoder/loss, FAR/uncertainty and failure
policies remain to be resolved. Concrete results/split review before fine-tuning
and applicable execution authorization are retained. No subject assignments,
pairs, segmentation, inference, training or external messages were performed.

Validation: documentation links, approval-state consistency, whitespace and
staged publication path/size/content checks pass. This is a documentation and
agent-guidance update; executable code, datasets, original scores/manifests,
model files and manuscript assets are preserved.

## 2026-10-11 — Precise documentation and navigation cleanup

README now leads with the current Precise direction, completed preparation and
first-milestone record; the earlier SLAPBench description and reproduction setup
are explicitly historical. The existing milestone document is the central
research plan, linking the architecture sequence, detailed protocol safeguards,
inventory/image inspection, collaborator discovery and completed VeriFinger
testing-score audit. The original filename is retained for link continuity.
This cleanup preserves the separately recorded professor approval of the
direction and protocol principles while keeping concrete decisions and execution
authorization pending.

REPO_MAP.md, the protocol sketch, research brief, workspace layout and reproduction
navigation now reflect the available, audited SDK testing exports. They retain
the distinction between retrospective evaluation-selected thresholds and a
development-fixed prospective protocol, separate thumbs from four-finger slaps,
and preserve the mismatch with the earlier VLM comparison population. Capture
independence, training/development exposure and SDK settings remain unresolved.
Dated reports and prior status entries remain historical records.

The central plan and workspace layout specify a future `code/precise_matching/`
package with data, encoder, verification-loss and evaluation modules. No package
is created or implemented by this task. Datasets, collaborator mappings, future
splits/manifests, minutiae, checkpoints and raw runs remain local; reviewed code,
documentation and aggregate reports are the public outputs. Existing scripts,
SD302b/RidgeBase assets, original results/manifests and paper files are unchanged.

Validation: local documentation links/anchors, Markdown fences, saved aggregate
counts/percentages/thresholds/overlap, whitespace and staged publication
path/size/content review pass. Changes are Markdown only; no analysis code or
result data changed, so experiment reruns and training were not needed. No split,
pair creation, segmentation, minutiae extraction, downloads or external messages
were performed.

## 2026-10-17 — offline Precise matcher protocol and evaluation implementation

The user requested progress toward the approved plan and clarified that
Konain's testing-file links remain his latest update; no new capture/session
definitions or exposure records have been supplied. The new standard-library
`code/precise_matching/` package implements private metadata/comparison
validation and evaluation of already generated continuous scores. It does
not construct a real split, generate pairs, load images, choose an encoder/loss,
run a matcher or train a model. Historical scripts and original assets remain
unchanged.

The validator checks subject separation across every listed image, whole-slap
hand compatibility, capture ownership, supplied raw/decoded hashes, labels,
self-pairs and duplicate unordered image/capture comparisons. Missing or
unreviewed capture/exposure evidence is reported as aggregate blockers, not
guessed from filename tokens. Metadata assertions and hashes are compared as
supplied; passing checks does not independently verify acquisition, near
duplicates, exposure history or training readiness.

The evaluator requires one score/status record per development/test comparison.
It selects a strict-greater-than cutoff from valid development impostors using
an exact rational FAR budget and applies it unchanged to test, rejecting ties
that exceed the budget. Reports include fixed-cutoff TAR/FAR, denominators,
coverage, failure statuses and descriptive valid-score ROC/DET/AUC/interpolated
EER. Failed attempts remain visible in an additional failure-as-reject view;
they do not relax the development budget. No independent-trial confidence
interval or population low-FAR claim is supplied.

The CLI writes only reviewed aggregate JSON under new `reports/` filenames,
refuses overwrites and resolved paths outside that tree, checks input-byte
stability and suppresses identifying input/error text. Private input contracts,
usage, limitations, remaining decisions and a short unsent Konain-request draft
are in `docs/notes/precise_matching_tooling_20261017.md`. The central plan,
README, repository map and workspace layout link the implemented tooling;
encoder/loss/training remain future work. Existing CI discovers its tests and
now parses Python syntax recursively, including the matcher package.

Validation: all 67 offline unit tests pass, including 27 new synthetic protocol,
score and CLI tests. End-to-end fixtures verify source preservation, aggregate
privacy, explicit blockers and a test FAR that changes without retuning the
development cutoff. All 300 deterministic synthetic ROC/AUC cases agree with
scikit-learn; 1,800 cutoff choices agree with exhaustive error-budget checks.
All 15 legacy metric files still reproduce, and the results table regenerates
byte-identically in temporary storage. Correctness lint, Python syntax, CI YAML,
documentation links/fences, whitespace and publication scope/size/content checks
pass. Only synthetic metadata/scores were processed by the new package; no
dataset, original result, manifest, model or manuscript file was changed.


## 2026-10-25 — complete collaborator training-list intake and FPBench reference

The intake begun on 2026-10-24 is organized under ignored
`local/precise/metadata/konain-intake-20261024/`: 23,103 genuine and 78,592
impostor training records, plus 2,100 genuine and 2,100 impostor testing records.
A fresh authenticated download of the supplied Drive link is byte-identical to
the previously preserved impostor test input. Original files remain unchanged;
private receipts retain source links, hashes and identifying overlap evidence.

The new standard-library `code/audit_precise_training_pairs.py` inventories the
two observed score formats and compares unordered pairs, image basenames and
filename subjects. No exact testing pair occurs in training, including reversed
pairs. However, all 4,869 testing images and all 337 testing subjects occur in
the supplied training lists. For four-finger slaps alone, all 3,283 testing
images and 330 subjects are shared. This confirms pair exclusion, not the
approved subject-disjoint protocol or actual consumption by a training job.
All supplied training impostors are right-hand only; 3,455 genuine training
rows have fused score zero without per-finger fields. Missing fields remain
unknown and zero is not converted into a failure or nonmatch target.

The complete FPBench reference checkout is preserved under ignored
`local/precise/references/FPBench/`, detached at
`ed27ae669947bfd157522d537a9f9c77b4bf3189`. Static inspection is documented in
`docs/notes/fpbench_reference_20261024.md`; the published answer-text fine-tuning
script does not contain Konain's Precise-specific binary conversion or a
subject-disjoint development protocol. His current binary fine-tuning is
collaborator-reported, not locally verified. The approved embedding-matcher
milestone remains unchanged. Navigation, the central plan and reproduction
instructions now distinguish supplied training membership from actual exposure.

Validation: all 75 offline unit tests pass, including eight new synthetic
parser, overlap, source-preservation and output-boundary tests. Correctness
lint passes and all 15 legacy metric files reproduce. An independent parser
confirms the aggregate pair/image/subject overlap. The pinned reference passes
Git object verification. Source records, original results and historical assets
are preserved. No new split, pair generation, model execution, fine-tuning,
paid API call or external message was performed. Public outputs contain code,
documentation and aggregate counts; private inputs and third-party assets stay
ignored. Final syntax, documentation-link, whitespace and staged publication
scope/content/size checks are completed before commit.

## 2026-10-25 — full-image readiness audit and concrete milestone review draft

The user supplied the ten-task first-milestone roadmap and confirmed that no
new acquisition/exposure clarification from Konain is available. The new
`code/audit_precise_readiness.py` performs strict full-image decoding, native
and grayscale decoded-content hashing, heuristic derivative screening and
private per-image capture/exposure bookkeeping without constructing a split or
experimental comparison list. The aggregate report is at
`reports/precise-readiness-20261025/`; identifying records and visual-review
assets remain under ignored `local/precise/metadata/readiness-20261025/`.

All 7,358 JPEGs decoded, all remain 1600×1500 grayscale, no constant images or
exact byte/native/grayscale duplicate groups were found, and all image bytes
plus six metadata inputs were rehashed unchanged. The coarse dHash/thumbnail
screen flags 2,685 nonidentical file relationships: 2,617 thumb, 30 right-slap
and 38 left-slap. Of these, 2,497 cross subject labels, including 30 four-finger
relationships. These are candidates, not confirmed duplicates or labeling
conflicts. A targeted private visual review of 16 relationships finds visible
impression/layout/contrast differences despite coarse similarity. It does not
adjudicate identity or acquisition independence; no exclusion or label change
was made. The complete candidate set and unknown capture/session/exposure fields
remain available for collaborator review.

The genuine training list contains all local images/subjects; the two historical
Precise comparison manifests together reference all 4,907 four-finger images and
all 338 subjects. Membership remains distinct from actual model consumption.
Right-only training impostors, 3,455 genuine zero-score rows missing component
fields, parent acquisitions, actual training/development use and SDK settings
remain unresolved; no scores or missing fields were recoded.

`docs/notes/precise_milestone_review_20261025.md` makes protocol/model decisions
reviewable: conditional 202/68/68 subject counts, seed/sampling, 0.1% FAR,
development selection/failure rules, proposed dependence-aware uncertainty and
a DINOv2-small candidate pinned to `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
It specifies frozen/adapted components, 384-dimensional cosine embeddings,
whole-image preprocessing, a supervised cosine contrastive loss and resource/
implementation design. These are proposals, not approved configuration or new
performance evidence. The central plan and navigation link the report/design.

Read-only compute inspection again finds the RTX 4080 SUPER on PCI but no usable
NVIDIA driver/device nodes; PyTorch CUDA is unavailable. Existing `.venv/`
packages, CUDA build and RAM/disk were recorded privately; the external `models`
target is unavailable. Public checkpoint metadata only was retrieved. Isolated
matcher dependency setup, runtime compatibility and synthetic real-model checks
remain pending. No system repair, weight download, real model instantiation,
new subject split, matching-list generation, segmentation, training, paid call,
external message or manuscript edit was performed. Later roadmap tasks remain
behind acquisition/exposure, protocol/results/split and execution review gates.

Validation: 82 offline tests pass, including seven new synthetic decode,
duplicate/hash-retrieval, source-change, malformed-history, privacy and output
boundary tests. All 15 legacy metrics reproduce; correctness lint, recursive
Python syntax, changed documentation links/fences, whitespace and publication
scope/content/size review pass. The results-table generator runs in temporary
storage without altering its original output. Original datasets, collaborator
lists, results/manifests, model files and paper assets are preserved.

## 2026-10-25 — isolated candidate matcher dependencies prepared

Continuing roadmap task 4, prepared a fresh Python 3.12.3 environment under
ignored `local/precise/environments/matcher-candidate-20261025/`. Dependency
preparation precedes the separately gated model-loading/forward/backward check
and does not approve the proposed model/protocol. The new
`code/precise_matching/environment/requirements.in` records direct candidate
choices; its Linux x86_64 / CPython 3.12 / CUDA 12.4 lock fixes all 50 runtime
wheel URLs and SHA-256 digests. Installation enforces hashes and binary wheels
with no package-index lookup. Package source hosts are reviewed PyPI, PyTorch
and NVIDIA endpoints; no checkpoint or dataset input was downloaded.

All 50 installed runtime versions and installation-report hashes match the
resolution. `pip check` passes. Isolated-mode, network-disabled imports of the
selected libraries and DINOv2 model/configuration classes succeed, with module
paths inside the new environment. No model object was created or forward/
backward executed. Setuptools' 11 import-visible vendored metadata entries were
verified inside its pinned wheel and distinguished from installed wheel count.
The historical `.venv/` package inventory is unchanged. Private logs/resolver/
installation/version/validation receipts stay under
`local/precise/metadata/environment-preparation-20261025/`; public aggregates
and exact restore commands are in `reports/precise-environment-20261025/`.

The environment occupies about 5.3 GiB. PyTorch includes CUDA 12.4 support, but
CUDA remains unavailable with zero devices, so usable driver/runtime compatibility,
device memory and model execution are not verified. System repair, checkpoint
access, model/protocol acceptance, acquisition/exposure clarification and later
results/split/execution review gates remain pending. Current reproduction,
workspace and review-plan notes link this follow-up; earlier audit reports
remain historical snapshots. No original dataset, result, manifest, model or
paper asset changed.

Validation: hash-enforced install, complete lock/version/hash agreement,
dependency consistency, offline imports, isolation and historical-environment
preservation pass. Documentation links/fences, JSON, whitespace and publication
scope/content/size are reviewed before commit. This change adds dependency and
documentation artifacts only; model execution remains unverified.


## 2026-10-25 — parallel source investigation and evidence handoff

The user assigned the ten-task first milestone to another worktree and asked
for parallel investigation of the upstream reference, Precise packages and
shared Drive material. This task adds a dedicated aggregate report at
`reports/precise-source-evidence-20261025/` and an implementation handoff at
`docs/notes/precise_parallel_source_brief_20261025.md`, without changing the
parallel readiness audit, matcher code, original lists or proposed protocol.

The 23,103 training and 2,100 testing genuine records together cover all 25,203
possible unordered distinct-file pairs within local subject/slap groups:
8,393 right, 8,418 left and 8,392 thumb pairs. Known consistent endpoints,
uniqueness and combinatorial capacity establish coverage without constructing
new pairs. This describes supplied membership, not the original generator,
independent captures, actual consumption or binary targets.

All 7,358 penultimate filename fields parse under two candidate calendar/time
formats (3,060 long-form and 4,298 short-form). All 1,014 subject/slap groups
have distinct candidate times; 12 span candidate calendar dates. Only 41 of
the 2,100 genuine test rows and 679 genuine training rows cross candidate dates.
Acquisition/export/processing meaning remains unverified; no session or capture
IDs were assigned. A private candidate mapping marks every interpretation false
and narrows the remaining acquisition question without manufacturing provenance.

Both local Precise ZIP central directories contain 7,358 JPEG files and no
metadata documents. Their path/CRC/size/compressed-size maps agree, and image
names/sizes match the dataset; full archive/image byte equality is not claimed.
All 7,358 JPEG headers expose only JFIF fields through Pillow, with no EXIF
tags, nonempty comments or header errors; no pixels were decoded by this probe.
The recommended upstream FPBench checkout/history remains at the pinned
revision. Its four reachable commits contain no Precise/binary/VeriFinger
keyword hits in inspected code/docs/JSON/YAML; its matching training/evaluation
schemas remain multi-image, four-option MCQs. The repository has one advertised
branch, no advertised tags, one artifact-release issue and no pull requests at
inspection. This is the repository Konain recommended, not an identified copy
of his private adaptation.

The upstream checkpoint Drive link returned 404 in the connected account.
Both explicitly shared test files remain directly accessible, with no comments
or exposed parent folder. Scoped Drive searches and six historical project
planning documents did not locate verified acquisition definitions or the
Precise binary conversion/consumption record. Search absence does not prove
absence; the directly accessible genuine test file did not surface in search.
Private receipts retain source IDs/dates/text, scope and access outcomes.
Historical document instructions were treated as evidence, not authorization;
the removed student guide was not restored to the active public tree.

Validation: independent parsing recomputes genuine-pair coverage and candidate
time counts; one-off private-script syntax, public-image-name/private-document-ID
privacy scan, local document links and whitespace pass. All 15 legacy metric
files reproduce. Staged publication scope, size and content are reviewed before
commit. No new subject assignment, comparison list, image extraction, model
execution, dependency/weight download, segmentation, training, paid call,
sharing change or external message was performed. Source-evidence work can
proceed alongside implementation; model/loss/FAR decisions still require the
project's review rather than inference from the collaborator's materials.

## 2026-10-25 — selected DINOv2 implementation and guarded training pipeline

The scheduled 9:30 PM Chicago resumption inspected the current checkout and
verified newer human decisions in the project's “Newer Models” chat: the user
selected the proposed DINOv2 matcher first and requested training/testing.
That chat ended at a usage limit after leaving encoder/image/loss modules and
a successful small CPU probe locally. This work continued those artifacts,
preserved the original CPU report bytes and recorded the updated authorization
scope. Model choice no longer needs another confirmation; capture/exposure,
protocol and concrete results/split review still do. AGENTS.md now records
that distinction without removing the research-integrity gates.

The selected local checkpoint's exact safetensors/configuration digests are
verified. Strict whole-image loading now checks original and decoded bytes,
letterboxes to 448×448 without cropping, and supplies the shared frozen/adapted
DINOv2 encoder. Normalized continuous cosine scoring and the balanced supervised
verification loss are implemented. The original CPU probe used four random
56px inputs and two optimizer updates; it showed trainable changes, frozen
parameter preservation and exact model/optimizer checkpoint restoration. It is
not Precise training, a full-resolution fit check or GPU evidence.

The new training entry point requires digest-bound private review assertions
and a metadata-consistent subject-disjoint protocol before model loading.
Training/development image paths exclude test inputs; the CLI has no test-score
or final-test scoring input. It implements same-hand training-only sampling,
AdamW, accumulation/clipping, warmup/cosine scheduling, deterministic seeds,
development-only threshold/checkpoint selection and immutable epoch checkpoints.
Resume restores optimizer/scheduler and Python/NumPy/PyTorch/CUDA/sampler states,
rejects changed inputs/configuration/provenance or corrupted files, and preserves
all earlier and failed attempts. Source/model/environment/input fingerprints,
development scores, losses, selection cutoffs and timestamps remain in private
run records. No real private run or review receipt was generated here.

Synthetic tests verify that resumed and uninterrupted weights and development
history match exactly even with dropout; test images and test-score rows never
enter training/selection. They cover canonical capture eligibility, decoded
hashes, loss/score contracts, failed attempts, changed inputs and checkpoint
corruption. A dedicated CPU CI job executes the Torch tests without checkpoint
downloads or private data. All 103 local tests pass, correctness lint passes,
and all 15 legacy published metric files reproduce. Final recursive syntax,
configuration/CI YAML, documentation links, unchanged table generation,
whitespace and staged privacy/publication reviews precede commit.

The implementation guide is `docs/notes/precise_training_implementation_20261025.md`;
the original CPU evidence and source digests are summarized under
`reports/precise-training-start-20261025/`. Current navigation, reproduction and
model-review notes now distinguish implemented synthetic preparation from
real-data readiness. CUDA still fails; the current kernel is `7.0.0-38-generic`.
The selected small checkpoint has its own local copy despite the unavailable
historical model mount. Administrator driver repair, reviewed acquisition/
exposure evidence, an approved concrete subject split and full resource checks
remain before real frozen/adapted/test execution. Original data, collaborator
lists, manifests, recorded scores, historical runners and paper assets are intact.


## 2026-10-25 — overnight worktrees, evaluation scheduling and keep-awake setup

Continued the user's outstanding overnight setup after the separate 9:30 p.m.
resumption published guarded DINOv2 training at f85ccb1. Model choice is DINOv2;
this task preserves that implementation and does not repeat its 103-test run,
weight download or small CPU probe. The active in-chat follow-up named
"Precise DINOv2 evaluation at 3 AM" is scheduled once for 2026-10-26 03:00
America/Chicago (08:00 UTC). The current chat settings were verified as
GPT-6.1 Sol with high reasoning. The scheduler rejects per-run model/effort
fields; it uses the existing chat setting, without a separately pinned override
or a workaround standalone cron. The already completed 9:30 p.m. resumption
was not duplicated.

Created and attached managed precise-training-prep and precise-evaluation
worktrees from dev. They remain detached with hooks enabled; reviewed changes
must integrate through primary dev before commit/push. Ignored per-worktree
context files point to primary read-only images, the existing interpreter and
pinned weights, with separate private output roots. No assets were duplicated,
extra agents started or new chats created. The handoff is recorded in
`docs/notes/precise_overnight_handoff_20261025.md`.

Verified the active temporary user service and logind sleep:idle inhibitor,
started at 17:01:54 Chicago on October 25 with a 24-hour limit, covering the
scheduled run through approximately 17:01 October 26. Screen locking/blanking
remains available and permanent power preferences are unchanged. The earlier
duplicate session-bound inhibitor was released. Scheduled long computation
must hold its own inhibitor if needed beyond that expiry.

CUDA is still unavailable. Current kernel 7.0.0-38 lacks the NVIDIA module;
older precompiled modules correspond to 6.8.0-45, Secure Boot is enabled, and
administrator authentication is required. No system repair or reboot occurred.
Capture/exposure evidence and actual protocol/results/split review remain
unresolved; no new research split, real training or final-test scores were
created. The scheduled task can continue synthetic evaluation tooling when
these real execution gates remain blocked.

Validation: app automation configuration/readback, both attached worktree paths
and detached base revisions, hooks, ignored context JSON and existing primary
input paths, active service/inhibitor, handoff links, whitespace and staged
publication scope/content/size are reviewed. No existing score, dataset,
checkpoint or matcher implementation was changed by this setup task.


## 2026-10-26 — scheduled final-test verification and scoring implementation

The requested 3 a.m. Chicago follow-up inspected current source, both attached
worktrees and ignored task contexts before continuing. CUDA is still unavailable
with zero PyTorch devices, the current kernel remains 7.0.0-38-generic, no real
approved split/comparison manifests or execution receipts exist, and no real
training run is present. The overnight user service/logind sleep inhibitor is
active. The completed audits, environment setup, weight download and original
small pretrained CPU probe were not repeated. No system repair or reboot occurred.

Implemented `code/precise_matching/testing.py` and `code/test_precise_matcher.py`
in the isolated evaluation worktree. The new path requires the latest completed,
frozen training attempt and a separate digest-bound final-test review. It verifies
selected/final checkpoint hashes, provenance and full epoch/scheduler/sampler
history, replays selected development scores, checks the recorded winning epoch
and applies its strict-greater-than cutoff unchanged to pooled and left/right
test outcomes. The CLI checks frozen source/environment compatibility and accepts
only test comparison endpoint paths; it never trains or selects from test.

A private exclusive adapted-test claim is created before test pixels. Existing
claims prevent silent repeats in new output directories. Missing/decode/invalid-
embedding failures remain attempted comparisons with explicit coverage and
failure-as-reject denominators. Fatal input/resource/runtime failures retain the
claim, flushed partial events and failure status without a successful report.
Original training artifacts are unchanged; the claim is a new private record.
Raw outcomes, image/pair identifiers and source mappings stay ignored. Optional
public output is aggregate-only and still requires publication review.

The training-preparation worktree supplied the readiness handoff; all reviewed
files were integrated through primary dev with hooks enabled. Guides are
`docs/notes/precise_final_test_20261026.md` and
`docs/notes/precise_execution_readiness_20261026.md`. Aggregate implementation
validation is at `reports/precise-final-test-tooling-20261026/`; navigation and
reproduction notes link the new tooling. No new monitor or recurring schedule
was created. The temporary keep-awake service remains available for the user.

Validation: 115 local unit tests pass, including 12 new synthetic final-test
contracts. All 41 CPU matcher-pattern tests pass in the isolated environment.
Tests cover corrupted/incomplete or later unfinished selections, changed inputs,
private path boundaries, development replay, test-only pixels, failure/partial
record preservation, exclusive claiming and changed test rates with an unchanged
cutoff. The full CLI test uses actual synthetic image decoding and a tiny fake
encoder, never Precise images or a new pretrained-model run. Existing CPU CI
includes the new suite. All 15 legacy metrics reproduce and correctness lint
passes; syntax/JSON/CI YAML, documentation, temporary table generation, whitespace
and staged publication scope/content/size checks complete review before commit.

Real execution still requires administrator GPU repair and actual capture/
exposure/protocol/results/split review. Frozen-control and common-list SDK runs,
full-resolution resource validation and actual adapted training/final-test
results remain future work. No subject allocation, new real pair list, research
score, paid call, model download or original-asset modification occurred.


## 2026-10-31 — common-list multi-method reporting

Continued roadmap task 9 with offline reporting code after checking the current
implementation and review boundaries. Added `code/precise_matching/comparison.py`
and the standard-library `python3 -m precise_matching compare` command. It accepts
two to four distinct frozen/adapted DINOv2 or VeriFinger arms using one validated
metadata/comparison list. Every arm must provide every development/test outcome
exactly once, including failures; training, unknown, duplicated or missing score
rows are refused. The method names are fixed public labels, not provenance proof.

Reports use each arm's pooled development-only cutoff unchanged for pooled and
left/right test summaries. Paired genuine/impostor acceptance tables and B-minus-A
rate differences retain the full attempted denominators. Paired coverage counts
separate both-valid, one-valid and neither-valid comparisons. Supplemental
both-valid differences explicitly retain their reduced denominator and original
cutoffs. The tool does not select a winning method or implement uncertainty;
the proposed subject-resampling method still needs statistical review. Supplied
scores and the FAR target require separate frozen-run/configuration review.

The CLI preserves existing output/privacy boundaries, rejects overwrite and
changed input bytes, and exports aggregate summaries without private identifiers,
paths or hashes. The guide is
`docs/notes/precise_common_list_comparison_20261031.md`; repository navigation,
reproduction instructions and the protocol proposal link this implementation.

Validation: all 123 local unit tests pass, including eight new comparison
contracts automatically discovered by regular CI. Tests cover score scales,
failure/coverage asymmetry, opposing paired decisions, no hand-specific/test
retuning, mismatched lists, metadata gates, all six four-arm comparisons,
empty/all-failed groups, input/order invariance and CLI privacy/mutation/overwrite
guards. All 15 legacy published metrics reproduce and correctness lint passes.
Syntax, links, temporary result-table regeneration, whitespace and staged
publication scope/content/size checks complete the publication review.

This is synthetic implementation evidence. No real split, comparison list,
matcher/SDK score, training run or uncertainty interval was produced; no model,
dataset, historical result or paper asset changed. Acquisition/exposure facts,
protocol/results/split review, usable GPU resources and subsequent real-run
evidence remain necessary for completion of the ten-task milestone.


## 2026-10-31 — frozen-control development and final-test path

Closed the missing selected DINOv2 frozen-control implementation path with
`code/precise_matching/baseline.py` and `code/run_precise_frozen.py`. It uses the
pinned pretrained encoder, all parameters frozen, FP32 inference and the shared
whole-slap preprocessing/CLS/cosine contract. It creates no optimizer and reads
no training pixels. The existing outcome scorer now supports one explicit
development/test partition while retaining the adapted test-only wrapper.

Development requires reviewed phase-only endpoints, metadata/configuration and
digest-bound capture/exposure/protocol/split/execution assertions. It preserves
raw outcomes/events and freezes a development-only cutoff only when every
development comparison succeeds. Final test requires a separate review binding
the exact successful development artifacts and original protocol/configuration;
it replays saved development scores, checks current source/environment/model
compatibility and uses the unchanged pooled cutoff for both hands. Test pixels
are unavailable in the development mapping and vice versa. Review flags still
require actual acquisition evidence and human review, not generated assertions.

An exclusive private frozen-test claim precedes test pixels. Repeat invocations
are refused, including with another output path. Fatal attempts retain their
claim, flushed partial events and failure record without a successful report.
Completed test image failures retain full attempted denominators and explicit
coverage. Failed development attempts cannot masquerade as frozen selections.

Both frozen and adapted final-test scorers now additionally export a private
`scores.csv` containing saved development outcomes and new test outcomes in the
existing comparison schema. Original JSON artifacts remain. Synthetic integration
tests pass these exports directly through the common-list evaluator without
recomputing development inference or manually relabeling pair IDs. The new
guide is `docs/notes/precise_frozen_control_20261031.md`; reproduction/navigation
and final-test notes link it and describe the exports.

Validation: all 135 local unit tests pass, including 12 new frozen-control
contracts; all 53 CPU matcher-pattern tests also pass in the isolated candidate
environment. They cover phase-only loading, immutable frozen weights/no optimizer,
review and digest/population gates, semantic development replay, environment
changes before model loading, incomplete/corrupt artifacts, explicit failures,
partial-event/claim retention, repeat refusal and test-score changes without
cutoff changes. The existing CPU CI pattern includes the new suite. All 15
legacy metrics reproduce; correctness lint, syntax, documentation links,
temporary byte-identical results-table regeneration, whitespace and staged
publication scope/content/size review are required before commit.

A fresh read-only preflight still finds zero files under private splits,
manifests and runs, and NVIDIA-SMI cannot communicate with the driver. No actual
Precise inference, full-resolution pretrained probe, training, split generation,
model download, dependency installation or system repair occurred. Existing
data/results and paper assets are unchanged. Real resource verification,
acquisition/exposure/protocol/results/split review, actual frozen/adapted/SDK
scores, agreed uncertainty and the resulting research package remain incomplete.


## 2026-11-01 — collaborator clarification, metadata offer and zero-score reconciliation

The user pasted new messages from Konain and the professor. Konain confirms
subject/hand filename positions but defers other filename details; reports using
approximately 19,648 genuine and 78,592 impostor training comparisons; explains
his 1:4 class ratio and acknowledges potentially right-only selection; and says
failed image-to-DAT conversion causes pair scores to be zero. He reports a
remaining pool of 23,103 genuine and 8,992,383 impostor comparisons. The professor
offers a ground-truth dataframe including resolution/subject fields and asks
whether fine-tuning has started. No dataframe or new execution/split approval
was supplied with these messages. Normalized correspondence/source/access
receipts remain private in `local/precise/metadata/clarification-20261101/`.

Recorded actual use as collaborator-reported evidence, distinct from the prior
independently checked list membership. Exact consumed rows, post-filter code,
development/selection history and parent acquisition independence remain open.
The new folder returned NOT_FOUND/404 in connected Drive; listing was empty and
web fallback inaccessible. This does not prove that the folder is empty or
deleted. No new large pool or source file was downloaded.

A targeted read-only reconciliation with the existing parser confirms unchanged
source hashes against the October 24 private intake receipt. Training genuine
contains 23,103 rows: 19,594 positive fused scores and 3,509 zeros, including
3,455 missing-component zeros and 54 component-bearing zeros. The reported
19,648 equals `23,103 - 3,455`, not literal positive-score filtering. The sampled
78,592 impostors contain 73,538 positive fused scores and 5,054 component-bearing
zeros. Positive component sums coexist with fused zero in 51 genuine and 5,017
sampled impostor training rows, and in 6 genuine/136 impostor evaluation rows.
The failure-to-zero convention therefore does not identify every failed image
or justify blanket zero removal. Obtain conversion status and the exact filter;
do not reconstruct missing fields or silently relabel original scores.

New aggregate outputs are `reports/precise-clarification-20261101/`; the evidence
and narrowed follow-up are in `docs/notes/precise_clarification_20261101.md`.
Central proposal/review, navigation and reproduction notes link the update.
The DINOv2 objective remains identity/capture supervision, not SDK distillation;
thumb exclusion and subject-disjoint principles are unchanged. A collaborator's
recommendation to mix thumbs/use nonzero pairs does not revise our protocol.
The professor's question does not waive the existing results/split review.

Validation: eight existing intake/parser synthetic tests pass; all 15 legacy
metrics reproduce. Aggregate arithmetic/JSON, current links, unchanged input
hashes, whitespace and staged publication scope/content/size are reviewed before
commit. No production Python behavior, old report or metric, data, model, split,
training run or paper asset was modified. The fresh NVIDIA-SMI check still fails.
An accurate professor update is data/protocol validation with implemented,
synthetically validated matcher/baseline/training/evaluation tooling; real Precise
fine-tuning has not begun. No external response was sent.

## 2026-11-01 — driver resolved and full-resolution synthetic GPU check passed

The user reported repaired GPU access. Independent read-only verification in
the existing pinned matcher environment confirms driver 595.99.02, PyTorch
2.6.0+cu124, CUDA, BF16 and one RTX 4080 SUPER with 15.57 GiB device memory.
The earlier driver blocker is resolved. No driver repair, dependency install,
model download or repeat of the original small CPU pretrained probe was needed.

Added `code/check_precise_gpu_runtime.py` and ran the authorized capacity check
with the existing pinned DINOv2-small weights/configuration: 448×448, 16 images
per microbatch, BF16, four-step accumulation and two AdamW updates. It exercises
eight synthetic grayscale-to-RGB microbatches, alternating artificial hands,
the actual loss, final two trainable blocks and final norm. Losses/gradients
remain finite; all 38 trainable tensors change while the frozen digest stays
unchanged. Frozen and adapted single-image FP32 inference return 384-dimensional
embeddings. The nominal LR is used without a scheduler; existing contract tests
cover checkpoint/resume separately.

Training peaks at 2.67 GiB allocated / 3.37 GiB reserved; sampled driver-free
memory remains at least 11.57 GiB, above the planned 2 GiB reserve. Allocator
caches persist between phases. These are short-run capacity measurements under
current competing usage, not long-run stability, throughput or biometric
results. Reviewed aggregate evidence is in
`reports/precise-gpu-runtime-20261101/`; timestamp, package/code/configuration
hashes and phase records remain private in
`local/precise/metadata/gpu-runtime-20261101/`.

Five new synthetic contracts cover refusal before loading/overwriting,
CUDA/BF16 preconditions, gradient accumulation and frozen/trainable invariants;
the existing CPU matcher CI pattern includes them. All 140 repository tests,
58 matcher-focused tests and all 15 legacy metric checks pass. Correctness lint,
Python syntax, JSON/aggregate arithmetic, new documentation links, whitespace
and staged publication content/scope/size are checked before commit.

Compute verification is now complete for this proposed workload. Capture
independence, exposure/claim scope, concrete protocol and actual results/split
review remain open; no real split, comparison generation, Precise image loading,
fine-tuning or final evaluation occurred. Konain's exact SDK filter is a
reproduction question, not an additional requirement for identity-supervised
DINOv2 training. Current plan/navigation/reproduction notes link the new
evidence; older dated compute reports and original research assets are retained.

## 2026-11-01 — professor requests LLM-focused DINOv2 plan and describes dataframe

The user pasted a professor reply listing Excel fields: dataset, finger, slap,
subject, quality/segmentation flags, original boxes and rotated geometry. The
reported finger codes distinguish four-finger right/left hands and thumbs.
This is a schema description, not workbook intake; no row unit, coverage,
capture/session mapping or physical PPI was verified. Multiple finger/rotation
annotations must not become multiple independent impressions. The normalized
message receipt remains private in
`local/precise/metadata/professor-plan-20261101/`.

The professor says DINOv2 may be a candidate, asks for its plan before experiments
and reaffirms the primary goal of investigating LLM slap verification, failures
and possible improvements. Prepared
`docs/notes/precise_dinov2_llm_plan_20261101.md` for review: selected DINOv2 as a
frozen/adapted visual control, a proposed direct Qwen3-VL comparison on identical
eligible lists, development-only input/output/supervision diagnostics and a
separately reviewed targeted VLM adaptation. DINOv2 alone or an LLM explanation
cannot establish LLM matching ability. Cross-model differences do not identify
a language-module defect; proposed within-VLM controls are needed to test causes.

The plan retains exact DINOv2 settings, existing shared-list/development/failure
safeguards, conditional SDK access and historical-exposure limitations. Verified
upstream DINOv2/Qwen model roles through primary model cards/paper; no literature
claim validates Precise performance. The previous synthetic GPU success stands;
it does not establish VLM memory fit. New VLM runs/adapters and all real-data
experiments remain subject to the professor's plan review and applicable
capture/exposure/protocol/results/split gates. Instructions and current entry
points link this latest direction while preserving earlier dated records.

Validation is documentation-only: review factual attribution, new local links,
whitespace, staged content/size/privacy and publication scope. No model job,
package installation, Excel read/import, new split/pair generation, source score
change or manuscript edit occurred. No message was sent to the professor or
collaborator. The full ten-task goal remains incomplete; this reviewable plan
addresses the new request without treating it as experiment approval.

## 2026-11-01 — supplied GitLab dataframe: complete coverage and padding aliases

The user supplied `Gitlab_SlapBench/` and requested continued work. Read-only
inventory finds a 118,658,976-byte reference folder containing code, original
exports, papers/manuscripts, an archive and a 96,804,926-byte annotation pickle,
`dataset/dataframe/CompleteWithRotated.P`, with matching schema documentation.
No Git metadata establishes its revision. The generic README's example remote,
force-push and setup instructions were treated as reference text, not executed.
The source folder is now ignored and rejected by the publication guard; no
reference assets were moved, edited or copied into public history.

Added `code/audit_precise_dataframe.py`: fixed reviewed pandas/numpy constructor
allowlist, exact schema validation, read-only image-name/finger/identity
reconciliation, generic failure messages and new-output refusal. Unknown pickle
constructors never trigger an unrestricted fallback. This is a format-specific
reader, not a general pickle sandbox. The existing verification environment
was used without installations or GPU work. Private source/version/digest
receipts and unchanged raw per-image identity representations stay in
`local/precise/metadata/gitlab-dataframe-20261101/`.

The dataframe has 457,005 rows and 26 columns. Its Precise slice has 24,530
finger annotations spanning exactly all 7,358 local JPEG names, with no missing
or extra image names, no multiple identities within one slap, no finger-set
conflicts with filename hand labels and no duplicate finger rows within a slap.
There are 2,452 right, 2,455 left and 2,451 thumb images. Rotation is zero in
20,900 Precise rows and missing in 3,630; eleven rows have missing geometry and
quality fields. Missing fields remain missing and no quality exclusion follows.
Other/augmented datasets remain outside the Precise intake population.

Crucially, 338 raw filename-prefix strings map to 277 numeric SubjectID values.
All 7,358 names agree numerically; 61 numeric groups each span exactly two raw
prefixes differing solely in ASCII leading-zero padding, while 216 have one.
This is candidate alias evidence, not proof of 277 distinct people or authority
to merge labels. Confirm the intended canonical identity mapping. The old
338-based subject-count assumption is no longer a safe allocation basis. If
277 were confirmed and fully eligible, the existing rounding rule would give
166/55/56 and trigger the draft's fewer-than-60 evaluation review condition.
No subjects were allocated and historical manifests/labels were not changed.

Reviewed aggregate evidence is in `reports/precise-gitlab-dataframe-20261101/`.
Updated the professor-review plan, central protocol, instructions/navigation and
reproduction notes: the dataframe is now available, so do not request it again.
The focused data questions are canonical padding aliases and independent parent
acquisitions. This schema still lacks explicit capture/session/parent, sensor,
physical PPI and prior exposure fields. Plan/protocol/results/split review
remains required before real experiments, with the LLM question primary.

Five synthetic intake tests cover identifier-free summaries, candidate alias
handling without canonicalization, conflict/missing-value accounting, rejected
constructors/schema/trailing data, original-byte preservation and overwrite
refusal. Existing publication tests cover the new local reference boundary.
All 145 repository tests and 15 legacy metric checks pass, together with
correctness lint, Python syntax, aggregate arithmetic, documentation links and
whitespace. Review staged publication content/size/privacy before commit. No source image bytes,
real model jobs, annotation fixes, splits/pairs or paper edits were produced.

## 2026-11-01 — professor approval, shared cohort and next-goal handoff

The user supplied later professor approval of DINOv2 and a request to use the
same training pairs as Konain, who handles Qwen3-VL. Reviewed all three new Drive
folders and the earlier remaining-pairs folder through the authenticated
connector. Six preserved local inputs match fresh raw Drive bytes exactly:
evaluation/fine-tuning scripts, training JSON, both testing lists and the full
genuine source. The large impostor pool and image archive were listed only.

Reviewed the existing uncommitted cohort audit and its tests; its code digest
matches its private receipt, and its aggregate reproduced exactly in memory.
The cohort contains 39,290 training pairs (19,645 per class), balanced by class
within right/left/two-thumb types, and 4,200 testing pairs. All references resolve.
Exact train/test pairs are disjoint, but all testing images/identities occur in
training. Positive component sum, not positive fused score, is the selection
rule. Preserve 51 genuine and 1,213 impostor fused-zero training pairs. There
are 24 training and 2 testing numeric-identity label disagreements; private
cases remain unchanged. No acquisition independence or actual run consumption
is inferred. Neither collaborator script was imported/executed.

Added docs/notes/precise_next_goal_handoff_20261101.md with source links,
professor-approved ownership, facts versus code defaults, remaining label and
checkpoint/threshold choices, fixed-pair implementation gaps, worktree asset
requirements and a suggested next-chat goal. Updated AGENTS.md and older plan
status so broad approval, pair lists and acquisition facts do not become stale
blanket blockers. The shared cohort includes thumbs; the earlier strict
subject-disjoint design is a separate extension. No existing runner guards
were weakened, new split generated, GPU job started, new chat/worktree created,
or message sent to a collaborator. Original inputs/results remain untouched.

Validation: all 149 repository tests and 15 legacy metric checks pass; correctness
lint, Python syntax, handoff/report links, aggregate arithmetic and whitespace
checks pass. The publication review covers code, synthetic tests, documentation
and aggregate JSON only; private source files and case lists stay ignored.

## 2026-11-01 — approved concrete shared-pair route and implementation

The user started the end-to-end DINOv2 goal, then directly reviewed and approved
the exact protocol and training/calibration/test execution in
`docs/notes/precise_shared_pair_protocol_20261101.md`. This is user review,
not a new professor attestation or confirmation of canonical identities. Preserve
all supplied labels/pairs, including thumbs and the 24 training/two testing
numeric-identity disagreements. Primary claims concern supplied-label agreement
on held-out pairs with seen images/identities. A secondary exclusion sensitivity
keeps frozen decisions and identifies its reduced testing denominator.

Implemented a separate explicit-pair route in `fixed_pairs.py`,
`fixed_training.py`, `fixed_evaluation.py` and `run_precise_shared_pairs.py`.
It retains every pair exactly once per epoch, including single-class/tail
microbatches and partial accumulation windows; no within-batch cross-pairs are
created. The approved schedule is three full epochs, final checkpoint fixed in
advance, with each arm calibrated on training impostors at nominal 1% FAR.
Training calibration is resubstitution, not independent validation or a promised
test FAR. The old strict subject-disjoint runner is unchanged.

Private preparation reuses the completed image/hash inventory and produces an
exact 39,290/4,200-pair bundle with 6,778 referenced images, without reading
pixels. Direct human approval is recorded privately against its exact bundle
and configuration hashes. No independent-capture/subject-separation assertion
is fabricated. Checkpoint/resume and immutable attempts preserve provenance;
calibration failures stop selection; testing verifies the training-only cutoff
and claims it once before pixels. Raw failures/diagnostics stay private.

Thirteen new synthetic contracts include full CLI train/calibrate/test on both
arms, deterministic dropout resume, supplied-target/thumb preservation, exact
pair visitation, hash/phase review gates, source-path guards, failure accounting,
selection replay, fixed pooled cutoffs and one-time testing. All 162 repository
tests and 71 matcher-pattern tests pass. All 15 legacy metric
checks, table generation, syntax, correctness lint and whitespace checks pass.
GPU availability was rechecked (RTX 4080 SUPER, driver 595.99.02); no package
installation, model download or repeated dataset audit was performed. Real-run
outputs and the final report will be recorded separately after execution.

## 2026-11-01 — shared-pair DINOv2 experiments and professor report complete

Executed the user's reviewed protocol from commit
`82d02111de2d13378c4be610cbc733923b0637de`: all 39,290 supplied pairs per epoch,
three full epochs / 3,684 updates, unchanged supplied targets including thumbs
and numeric-identity disagreements. The fixed final checkpoint was sealed before
calibration/testing. Offline comparison with the pinned encoder loader confirms
185 frozen tensors unchanged and all 38 selected tensors changed. No model,
package or dataset download/install/audit was repeated.

Both arms calibrated on all 19,645 training impostors, accepting 196 at nominal
1% training FAR, and then scored exactly 4,200 testing attempts without failures.
Frozen: cutoff 0.8937185406684875, TAR 41.71% (876/2,100), FAR 0.81% (17/2,100),
AUC 0.8776 and descriptive EER 19.76%. Adapted: cutoff 0.9320146441459656,
TAR 56.29% (1,182/2,100), FAR 1.10% (23/2,100), AUC 0.9800 and EER 6.52%.
Observed TAR increases 14.57 percentage points while FAR increases 0.29 points;
the adapted model still rejects 918 genuine attempts at its fixed cutoff.
All numbers are finite-cohort descriptive results with training calibration.
There is no independent development estimate or test-based retuning.

Published reporting code replays private calibration/outcomes, validates exact
common testing pairs for supplied SDK and historical Qwen exports, and reports
pooled/per-type metrics, paired decisions, failures/coverage, reuse counts and
the two-case label-ambiguity exclusion sensitivity. Original labels are unchanged;
both flagged testing rows are rejected by both DINO arms. All testing images and
numeric groups occur in training. No subject-disjoint/capture-independence,
independent-trial interval, significance or population low-FAR claim is made.

Supplied SDK fused/sum descriptive EER is 2.05%, with zero values preserved and
extraction status unresolved. Historical Qwen score-export AUC is 0.7612 / EER
34.03%; this uses a different 0-100 prompt with unverified model/adapter/run
provenance. The user confirmed matching fine-tuned A/B outputs are not ready.
No local Qwen fine-tuning was duplicated and no trained-Qwen superiority claimed.

The professor-ready four-page PDF, editable Markdown, metric CSV, ROC SVG and
aggregate JSON are at `reports/precise-dinov2-shared-20261101/`. Every final PDF
page was rendered and visually checked. Independent NumPy/scikit-learn checks
reproduce DINO counts/AUC/EER to numerical precision; original audited source
hashes remain unchanged. All 170 repository tests, 74 matcher CPU contracts,
15 legacy metric checks, table regeneration, syntax, lint and whitespace checks
pass. The report README records operational handling of an idle launcher exit
and offline state-key verifier correction; no model phase was repeated.

Private approvals, image mappings, original inputs, checkpoints, scores and
diagnostics remain ignored. The manuscript and historical assets are unchanged.
Publication includes reviewed source, synthetic tests and aggregate documents
only; commit/push and remote SHA verification complete this publication step.

## 2026-11-01 — prepare the remaining trained-Qwen A/B comparison path

The goal continuation verified the clean published DINO state at `36013bb` and
the user's preserved statement that Konain's matching fine-tuned A/B outputs
and actual run/checkpoint evidence are not ready. The completed DINO report
already flags this external gap; it does not fulfill the remaining trained-Qwen
comparison or justify marking the entire original goal complete.

Added `code/compare_precise_qwen_ab.py` for the supplied script's actual parsed
A/B output format, plus a documented future evidence/command contract at
`docs/notes/precise_qwen_ab_comparison_20261101.md`. It verifies exact full pair
coverage and class membership, preserves case variants and explicit failures,
requires digest-bound review of actual consumed-training/prompt/environment/
checkpoint/selection evidence, and replays both saved DINO cutoffs/outcomes.
It exports pooled/per-type/sensitivity operating points and paired decisions,
with no binary Qwen ROC/AUC/EER or test-based tuning. It creates no real review
assertion and performs no model inference, training, image access or download.

Seven new synthetic contracts pass, including a CLI roundtrip and mutation/
overwrite/privacy cases. The regular CI suite includes them automatically.
Completed DINO runs, frozen decisions, original scores and the professor PDF
remain unchanged. This prepares independent implementation while actual Qwen
outcomes/provenance remain the same external blocker.

Validation: all 177 repository tests and all 15 legacy metric checks pass;
results-table regeneration, Python syntax, correctness lint, whitespace and
explicit staged publication/content review precede commit/push. No new result,
real comparison review, score file or professor-report revision was generated.

## 2026-11-07 — new Drive cohort found; replacement approved

The user resumed the full goal and supplied replacement repository instructions.
The checkout remains on published `dev` at `3800ce0`. A focused check of
the previously supplied Drive folders found five new November 5 files in the
testing/data folder; the evaluation and fine-tuning-code folders still contain
their previously supplied scripts/JSON, not matching trained-Qwen outputs.

Saved the complete five raw new sources (about 28.0 MiB) under ignored
`local/precise/collaborator/konain-followup-20261107/`, checking provider byte sizes
and recording Drive IDs/modified dates and SHA-256 values. No archive/model
download, original replacement or dependency installation occurred. The new
`code/audit_precise_revised_cohort.py` reads these inputs, reuses the completed
image inventory and compares to the prior bundle bound to the actual DINO
training record. It does not decode pixels, allocate identities or run models.

New JSON/training lists match exactly: 37,194 pairs, 18,597 per class, balanced
within right/left/thumb types. Testing has 4,200 pairs, with 700 per class/type.
New training/testing share zero exact pairs, images or numeric prefix groups;
there are 221 training groups and 55 testing groups. No raw-prefix/numeric label
disagreements, missing references or duplicate pairs occur. All 61 inventory
padding-alias groups occur in training only. This establishes numeric-group
separation in the supplied rows, not confirmed biological identities or capture
independence. Positive component sums retain 47/1,214 training and 7/142 testing
genuine/impostor fused-zero scores; extraction status remains unverified.

Only 189 new testing pairs match the old testing list; 4,011 changed. Critically,
all 1,057 new testing images and all 55 groups, including 1,918 exact new testing
pairs, were used by the already completed adapted model's training. Reusing that
checkpoint cannot support an unseen-group claim. A newly reviewed numeric-group
study would need fresh adaptation from the pinned base and separate outputs.

The aggregate/README are at `reports/precise-revised-cohort-20261107/`. A concrete
replacement proposal in `docs/notes/precise_revised_cohort_review_20261107.md`
retains fixed three-epoch selection and nominal 1% training calibration, using
the new sources after a separate scope/results/protocol review. The user then
explicitly replied: “Yes—use the new lists and run fresh DINOv2 training,
calibration and testing.” The active benchmark is now the revised cohort; this
is actual user authorization, not approval inferred from folder content or the
resumed goal. Completed prior runs/report,
frozen decisions, labels, old metadata and private assets are unchanged.

Four new synthetic intake contracts pass. Full offline checks and explicit
publication review precede commit/push. This is new-source intake, not a repeat
of completed dataset/image audits. The separately authorized fresh study is
prepared below.

The OS now reports Ubuntu 26.04.1 with default Python 3.14, while the original
verification and matcher packages remain under Python 3.12 site directories.
Initial verification therefore reported missing packages under the changed
interpreter. Created dated Python 3.12.14 compatibility environments that reuse
the existing package directories without installing packages or editing the
original environments. The pinned matcher packages pass dependency checks;
CUDA/BF16 is available on the RTX 4080 SUPER. No driver repair was required.
The missing `nvidia-smi` management executable is handled for provenance only by
reading the loaded NVIDIA module's version; model/device/precision never change.

Added explicit `numeric_group_disjoint_fixed_pairs` support to preparation,
digest-bound review, metadata validation and final-test reporting. It checks
numeric-prefix/image separation, including padding aliases, before model access;
the older seen-image scope retains its behavior. Synthetic full CLI contracts
cover fresh training/calibration/testing for both scopes and retain all prior
matcher tests. The new bundle has exactly 37,194/4,200 pairs and 6,778 images;
preparation reads no pixels and generates no approval. Actual user review is
preserved privately against proposal/audit bytes and bound separately to the
new bundle/configuration before execution.

The November 7 full-resolution synthetic check passes under the compatible
runtime (driver 595.99.02, Torch 2.6.0+cu124, BF16, 16 images at 448 pixels):
2.67 GiB peak allocated / 3.37 GiB reserved. It addresses the changed interpreter/
OS runtime; no fingerprint image or real model score was used. All 187 repository
tests, 80 matcher CPU contracts and 15 legacy metric checks pass after runtime
compatibility was restored. Source/content/publication checks and commit/push
precede fresh authorized real-data execution.


## 2026-11-07 — fresh numeric-group-holdout DINO study completed

Following the direct replacement/run approval recorded above, committed and
verified source `b5a6a128b7595afaa5b4dbac4d73cf09cd65788e` before real execution.
Fresh adaptation from the pinned DINOv2-small base completed three full epochs
of all 37,194 supplied pairs: 111,582 pair presentations and 3,489 updates, with
one attempt and no resume, fallback or retry. Mean pair losses were 0.1456442,
0.1024788 and 0.0911037. The predeclared final checkpoint SHA-256 is
`106a860fbba8e0ac28163a34faf0510df309e80ff28a39e375ccb945376f6b09`.
Offline tensor/scheduler verification confirms 185 frozen tensors unchanged,
38 selected trainable tensors changed and the complete epoch/update schedule.

Both training-only calibrations then completed, each accepting 185 of 18,597
training impostors under the nominal 1% rule. Cutoffs were sealed before either
new final test: frozen 0.8948034048080444 and adapted 0.9250705242156982, strict
score-greater-than comparison. Both one-time final tests completed all 4,200
attempts with zero failures. Frozen accepts 730/2,100 genuine and 20/2,100
impostor pairs (TAR 34.76%, FAR 0.95%); adapted accepts 727 and five (TAR 34.62%,
FAR 0.24%). Descriptive EER changes from 22.67% to 10.95% and AUC from 0.8495
to 0.9416. Adaptation improves score ordering without a TAR gain at these
training-calibrated operating points; no matched-test-FAR or superiority claim
and no subsequent test tuning is made.

The new aggregate exporter replays sealed outcomes/calibrations, verifies fresh
initialization and exact source/review/cohort binding, and confirms zero
cross-phase raw/decoded image-hash overlap. Numeric groups are held out, with
221 training/55 testing groups and 5,721/1,057 images. All images recur within
phase; testing reuses each image across up to 16 pairs. Biological subject and
capture independence remain unconfirmed. All 61 inventoried padding-alias
groups occur in training, limiting holdout representativeness. Zero label
conflicts mean exclusion sensitivity equals the full population. Prior dataset
findings are known; the old adapted checkpoint, which saw every new testing
image/group and 1,918 exact pairs, was not reused.

Supplied new SDK scores match all new testing pairs: fused/sum descriptive EER
3.57%, with all seven genuine/142 impostor fused-zero scores retained. Positive
component-sum source selection and unknown extraction status remain explicit;
no SDK cutoff was selected on testing. Historical Qwen scores cover a different
testing list and are excluded. Matching fine-tuned Qwen outputs/run provenance
for this cohort remain unsupplied; Konain owns fine-tuning. The offline A/B
importer now accepts both reviewed cohort scopes, retaining exact coverage,
unchanged DINO cutoffs, failures, binary-only reporting and actual-evidence
review requirements. No VLM experiment or message to the professor was sent.

New results, tables, ROC and the four-page professor PDF/editable report are in
`reports/precise-dinov2-numeric-holdout-20261107/`. Every final PDF page was
rendered and visually inspected. A separate NumPy/scikit-learn verifier
reproduces pooled/per-type DINO confusion counts, failures, training-cutoff
replay, AUC and interpolated EER. New synthetic contracts cover reporting
provenance, full schedules, new source names, exact outcome coverage and tied
scores/failures. All 196 repository tests, 80 matcher CPU contracts, 15 legacy
metric checks, correctness lint, syntax, table regeneration, links and
publication/content checks pass. Publishable code, aggregate outcomes and
notes are reviewed for commit/push to origin/dev; originals and private assets
remain unchanged and ignored. This completes the authorized DINO study; the
matching trained-Qwen comparison remains pending actual collaborator evidence.

Post-publication CI exposed a filesystem-order-dependent synthetic intake test:
it deleted the first directory entry, which could be an inventory-only image
absent from the pair lists. The audit correctly counts missing referenced
images, so CI observed zero instead of the test's expected one. The fixture now
explicitly checks both an unused inventory image (zero missing references) and
a training-JSON endpoint (one missing reference). This changes only the test;
sealed runs, scientific policy, source audit and professor report are unchanged.
Local regression checks and hosted CI are checked again for the corrective
commit rather than repeating any model phase.


## 2026-11-08 — new Qwen outputs verified; native-large route prepared

The user supplied Konain's newly uploaded evaluation and fine-tuning/checkpoint
folders and relayed the professor's request for a larger DINOv2 model with
original high-resolution images. This is new evidence/direction after the
completed November 7 study. It does not overwrite or repeat that study.

Preserved both new Qwen summaries, four complete prediction files, updated
matched evaluation code, training-source JSON/script and adapter/checkpoint
metadata under ignored `local/precise/collaborator/konain-results-20261108/`.
Raw provider IDs/byte hashes are recorded privately. Downloaded adapter bytes
are 132,149,144; no Qwen model was executed, trained or installed. Training
arguments were inspected as pickle opcodes without executing the pickle.

The source audit joins every new output to exactly 4,200 supplied new test pairs
with labels, zero duplicates/missing/extra rows and zero INVALID/ERROR outcomes.
It reproduces base TP 1927 / TN 732 / FP 1368 / FN 173 (accuracy 63.31%, TAR 91.76%, FAR 65.14%)
and fine-tuned TP 2040 / TN 1726 / FP 374 / FN 60 (accuracy 89.67%, TAR 97.14%, FAR 17.81%).
The training JSON matches the DINO bundle byte-for-byte. Both prompts match the
training JSON; the summaries' list-based prompt hashes reproduce. The supplied
checkpoint state records epoch 3 / step 27897 and is consistent with the declared
37,194-example, batch 1 / accumulation 4 / three-epoch schedule. Actual
`binary_training_config.json`, resolved base revisions/runtime versions, raw
replies and selection/evaluation-digest evidence remain incomplete. These facts
are separated from verified saved-outcome arithmetic. Different DINO/Qwen
operating policies are not treated as matched test FAR. The new aggregate,
per-type table and source review are in `reports/precise-qwen-supplied-results-20261108/`.

Verified/downloaded official `facebook/dinov2-large` at pinned revision
47b73eefe95e8d44ec3623f8890bd894b6ea2d6c (304,368,640 parameters). Existing
benchmark inventory records all 6,778 images at 1600 x 1500. Added native preparation
that preserves every grayscale pixel, centered in a 1610 x 1512 white canvas with
no resampling/cropping. Added separate native-large train/calibrate/test gates,
explicit BF16 scoring, Flash-only SDPA, nonreentrant gradient checkpointing and
native offline checkpoint verification. Only blocks 22/23 and final norm adapt.
Old model/configuration defaults and all original assets remain intact.

A preliminary synthetic capacity check passed, followed by an exact-policy
check adding deterministic execution and both BF16 inference contexts. The
exact check completes 64 microbatches / two AdamW updates with frozen tensors
unchanged, finite gradients and about 3.01 GiB peak allocated / 3.26 GiB reserved on
the RTX 4080 SUPER. It uses synthetic pixels only. Measured 1.26 pairs/second implies
about 24.6 hours of model computation for the proposed three epochs, plus actual
pipeline overhead/calibration/testing. No performance or long-run fit is inferred.

The concrete new proposal is `docs/notes/precise_large_native_protocol_20261108.md`:
same 37,194 / 4,200 pairs, fresh pinned large base, native pixel preservation,
1 pair / accumulation 32, three full epochs/final checkpoint, nominal 1% training FAR
and one-time tests. The user then directly approved the exact proposal and its training, calibration
and test execution. The original proposal bytes/response are preserved privately
and bound to the new bundle/configuration; no identity/capture facts are inferred.
Real execution follows source checks/commit/publication.
New tests cover exact pixel preservation, source hashes, summary arithmetic,
explicit INVALID failures, static training-argument reading, native phase
provenance/precision dispatch and explicit large tensor-group verification.
Full verification/content review and publication precede real execution.

Local verification passes: 206 repository tests, 83 matcher CPU contracts and
15 legacy metric checks, plus correctness lint, changed-file syntax, generated
table consistency, 84 relative links and publication/privacy/content checks.
The exact native-large execution review is recorded; new real phases follow
the reviewed source commit and verified push.

## 2026-11-08 — approved native training active; offline final-report path ready

After source publication at `06d12c249ec7f5db53124f673b2c6947c4b144b7`, all
four hosted CI jobs and CodeQL passed. The approved Large/native adaptation
started at 22:31:35 UTC in a new ignored run directory. Its existing process,
session and continuation details are recorded privately under
`local/precise/metadata/native-large-execution-20261108/`. Monitor this run;
do not start another because a terminal observation expires. Training remains
active; native calibration, final tests and final research results are pending.
The protected runtime/configuration/input/review bytes remain unchanged.

Added an offline exporter for the eventual sealed native outcomes, final large
checkpoint verification and exact-cohort small/Qwen/SDK comparison. It replays
both calibrations/tests, checks committed execution source, rechecks Qwen pooled
and per-type source outcomes, and preserves binary-Qwen provenance limits. Added
a separate six-page professor-report renderer with before/after accuracy and
confusion matrices, native preprocessing, actual training elapsed time, per-type
rates, paired decisions, continuous DINO/SDK ROC only and reproducibility limits.
A clearly marked synthetic layout preview was inspected on every page under
ignored metadata. It contains no new native-model measurements and is not a
public result. Prior reports remain unchanged.

The user also relayed Konain's clarification about optional prompt JSON and his
confirmation that the training-data JSON was already uploaded. The supplied
code distinguishes optional evaluation `--prompt_json` from the automatically
saved `binary_training_config.json`, which contains selected counts and actual
argument values as well as prompts. The training-data JSON is already present
and verified byte-for-byte; it is not requested again. No new saved config or
training command accompanied these replies. See
`docs/notes/precise_qwen_config_clarification_20261108.md`. These limits do not
hold up the approved native run or authorize a restart/new experiment.

Verification for this reporting change passes 209 repository tests, all 15
legacy metric checks, correctness lint, changed-Python syntax and unchanged
results-table regeneration. Final native results still require completed real
phases, independent metric verification and final report review/publication.

## 2026-11-10 — Large/native interruption verified; one resume approved

Continuation inspected the existing process and recovered terminal exit139 /
segmentation fault, last reported in epoch3 at 19,072 of 37,194 pairs and update
2,922 of 3,489. The original process is absent and no final selection exists.
The log's last write is November 9 at 19:51:42 UTC. Accessible diagnostics do not
establish the crash cause; no OOM, model-performance or driver-repair conclusion
is inferred. No calibration or final test had started.

CPU-only inspection verified the original epoch2 checkpoint SHA-256
`e8afe81b4963b3315e545eda22ee71e4bed14973f07662368e9565322564e762`:
both completed epochs contain all 37,194 pairs, scheduler/optimizer state records
2,326 saved updates with finite moments, RNG metadata is valid, all 401 frozen
tensors remain unchanged and all 38 selected trainable tensors changed. Protected
runtime bytes still match source 06d12c249ec7f5db53124f673b2c6947c4b144b7.
The new `reports/precise-native-large-interruption-20261110/` exports only the
reviewed aggregate; original assets and incomplete-run files remain untouched.

The original approved protocol forbids automatic retries. After the concrete
checkpoint-bound proposal was supplied, the user directly answered
“Approve one unchanged-config resume.” The actual response and proposal snapshot
are bound privately to the original input/config/review/run/checkpoint and
interruption-verification digests. This approves one epoch2 restoration and full
epoch3 replay, then the previously approved calibrations/tests. No further
automatic retry, changed pixels/backend/precision, extra epoch or study is inferred.

Added offline recovery review/chain checks and optional final-report support
for this actual reviewed two-attempt history. Fresh-run gates stay strict;
extra attempts, changed preserved history/checkpoint, missing review and failed
final integrity checks remain rejected. Timing explicitly includes interruption
downtime in the full wall interval and separately reports resumed-attempt elapsed
time. All 213 repository tests, 15 legacy metric checks, correctness lint and
unchanged results-table regeneration pass. Four synthetic recovery contracts
cover the new behavior; real continuation follows source
checks, commit and verified publication. The main final research report remains
pending completed training, both calibrations and both one-time tests.

Recovery changes were published/remote-verified at
`6da72a2c8703888e0e9db0b88f2d71ef7fb7c128`; CI and CodeQL both pass.
The single approved resume created `attempt-0002` at November 11 02:26:52 UTC
(November 10 evening in Chicago), restoring the exact epoch2 digest. Observed
progress advances from update 2,326 to 2,327 and onward in the replayed third
epoch. The original interrupted console/checkpoints remain unchanged. Private
continuation metadata identifies the actual live process/session; monitor it
without launching another attempt.

Final reporting distinguishes 111,582 pair presentations in the retained
three-epoch trajectory from at least 130,654 presentations across both attempts,
including the last reported 19,072 discarded partial-epoch presentations. This
is a lower bound on executed work because crash progress was logged periodically.
The model configuration/optimizer schedule and protected runtime remain unchanged.
