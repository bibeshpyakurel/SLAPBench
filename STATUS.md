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
