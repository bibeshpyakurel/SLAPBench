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
