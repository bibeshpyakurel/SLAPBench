# Research status and audit log

## 2026-09-19 — approved documentation and offline-validation audit

The user confirmed the supervisor guide is no longer available and approved the
proposed documentation/validation plan. No experiment, API call, model download or
training run was performed. This is an evidence audit, not certification against
unavailable requirements. Future entries must be appended rather than replacing this log.

### Scope and gap table

The nine deliverables in the missing guide's §12 and eight steps in §16 cannot be
enumerated or classified faithfully. All 17 requirement-level assessments remain
**Unclear: specification unavailable**. The following is an observed-capability table,
not an invented substitute for those requirements.

| Capability | Status | Evidence / limitation |
|---|---|---|
| Dataset inventory | Done for mounted data | REPO_MAP.md; three datasets under `datasets/` |
| Pair manifests | Present; validation partial | `results/{precise,ridgebase}/pairs_*_eval.csv`; coverage report below |
| Image-only VLM results | Partial | `results/*/*/latest/`; missing scores, invalid answers and unmatched IDs |
| Embedding comparisons | Present | `results/{precise,ridgebase}/*/embed_*_eval.csv`; not a conventional matcher baseline |
| Conventional matcher baseline | Missing evidence | No NBIS commands on PATH or `.xyt` under datasets; manuscript discusses baseline as future work |
| Subject split | Partial | RidgeBase Task2 Train/Test verified; no established development split for all experiments |
| Minutiae text, overlays, encoder/fusion | Missing evidence | No corresponding implementation/result artifact established in `code/` or result trees |
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
   subjects, hands and paths exist; the missing guide's schema cannot be checked.
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
7. **Low-FAR claims:** Inspected paper/guide passages report FAR=0.1%, not 0.01%.
   The 592 and 676 primary-impostor sets have one-error resolutions of about 0.169%
   and 0.148%. Interpolation is not evidence of reliable measurement below these
   increments. With zero errors, the approximate 95% rule-of-three upper bounds
   would be 0.507% and 0.444%; for 7,656 trials about 0.0392%. These assume
   independent trials, while repeated subjects/images further limit inference.
   No claim of validated FAR=0.01% is supported by these counts alone.

### Milestone position

No formal M1–M5 completion is certified without the specification. Even under the
onboarding checklist's descriptions, M1 lacks demonstrated conventional-baseline
results and a reviewed evaluation protocol. Existing whole-SLAP and embedding
experiments do not establish that later milestone gates were passed. There is no
basis here to authorize fine-tuning or conclude that training ran ahead of a gate:
checkpoint contents were unavailable.

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
3. Establish development/test protocol, authorized dataset scope and acceptable FAR
   reporting with the supervisor; recover or replace the missing written requirements.
4. Establish conventional matcher availability/licensing and compare on agreed pairs.
5. Only after protocol/baseline review, consider new model runs or advanced methods.

Unresolved before experiments: authorized datasets; conventional matcher/license;
chosen hosted model and budget; split-review status; whether superseding analyses
are permitted; compute host and model-storage availability. These do not block
completion of this approved documentation task.

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

## 2026-09-19 — source instructions supplied; export and parsing audit

The user supplied the professor's full 16-section STUDENT RESEARCH INSTRUCTIONS
and authorized validation of the Qwen3 exports, response parsing and protocol
preparation. All 16 sections have been read. The source is preserved verbatim in
`docs/Fingerprint_Foundation_Model_Verification_Student_Guide.md`; earlier entries
saying it was unavailable are historical. The source has no Appendix A or C1–C7
orientation table, so those assertions in the onboarding checklist are not repeated.

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
same-person bias. No causal claim that missing minutiae alone explains this bias
is established; testing H2 is the intended controlled next experiment (§§1,15).

### Reconciliation with the actual §12 deliverables

| §12 item | Status | Evidence and exact remaining gap |
|---|---|---|
| 1 Dataset/split summary | Partial | REPO_MAP.md has mounted inventories and RidgeBase split evidence; no approved single-finger development/final-test subject lists |
| 2 Exact pair manifest | Partial | `results/precise/pairs_precise_eval.csv`, `results/ridgebase/pairs_ridgebase_eval.csv`; existing whole-slap manifests, not the required controlled single-finger pilot |
| 3 Bozorth3 scores/metrics | Missing evidence | No conventional output in inventoried results; NBIS commands were not found on PATH (REPO_MAP.md) |
| 4 Image-only VLM on same pairs | Partial | `results/*/*/latest/`; source and parsing results exist, but not paired with conventional/minutiae arms under the required reviewed split |
| 5 VLM image+minutiae | Missing evidence | Existing prompt/runner code has no explicit minutiae arm; no corresponding result artifact in REPO_MAP.md |
| 6 Joint per-pair score/error CSV | Partial | New `reports/response-audit-20260919/parsed/` gives per-run traceability; no joint Bozorth3/image-only/minutiae score table |
| 7 ROC/DET/distributions | Partial | `paper/manuscript/figures/fig_roc_curves.*`, score-distribution figures; no complete matched-pilot ROC/DET comparison |
| 8 TAR at supported fixed FAR | Partial | Existing manuscript metrics are historical; no development-fixed thresholds and uncertainty for the new controlled pilot; see protocol draft |
| 9 Minutiae benefit/failure analysis | Missing for H2 | New audit explains observed parsing/output failures; no image+minutiae experiment from which to estimate benefit |

### Reconciliation with the actual §16 first assignment

| §16 step | Status | Evidence and remaining gap |
|---|---|---|
| 1 Manageable development subset | Partial | Three mounted datasets inventoried; authorized subset with repeated single-finger impressions and approved split not fixed |
| 2 Genuine/larger impostor lists | Partial | SD302b exhaustive list exists (176/7,656) but is a resolution-copy control; existing Precise/RidgeBase balanced lists do not fulfill a new single-finger pilot |
| 3 Bozorth3 all-pair raw scores | Missing evidence | No result artifact; baseline installation/version still pending |
| 4 Hosted VLM Prompt A | Partial | Existing hosted SD302b CSVs use older prompts/protocol; no approved pilot with frozen source Prompt A |
| 5 Minutiae + Prompt B | Missing evidence | No explicit-minutiae run in recorded outputs |
| 6 ROC/same-FAR comparison | Missing for pilot | Old figures do not compare the three required arms |
| 7 Benefit/refusal report | Missing for H2 | Offline refusal-candidate/error audit exists, but no measured minutiae benefit |
| 8 Review before fine-tuning | Gate still pending | No reviewer sign-off is recorded; no fine-tuning was initiated by this work |

M1 (§14) is not demonstrated: conventional baseline and a reviewed controlled
single-finger protocol remain missing. Existing slap-level experiments do not
complete M5 (§8/§14), which requires per-finger matching and fusion comparisons.

### Protocol and next actions

`docs/EVALUATION_PROTOCOL_DRAFT.md` now implements the provided source rather than
inventing a replacement specification. The guide settles the need for subject-
disjoint development/final test, NBIS baseline, same pairs, minutiae arm, frozen
thresholds and count-supported low-FAR claims (§§3–6,9–12,16). It does not settle
lab data authorization, concrete subject splits, provider/model/budget, NBIS
installation/version or optional VeriFinger access. No supervisor agreement on
those choices is claimed, and no message was sent to the supervisor.

Next: confirm those decisions, validate per-finger data and image/minutiae coordinate
handling, establish Bozorth3 on the frozen development manifest, then run the permitted
image-only/minutiae pilot. Investigate resizing/template/decoding as controlled
hypotheses, not post-hoc explanations. Keep final test untouched; previously inspected
subjects cannot silently become an unexamined confirmatory test set.

Validation for this entry: six regression tests pass (including corrupted-ID
normalization, contradictory/error responses, detection of an altered export score,
and preserving existing output directories); CI now runs them. All 15 legacy
metric files reproduce, correctness lint passes, and source/document hash checks
confirm unchanged originals and a verbatim supervisor-instruction copy. The new
parsed files are an explicitly post-hoc conservative analysis; production inference
parsers have not been changed. Any formal new run needs the reviewed JSON parser
and protocol before execution.
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
