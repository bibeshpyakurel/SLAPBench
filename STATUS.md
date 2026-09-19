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
