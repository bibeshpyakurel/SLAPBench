# Repository map

Inventory date: 2026-09-19. This describes observed files, not compliance with a
supervisor specification. That specification is unavailable, confirmed by the user.
The onboarding checklist was supplied outside the repository; its section references
cannot substitute for the missing guide. See STATUS.md for findings and next steps.

## Layout and entry points

| Location | Observed purpose |
|---|---|
| `code/build_master_df.py`, `code/build_slap_images_df.py` | SD302b metadata preparation |
| `code/archive_non_slap.py` | Moves non-SLAP images; not run in this audit |
| `code/ridgebase_pairs.py`, `code/ridgebase_pairs_full.py` | RidgeBase pair construction; full variant supports Train/Test and device categories |
| `code/precise_pairs.py` | Precise genuine, primary-impostor and diagnostic pair construction |
| `code/run_verification.py` | Model loaders, prompts, response parsing, SD302b inference and metrics |
| `code/run_ridgebase_prompted.py` | Manifest-driven prompted inference, also used for Precise |
| `code/embed_verify.py` | Vision-encoder embedding comparisons saved as cosine scores |
| `code/test_paid_models.py`, `code/count_tokens_check.py` | Hosted-model preflight and token checks; may call paid services |
| `code/setup_models.py` | Checks/downloads model weights |
| `code/resume_precise_pixtral.py` | Repairs an interrupted CSV and resumes inference; not an audit command |
| `code/export_precise_qwen3_scores.py` | Joins scores to manifests and writes two 676-line exports |
| `code/analyze_ridgebase.py` | Summarizes saved RidgeBase or Precise scores; supports separate output directories |
| `code/verify_metrics.py` | Verifies legacy SD302b metric JSONs against sibling CSVs |
| `code/generate_figures.py`, `code/generate_ridgebase_figures.py`, `code/generate_gradient_figure.py` | Paper figures |
| `code/generate_results_table.py` | SD302b HTML results table |
| Root `build_*`, `run_matched.py`, `generate_figures.py`, `fairness_analysis.py` | Matched-pair construction, inference and paper analyses |
| `run_overnight.sh` | Experiment launcher; not run |
| `github/`, flat `results/<model>/` | Earlier public-release copies; coexist with newer dataset-specific folders |
| `results_current/`, `results_matched/`, `results_matched_reversed/` | Alternative pair manifests and matched results |
| `reports/audit-20260919/` | Offline audit counts and new provisional summaries |

## Local datasets and models

Excluded from Git. Counts below describe the mounted files observed during this audit.

| Directory | Size | Files / formats |
|---|---|---|
| `datasets/sd302b/` | 0.60 GiB | 596 files; 584 PNG images, metadata CSVs and notes |
| `datasets/Precise/` | 5.44 GiB | 7,358 JPEG images |
| `datasets/ridgebase/` | 21.31 GiB | 21,625 files; PNG, BMP, WSQ and supporting metadata |

Subject/hand information is recoverable from filenames and manifests; these are
sensitive metadata, not anonymous model prompts. `code/precise_pairs.py` and
`code/ridgebase_pairs_full.py` encode the filename conventions.
`models` is a symlink to `/media/bibesh/DATA/models`; its target was unavailable.
Consequently checkpoint inventory and training provenance could not be inspected.
No `.xyt` files were found under datasets. `mindtct` and `bozorth3` were not on PATH;
this does not prove they are absent from every disk. No usable commercial SDK was
established. Embedding score CSVs exist, but scores do not establish retained feature vectors.

## Results and conventions

`results/sd302b/`, `results/ridgebase/`, `results/precise/` are current dataset trees.
Prompted files live under `<model>/latest/` with model, strategy and timestamp in
filenames. Historical SD302b files also live under `previous/<date>/`.
Embedding comparisons use `embed_<model>_eval.csv`. Pair manifests carry identities,
labels and image paths. Prompted CSVs contain raw response text, latency, model key,
prompting setting and timestamp; score prompts and binary prompts have different fields.
Crash skips can be recorded in `.csv.skipped`; recovery backups are ignored.
The most recent timestamped result filename inspected is the Precise Pixtral
similarity run `20260908_2100`; this is a filename timestamp, not independent proof
of execution time. Existing summary files predate inclusion of Pixtral in the generator.

## Environment and documents

`requirements.txt` lists GPU/model and analysis dependencies; `requirements-verify.txt`
is the smaller verification environment. `venv/` is local and ignored. No root lockfile,
Dockerfile, conda environment definition or Makefile was found. Dependency ranges
are not a complete record of versions used for historical runs.

- `README.md`: benchmark explanation, historical results and reproduction notes.
- `project_guide.md`: earlier workflow and metrics guidance; not the missing supervisor guide.
- `paper_metrics_tracker.md`, `plan /SlapBench_Plan.txt`: historical planning/metric notes.
- `prompt_templates.txt`: prompt reference; executable prompts also live in code.
- `manuscript/`: active LaTeX manuscript, bibliography and figures, including fingerprint examples.
- `eccv2026_submission/`: separate submission sources and figures.
- `ECCV_2026_Paper_Template/`: third-party template, styles and documentation.
- `related_papers/`: excluded third-party reference material; background only.
- `CITATION.cff`, `LICENSE`, `CHANGELOG.md`, `SECURITY.md`: public-repository metadata.
- `AGENTS.md`, `CONTRIBUTING.md`, `.githooks/`: dev-only publication rules and safeguards.
- `.github/workflows/`: verification, lint, security analysis and release workflows.

## Version control and ambiguities

Audit started on clean `dev` at `ceb43ca`, matching `origin/dev`. Recent commits
established dev-only publication and integrated local research with the older public
repository. Multiple copies of code/results are intentional historical material;
which copy is authoritative must be established per analysis, not guessed.
Dataset-specific summaries are snapshots, not evidence every current model was included.
No original results, datasets, paper sources or checkpoints were changed in this audit.
