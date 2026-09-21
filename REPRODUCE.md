# Reproducing SLAPBench

Every table and headline number in
[arXiv:2607.15517](https://arxiv.org/abs/2607.15517), mapped to the file in
this repository that holds it and the command that produces it.

This document is grounded in the code as committed. Where a fact about the
published run is not recorded anywhere in this repository, it is written as
**`UNKNOWN — to be supplied by the author`** rather than guessed. Those gaps
are listed together in [What is not reproducible from this repository](#what-is-not-reproducible-from-this-repository).

Companion documents:

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — the publication
  boundary: which assets are deliberately local-only and why.
- [`STATUS.md`](STATUS.md) — research status, evidence and open limitations.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — branch, commit and check policy.

---

## Three tiers of reproduction

| Tier | Needs | What you can reproduce |
|---|---|---|
| **A — offline** | Python 3.11 or 3.12, `requirements-verify.txt` | That every published metric follows from the committed per-pair scores; the results table; the pair-manifest invariant. No GPU, no dataset, no API key. |
| **B — local inference** | Tier A + a CUDA GPU + NIST SD302b + model weights | The four open-weight models' per-pair scores from the images. |
| **C — API inference** | Tier B + `ANTHROPIC_API_KEY` | The Claude Opus 4.8 rows. Not bit-reproducible; see [below](#what-is-not-reproducible-from-this-repository). |

Tier A is what CI runs on every push and pull request
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Start there.

---

## Tier A — verify the published numbers without a GPU

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-verify.txt   # pandas, numpy, scikit-learn only

python scripts/check_publication_scope.py   # repository boundary
python code/verify_metrics.py               # the reproducibility check
python -m unittest discover -s tests -v     # offline parser/export audit
python code/generate_results_table.py       # regenerates results/sd302b/SLAPBench_results.html

pip install ruff && ruff check code/ tests/ --select F,E9,B   # separate CI job
```

`code/verify_metrics.py` walks every `results/**/*.metrics.json`, recomputes
AUC, EER and the genuine/impostor counts straight from the sibling per-pair
CSV, and exits non-zero on disagreement (tolerances: AUC `5e-3`, EER
`0.75` percentage points). It also asserts the manifest invariant the paper's
abstract states: `7,832 = 176 genuine + 7,656 impostor`. This is the check that
catches a results file being swapped without the analysis being rerun.

> **Do not use `python code/run_verification.py --metrics <csv>` to "check" a
> published run.** That mode recomputes the metrics *and overwrites the sibling
> `.metrics.json`* (`save_metrics_json`, `run_verification.py`). It is a
> post-run tool, not a verifier. `code/verify_metrics.py` is read-only.

### Independent check: pair order

Every published SD302b run visits pairs in exactly the order recorded in
`results/sd302b/task8_pairs_all.csv`. That is checkable offline and does not
need the images:

```bash
python - <<'PY'
import csv, glob
manifest = [r["pair_id"] for r in csv.DictReader(open("results/sd302b/task8_pairs_all.csv"))]
for path in sorted(glob.glob("results/sd302b/*/latest/*.csv") + glob.glob("results/sd302b/*/previous/*/*.csv")):
    ids = [r["pair_id"] for r in csv.DictReader(open(path))]
    print(f"{'OK ' if ids == manifest else 'DIFF'}  {path}")
PY
```

---

## 1. Obtaining NIST SD302b

The images are **not redistributed in this repository** and never will be; the
publication-boundary check (`scripts/check_publication_scope.py`) rejects any
commit that stages `datasets/`.

- Dataset home page: <https://www.nist.gov/itl/iad/btg/nist-special-database-302>
  (this is the URL cited as `nist_sd302` in
  [`paper/manuscript/references.bib`](paper/manuscript/references.bib))
- Request / download form: <https://nigos.nist.gov/datasets/sd302/request>
- Technical note describing the collection: NIST TN 2007,
  <https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2007.pdf>

SD302 is distributed as nine lettered subsets. **SLAPBench uses SD302b only** —
the operator-assisted rolled and slap impressions, PNG, approximately 4.5 GB.

**Terms of use.** This repository does not restate SD302's terms, because they
are not quoted on the pages above in a form that can be reproduced verbatim
here. Read the conditions presented on the request form at
<https://nigos.nist.gov/datasets/sd302/request> and in whatever agreement or
README NIST supplies with the download, and follow those. The request form is
reviewed by a person at NIST and access has historically required an
institutional email address. The dataset's terms are separate from this
repository's MIT licence, which covers the code only.

The same applies to the v2 datasets: **RidgeBase** requires a signed licence
agreement from the University at Buffalo CUBS lab
(<https://www.buffalo.edu/cubs/research/datasets.html>), and **Precise** is an
authorized-access JPEG collection. Neither is redistributed here.

## 2. Preparing the dataset tree

The pipeline expects the extracted SD302b tree at `datasets/sd302b/` with the
NIST `images/baseline/` layout, `participants.csv`, and the per-directory
`segmentation_*.csv` / `checksum_*.csv` files shipped by NIST. Then:

```bash
python code/archive_non_slap.py          # dry run — prints the plan, moves nothing
python code/archive_non_slap.py --run    # moves rolls, FRGP-15 thumbs and segmented crops
python code/build_master_df.py           # → datasets/sd302b/master_dataframe.csv   (2,336 rows)
python code/build_slap_images_df.py      # → datasets/sd302b/slap_images.csv        (584 rows)
python code/run_verification.py --dry-run   # confirms pairs build and images resolve
```

`build_master_df.py` hard-codes the seven errata subjects (`00002302`,
`00002354`, `00002361`, `00002420`, `00002497`, `00002534`, `00002561`) and
marks them `has_errata`; `build_pairs()` filters them out of both the 500 PPI
and 1000 PPI Device-R pools. What remains is 88 subjects present at both
resolutions for each of FRGP 13 and 14 — which is where 176 genuine pairs and
`C(88,2) = 3,828` impostor pairs per hand come from. `--dry-run` prints those
counts, so you can confirm the split on your own copy of the data before
spending GPU time.

`slap_images.csv` is the only dataset file `run_verification.py` reads.

## 3. The committed pair manifests

Two manifests are committed under `results/sd302b/`:

| File | Rows | Composition | Used by |
|---|---|---|---|
| `task8_pairs.csv` | 352 | 176 genuine + 176 impostor (balanced, 88 per FRGP per class) | the legacy balanced protocol described in `README.md` |
| `task8_pairs_all.csv` | 7,832 | 176 genuine + 7,656 impostor (`C(88,2) = 3,828` impostors per FRGP) | **every published SD302b run and every number in the paper** |

### How they were generated

`build_pairs()` in [`code/run_verification.py`](code/run_verification.py):

- **Seed: `42`** — `--seed` defaults to `42`; `build_pairs(..., seed=42)` calls
  `random.seed(seed)` once, before any sampling.
- Genuine pairs: for each FRGP (13 = right, 14 = left), the subjects present at
  both Device R / 500 PPI and Device R / 1000 PPI are intersected, the list is
  `random.shuffle`d, and all 88 are kept (`--n-genuine` defaults to `None` =
  all). Each pair is that subject's 500 PPI image versus their 1000 PPI image.
- Impostor pairs, **exhaustive** (`--all-impostors`): `itertools.combinations`
  over the *sorted* subject list — deterministic, no sampling. 3,828 per FRGP.
- Impostor pairs, **balanced** (default, `--n-impostor 88`): the subject list is
  `random.shuffle`d and the first 88 unique unordered pairs per FRGP are taken.
  This branch is seed-dependent.
- Finally the whole list is `random.shuffle`d — so the *row order* of both
  manifests is seed-dependent even in the exhaustive case.

Regenerating the exhaustive manifest therefore reproduces both the pair set and
its order, given the same `slap_images.csv`:

```bash
python code/run_verification.py --pairs-only --all-impostors   # → results/sd302b/task8_pairs_all.csv
python code/run_verification.py --pairs-only                   # → results/sd302b/task8_pairs.csv
```

> **Both commands overwrite a committed file.** Run them on a scratch checkout,
> or `git restore results/` afterwards. The repository policy is that `results/`
> is never modified except by a deliberate, reviewed re-run.

> **Schema note.** The two committed manifests do not have the same columns.
> `task8_pairs.csv` is the 15-column form the runner writes (including
> `img1_path`/`img2_path` — absolute paths from the original workstation — and
> per-subject demographics). `task8_pairs_all.csv` is a 7-column redaction
> (`pair_id, label, frgp, which_hand, subject1, subject2, ground_truth`) with no
> image paths. Regenerating it with `--pairs-only --all-impostors` produces the
> 15-column form, so the file will differ from the committed one in schema even
> when the pair set and order match. Compare on `pair_id` order, not bytes.
> `results/sd302b/current/task8_pairs_current_auditable.csv` is a third,
> path-relative auditable export.

## 4. Running the models

### Setup

```bash
pip install -r requirements.txt
hf login
python code/setup_models.py --check-only     # reports GPU, disk, packages
python code/setup_models.py --download       # downloads weights (see note below)
```

Weights are resolved through [`code/model_registry.py`](code/model_registry.py):
`SLAPBENCH_MODELS_DIR` if set, otherwise `<repo>/models/`.

| Runner key | Hugging Face repository | Local subdirectory | Load |
|---|---|---|---|
| `internvl3` | `OpenGVLab/InternVL3-8B` | `internvl3-8b/` | 4-bit NF4 (bitsandbytes), compute dtype bfloat16 |
| `qwen25vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen25vl-7b/` | 4-bit NF4, compute dtype float16 |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3vl-8b/` | 4-bit NF4, compute dtype bfloat16 |
| `gemma3` | `google/gemma-3-12b-it` | `gemma3-12b/` | 4-bit NF4, compute dtype bfloat16 |
| `pixtral` | `unsloth/Pixtral-12B-2409-bnb-4bit` | Hugging Face cache (loaded by repo ID) | pre-quantized 4-bit in `config.json` |
| `anthropic` | Anthropic API | — | `--anthropic-model claude-opus-4-8` |
| `openai` | OpenAI API | — | `--openai-model <name>` |

Pixtral has **no SD302b run**; it appears only in the RidgeBase and Precise
trees. The five models in the paper's SD302b table are `qwen3vl`, `anthropic`,
`gemma3`, `internvl3`, `qwen25vl`.

> **InternVL3 is loaded in 4-bit, not bfloat16.** The `MODELS` registry entry
> carries `"load_4bit": False`, but that flag is never read on the `internvl3`
> branch of `load_model()`, which unconditionally builds a
> `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`. The
> "Loaded in bfloat16 / ~16 GB VRAM" line in `README.md` describes an earlier
> configuration. All four open-weight models quantize on CPU
> (`device_map={"": "cpu"}`) and are then moved to CUDA, which is what keeps
> the quantization kernel from OOM-ing on a small GPU.

### The three prompting strategies

`PROMPTS` in `run_verification.py` defines exactly three keys, and they are the
only values `--prompting` accepts:

- `zero_shot` — binary A/B, no context
- `task_description` — binary A/B, with a sentence of domain context
- `similarity_score` — a 0–100 integer confidence, parsed by
  `parse_score_response()` and clamped to `[0, 100]`

The `chain_of_thought` strategy described in older parts of `README.md` is
**not** implemented in the current code and produced none of the published
numbers.

### Per-model commands (SD302b, exactly as published)

Every published SD302b run used the **exhaustive** manifest. `--all-impostors`
is required; omitting it silently runs the 352-pair balanced set instead and
will not reproduce the paper.

```bash
# Qwen3-VL-8B-Instruct
python code/run_verification.py --model qwen3vl   --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model qwen3vl   --prompting task_description --all-impostors --run
python code/run_verification.py --model qwen3vl   --prompting similarity_score --all-impostors --run

# Gemma-3-12B-IT
python code/run_verification.py --model gemma3    --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model gemma3    --prompting task_description --all-impostors --run
python code/run_verification.py --model gemma3    --prompting similarity_score --all-impostors --run

# InternVL3-8B-Instruct
python code/run_verification.py --model internvl3 --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model internvl3 --prompting task_description --all-impostors --run
python code/run_verification.py --model internvl3 --prompting similarity_score --all-impostors --run

# Qwen2.5-VL-7B-Instruct
python code/run_verification.py --model qwen25vl  --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model qwen25vl  --prompting task_description --all-impostors --run
python code/run_verification.py --model qwen25vl  --prompting similarity_score --all-impostors --run

# Claude Opus 4.8 (Anthropic API; needs ANTHROPIC_API_KEY in .env or the environment)
python code/run_verification.py --model anthropic --anthropic-model claude-opus-4-8 --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model anthropic --anthropic-model claude-opus-4-8 --prompting task_description --all-impostors --run
python code/run_verification.py --model anthropic --anthropic-model claude-opus-4-8 --prompting similarity_score --all-impostors --run
```

`--anthropic-model` defaults to `claude-sonnet-4-6`, which is **not** the
published model. Pass `claude-opus-4-8` explicitly. The model identifier
actually used is recorded in each run's metrics file — see
[Environment of the published run](#environment-of-the-published-run).

Output goes to `results/sd302b/<model_key>/latest/task8_<model_key>_<prompting>_<YYYYMMDD_HHMM>.csv`,
with the previous contents of `latest/` rotated into `previous/<YYYY-MM-DD>/`.
The run checkpoints after every pair (`f_out.flush()`); resume with
`--resume <that csv>`.

### RidgeBase and Precise (paper Table II / v2)

```bash
# manifests
python code/ridgebase_pairs_full.py \
    --root datasets/ridgebase/Fingerprint_Train_Test_Split --split Test --seed 42 \
    --out results/ridgebase/pairs_ridgebase_full.csv \
    --balanced-out results/ridgebase/pairs_ridgebase_eval.csv

python code/precise_pairs.py --root datasets/Precise --seed 42 \
    --out results/precise/pairs_precise_full.csv \
    --balanced-out results/precise/pairs_precise_eval.csv

# prompted inference (loads the model once, runs all three strategies)
python code/run_ridgebase_prompted.py --model qwen3vl \
    --pairs results/ridgebase/pairs_ridgebase_eval.csv

# prompt-free embedding matching (cosine on the vision tower)
python code/embed_verify.py --model qwen3vl \
    --pairs results/ridgebase/pairs_ridgebase_eval.csv \
    --out results/ridgebase/qwen3vl/embed_qwen3vl_eval.csv

# summaries — never overwrite an existing one; write to a new dated directory
python code/analyze_ridgebase.py --dir results/ridgebase \
    --output-dir reports/ridgebase-summary-YYYYMMDD
```

`ridgebase_pairs_full.py` and `precise_pairs.py` both default to `--seed 42`
and use a dedicated `random.Random(seed)`; `ridgebase_pairs.py` (the earlier
CL2CL builder) hard-codes `random.seed(42)` with no flag.

---

## Paper claim → file → command

Numbers below are quoted from the committed `.metrics.json` files, which is
where `code/verify_metrics.py` and `code/generate_results_table.py` read them.
Every "Command" is a Tier-A command unless it says otherwise.

### Table I — SD302b verification, 7,832 pairs (`tab:main_results`)

Rendered from the metrics JSONs by `code/generate_results_table.py`; verified
against the CSVs by `code/verify_metrics.py`.

| Paper claim | File in `results/` | Command |
|---|---|---|
| Qwen3-VL-8B ZS 74.1% acc / FAR 26.5% | `sd302b/qwen3vl/previous/2026-06-13/task8_qwen3vl_zero_shot_20260613_1557.{csv,metrics.json}` | `python code/verify_metrics.py` |
| Qwen3-VL-8B TD 3.3% acc / FAR 98.9% (collapsed) | `sd302b/qwen3vl/previous/2026-06-13/task8_qwen3vl_task_description_20260613_1635.*` | `python code/verify_metrics.py` |
| Qwen3-VL-8B similarity AUC **1.000**, EER **0.00%** | `sd302b/qwen3vl/latest/task8_qwen3vl_similarity_score_20260613_1712.*` | `python code/verify_metrics.py` |
| Claude Opus 4.8 ZS 80.3% acc / **FAR 20.2%** (best binary) | `sd302b/anthropic/previous/2026-06-20/task8_anthropic_zero_shot_20260620_1850.*` | `python code/verify_metrics.py` |
| Claude Opus 4.8 TD 50.2% acc / FAR 50.9% (sole non-collapse) | `sd302b/anthropic/latest/task8_anthropic_task_description_20260620_2250.*` | `python code/verify_metrics.py` |
| Claude Opus 4.8 similarity AUC 0.953, EER 11.75% | `sd302b/anthropic/previous/2026-06-20/task8_anthropic_similarity_score_20260620_1437.*` | `python code/verify_metrics.py` |
| Gemma-3-12B ZS 5.7% acc / FAR 96.4% (collapsed) | `sd302b/gemma3/previous/2026-06-20/task8_gemma3_zero_shot_20260620_1346.*` | `python code/verify_metrics.py` |
| Gemma-3-12B TD FAR 100.0% (collapsed) | `sd302b/gemma3/previous/2026-06-20/task8_gemma3_task_description_20260620_1449.*` | `python code/verify_metrics.py` |
| Gemma-3-12B similarity AUC 0.837, EER 15.10% | `sd302b/gemma3/latest/task8_gemma3_similarity_score_20260620_1552.*` | `python code/verify_metrics.py` |
| InternVL3-8B ZS 29.3% acc / FAR 72.3% | `sd302b/internvl3/previous/2026-06-14/task8_internvl3_zero_shot_20260614_1352.*` | `python code/verify_metrics.py` |
| InternVL3-8B TD FAR 100.0% (collapsed) | `sd302b/internvl3/previous/2026-06-14/task8_internvl3_task_description_20260614_1422.*` | `python code/verify_metrics.py` |
| InternVL3-8B similarity AUC 0.590, EER 48.09%, **Δ = −21.4 (inverted)** | `sd302b/internvl3/latest/task8_internvl3_similarity_score_20260614_1456.*` | `python code/verify_metrics.py` |
| Qwen2.5-VL-7B ZS 32.0% acc / FAR 69.6% | `sd302b/qwen25vl/previous/2026-06-14/task8_qwen25vl_zero_shot_20260614_1535.*` | `python code/verify_metrics.py` |
| Qwen2.5-VL-7B TD FAR 100.0% (collapsed) | `sd302b/qwen25vl/previous/2026-06-14/task8_qwen25vl_task_description_20260614_1609.*` | `python code/verify_metrics.py` |
| Qwen2.5-VL-7B similarity AUC 0.567, EER 43.34% | `sd302b/qwen25vl/latest/task8_qwen25vl_similarity_score_20260614_1644.*` | `python code/verify_metrics.py` |
| TAR@FAR = 0.1% column | computed on the fly from the similarity-score CSVs | `python code/generate_results_table.py` → `results/sd302b/SLAPBench_results.html` |
| Per-FRGP FAR splits quoted in the text (e.g. Qwen3-VL 29.0% / 24.0%) | the `per_frgp` block of each binary `.metrics.json` | `python -c "import json;print(json.load(open('results/sd302b/qwen3vl/previous/2026-06-13/task8_qwen3vl_zero_shot_20260613_1557.metrics.json'))['per_frgp'])"` |
| Abstract: "7,832 pairs (176 genuine, 7,656 impostor)" | `sd302b/task8_pairs_all.csv` | `python code/verify_metrics.py` (asserts the invariant) |
| Collapse in 5 of 10 binary configurations, FAR 96.4–100% | the `collapsed` flag in the ten binary `.metrics.json` files | `python code/verify_metrics.py` |

Reproducing any row **from the images** is the Tier-B/C command for that model
and prompting strategy in [section 4](#per-model-commands-sd302b-exactly-as-published).

### Table II — RidgeBase, 1,484-pair balanced manifest (`tab:ridgebase_main`)

| Paper claim | File in `results/` | Command |
|---|---|---|
| Manifest: 592 genuine + 592 primary impostor + 300 diagnostic impostor | `ridgebase/pairs_ridgebase_eval.csv` | — (inspect the `category` column) |
| Embedding AUC 0.469 / 0.400 / 0.434 / 0.509 (Qwen3-VL, Qwen2.5-VL, InternVL3, Gemma-3) | `ridgebase/<model>/embed_<model>_eval.csv` | `python code/analyze_ridgebase.py --dir results/ridgebase --output-dir reports/ridgebase-summary-YYYYMMDD` |
| Prompted similarity AUC 0.504–0.558, EER 43.6–51.4% | `ridgebase/<model>/latest/rb_<model>_similarity_score_*.csv` | same |
| Snapshot of those metrics as published | `ridgebase/summary_metrics.json`, `ridgebase/SUMMARY.md` | — |
| Claude Opus 4.8 and GPT-5.6-sol rows | **not run** — the paper prints them as pending | — |

`code/verify_metrics.py` does **not** cover RidgeBase or Precise: those trees
carry `summary_metrics.json`, not the per-run `.metrics.json` the verifier
recomputes. The equivalent trees for Precise are `results/precise/` with
`--dir results/precise`.

### Table III — demographic fairness (`tab:fairness`)

| Paper claim | File | Command |
|---|---|---|
| Per-subgroup AUC/EER for Qwen3-VL, Claude Opus 4.8, Gemma-3 | derived from the three similarity-score CSVs **plus** `datasets/sd302b/participants.csv` | `python code/paper/fairness_analysis.py` |

This is the one paper table that is **not** Tier A: `fairness_analysis.py` reads
demographics from `datasets/sd302b/participants.csv`, which is not committed.
The similarity-score CSVs alone do not carry subject demographics.

### Figures

All figure scripts write to `paper/manuscript/figures/`. The committed PDFs and
PNGs are the published versions. **`matplotlib` is required by every figure
script but is declared in neither `requirements.txt` nor
`requirements-verify.txt`** — install it separately (`pip install matplotlib`)
until that is fixed.

| Figure | Script | Inputs |
|---|---|---|
| `fig_score_dist_all/strong/failed.pdf`, `fig_roc_curves.pdf` | `python code/paper/generate_figures.py` | the five SD302b similarity-score CSVs (Tier A) |
| `fig_fairness.pdf` | `python code/paper/fairness_analysis.py` | above **+ `datasets/sd302b/participants.csv`** |
| `fig_qualitative.pdf`, `fig_slap_anatomy.pdf`, `slap_*.png` | `python code/paper/build_image_figures.py` | **requires SD302b images** |
| `fig_capture_gradient.pdf` | `python code/generate_gradient_figure.py` | the committed `embed_<model>_eval.csv` files across `results/{sd302b,precise,ridgebase}/` (Tier A) |
| `fig_ridgebase_roc.pdf`, `fig_ridgebase_diagnostic.pdf`, `fig_sd302b_vs_ridgebase.pdf` | `python code/generate_ridgebase_figures.py` | RidgeBase result CSVs (Tier A) |

### Resolution-matched ablation (Section "the perfect score")

| Paper claim | File | Command |
|---|---|---|
| Matched-resolution manifest | `sd302b/matched/task8_pairs_matched.csv`, `sd302b/matched_reversed/task8_pairs_matched.csv` | `python code/paper/build_matched_pairs.py` (needs the dataset) |
| Qwen3-VL matched-resolution scores | `sd302b/matched/qwen3vl/task8_qwen3vl_similarity_score_matched.csv` | `python code/paper/run_matched.py --model qwen3vl --manifest results/sd302b/matched/task8_pairs_matched.csv` (Tier B) |

### Building the paper

```bash
# paper/manuscript/  — IEEE journal version (IEEEtran.cls is in that directory)
# paper/eccv2026_submission/ — ECCV/LNCS version + supplementary
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Environment of the published run

This section is the environment record. Facts that the repository actually
establishes are stated with the file that establishes them. Everything else is
marked `UNKNOWN — to be supplied by the author`.

### Decoding and sampling — established by the code

| Setting | Value | Where |
|---|---|---|
| Open-weight decoding | `do_sample=False` (greedy) for all four local backends | `run_verification.py`, the `_infer_*` functions |
| Open-weight token limit | `max_new_tokens=48` | `_infer_qwen25vl`, `_infer_qwen3vl`, `_infer_gemma3` defaults |
| Open-weight token limit, Pixtral | `max_new_tokens=96` | `_infer_pixtral` default |
| Open-weight token limit, InternVL3 | `max_new_tokens=150` (`gen_config = dict(max_new_tokens=150, do_sample=False)`) | `_infer_internvl3` |
| OpenAI backend | `max_tokens=50, temperature=0` (non-reasoning models); `max_completion_tokens=2000, reasoning_effort="minimal"` for `gpt-5*`/`o*` | `_infer_openai` |
| **Anthropic backend** | `max_tokens=50`. **No `temperature` is passed**, so the API default applies — the published Claude Opus 4.8 rows are *not* greedy-decoded and are not expected to be bit-reproducible | `_infer_anthropic` |
| Retries | up to 5, exponential backoff, on rate-limit/overload only | `_infer_openai`, `_infer_anthropic` |
| Pair-construction seed | `42` (`--seed`, default) | `build_pairs()` |
| RidgeBase / Precise pair seed | `42` (`--seed`, default; `random.Random(seed)`) | `ridgebase_pairs_full.py`, `precise_pairs.py` |
| Inter-call delay | `--delay` default `0.0` | `run_verification.py` |
| Image preprocessing | grayscale → `ImageOps.autocontrast` → white square pad → resize to **448×448** → RGB; encoded as PNG base64 for API models | `load_pil()`, `image_to_base64()` |
| System prompt | `"You are an expert fingerprint examiner."` | `SYSTEM_PROMPT` |

Note: `--seed` controls pair construction only. It is not applied to
`torch`/`numpy`, which is harmless because all local decoding is greedy.

### Model identifiers — established by the results

Each run records the model identifier it used in its `.metrics.json` `"model"`
field, and in the `model` column of every row of the per-pair CSV:

| Results directory | `"model"` recorded |
|---|---|
| `results/sd302b/anthropic/` | **`claude-opus-4-8`** |
| `results/sd302b/qwen3vl/` | `Qwen3-VL-8B-Instruct` |
| `results/sd302b/gemma3/` | `Gemma-3-12B-IT` |
| `results/sd302b/internvl3/` | `InternVL3-8B-Instruct` |
| `results/sd302b/qwen25vl/` | `Qwen2.5-VL-7B-Instruct` |

```bash
python -c "import glob,json;[print(p.split('/')[2], json.load(open(p))['model']) for p in sorted(glob.glob('results/sd302b/*/*/*.metrics.json'))+sorted(glob.glob('results/sd302b/*/*/*/*.metrics.json'))]"
```

For the open-weight models the recorded value is the *display name*, not a
Hugging Face revision. Inference loaded them **from local directories** under
`SLAPBENCH_MODELS_DIR` or `<repo>/models/` (`code/model_registry.py`), not by
repository ID from the Hub — the sole exception is Pixtral, which is loaded by
the repo ID `unsloth/Pixtral-12B-2409-bnb-4bit` from the Hugging Face cache.
`docs/REPRODUCIBILITY.md` records that on the original workstation `models/`
was a symlink to `/media/bibesh/DATA/models`, whose target was unavailable at
audit time, so the weight files and revisions **could not be inventoried**.

### Runtime — partially established

| Fact | Value | Source |
|---|---|---|
| Operating system | Linux | interpreter paths in `logs/matched_qwen3vl.log` |
| Python version of the published run | 3.12 | `venv/lib/python3.12/site-packages/...` in `logs/matched_qwen3vl.log` |
| CI-verified Python versions | 3.11 and 3.12 | `.github/workflows/ci.yml` |
| `torch` at the time of the Gemma-3 run | `< 2.6` — the first attempt failed with *"Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6"*, so the successful run used a later environment than that log | `logs/gemma3_experiments.log` |
| Declared dependency floors | `torch>=2.5.1`, `transformers>=4.51,<5.0`, `bitsandbytes>=0.43.0`, `accelerate>=0.27.0`, `scikit-learn>=1.3.0`, `anthropic>=0.25.0` | `requirements.txt` |

### To be supplied by the author

These are not recorded anywhere in this repository. They are the gaps a reader
needs filled before calling the published run exactly reproducible.

| Fact | Status |
|---|---|
| GPU model and VRAM used for the published runs | `UNKNOWN — to be supplied by the author` |
| CUDA toolkit and NVIDIA driver versions | `UNKNOWN — to be supplied by the author` |
| Exact `torch` version of the published runs | `UNKNOWN — to be supplied by the author` |
| Exact `transformers` version of the published runs | `UNKNOWN — to be supplied by the author` |
| Exact `bitsandbytes`, `accelerate`, `qwen-vl-utils`, `timm`, `einops` versions | `UNKNOWN — to be supplied by the author` |
| Exact `anthropic` SDK version of the Claude Opus 4.8 runs | `UNKNOWN — to be supplied by the author` |
| Hugging Face commit revision of each open-weight checkpoint | `UNKNOWN — to be supplied by the author` |
| Anthropic API snapshot / model version date behind `claude-opus-4-8` | `UNKNOWN — to be supplied by the author` |
| SHA-256 of the SD302b download used | `UNKNOWN — to be supplied by the author` |
| SHA-256 / version of the RidgeBase and Precise collections used | `UNKNOWN — to be supplied by the author` |
| Wall-clock cost and total API spend for the Claude runs | `UNKNOWN — to be supplied by the author` |

`requirements.txt` states lower bounds, not pins. It is deliberately left
unpinned here; pinning it to the versions above is the right follow-up once
those values are known, and should be done in the same change that fills this
table.

---

## What is not reproducible from this repository

Stated plainly, so nobody discovers it after a download:

1. **The images.** NIST SD302b, RidgeBase and Precise are not redistributed.
   Without them, nothing in Tier B or Tier C can run, and neither
   `code/paper/fairness_analysis.py` nor `code/paper/build_image_figures.py`
   will execute. Only the recorded scores are inspectable.
2. **Model weights.** They are loaded from local directories
   (`SLAPBENCH_MODELS_DIR` / `models/`), never committed, and **no revision is
   pinned**. `code/setup_models.py --download` fetches the current head of each
   Hugging Face repository, which may not be what produced the published
   numbers.
3. **The API models are external services.** Claude Opus 4.8 is reached over the
   Anthropic API. The service can change or be retired behind a fixed model
   string, the call passes no `temperature`, and the published rows therefore
   cannot be expected to reproduce exactly — only statistically. Re-running them
   also costs money: 7,832 pairs × 3 strategies = 23,496 two-image requests.
4. **Absolute paths.** `results/sd302b/task8_pairs.csv` embeds absolute paths
   from the original workstation
   (`/home/bibesh/Desktop/MultiModel LLM's/datasets/sd302b/...`). Remap them in
   a **new** manifest; leave the committed one intact.
5. **The environment.** See the table above — GPU, CUDA, and the exact package
   and checkpoint versions of the published run are not recorded.
6. **RidgeBase/Precise metrics are not CI-verified.** `code/verify_metrics.py`
   covers the fifteen SD302b `.metrics.json` files only. The v2 summaries are
   regenerated by `code/analyze_ridgebase.py` and carry the caveats in
   `STATUS.md` and `docs/EVALUATION_PROTOCOL_DRAFT.md`.
7. **The SD302b genuine pairs are near-duplicates.** This is a limitation of the
   published protocol, not of the repository: each mated pair is the 500 PPI and
   1000 PPI rendering of one capture. The similarity-scoring columns of Table I
   index near-duplicate sensitivity, not verification across independent
   impressions. Do not cite them as the latter. The binary/FAR columns, computed
   on same-resolution impostor pairs, are unaffected.

## Citing this artifact

See [`CITATION.cff`](CITATION.cff). The preprint is
[arXiv:2607.15517](https://arxiv.org/abs/2607.15517),
DOI [10.48550/arXiv.2607.15517](https://doi.org/10.48550/arXiv.2607.15517).
Tag [`v0.1.0`](https://github.com/bibeshpyakurel/SLAPBench/releases/tag/v0.1.0)
is the state of the code behind the preprint. A Zenodo DOI for the archived
snapshot is pending — see [`docs/ZENODO.md`](docs/ZENODO.md).
