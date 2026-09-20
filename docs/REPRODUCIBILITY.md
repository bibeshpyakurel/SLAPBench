# Reproducing SLAPBench from a clone

The repository preserves source code, dependency lists, fixed pair manifests,
published per-pair scores, derived audit reports, manuscript sources, reviewed
run logs and interrupted-run backups. These allow offline inspection of the
recorded analyses. A clone alone cannot regenerate model outputs: the licensed
fingerprint datasets, downloaded model weights, credentials and GPU environment
are local assets and are intentionally absent from Git.

This is a research repository, not a byte-for-byte backup of the workstation.
The 28 GiB fingerprint dataset tree, downloaded model checkpoints, the 8 GiB
Python environment, local credentials and personal automation settings stay
here. Model weights are large, may have separate terms, and their external drive
was unavailable at audit time; the code, model identifiers and download procedure
are published instead. Without the datasets, a clone cannot train or reproduce
the fingerprint experiments even if it can install the software. This is a
practical separation of assets, not a technical access control: someone who
independently obtains authorized data and weights could run the public code.

The `related_papers/` directory holds third-party full-text copies and stays
local. The public [`references.bib`](../paper/manuscript/references.bib),
[`STATUS.md`](../STATUS.md), and protocol draft preserve source citations,
findings and research direction without redistributing those copies.

## Local assets and paths

| Asset | Path expected by code | What a new machine needs |
| --- | --- | --- |
| SD302b | `datasets/sd302b/` | Authorized source data plus the local metadata/processed images expected by `code/build_master_df.py` and `code/run_verification.py` |
| RidgeBase | `datasets/ridgebase/Fingerprint_Train_Test_Split/` | Data obtained under its own access terms; pair builder accepts `--root` |
| Precise | `datasets/Precise/` | Authorized JPEG collection; pair builder accepts `--root` |
| Local VLM weights | `models/` or `SLAPBENCH_MODELS_DIR` | Download separately using `code/setup_models.py --download` or restore an existing weight directory |
| Pixtral weights | Hugging Face cache | Separately cache the repository listed below; inference loads it by ID |
| API credentials | local `.env` or environment variables | Only if running paid API experiments; never commit credentials |

The current workstation has a `models` symlink to
`/media/bibesh/DATA/models`. Its target was unavailable during this audit, so
the actual weight files and revisions could not be inventoried or verified.
The symlink is not committed. On another computer, use a regular ignored
`models/` directory or set `SLAPBENCH_MODELS_DIR` to its weight directory. If
the workstation's drive is unmounted, mount it before downloading; the setup
script refuses to write through a dangling symlink.

`code/model_registry.py` is the shared mapping used by the setup and SD302b
inference scripts:

| Runner key | Model repository | Local subdirectory |
| --- | --- | --- |
| `internvl3` | `OpenGVLab/InternVL3-8B` | `internvl3-8b/` |
| `qwen25vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen25vl-7b/` |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3vl-8b/` |
| `gemma3` | `google/gemma-3-12b-it` | `gemma3-12b/` |
| `pixtral` | `unsloth/Pixtral-12B-2409-bnb-4bit` | Hugging Face cache; no local subdirectory |

The model setup script downloads the four local models to exactly these folder
names and Pixtral to the Hugging Face cache. The repository IDs document the
intended sources, but do not pin revisions. For an exact rerun, record the
resolved model revisions, package versions, seeds, hardware, dataset checksums
and pair manifest hashes in a run record before inference. Existing historical
runs may not have all of this provenance.

## Practical setup

```bash
git switch dev
git config core.hooksPath .githooks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/setup_models.py --check-only
# Set SLAPBENCH_MODELS_DIR first if weights live outside this checkout.
# Run only when the selected storage is ready and download is intended:
python code/setup_models.py --download
python code/run_verification.py --dry-run
```

`--check-only` and `--dry-run` do not download weights or run inference. Some
model sources may require separate access or authentication. Dataset acquisition
and preparation remain manual and must follow their respective terms.

Some saved SD302b manifests contain absolute paths from the original machine;
RidgeBase and Precise manifests use paths recorded when the pairs were made.
Check each manifest's `img1_path` and `img2_path` before a new run. Remap paths
in a **new** manifest when moving computers; keep committed originals intact and
preserve the pair IDs, labels, sampling rules and audit trail. The recorded score
tables remain inspectable without local images. Existing manifests and score
tables include subject IDs and source filenames, while some manuscript figures
show fingerprint examples. Review these public artifacts and applicable data
terms before promoting a working-branch snapshot to a public release.

The logs under `logs/` and `.interrupted-backup` files under `results/` preserve
run history and crash recovery state. They are not additional verified result
sets and should not be merged into published metrics without a separate audit.
The archived backups were kept byte-for-byte. New logs and backups must be
checked for credentials and accidental dataset images before publication.
SHA-256 checksums and sizes for the seven newly published artifacts are in
[`artifact-inventory-20260920.csv`](../reports/artifact-inventory-20260920.csv).

The current research status and unresolved protocol decisions are in
[`STATUS.md`](../STATUS.md) and
[`EVALUATION_PROTOCOL_DRAFT.md`](EVALUATION_PROTOCOL_DRAFT.md). Existing
Precise/RidgeBase metrics and the post-hoc parser audit have known limits; do
not treat the old SD302b near-duplicate pairs as independent-capture biometric
evidence or make unsupported low-FAR claims.

## Publication check

`scripts/check_publication_scope.py` checks the Git index for dataset/model/
environment/credential paths and files over 50 MiB. It runs in CI and, when
`git config core.hooksPath .githooks` is set, before each commit. Agents must
still review the contents of new text results, logs and figures, because file
names alone cannot prove that a file has no credentials or raw fingerprint data.
