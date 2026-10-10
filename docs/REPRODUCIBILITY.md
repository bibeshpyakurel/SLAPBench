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
local. The public [`references.bib`](../paper/manuscript/references.bib) and
[`STATUS.md`](../STATUS.md) preserve source citations and findings without
redistributing those copies.

## Local assets and paths

| Asset | Path expected by code | What a new machine needs |
| --- | --- | --- |
| SD302b | `datasets/sd302b/` | Authorized source data plus the local metadata/processed images expected by `code/build_master_df.py` and `code/run_verification.py` |
| RidgeBase | `datasets/ridgebase/Fingerprint_Train_Test_Split/` | Data obtained under its own access terms; pair builder accepts `--root` |
| Precise | `datasets/Precise/` | Authorized JPEG collection; pair builder accepts `--root` |
| Private Precise preparation/training artifacts | `local/precise/` | Re-create the empty folders from [WORKSPACE_LAYOUT.md](WORKSPACE_LAYOUT.md); metadata, crops, minutiae, subject splits, checkpoints and raw runs remain local |
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

### October 2026 local model additions

On 2026-10-04 the user separately authorized downloading and setting up these
three models. This authorizes setup, not a fingerprint experiment or training.
The sources and immutable revisions are in `code/model_registry.py`; complete
file sizes and SHA-256 digests are in
[`snapshots.json`](../reports/model-setup-20261004/snapshots.json).

| Runner key | Official repository | Directory | Environment |
| --- | --- | --- | --- |
| `qwen35` | `Qwen/Qwen3.5-9B` | `qwen35-9b/` | Separate Transformers 5 environment (`.venv/`) |
| `gemma4` | `google/gemma-4-12B-it` | `gemma4-12b-it/` | Separate Transformers 5 environment (`.venv/`) |
| `internvl35` | `OpenGVLab/InternVL3_5-8B` | `internvl35-8b/` | Historical Transformers 4 environment (`venv/`, observed 4.57.6) |

The workstation stores these snapshots at
`/home/bibesh/.local/share/slapbench/models/`. The historical `models` symlink is
preserved. `.cache/local-model-storage.json` stores the new models' machine-local
directory; `SLAPBENCH_MODELS_DIR` takes precedence. The config, environments and
weights are ignored and must not be published. Historical model paths and the
default `code/setup_models.py` download set are unchanged.

For Qwen and Gemma, create a separate environment rather than upgrading the
environment used by earlier experiments:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local-models.txt
.venv/bin/python code/setup_next_models.py --download --verify \
    --models-dir "$HOME/.local/share/slapbench/models" --save-location
# No weights loaded, no inference, only synthetic images/configuration:
.venv/bin/python code/check_next_models.py --models qwen35 gemma4
```

Use the Transformers 4 environment for the original InternVL checkpoint's
custom model code. A fresh historical environment is installed from
`requirements.txt`; the setup check recorded here used Transformers 4.57.6.

```bash
venv/bin/python code/check_next_models.py --models internvl35
```

All three keys are registered in `run_verification.py` and in the manifest-driven
`run_ridgebase_prompted.py` runner used for Precise. The new loaders require the
downloaded local directories and working CUDA, and configure 4-bit NF4 loading.
They do not fall back to downloading another checkpoint during inference.
InternVL uses the original checkpoint's locally downloaded Python code
(`trust_remote_code=True`), with FlashAttention disabled; its revision and code
digests are recorded alongside the weights. Qwen and Gemma use native model
classes and templates with thinking disabled. The image adapters preserve the
historical 448px preprocessing; this is an implementation choice to review
alongside the new protocol, not evidence that it retains sufficient ridge detail.

The workstation currently has no NVIDIA device nodes and `nvidia-smi` cannot
communicate with its driver. Therefore GPU weight loading, generation, runtime
memory use and fingerprint performance remain unverified. Restore CUDA access
before attempting inference. Existing parsers also need the previously required
review before a formal run; no pair generation, segmentation, fine-tuning or
fingerprint experiment was performed during setup.

### Current Precise preparation (updated 2026-10-10)

See [WORKSPACE_LAYOUT.md](WORKSPACE_LAYOUT.md) for the prepared local folders,
historical asset locations and reviewed-publication workflow. This checkout is
the primary workspace; future authorized training runs locally.

The next study uses `datasets/Precise/`. SD302b and RidgeBase assets and commands
remain available for reproducing historical work. Do not move or delete those
assets to change the research focus; existing manifests depend on their paths.
Precise source/acquisition documentation, session meanings, physical resolution,
data terms and the collaborator's training/evaluation subjects still need to be
established before freezing a protocol. See the
[preparation note](notes/precise_preparation_20261004.md).

The aggregate inventory needs only Python's standard library:

```bash
python3 code/audit_precise_dataset.py --root datasets/Precise
# To retain another audit, choose a new dated path; existing reports are refused:
# python3 code/audit_precise_dataset.py --root datasets/Precise --output reports/precise-readiness-YYYYMMDD/inventory.json
```

This reads filenames and complete file bytes, without image decoding, GPU work,
network requests or dataset writes. It reports byte duplicates and aggregate
subject/hand coverage, not capture independence or single-finger labels.
Subject identifiers, source filenames, paths and per-image hashes are omitted.

The separate image-inspection utility requires Pillow. Its completed
[2026-10-10 report](../reports/precise-inspection-20261010/README.md) records
7,358 headers and 35 fully decoded sample images. Both output directories must
be new; choose a new date or suffix when reproducing:

```bash
python3 code/inspect_precise_images.py --output-dir reports/precise-inspection-YYYYMMDD --private-dir local/precise/metadata/inspection-YYYYMMDD
```

Only aggregate JSON goes to `reports/`. Identifying header/sample records and
contact sheets stay under ignored `local/`; source images are opened read-only.
The sample is not a subject split or a matching manifest. Pixel dimensions are
verified, but actual acquisition PPI and capture independence remain unknown.
The [research brief](notes/precise_research_brief_20261010.md) links the proposed
question, pipeline primer and protocol sketch awaiting joint review.

### Historical full experiment environment

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

The current research status is in [`STATUS.md`](../STATUS.md). Existing
Precise/RidgeBase metrics and the post-hoc parser audit have known limits; do
not treat the old SD302b near-duplicate pairs as independent-capture biometric
evidence or make unsupported low-FAR claims.

## Publication check

`scripts/check_publication_scope.py` checks the Git index for dataset/model/
environment/credential paths and files over 50 MiB. It runs in CI and, when
`git config core.hooksPath .githooks` is set, before each commit. Agents must
still review the contents of new text results, logs and figures, because file
names alone cannot prove that a file has no credentials or raw fingerprint data.
