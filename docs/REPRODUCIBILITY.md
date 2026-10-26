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

The selected first-milestone DINOv2-small checkpoint is independently restored
under `local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056/`.
Its pinned source, exact configuration/weight digests and original synthetic CPU
probe are in the [runtime report](../reports/precise-training-start-20261025/README.md).
The [training guide](notes/precise_training_implementation_20261025.md) documents
reviewed inputs, configuration, environment, private outputs and resume. This
small local copy does not restore the unavailable historical model drive or
establish GPU/real-data execution readiness.

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

### Current Precise preparation (updated 2026-10-24)

See [WORKSPACE_LAYOUT.md](WORKSPACE_LAYOUT.md) for the prepared local folders,
historical asset locations and reviewed-publication workflow. This checkout is
the primary workspace; future authorized training runs locally.

The next study uses `datasets/Precise/`. SD302b and RidgeBase assets and commands
remain available for reproducing historical work. Do not move or delete those
assets to change the research focus; existing manifests depend on their paths.
Precise source/acquisition documentation, session meanings, physical resolution,
data terms and the concrete evaluation protocol still need review. Supplied
collaborator training/testing membership is now audited, while actual training
consumption and development allocation remain unverified. Start with the
[central research plan](notes/precise_first_milestone_proposal_20261011.md);
the [preparation note](notes/precise_preparation_20261004.md) preserves the
professor's feedback and dated follow-ups.

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
The [central plan](notes/precise_first_milestone_proposal_20261011.md) connects
these completed audits, the architecture sequence and protocol decisions.
The [research brief](notes/precise_research_brief_20261010.md) and protocol
sketch provide supporting detail.

The collaborator's separately shared VeriFinger testing exports are retained
under ignored `local/precise/metadata/verifinger-intake-20261011/`, alongside
private source metadata and SHA-256 checks. Public Git contains the offline
audit utility and reviewed aggregates, not the identifying raw records. Obtain
these two inputs from the collaborator's authorized sharing to reproduce the
[2026-10-11 score audit](../reports/precise-verifinger-audit-20261011/README.md);
that report provides the exact command and requires new output directories.
No SDK license, matcher execution or model inference is needed to recompute
these supplied scores. The training-list receipt below supersedes earlier
statements that those lists are missing; SDK acquisition/configuration and the
actual binary training conversion remain separate unknowns.

### Collaborator training lists and pinned FPBench reference (2026-10-24)

The ignored `local/precise/metadata/konain-intake-20261024/` bundle contains:

- `genuine_filtered_minus2100.txt`: 23,103 supplied genuine training records.
- `imposter_filtered_minus2100_78592.txt`: 78,592 supplied impostor training records.
- `testing_genuine.txt` and `testing_impostor.txt`: 2,100 records each, preserved
  from the original intake and a fresh authenticated impostor-file download.
- `source_provenance.json` and `audit/audit_details.json`: private origin/hash
  records and identifying overlap evidence.

Obtain the training inputs from Konain's authorized sharing. The test source IDs
are `1rru-nFStR6cgzJszI6YKjtQmlkRIV4Kj` (genuine) and
`1MAWGm71PZNYtlLnAA8WAY-84alA4mDdO` (impostor); the canonical Drive links and
hashes are retained in the private receipts. Originals in Downloads and the
earlier intake were not moved or overwritten. Follow the
[new intake report](../reports/precise-training-intake-20261024/README.md) to
reproduce the standard-library audit into fresh output directories.

The new lists exclude exact testing pairs, but share every testing image
basename and subject. Also, the supplied training impostors are right-hand
only, and 3,455 genuine training rows have fused score zero without component
fields. These inputs are collaborator records, not our approved subject split
or a ready-to-run binary training dataset.

The full FPBench repository is available at ignored
`local/precise/references/FPBench/`, pinned to
`ed27ae669947bfd157522d537a9f9c77b4bf3189`. The
[reference guide](notes/fpbench_reference_20261024.md) records restoration
commands, the Qwen entry-point digest, source terms and integration limitations.
No third-party dependencies or model checkpoints were installed or executed.

### Historical full experiment environment

The [2026-10-25 readiness report](../reports/precise-readiness-20261025/README.md)
documents the new full-decode/provenance audit and exact reproduction command.
Its inputs are authorized local JPEGs, the four preserved collaborator lists
and the two existing Precise comparison manifests. Private image/hash/candidate
and exposure records stay under `local/precise/metadata/readiness-20261025/`;
compute/package/model-metadata receipts stay under `metadata/compute-20261025/`.
The [review design](notes/precise_milestone_review_20261025.md) pins a candidate
DINOv2 revision without acquiring its weights. The inspected `.venv/` is not a
verified matcher environment: CUDA is unavailable, the external model path is
unavailable and synthetic model checks remain pending. A subsequent
[candidate environment preparation](../reports/precise-environment-20261025/README.md)
installs 50 hash-pinned runtime wheels under ignored
`local/precise/environments/matcher-candidate-20261025/` and verifies dependency
consistency and offline DINOv2 class imports. Follow that report and its platform
lock for isolated restoration. This does not establish GPU execution readiness
or acceptance of the proposed model/protocol.

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
