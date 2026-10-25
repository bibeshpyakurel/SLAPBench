# Precise candidate dependencies — 2026-10-25

**The isolated dependency layer installs and imports successfully. GPU execution
remains unavailable and the matcher configuration still requires review.**
This completes additional preparation from roadmap task 4 after the
[initial inspection](../precise-readiness-20261025/environment.md). Dependency
preparation is permitted before the separately gated loading/forward/backward
check. Preparing this candidate does not accept the
[proposed protocol/model](../../docs/notes/precise_milestone_review_20261025.md).

## Prepared and verified

- Created a fresh virtual environment at ignored
  `local/precise/environments/matcher-candidate-20261025/`, using Python 3.12.3
  with system site packages disabled.
- Resolved and installed 50 runtime wheels. The
  [input requirements](../../code/precise_matching/environment/requirements.in)
  record the seven direct dependency choices; the
  [Linux/Python 3.12/CUDA 12.4 lock](../../code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock)
  pins every resolved wheel URL and SHA-256. Installation used hash enforcement,
  binary wheels only and no package index lookup.
- Checked installed distribution versions and installation-report hashes against
  the complete resolution: all 50 match. `pip check` reports no broken
  requirements. The installer is pip 24.0 from this machine's `venv`/`ensurepip`;
  pip itself is recorded separately from the runtime lock.
- Imported torch, torchvision, Transformers, Pillow, NumPy, safetensors,
  huggingface-hub, `Dinov2Model` and `Dinov2Config` successfully in Python
  isolated mode, with Hugging Face network access disabled for that check.
  The seven selected modules resolve inside the new environment.
- Compared the historical `.venv/` package-version inventory to the preserved
  pre-preparation snapshot: unchanged. New environment files and receipts are
  ignored by Git.

| Component | Installed version |
|---|---|
| Python | 3.12.3 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 5.18.0 |
| Pillow | 12.3.0 |
| NumPy | 2.5.3 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.33.0 |

The torch/torchvision/CUDA choice follows the corresponding
[official PyTorch 2.6 installation combination](https://pytorch.org/get-started/previous-versions/#v260).
Versions here preserve the review proposal's candidate stack rather than
selecting the latest framework release. The complete transitive package pins
and hashes are in the lock, including CUDA runtime libraries packaged in the
virtual environment. No system driver or CUDA toolkit was changed.

The environment occupies approximately 5.3 GiB locally. Resource observations
are snapshots, not reservations; the generated
[validation.json](validation.json) records available disk bytes after preparation.
Private resolution/installation reports, logs, version inventories and scope
receipts are under `local/precise/metadata/environment-preparation-20261025/`.

Setuptools exposes 11 bundled metadata entries when imported indirectly; their
locations were verified under the pinned setuptools wheel's `_vendor/` directory.
They are not additional independently installed wheels. The runtime wheel count
is taken before those imports and checked against the 50-wheel installation
report. The first exploratory private receipt counted those entries separately;
`validation-final.json` records the corrected counting convention.

## Restore on the matching platform

This lock is specific to Linux x86_64 / CPython 3.12 and CUDA 12.4 wheels; it is
not a cross-platform dependency promise. Wheel tags also impose their glibc
requirements. Use a **new** environment path and enough local disk space. From
the repository root:

```bash
python3.12 -m venv local/precise/environments/matcher-restore-YYYYMMDD
local/precise/environments/matcher-restore-YYYYMMDD/bin/python -m pip install \
  --require-hashes \
  -r code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock
local/precise/environments/matcher-restore-YYYYMMDD/bin/python -m pip check

# Import classes only: no checkpoint loading or model object creation.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
local/precise/environments/matcher-restore-YYYYMMDD/bin/python -I -c \
  'import torch, torchvision; from transformers import Dinov2Model, Dinov2Config'
```

The lock uses explicit public HTTPS wheel URLs from PyPI, PyTorch and NVIDIA,
with SHA-256 verification; restoring requires those files or a separately
preserved authorized package cache. No weights or dataset inputs are included.
Record the bootstrap installer version on another machine and repeat dependency
and import verification. Do not treat a successful package installation as GPU
or training readiness.

## Remaining gates

The installed torch build includes CUDA 12.4 support, but
`torch.cuda.is_available()` remains false and device count is zero. The earlier
NVIDIA driver failure and unavailable external weight mount remain unresolved.
Driver/runtime compatibility and device memory cannot be certified from these
imports. System repair is a separate scope.

No checkpoint was downloaded/loaded, no model object instantiated, no model
forward/backward executed and no dataset inputs loaded. After applicable review
and execution authorization, perform the small synthetic real-model check and
record output/gradient contracts and peak memory before any real-data run.
Acquisition/exposure clarification and protocol/results/split review remain
necessary for the later roadmap tasks.
