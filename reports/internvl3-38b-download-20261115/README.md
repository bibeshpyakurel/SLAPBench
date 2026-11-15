# InternVL3-38B local download

2026-11-15. The user requested downloading InternVL3-38B for this Ubuntu
computer. This is model acquisition, not a fingerprint experiment, training
run or claim of GPU inference readiness.

The download and complete checksum verification finished successfully. All 36
files match the pinned publisher metadata. The [verified inventory](snapshot.json)
records file sizes and SHA-256 digests. Configuration JSON parses successfully,
all 16 shards named by the weight index exist, and no partial downloads remain.

## Source and storage

- Official source: <https://huggingface.co/OpenGVLab/InternVL3-38B>.
- Pinned revision: `b99bcba4fe0c51d9e475af25ec000dbfea6284af`.
- Local directory: `/home/bibesh/.local/share/slapbench/models/internvl3-38b/`.
- Verified snapshot: 36 files, 76,790,898,977 bytes, including 16 Safetensors
  weight shards, tokenizer/configuration files, upstream Python source and
  upstream examples. All downloaded assets stay outside Git.
- Private acquisition script, transfer log and checksum receipt:
  `local/precise/metadata/internvl3-38b-download-20261115/`.

## Restoration and verification

Use an environment with `huggingface_hub`, such as the existing
`local/precise/environments/matcher-compat-20261107/` environment. From the
repository root, the following restores the exact upstream snapshot and checks
all file sizes and publisher digests using the existing verification helper:

```python
from pathlib import Path
import sys
from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, "code")
from setup_next_models import verify_snapshot

repo = "OpenGVLab/InternVL3-38B"
revision = "b99bcba4fe0c51d9e475af25ec000dbfea6284af"
directory = Path.home() / ".local/share/slapbench/models/internvl3-38b"
metadata = HfApi().model_info(repo, revision=revision, files_metadata=True)
assert metadata.sha == revision
snapshot_download(repo, revision=revision, local_dir=str(directory), max_workers=2)
files = verify_snapshot(directory, metadata.siblings)
assert len(files) == 36
assert sum(item["bytes"] for item in files) == 76_790_898_977
```

Verification uses SHA-256 for Hub LFS files and Git blob SHA-1 for other files;
the receipt also records local SHA-256 values for every file. This verifies
snapshot integrity without importing or executing the downloaded model code.

## Runtime limits

The observed machine has a 16 GiB RTX 4080 SUPER and approximately 61 GiB of
system RAM. The original weight snapshot exceeds GPU memory, so this download
alone does not establish a working inference configuration. Quantization,
offloading or additional hardware and an appropriate runtime would need
separate preparation and validation. No GPU inference, model-code execution,
fingerprint processing or project fine-tuning is part of this acquisition.
Existing environments, model registries, historical storage links and completed
experiment assets are preserved.
