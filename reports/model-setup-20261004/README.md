# Local model setup — 2026-10-04

The user authorized downloading and preparing three new local models. No
fingerprint experiment, dataset processing, paid model call, split construction
or training was performed. Historical checkpoints, environments, results and
the unavailable external-drive model symlink are preserved.

| Key | Source | Downloaded bytes | Files |
| --- | --- | ---: | ---: |
| `qwen35` | `Qwen/Qwen3.5-9B` | 19,329,393,661 | 16 |
| `gemma4` | `google/gemma-4-12B-it` | 23,951,776,979 | 9 |
| `internvl35` | `OpenGVLab/InternVL3_5-8B` | 17,072,800,269 | 24 |

Weights live at `/home/bibesh/.local/share/slapbench/models/`, outside Git.
`snapshots.json` records the immutable Hub revisions, snapshot file sizes and
SHA-256 digests. Verification compares weight digests with the Hub's LFS SHA-256
values and small-file Git blob hashes with Hub metadata; it checks every file
in the snapshot. It does not establish research performance or the absence of
training-data overlap. Ignored Hub bookkeeping files are not model artifacts.

`preflight.json` records offline setup checks: configurations and tokenizers
load, model classes construct on the meta device without loading weights, and
Qwen/Gemma processors accept two synthetic 448px images. InternVL's original
custom model code was inspected and tested with Transformers 4.57.6; Qwen and
Gemma were tested with Transformers 5.18.0 in a separate `.venv`. The configuration
parameter counts describe the empty architecture, not an inventory of loaded
weights. `environment.json` lists installed packages in that separate environment.

The machine exposes an RTX 4080 SUPER in PCI hardware information, but has no
NVIDIA device nodes and `nvidia-smi` cannot access the driver. All recorded setup
checks therefore report CUDA unavailable. GPU loading, generation, 4-bit runtime
compatibility, memory headroom and fingerprint performance are unverified.

The project now recognizes the three runner keys. See
[setup instructions](../../docs/REPRODUCIBILITY.md#october-2026-local-model-additions).
The adapters retain historical 448px preprocessing and existing parsers; formal
experiments still require the protocol/parser review and separate authorization.
