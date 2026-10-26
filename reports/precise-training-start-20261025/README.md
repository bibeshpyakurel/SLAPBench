# Selected DINOv2 runtime and training implementation — 2026-10-25

The user selected the proposed DINOv2 matcher first and requested training and
testing in the “Newer Models” chat. The scheduled resumption verified that newer
decision and continued the interrupted implementation. The
[implementation guide](../../docs/notes/precise_training_implementation_20261025.md)
documents loaders, encoder/loss, guarded training, immutable resume and required
private review inputs. No real Precise training/evaluation result is claimed.

## Original CPU probe

[cpu_runtime.json](cpu_runtime.json) is the original output of the previously
completed, authorized synthetic pretrained-encoder check. Its bytes were retained;
the scheduled resumption did not repeat this probe. It records:

- Local DINOv2-small at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- Four random 56×56 RGB inputs, two CPU optimizer updates, 384-dimensional
  embeddings and finite gradients/losses.
- 22,056,576 total / 3,551,232 trainable parameters; 38 trainable tensors changed.
- Frozen parameters unchanged; restored model embeddings exactly identical;
  38 optimizer-state entries restored from the synthetic checkpoint.

This is neither the full 16-image/448px resource check nor a biometric experiment.
The reported synthetic losses are runtime diagnostics, not fingerprint matching
performance. Private checkpoint/source and original probe artifacts are under
ignored `local/precise/metadata/training-start-20261025/`.

## Checkpoint source and reproduction

The selected checkpoint is locally available at
`local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056/`.
The loader verifies these exact digests and refuses network downloads or remote
code during loading:

| File | SHA-256 |
|---|---|
| `config.json` | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| `model.safetensors` | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |

The weights contain 88,249,960 bytes. Restore separately from the
[pinned public revision](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
under its applicable terms; they are not committed to Git. The dependency
environment is documented in the [environment report](../precise-environment-20261025/README.md).
For a separately authorized reproduction of the small probe, use new output paths:

```bash
local/precise/environments/matcher-candidate-20261025/bin/python code/check_precise_training_runtime.py \
  --weights local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --device cpu \
  --output reports/precise-runtime-YYYYMMDD/cpu_runtime.json \
  --private-dir local/precise/metadata/runtime-YYYYMMDD
```

## Training pipeline validation and gates

Synthetic tests exercise preprocessing/digest/error handling, frozen/adapted
parameter boundaries, unit-vector similarity, loss counts, reviewed-input/path
gates, same-hand training-only sampling, development-only selection and exact
epoch-boundary resume. They verify that corrupted checkpoints, changed provenance
or source/configuration inputs are rejected and failed attempts are retained.
A dedicated CPU CI job runs the Torch contracts with no weights or private data.
The [pipeline validation summary](pipeline_validation.json) records 103 passing
local tests, including 29 tests selected by the synthetic training CI pattern,
exact resume and successful CLI wiring with synthetic images/a fake encoder.
All 15 legacy metric files reproduce and correctness lint passes.

CUDA is still unavailable: `nvidia-smi` fails to communicate with the driver.
The original external model mount is unavailable, but the selected small
checkpoint has its own verified local copy. Capture independence, exposure
eligibility, final protocol and concrete split approval remain unresolved.
No real subject allocation, matching-list generation or Precise training/test
scoring was performed. No system repair was attempted by the scheduled resumption.
