# Synthetic runtime check after the interpreter/OS change

2026-11-07. **Passed**, using the existing pinned DINOv2-small weights and
packages, without fingerprint inputs or a model experiment. The
[aggregate](gpu_runtime.json) records the workload and observed allocator data.

The OS now reports Ubuntu 26.04.1 and default Python 3.14. Original environments
still contain their installed Python 3.12 packages; they were not reinstalled
or altered. A new compatibility environment uses bundled Python 3.12.14 with
the preserved matcher site-packages: Torch 2.6.0+cu124, Transformers 5.18.0,
NumPy 2.5.3 and Pillow 12.3.0. Dependency checks and CUDA/BF16 availability pass.

The loaded NVIDIA driver remains 595.99.02 on the RTX 4080 SUPER. The management
executable is absent, so read-only version provenance uses
`/proc/driver/nvidia/version`; CUDA operations still use the same declared
device/precision. No driver repair, system change or numerical fallback occurred.

The full-resolution synthetic adapted workload uses 16 images at 448 x 448,
BF16 and accumulation four. Peak allocation was **2.67 GiB**, with **3.37 GiB**
reserved. Frozen and adapted inference phases also passed; losses/gradients were
finite and frozen parameters were preserved. These random tensors provide
capacity/update evidence only, not accuracy or a guarantee of future free memory.

This check was warranted by the changed runtime; it does not repeat a dataset
audit or original real-data experiment. It reuses the earlier synthetic
subject-batch capacity configuration, separate from the newly approved real
fixed-pair configuration and its explicit-pair synthetic contracts. The latter
uses up to 16 images per microbatch as well. The actual new cohort/protocol
authorization is recorded in the [November 7 review](../../docs/notes/precise_revised_cohort_review_20261107.md).

For another authorized runtime change, use new output/receipt paths:

```bash
local/precise/environments/matcher-compat-20261107/bin/python code/check_precise_gpu_runtime.py \
  --weights local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --config code/precise_matching/configs/dinov2_first_milestone.json \
  --output reports/precise-gpu-runtime-NEWDATE/gpu_runtime.json \
  --private-dir local/precise/metadata/gpu-runtime-NEWDATE
```

Private execution metadata and phases stay under ignored
`local/precise/metadata/gpu-runtime-20261107/`. No raw inputs, images, identities
or model weights are included in this aggregate report.
