# Precise matcher: synthetic CUDA capacity — 2026-11-01

**Passed:** the selected local DINOv2-small encoder fits the proposed 448×448
workload on the RTX 4080 SUPER. Driver 595.99.02, PyTorch 2.6.0+cu124 and BF16
were verified in the existing pinned matcher environment. No driver repair,
package installation, model download or Precise image access was performed.

The machine exposes 15.57 GiB of device memory, with 15.05 GiB initially free.
The [aggregate result](gpu_runtime.json) records exact bytes and configuration.

| Phase | Workload | Peak allocated | Peak reserved |
|---|---|---:|---:|
| Frozen inference | One 448×448 RGB tensor, FP32, no gradients | 0.17 GiB | 0.20 GiB |
| Adapted training | 16 synthetic grayscale-to-RGB images per microbatch, BF16, accumulation 4; eight microbatches / two AdamW updates | 2.67 GiB | 3.37 GiB |
| Adapted inference | One 448×448 RGB tensor, FP32, no gradients | 0.20 GiB | 3.37 GiB |

Allocator caches were retained between phases; the adapted-inference reserved
peak includes training's cache. Free device memory sampled after operations
stayed at or above 11.57 GiB, exceeding the planned 2 GiB external reserve.
The estimate using initial external usage and peak allocator reservation leaves
11.68 GiB. These observations depend on other processes and are not a promise
of capacity under future competing workloads. Driver sampling does not measure
every transient allocation outside PyTorch.

All eight synthetic losses and gradients were finite. The two-update sequence
preserved the frozen parameter digest and changed all 38 trainable tensors (3,551,232 of
22,056,576 parameters). AdamW initialized state for all 38 tensors. Each batch
exercised the implemented loss with eight positive and 112 negative unordered
pairs and alternating artificial left/right labels. No fingerprint identifiers,
images, source lists, real splits, checkpoints or biometric scores were used.

This short check uses the nominal learning rate without a scheduler. It verifies
capacity and update invariants, not long-run stability, training throughput,
checkpoint selection or fingerprint performance. The earlier
[small CPU probe](../precise-training-start-20261025/README.md) and synthetic
contract tests cover checkpoint/resume separately; that probe was not rerun.

## Reproduction

Run from the repository root with the existing hash-pinned local environment and
checkpoint. Choose **new** output/private directories for another authorized
probe; the command refuses to overwrite existing artifacts and has no automatic
batch-size, precision or device fallback.

```bash
local/precise/environments/matcher-candidate-20261025/bin/python code/check_precise_gpu_runtime.py \
  --weights local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --config code/precise_matching/configs/dinov2_first_milestone.json \
  --output reports/precise-gpu-runtime-20261101/gpu_runtime.json \
  --private-dir local/precise/metadata/gpu-runtime-20261101
```

The private directory holds the execution timestamp, Python/package versions,
code/configuration hashes and phase records. Failure records are retained with
no automatic retry. CI tests only tiny synthetic CPU models and early refusal
contracts; it does not run the full-resolution GPU probe or obtain weights.

The driver and proposed-workload fit are now verified. Independent parent
captures, exposure/claim scope, concrete protocol and actual results/split review
remain prerequisites for real-data execution. Konain's exact SDK filtering is
needed to reproduce his system; it is not an extra requirement for the
identity-supervised DINOv2 objective. See the
[current milestone review](../../docs/notes/precise_milestone_review_20261025.md).
