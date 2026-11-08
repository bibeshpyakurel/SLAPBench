# DINOv2-Large at original image resolution: synthetic capacity

2026-11-08. The user relayed the professor's request for a larger DINOv2 model
and original high-resolution images. This report verifies a concrete candidate
without opening fingerprint pixels or executing real-data training/testing.
The [separate run proposal](../../docs/notes/precise_large_native_protocol_20261108.md)
retains the applicable human protocol/results/cohort review requirement.

Candidate: `facebook/dinov2-large`, pinned revision
`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`, 304,368,640 parameters.
Only the final two blocks and final layer norm are trainable (25,198,592
parameters). The official 1,217,522,888-byte safetensors file was downloaded
locally and SHA-256 verified against the provider's published digest:
`399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23`.
No existing small-model weights, environment or original images were changed.

The existing image inventory, restricted to the exact new-cohort endpoints,
records all 6,778 images at **1600 x 1500**. Native preparation converts to
grayscale as before, centers the unchanged raster in a white **1610 x 1512**
canvas, replicates channels and applies the same DINO normalization. It performs
no resampling, resizing or cropping. A synthetic pixel-equality test recovers
every original pixel by the inverse crop. Each image has 12,420 complete
14-pixel patches, versus 1,024 at the completed 448-pixel setting.

## Exact-policy check

[capacity.json](capacity.json) records the second, exact-policy synthetic check:
CUDA/BF16, one explicit pair (two images) per microbatch, accumulation 32,
nonreentrant gradient checkpointing, deterministic algorithms, and SDPA with
**Flash attention as the only allowed backend**. Both pre-update and post-update
BF16 inference produce finite FP32 embeddings; two full accumulated backward /
AdamW updates produce finite selected gradients and preserve frozen tensors.
No smaller shape/batch, CPU or numerical/backend fallback is used.

| Observation | Measured |
| --- | --- |
| GPU | NVIDIA RTX 4080 SUPER, approximately 16 GiB VRAM |
| Peak allocated | 3.01 GiB |
| Peak reserved | 3.26 GiB |
| Minimum sampled free | More than 11 GiB |
| Synthetic updates / microbatches | 2 / 64 |
| Elapsed model-work interval | 50.78 seconds |
| Synthetic pairs / second | Approximately 1.26 |
| Frozen parameters | Unchanged |
| Selected parameter tensors with optimizer state | 38 |

A preliminary allocation check also passed; its original artifacts/source
snapshots remain private. The second check adds the deterministic and BF16
scoring contexts required by the proposed complete pipeline. These are capacity
checks, not biometric experiments or a search over test performance.

At the exact-policy synthetic throughput, 111,582 pair presentations correspond
to roughly **24.6 hours of model computation**. Actual training additionally
includes image decoding, normalization, disk I/O, integrity checks and checkpoint
writes, and shares the GPU with any external processes. This is an estimate,
not a promised completion time. Calibration/testing are additional phases.
Short synthetic success does not establish long-run stability or model accuracy.

## Code and reproduction

- `code/precise_matching/native_large.py`: pinned loader and pixel-preserving
  native preparation; SDPA, final-two-block adaptation and gradient checkpointing.
- `code/check_precise_native_large.py`: synthetic exact-policy capacity check.
- `code/run_precise_native_large.py`: separately gated train/calibrate/test entry
  point; the completed small-model route remains separate.
- `code/verify_precise_native_checkpoint.py`: offline base/final tensor and
  optimizer-schedule verification.

The real-data entry point requires a new digest-bound review for this exact
model/native-pixel profile. The larger model and resolution change together;
comparison with small/448 cannot isolate their individual causal effects.
See the proposal for schedule, calibration, test claiming and failure policy.
Existing completed audits/experiments are not run again by this preparation.
