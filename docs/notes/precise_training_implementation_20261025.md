# DINOv2 implementation and execution gates — 2026-10-25

The user selected **“Use the proposed DINOv2 matcher first”** in the project's
“Newer Models” chat and requested training/testing. That chat stopped at a usage
limit, leaving an implementation and successful small CPU runtime check locally.
The scheduled 9:30 PM Chicago resumption verified this user decision, continued
the existing code and implemented the training pipeline. Model choice no longer
awaits another confirmation. Dataset acquisition/exposure facts, the final
protocol and concrete results/split review remain unresolved.

**Later update — 2026-10-26:** a separate
[final-test entry point](precise_final_test_20261026.md) is now implemented and
synthetically validated. The training CLI remains development-only; its absence
of test scoring is deliberate. Earlier statements below that final-test execution
is future work describe the state before that separate implementation. Actual
real-data reviews and usable compute are still required.

## Implemented contracts

`code/precise_matching/images.py` strictly loads reviewed bytes, checks optional
decoded-grayscale SHA-256 using the earlier audit convention, retains the whole
image, converts to grayscale then RGB, and letterboxes to 448×448 with a white
canvas. It applies the documented mean/std normalization and no augmentation.
1600×1500 inputs retain their full 448×420 resized image, with 14 white rows on
each vertical side. Decode/digest failures stop training; no input is silently
dropped, replaced or given an imputed score.

`encoders.py` verifies both exact checkpoint configuration and safetensors
digests, loads local files only without remote code, and offers frozen/adapted
modes. The final two blocks and final layer norm are trainable in adapted mode:
3,551,232 of 22,056,576 parameters in the checked model. Class-token embeddings
are normalized float32 vectors; invalid output shapes, nonfinite/zero vectors
and incomplete spatial patches are rejected. `cosine_scores` returns one finite
continuous score per aligned pair of unit embeddings. Frozen inference callers
use evaluation mode and no-gradient/inference context.

`losses.py` implements the documented equally weighted positive/negative cosine
contrastive objective with margin 0.2. A microbatch uses one hand, distinct
reviewed captures and both matching/nonmatching subject classes. The full
8-subject × 2-capture design gives eight positive and 112 negative unordered
pairs. These are synthetic contract counts, not new real comparison manifests.

`training.py` and `code/train_precise_matcher.py` implement:

- Training-only subject/capture sampling with alternating hands and one reviewed
  canonical image per capture. Both hands must have enough subjects with two
  verified captures; ambiguous/missing eligibility is rejected rather than
  silently changing the population.
- Reviewed configuration, AdamW, gradient accumulation/clipping, linear warmup
  and cosine decay, deterministic seeds, explicit CPU/FP32 or CUDA/BF16 execution
  and no automatic CPU/precision fallback.
- Development-only scoring, strict-greater-than cutoff selection at the reviewed
  FAR, maximum development TAR then lower EER then earlier epoch for checkpoint
  selection. Development decode/encoder failures stop selection for investigation.
- Atomic, exclusive epoch checkpoints containing model/optimizer/scheduler,
  Python/NumPy/PyTorch/CUDA and sampler state; separate digest sidecars. Resume
  checks input/configuration/provenance and epoch/step counters and uses
  `weights_only=True` for this run's local checkpoints.
- Immutable per-attempt outputs, all development scores/reports, losses, learning
  rates and failure records. Resume creates another attempt, preserving earlier
  attempts and even directories left incomplete by a crash.
- Run records linking source bytes/commit, model revision/configuration/weight
  digests, environment/platform, input paths/digests, configuration and seeds.
  Completed training emits a frozen selected-checkpoint digest and development
  cutoff; incomplete attempts explicitly leave selection unfrozen.

The training CLI has no test-score input or test-scoring command. It validates
the metadata's global partitions but accepts image paths only for training and
development. Test labels/identifiers can occur in the global metadata preflight;
test image pixels and test scores do not enter training or selection. Future
final-test execution must verify the completed selection record and the approved
evaluation scope; it is not implemented as a hidden stage of training.

## Required private inputs and review receipt

Use the existing [metadata/comparison CSV schemas](precise_matching_tooling_20261017.md).
Provide one reviewed canonical training image per capture; keep all subject
derivatives together in the separately reviewed provenance/split record. The
additional UTF-8 path CSV has exactly `image_id,path`; it must contain every
training metadata row and every development comparison endpoint exactly once,
with **no test or extra image paths**. Paths must resolve under the explicit
authorized source root. No image file is opened during metadata preflight.

The [candidate configuration](../../code/precise_matching/configs/dinov2_first_milestone.json)
records implementation defaults. Concrete protocol values, including its 0.1%
FAR target, still need review before real execution. Any setting change requires
a fresh receipt bound to the new exact bytes; do not alter a running configuration.

The private review JSON must contain the following assertions and actual SHA-256
values. This is a schema illustration, **not an issued approval receipt**:

```json
{
  "protocol_reviewed": true,
  "split_reviewed": true,
  "capture_exposure_reviewed": true,
  "execution_authorized": true,
  "review_record": "reference-to-the-actual-private-review",
  "input_sha256": {
    "images": "sha256-of-exact-images-csv-bytes",
    "pairs": "sha256-of-exact-comparisons-csv-bytes",
    "paths": "sha256-of-exact-path-mapping-csv-bytes",
    "config": "sha256-of-exact-config-json-bytes"
  }
}
```

An asserted boolean is not evidence by itself. A human must review the underlying
acquisition/exposure evidence, eligibility, comparison lists and applicable
execution scope before those assertions are supplied. The trainer never writes
them for real data. Metadata consistency does not prove acquisition independence
or an untouched cohort. No real approval receipt, subject split or new comparison
manifest was created by this implementation.

## Future reviewed execution

The command below is a future interface using placeholders. It will not run
without the actual reviewed inputs/receipt, committed matcher source and usable
device. Keep all run artifacts under ignored `local/precise/runs/`:

```bash
local/precise/environments/matcher-candidate-20261025/bin/python code/train_precise_matcher.py \
  --images local/precise/metadata/approved_images.csv \
  --pairs local/precise/manifests/approved_comparisons.csv \
  --paths local/precise/metadata/approved_training_development_paths.csv \
  --config code/precise_matching/configs/dinov2_first_milestone.json \
  --review local/precise/metadata/approved_execution_review.json \
  --source-root datasets/Precise \
  --weights local/precise/models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --output local/precise/runs/approved-first-run
```

Resume uses the same arguments plus `--resume` pointing to an existing epoch
checkpoint within that run. Existing output directories are refused for a fresh
run. A resume must preserve the original configuration, source/model/environment
and input provenance; a completed run cannot be resumed for additional tuning.
Epoch-boundary resume repeats work since the most recent saved epoch if the
process was interrupted mid-epoch, preserving incomplete attempts. Never load
an untrusted external training checkpoint.

## Evidence and remaining limits

The [runtime report](../../reports/precise-training-start-20261025/README.md)
preserves the original small pretrained CPU check: four random 56×56 inputs,
two optimizer updates, frozen parameter preservation and checkpoint restoration.
It is not a fingerprint experiment, GPU check or full 16-image/448px memory test.
The later pipeline tests use tiny fake encoders and synthetic metadata/images;
they establish exact resumed-versus-uninterrupted weights/history even with
dropout, failure preservation, input/provenance change rejection and test-image/
score exclusion. CI now explicitly executes these Torch contracts in a CPU job,
without private data, checkpoint downloads or model-hub access.

CUDA remains unavailable and the original external `models` mount is still
unavailable. The selected small checkpoint is now independently available under
ignored `local/precise/models/` with verified source digests; the unavailable
external drive is therefore not a blocker for this selected encoder. GPU driver
repair remains separate administrator work. Full-resolution memory/resource
verification, reviewed capture/exposure eligibility and a concrete approved
subject split still block the real frozen baseline/adapted/final-test runs.
