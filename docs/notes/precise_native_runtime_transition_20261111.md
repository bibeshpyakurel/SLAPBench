# Exact restored-runtime scoring proposal

2026-11-11. The user confirmed “it is restored. please complete your task for
me.” CUDA is available again on the same RTX 4080 SUPER / NVIDIA 595.99.02.
The restored host remains on kernel 7.0.0-43 rather than the training kernel
7.0.0-42. The other observed difference is driver-version discovery:
`nvidia-smi` is now available, whereas training read `/proc/driver/nvidia`.
Python, package versions, interpreter bytes, compatibility binding, GPU, driver
version, compiled CUDA and the protected model/processing source all match.

This confirmation authorizes continued work, but is **not recorded as specific
review of this new runtime bridge**. The prior preflight note explicitly says
not to infer approval of a different runtime or weaken the provenance gate.
The original native scorer rejects the changed hardware metadata by design.
No further calibration/model/image access has been attempted after restoration.

## Exact change proposed for review

Use a separate scoring-only entry point, `code/score_precise_native_restored.py`,
with `code/precise_native_runtime_review.py`. It permits only the two exact
recorded differences above. Every other training/scoring provenance field must
match; another GPU, driver version, package/interpreter, source, model, data or
numerical setting fails. The new entry has no training command.

The protected original 19 source files stay byte-identical to execution commit
`06d12c249ec7f5db53124f673b2c6947c4b144b7`. The scoring entry reuses their encoder,
native preprocessing, scoring, calibration and one-time testing functions. It
records its own committed source/hash identity, the true scoring environment,
the original training source/record/checkpoint identity and this explicit
transition. Neither the original checkpoint nor any original record, receipt
or approval is rewritten to make the environments appear identical.

Retain the pinned Large weights and final checkpoint
`c24099fa41f38a1008b91454597e054013812a885f8d2433ab1598c23ed34c15`,
all 37,194 / 4,200 supplied pairs and labels/thumbs, 1600 x 1500 pixels padded
without resizing to 1610 x 1512, CUDA/BF16 scoring, FP32 normalized CLS/cosines,
forced Flash attention, deterministic algorithms, final-epoch selection and
nominal 1% training-impostor FAR calibration (budget185; strict greater-than).
No retraining, changed threshold policy, test-based tuning, fallback, extra
study or bitwise-equivalence claim is included.

Authorize **one frozen-calibration retry**, preserving its prior pre-model
CUDA failure, then the previously approved adapted calibration and two one-time
final tests under this exact bridge. Bind four unique output paths to the
actual review. Freeze both training selections before either test. Do not
retry a failed phase automatically. The retry gets a separate immutable dated
attempt directory/console; a documented canonical symlink can resolve to the
sole completed calibration for the existing report layout. No copied selection
or extra independently claimable test tree is created.

All paths below are under `local/precise/runs/large-native-20261108/`:

| Phase | Sole authorized output directory |
| --- | --- |
| Frozen calibration retry | `frozen-calibration-retry-20261111` |
| Adapted calibration | `adapted-calibration` |
| Frozen final test | `frozen-test` |
| Adapted final test | `adapted-test` |

The actual human response/proposal/input/checkpoint/verification and bridge
source hashes must be bound privately before execution. Synthetic checks,
source review, commit and verified publication precede real scoring. The final
aggregate and professor report will disclose both kernels, discovery methods,
source identities, restoration and the earlier interruption. All earlier
results and Qwen provenance limitations remain intact.
