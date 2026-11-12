# DINOv2-Large/native training completed; evaluation awaits original GPU runtime

2026-11-11. The single explicitly approved epoch-2 continuation completed the
full third epoch. The final checkpoint was selected under the unchanged
predeclared rule, not from testing results. This is a training-completion
record, **not an accuracy/evaluation report**.

[training_verification.json](training_verification.json) exports the reviewed
offline checkpoint integrity and recovery-chain summary. It verifies:

- All three epochs contain the full 37,194 supplied training pairs; the final
  optimizer schedule contains 3,489 updates.
- All 401 frozen tensors remain unchanged; all 38 selected trainable tensors
  changed. All model tensors pass the verifier's finiteness checks.
- The final checkpoint SHA-256 is
  `c24099fa41f38a1008b91454597e054013812a885f8d2433ab1598c23ed34c15`.
- The epoch-2 restoration follows the single actual user-approved recovery
  record, preserving the original attempts and the discarded partial epoch.
- The resumed attempt took **8h 29m 23s**. Observed training-attempt intervals
  total at least **29h 49m 30s**, including discarded partial work. The full wall
  interval from first start to final selection is **60h 24m 40s**, including the
  interruption gap. These are pipeline times, not GPU kernel timing.
- The retained final trajectory contains 111,582 pair presentations; at least
  130,654 presentations were observed across both attempts. Periodic logging
  makes the latter a lower bound on total work.

No training or checkpoint is repeated to address the subsequent environment
issue. The supplied labels, all left/right/thumb pairs, model/configuration,
protected execution source and final-checkpoint rule remain unchanged.

## Calibration has not produced scores

The first frozen-calibration command stopped at the CUDA availability preflight,
before model loading or image processing. No calibration scores/thresholds or
final-test outputs exist. Private failure files remain intact.

Read-only inspection found a recent reboot into kernel `7.0.0-43-generic`, which
has no NVIDIA compute module. The prior working `7.0.0-42-generic` kernel and its
NVIDIA modules remain installed. Python/package/interpreter/compatibility bytes
still match training. This explains the current CUDA blocker; it does not explain
the earlier segmentation fault. No system repair or further retry was performed.

The [concrete restoration/preflight note](../../docs/notes/precise_native_cuda_preflight_20261111.md)
describes returning to the original runtime and the separate actual decision
needed for one preflight retry. Administrative authentication is required and
is unavailable in this session. The main goal remains incomplete until both
training calibrations, both one-time final tests and the professor report are
completed and reviewed/published. No Large/native accuracy is fabricated from
training loss, checkpoint integrity or report-layout fixtures.
