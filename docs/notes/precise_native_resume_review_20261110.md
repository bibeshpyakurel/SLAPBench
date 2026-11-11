# One explicit resume of the interrupted Large/native run

2026-11-10. This is a concrete recovery proposal, **not an approval record**.
The user requested continuation; inspection then established that the original
process had exited with code 139 / segmentation fault during epoch 3. No new
training, calibration or testing has been started after this failure.

The [approved native protocol](precise_large_native_protocol_20261108.md) says
“No automatic retry after failure.” [AGENTS.md](../../AGENTS.md) retains this
rule. The new failure was not known when the user asked to continue; the request
is therefore not represented as an informed, specific retry approval.

## Verified saved state

The original run retains complete epoch-1 and epoch-2 checkpoints and its
unchanged console. The terminal reports SIGSEGV, but accessible evidence does
not establish its cause. No Python failure receipt, final selection, calibration
or test output exists. This is not evidence of an out-of-memory error or a reason
to change GPU drivers, pixels, attention backend, precision or optimizer.

The [offline integrity report](../../reports/precise-native-large-interruption-20261110/README.md)
verifies epoch 2: all 37,194 training pairs in each completed epoch, 2,326 saved
optimizer updates, consistent finite optimizer state and saved RNG metadata,
401 frozen tensors unchanged, and all 38 selected trainable tensors changed.
The checkpoint SHA-256 is
`e8afe81b4963b3315e545eda22ee71e4bed14973f07662368e9565322564e762`.
Its source/configuration/provenance bind the original reviewed run. This verifies
an unfinished boundary; it does not select a scientific final checkpoint.

## Exact proposed action

Use the existing CLI's explicit `--resume` route to restore **that epoch-2
checkpoint in the same run directory**, preserving original files and writing
a new `attempt-0002` plus a separate resume console. Restore model, optimizer,
scheduler and saved RNG state. Keep the original protected execution source
`06d12c249ec7f5db53124f673b2c6947c4b144b7`, model/configuration, pair lists,
review and preprocessing unchanged.

Run the full third epoch (37,194 supplied pairs) and retain the original
three-epoch/final-checkpoint rule and 3,489-update final schedule. The last
reported 19,072 pairs of the interrupted third epoch are not checkpointed;
they must be replayed from the saved boundary, not silently treated as retained
updates. No partial model state survives to initialize this continuation.
The partial work remains part of the execution history and compute cost.

The first two epochs took roughly 17 hours. Allow approximately another
8.5 hours for the full third epoch, with normal runtime variation; both training
calibrations and tests are additional. No automatic second recovery attempt,
kernel fallback, resolution reduction, extra epoch or test-based selection is
included. A repeat failure remains a new decision.

After successful final selection, verify the checkpoint offline and perform
the already approved two training calibrations followed by the two one-time
final tests. Retain the original cutoffs/claim limits and the pending Qwen
run-provenance limitations. Reporting will explicitly disclose the interruption,
replayed epoch, retained epoch-boundary state and actual elapsed execution time.

Separate actual human approval of this **one explicit unchanged-configuration
resume** must be recorded against the proposal, original inputs, run record,
interruption verification and selected epoch-boundary digest before launch.
The original approval/review bytes are preserved; they are not edited to assert
new permission. Reporting/source checks and publication can progress meanwhile.
