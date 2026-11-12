# CUDA preflight after successful Large/native training

2026-11-11. Training is complete and its final checkpoint has passed offline
verification. This note records a subsequent environment blocker, not a new
model failure or permission to train again.

## Observed state

- The approved resumed third epoch completed all 37,194 supplied pairs and
  sealed the original three-epoch / 3,489-update final selection.
- The frozen training-calibration command exited before model loading or image
  processing: CUDA was unavailable. Its separate private console and failure
  sidecar are preserved. No calibration output directory, scores, threshold
  selection or test claim was created. No final testing has occurred.
- This computer has rebooted into Linux `7.0.0-43-generic`; observed uptime was
  about seven minutes. Training used `7.0.0-42-generic`.
- PCI enumeration still finds the RTX 4080 SUPER. The current kernel has no
  loadable NVIDIA compute module: read-only `modinfo` and `modprobe --dry-run`
  both report that module `nvidia` is absent. `/dev/nvidia0`, the NVIDIA kernel
  version record and a usable CUDA device are absent. The userspace CUDA/NVML
  595.99.02 libraries remain present.
- The existing `7.0.0-42-generic` installation still contains the NVIDIA
  595-open kernel modules and its boot image. Python, package versions,
  interpreter bytes and compatibility binding still match the recorded run.
- Administrative commands require interactive authentication. This session
  cannot change boot selection or reboot the machine with existing privileges.

These observations explain the CUDA availability failure under the new kernel;
they do not establish the cause of the earlier epoch-3 segmentation fault.
No driver installation, repair, module loading, boot configuration change,
restart or environment-provenance rewrite has been performed.

## Least-change continuation

Have the user or lab administrator boot the **existing working
`7.0.0-42-generic` kernel**, then verify that the original NVIDIA 595.99.02 / RTX
4080 SUPER runtime is available again. The usual boot menu route is Advanced
options for Ubuntu, followed by the normal entry for that kernel. This is a
user/admin action, not an automatically authorized reboot.

Restoring the original kernel matters beyond CUDA availability: the approved
entry point requires scoring provenance to match the completed adapted run.
Installing a driver for the different kernel would not by itself make that
recorded platform identical. Do not weaken the provenance gate, change model
source, edit the original record or infer approval of a different runtime.

The approved protocol says “No automatic retry after failure,” and AGENTS.md
keeps system-level GPU repair separate. After actual confirmation/authorization
of **one frozen-calibration preflight retry on the restored original runtime**,
continue the already approved two calibrations and two one-time tests. Do not
repeat training or any completed inference phase. A new failure requires a new
decision; no backend/precision/resolution fallback is included.

Preserve the failed preflight files. Use a new dated calibration-attempt
directory and console for the authorized retry. After a successful calibration,
an explicitly documented canonical symlink may point `frozen-calibration` to
that sole immutable attempt directory. Paths are resolved before claiming the
one-time test, so the alias does not duplicate a selection or bypass a test
claim. Never copy scores/selections into a second independently claimable tree.

Independent training-completion documentation and source publication can
finish while this environment action is pending. The final research report and
Large/native accuracy remain pending actual calibration and final testing.
