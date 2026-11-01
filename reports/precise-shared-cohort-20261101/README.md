# Supplied shared-cohort reconciliation

Read-only intake, 2026-11-01. See
[cohort_audit.json](cohort_audit.json) for aggregate evidence and the
[next-goal handoff](../../docs/notes/precise_next_goal_handoff_20261101.md)
for source links, latest professor direction, code review and remaining work.

- Training JSON: 39,290 unique pairs, 19,645 per class, with class balance
  within right/left/two-thumb types; all referenced images exist locally.
- Testing: 2,100 genuine plus 2,100 impostor pairs, unchanged from prior intake.
- Exact training/test pair overlap: zero. All 4,869 testing images and 276
  numeric identity groups also occur in training. This is not subject-disjoint.
- Positive per-finger sum is the JSON's selection rule. Fused-zero training
  pairs remain: 51 genuine and 1,213 impostor. No score was repaired or removed.
- Numeric-identity equality conflicts with 24 training and 2 testing impostor
  labels. Originals are preserved; private cases require a declared treatment.
- Full genuine-source reconciliation reproduces all 19,645 selected genuine
  pairs and their source lines/scores. The full 1.6 GB impostor pool was listed
  on Drive, not downloaded or recounted here.

The audit script and aggregate initially existed as uncommitted work. This
handoff reviewed the code, verified its digest against its private receipt,
reproduced the aggregate exactly in memory, and verified six local source files
byte-for-byte against fresh authenticated Drive downloads. Four synthetic tests
cover overlap/label privacy, score semantics/provenance, invalid targets and
overwrite refusal. No original labels or files changed, and no model ran.

Private inputs: `local/precise/collaborator/konain-followup-20261101/`.
Private cases/receipts: `local/precise/metadata/shared-cohort-20261101/`.
To reproduce only when source/code changes justify another audit, use the
verification environment and new output paths:

```bash
.venv/bin/python code/audit_precise_shared_cohort.py \
  --intake-dir local/precise/collaborator/konain-followup-20261101 \
  --images datasets/Precise \
  --previous-intake local/precise/metadata/konain-intake-20261024 \
  --output reports/precise-shared-cohort-NEWDATE/cohort_audit.json \
  --private-dir local/precise/metadata/shared-cohort-NEWDATE
```

Source scripts/configuration establish available procedures, not which flags,
checkpoints or rows were consumed in a completed collaborator run. Capture
independence and a development-selection history are not supplied by this JSON.
