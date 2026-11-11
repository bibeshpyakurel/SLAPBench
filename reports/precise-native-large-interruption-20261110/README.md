# Large/native run interrupted before final selection

2026-11-10 read-only inspection. The approved native run completed two full
epochs and then exited with code 139 / segmentation fault during epoch 3.
The last saved progress event reports 19,072 / 37,194 pairs in that third epoch
and update 2,922 / 3,489; it is a progress observation, not a saved model state.
The console's last-write timestamp is November 9 at 19:51:42 UTC.

**No final Large/native checkpoint, calibration, testing or accuracy result
exists.** The completed small-model and source-verified Qwen results remain
unchanged. No accuracy is inferred from training loss or a synthetic report
layout. The crash cause is unconfirmed. System kernel-log access was restricted,
and no attributable local crash report was available.

The [offline aggregate](integrity.json) reproduces the preserved epoch-2
checkpoint's digest and original run provenance. It finds:

- Two complete epochs, each containing all 37,194 supplied training pairs.
- 2,326 saved optimizer updates, consistent scheduler/optimizer step counts,
  finite optimizer moments and syntactically valid saved RNG states.
- Zero changed frozen tensors out of 401 (279,170,048 parameters).
- All 38 selected trainable tensors changed (25,198,592 parameters).
- The original protected runtime source remains identical to execution commit
  `06d12c249ec7f5db53124f673b2c6947c4b144b7`.

This CPU-only inspection reads no image pixels, performs no inference or
training, and does not choose a final model or authorize a retry. All original
checkpoints and logs remain private and unchanged. The verifier is
`code/verify_precise_native_interruption.py`; its private receipt lives under
`local/precise/metadata/native-large-interruption-20261110/`.

The approved protocol forbids automatic retries. The
[concrete one-resume proposal](../../docs/notes/precise_native_resume_review_20261110.md)
describes restoring epoch 2 and replaying the complete third epoch with the
same configuration, source, labels, preprocessing and final-checkpoint rule.
It requires an actual specific decision; no decision is inferred from this
integrity report. Independent reporting/reproducibility work can continue.

The user subsequently answered **“Approve one unchanged-config resume”** to
the exact recovery question on November 10. That response and proposal bytes
are recorded separately with the original input/run/checkpoint/evidence hashes.
This authorizes one explicit epoch-2 restoration and full epoch-3 replay, then
the previously approved calibration/tests. It does not authorize automatic
further retries or configuration changes. Execution follows source checks and
publication; the offline integrity receipt itself remains a non-approval record.
