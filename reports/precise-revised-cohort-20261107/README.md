# Newly supplied collaborator cohort: read-only intake

2026-11-07. Five new files, modified on Drive on November 5, were found in the
previously supplied [testing/data folder](https://drive.google.com/drive/folders/1BveHqYTBQPqP9kQB0aPoeHNsNJXNsEzQ).
They are pair inputs, not matching trained-Qwen results. Originals were saved
unchanged under ignored `local/precise/collaborator/konain-followup-20261107/`.
Source IDs, raw-download receipts and hashes remain there. The existing
November 1 inputs, runs, cutoffs and professor report are unchanged.

The [aggregate evidence](cohort_audit.json) comes from
`code/audit_precise_revised_cohort.py`. This reads the new files, reuses the
existing image inventory, checks current path existence, and compares against
the private bundle bound to the actual completed DINO training record. It does
not decode images, allocate subjects, generate pairs, change labels or run models.

## The new cohort is different

| Property | New training | New testing |
| --- | ---: | ---: |
| Pairs | 37,194 | 4,200 |
| Genuine / impostor | 18,597 / 18,597 | 2,100 / 2,100 |
| Right genuine / impostor | 6,569 / 6,569 | 700 / 700 |
| Left genuine / impostor | 6,247 / 6,247 | 700 / 700 |
| Two-thumb genuine / impostor | 5,781 / 5,781 | 700 / 700 |
| Images | 5,721 | 1,057 |
| Numeric filename-prefix groups | 221 | 55 |
| Raw prefix strings | 282 | 55 |
| Numeric-identity label disagreements | 0 | 0 |
| Duplicate unordered pairs / missing references | 0 / 0 | 0 / 0 |

New training/testing share **zero exact pairs, images or numeric groups**.
The training JSON exactly matches both new training lists, including class,
fused score, component sum and finger count. Declared class/type/group counts
match observations. System/user prompts and A/B options match the earlier JSON.
All referenced images exist locally; none was loaded for this intake.

The source declares raw-prefix labeling, numeric-prefix grouping, exclusion of
numeric-equal alias impostors, positive per-finger sums and all three slap types.
All 61 padding-alias groups from the existing inventory occur in new training,
and none occurs in new testing. This is structural evidence for a
**numeric-prefix-group-disjoint supplied cohort**, not independent verification
of biological identities or capture/session independence. No identity relabeling
or allocation was performed here.

Fused zeros remain: 47 training genuine, 1,214 training impostor, seven testing
genuine and 142 testing impostor. Every supplied list row has component fields
with a positive sum. These are retained values, not verified extraction statuses.
No upstream full-pool count, generation history, actual model consumption or
checkpoint-selection history is established by the source metadata.

## Why the completed adapted checkpoint cannot establish a new holdout

Only **189 of 4,200** new testing pairs match the approved old testing list;
4,011 testing pairs changed. New training retains 16,838 old training pairs,
adds 20,356 and omits 22,452. It is not the approved 39,290-pair benchmark.

The completed DINO adaptation already trained on **all 1,057 new testing images
and all 55 new testing numeric groups**, including 1,918 exact new testing pairs.
Calling that checkpoint's scores unseen-group evaluation would therefore be
incorrect, even though the new source files are internally group-disjoint.
A model for a genuinely held-out new numeric-group analysis must start again
from the pinned pretrained base and use only the new training cohort.

The source-folder update alone did not authorize replacing the benchmark or
rerunning models. The user subsequently reviewed and explicitly approved the
[replacement proposal and fresh runs](../../docs/notes/precise_revised_cohort_review_20261107.md)
in the resumed goal chat. That approval selects the new cohort and is separate
from these read-only observations. The original results remain preserved; broad
DINO approval and the newly granted concrete authorization need no further request.

## Sources and reproduction

| New supplied source | Drive reference |
| --- | --- |
| `slap_subject_disjoint_train_ab.json` | [Training JSON](https://drive.google.com/file/d/1YBla7ejXiiOruP2KZP5Iy6P8MvqZi1dZ/view) |
| `train_genuine_nonzero.txt` | [Training genuine list](https://drive.google.com/file/d/1XCWYC9I9W9pMbtSQxAMEqybr70XC6eK1/view) |
| `train_impostor_nonzero.txt` | [Training impostor list](https://drive.google.com/file/d/1JmjAZPWPY7mEmQmYDNqsbwdrplolClJi/view) |
| `test_genuine_2100_nonzero.txt` | [Testing genuine list](https://drive.google.com/file/d/1Bh8M1gmYsad4EHixawZpQj2PjHDxpyzx/view) |
| `test_impostor_2100_nonzero.txt` | [Testing impostor list](https://drive.google.com/file/d/1ugqNA9-6xd4VB4kHIfsnquWtE27mG7f2/view) |

New file names and byte hashes are recorded in the aggregate. The five raw
sources total about 28.0 MiB. They were fetched via the connected Google Drive
raw-file route, checked against provider byte sizes and stored privately without
overwriting any prior input. No image/model archive was downloaded again.
Private policy text, source paths and any identifying review cases remain under
`local/precise/metadata/revised-cohort-20261107/`.

Only repeat this intake when sources or code change; use new report/private paths:

```bash
.venv/bin/python code/audit_precise_revised_cohort.py \
  --intake local/precise/collaborator/konain-followup-20261107 \
  --inventory local/precise/metadata/readiness-20261025/readiness_records.json \
  --approved-bundle local/precise/manifests/shared-pairs-20261101/bundle.json \
  --approved-record local/precise/runs/shared-pairs-20261101/adapted-training/run_record.json \
  --image-root datasets/Precise \
  --earlier-training-json local/precise/collaborator/konain-followup-20261101/slap_verification_binary_ab_13_14_15_balanced.json \
  --output reports/precise-revised-cohort-NEWDATE/cohort_audit.json \
  --private-dir local/precise/metadata/revised-cohort-NEWDATE
```

Four synthetic contracts cover disjoint new groups versus prior model exposure,
retained zeros, JSON/list disagreement, incorrect targets, changed prior-bundle
provenance, missing files, prompt equality, aggregate privacy, no image decoding,
source immutability and overwrite refusal. Regular CI discovers them.
