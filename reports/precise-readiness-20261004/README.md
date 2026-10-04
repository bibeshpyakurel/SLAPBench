# Precise readiness inventory — 2026-10-04

This is an offline preparation report, not an experiment, approved split or
claim of biometric performance. Source: the local `datasets/Precise/` collection.
No source image, existing manifest or result was modified.

## Observations

| Item | Observed count |
|---|---:|
| JPEG files | 7,358 |
| Subjects from recognized filename labels | 338 |
| Right four-finger slap files (position 13) | 2,452 |
| Left four-finger slap files (position 14) | 2,455 |
| Four-finger slap files combined | 4,907 |
| Thumb files (position 15) | 2,451 |
| Subjects with both slap hands | 338 |
| Slap subject/hand groups with multiple distinct byte contents | 676 |
| Exact byte-duplicate groups across all JPEG files | 0 |
| Three-token filenames | 9 |
| Four-token filenames | 7,349 |
| Unrecognized filenames under the inventory's rules | 0 |

Every observed subject/position group has at least two files, including thumb
groups. These are filename-based counts, not verified biometric annotations.
The three-token variant is retained in inventory because the existing pair
builder uses only the first and last tokens. Its acquisition meaning needs
checking rather than silently excluding or relabeling those nine files.

## Method and provenance

Generated from the unchanged local collection with:

```bash
venv/bin/python code/audit_precise_dataset.py --output reports/precise-readiness-20261004/inventory.json
```

The script uses only the Python standard library; `python3` also works. It
reads top-level JPEG filenames and computes SHA-256 over all file bytes. The
first and last tokens are interpreted consistently with `code/precise_pairs.py`;
middle tokens are not interpreted. The report records aggregate counts and
group-size histograms, omitting subject IDs, original filenames, per-image
hashes and absolute workstation paths. An existing output file is refused.

The machine-readable observations are in [inventory.json](inventory.json).
For comparison, the earlier repository inventory already recorded 7,358 JPEGs.
This audit establishes aggregate byte/label coverage, not the meaning of the
previous experiment's genuine-pair labels.

## What remains unverified

Different file bytes can encode the same pixels or different derivatives of
one capture. No image decoding, near-duplicate analysis, image quality check,
resolution validation or acquisition/session verification was performed.
Hand codes do not establish individual-finger segmentation or labels.
Dataset provenance/access terms and the collaborator's previously used subjects
remain pending. No pair list or subject split was generated, and no baseline,
GPU job, API call, model download or training run was performed.

A local PATH check did not find `mindtct` or `bozorth3`; that does not establish
their absence elsewhere. This preparation did not install NBIS or establish
VeriFinger SDK access.

Next decisions are recorded in the
[preparation note](../../docs/notes/precise_preparation_20261004.md). Keep the
existing `datasets/`, `results/` and paper paths intact for reproducibility.
