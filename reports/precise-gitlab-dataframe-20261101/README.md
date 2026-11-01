# GitLab annotation dataframe intake — 2026-11-01

The user supplied `Gitlab_SlapBench/` as a local reference. It contains
`dataset/dataframe/CompleteWithRotated.P` and an accompanying schema description.
The description matches the professor's message. The actual file is a pandas
pickle, not an Excel workbook. Upstream GitLab revision/original export time
was not established: the folder has no Git metadata, and its root README is a
generic GitLab template. Its sample remote/force-push commands were not executed.

The original 96,804,926-byte dataframe is retained unchanged under the ignored
reference folder. Its SHA-256 is
`67c94c7cbf65c8100c4ef04aea221e7e25dc14473021d0f76a0d4524b6d894a1`.
The [aggregate report](dataframe_summary.json) contains no image names, subject
values, row-level labels or fingerprint images. Private image mappings and
version/source receipts are in `local/precise/metadata/gitlab-dataframe-20261101/`.

## Verified structure and coverage

| Finding | Observed value |
|---|---:|
| All annotation rows / columns | 457,005 / 26 |
| Precise annotation rows | 24,530 |
| Distinct Precise slap names | 7,358 |
| Exact local JPEG basename coverage | 7,358 of 7,358 |
| Local images lacking an annotation / annotation names lacking a local image | 0 / 0 |
| Multiple SubjectID values within one slap | 0 |
| Finger-set conflicts with filename hand labels | 0 |
| Repeated finger codes within a Precise slap | 0 |

There are 2,452 right slaps, each with codes 2–5; 2,455 left slaps, each with
codes 7–10; and 2,451 thumb slaps, each with codes 11/12. The 24,530 rows are
finger annotations across those images, not 24,530 independent captures.
The complete table also contains rows labeled Children (22,370), LiveDet
(5,774), augmented (404,230), and 101 other/missing-dataset rows. Those rows
were not folded into the Precise population. The schema has no explicit
source-dataset/parent-capture field for rows labeled augmented.

Within Precise, 20,900 annotation rows have rotation angle zero, 3,630 have a
missing angle and none have an observed nonzero angle. Missing angles remain
missing. Eleven rows have missing original/rotated geometry and quality flags;
their dataset/subject/slap/finger fields are present. No quality exclusion,
geometry repair, segmentation or augmentation was performed.

## Identity finding: padding aliases require confirmation

The dataframe contains **277 numeric SubjectID values**, while the earlier
filename inventory counted **338 distinct prefix strings**. Of the 277 numeric
groups, 216 have one raw prefix and 61 each have two prefixes. All 61 two-prefix
groups differ solely in ASCII leading-zero padding. Every image's prefix agrees
numerically with its dataframe SubjectID; 4,628 agree as exact strings and 2,730
agree only after numeric conversion. No prefix maps to multiple dataframe IDs.

For illustration only, a synthetic prefix such as `0007` and `7` can map to the
same numeric ID. No actual subject values are shown here. This is evidence for
candidate formatting aliases, not independent verification that each numeric
ID is one person. The numeric dataframe may have discarded meaningful original
formatting. Confirm the intended canonical subject mapping with the professor
before merging identities, constructing genuine/impostor labels or allocating
subjects. Do not treat 338 raw strings as an established count of distinct
people, or silently replace historical pair labels with numeric equality.

The previous 202/68/68 split proposal assumed 338 eligible distinct subjects.
If 277 canonical subjects were confirmed and all qualified, the existing
60/20/20 rounding rule would yield 166/55/56. This is a **hypothetical size
calculation**, not allocation: development and test would both fall below the
draft's 60-subject review condition. With both hands fully eligible, the old
impostor rule would allow at most 2,970 development / 3,080 test comparisons.
Revisit statistical support and protocol sizes rather than silently accepting
that scenario. No subject has been assigned to any partition.

## What this resolves and what remains open

The actual annotation table is now available and its coverage, row unit,
finger/hand consistency and numeric identity relationship have been checked.
This supersedes the earlier statement that only a column description was
available. The field meanings remain the professor's reported semantics;
consistent annotations alone do not independently verify biometric identity.

The schema has no explicit acquisition/session/parent identifier, sensor,
physical PPI or historical model-consumption fields. No independent-capture
facts can be inferred from zero rotation, different filenames or annotation
rows. The focused data questions are now whether the padding aliases denote
the same participant and how to identify independent original acquisitions.
Professor review of the [LLM-focused DINOv2 plan](../../docs/notes/precise_dinov2_llm_plan_20261101.md)
and concrete protocol/results/split review still precede real experiments.

## Method and reproduction

`code/audit_precise_dataframe.py` uses a fixed allowlist of pandas/numpy
reconstruction globals for this observed format. Unknown constructors and
schema changes fail without unrestricted-unpickle fallback. It does not execute
the reference repository's scripts. This format-specific restriction is not a
general-purpose pickle sandbox. The successful intake used Python 3.12.3,
pandas 3.0.6 and NumPy 2.5.3 in the existing verification environment.

```bash
.venv/bin/python code/audit_precise_dataframe.py \
  --dataframe Gitlab_SlapBench/dataset/dataframe/CompleteWithRotated.P \
  --images datasets/Precise \
  --output-dir reports/precise-gitlab-dataframe-20261101 \
  --private-dir local/precise/metadata/gitlab-dataframe-20261101
```

Both output directories must be new; use different dated paths for another
authorized intake. The tool validates the original digest and image-name set
again before writing. It reads image names only, not their bytes. The private
CSV preserves the conflicting/raw identity representations and marks every
identity/capture review unresolved; it is not a training manifest. No original
data, existing scores, paper or historical manifest was changed.
