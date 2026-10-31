# Precise common-list score comparison — 2026-10-31

`code/precise_matching/comparison.py` extends the existing offline evaluator
with aggregate comparisons of two to four supplied frozen-method score tables.
This advances roadmap task 9 without allocating subjects, generating comparisons,
reading fingerprint pixels or executing a matcher. No new research scores are
available. Acquisition/exposure and protocol/results/split review and GPU repair
still precede real execution.

## Input and provenance contract

All methods use one private images table and one private comparison table, with
the [existing exact CSV contracts](precise_matching_tooling_20261017.md).
Each method must supply exactly one outcome for every development/test pair,
including failed attempts. Missing, extra, repeated or training score rows are
rejected; no intersection join silently removes a method's failures. Subjects,
hands, captures, labels and partitions come from the common validated metadata.
Unresolved capture/exposure flags block comparison before score-table parsing.

Supported public arm names are `frozen_dinov2`, `adapted_dinov2`,
`verifinger_fusion` and `verifinger_sum`. At least two distinct arms are required.
Fixed names prevent exporting arbitrary private labels through the method-name
field. An arm name describes the intended method; it does not prove that its
scores came from that method. Independently review configuration, model/SDK
version, original score provenance, frozen selection and execution authorization.
The adapted scorer's [selection verification and one-time test claim](precise_final_test_20261026.md)
remain separate from this offline reporting command. Historical collaborator
scores cannot stand in for new common-list scores merely by renaming pair IDs.

For each arm, the evaluator chooses its own strict `score > cutoff` boundary
using only valid pooled development impostor scores and the explicit FAR target.
Use the same FAR target frozen in the reviewed experiment; the CLI does not
independently authenticate that choice against each method's original run record.
The exact integer budget and tie handling are unchanged. Successful development
scores must include both classes. Development failures remain visible and do
not relax the valid-impostor budget. This reporting capability does not relax
the training pipeline's stricter requirement to resolve development failures
before checkpoint selection.

## Aggregate report

Each method reports its threshold and development summary, then pooled, left
and right test summaries. All test summaries use that method's same pooled
development cutoff. Counts, TAR/FAR, coverage, failures and descriptive valid-score
ROC/DET, AUC and interpolated EER reuse the existing evaluator. Missing hand or
class support produces null rates rather than fabricated values.

Every unordered method pair additionally reports, by hand and pooled, separate
genuine and impostor counts for both accept, A-only accept, B-only accept and
both reject. Failures remain rejects in these primary all-attempt counts. The
acceptance-rate difference is **B minus A**, divided by the unchanged number
of attempted comparisons. For genuine pairs this is the TAR difference; for
impostors it is the FAR difference, so a positive value increases false accepts.
Arm ordering is lexicographic and explicitly recorded; no winner is selected.

Paired coverage counts distinguish both valid, A-only valid, B-only valid and
neither valid. A supplemental both-valid comparison gives its reduced denominator,
acceptance counts and rate difference using the original method cutoffs. It is
not the primary population and can favor selectively failing methods. Failed
impostors rejected by the software are not evidence of biometric discrimination.

These are descriptive comparisons. Repeated subjects/images invalidate an
independent-trial interpretation. No significance test, confidence interval or
population low-FAR bound is produced. The subject-resampling proposal in the
[protocol review](precise_milestone_review_20261025.md) remains unapproved and
unimplemented. Test curves, differences and EER must not be used to retune the
selected models or thresholds.

## Command for future reviewed scores

The paths and FAR below are examples, not actual approved experiment artifacts
or a finalized numerical protocol. The command uses only the standard library.

```bash
PYTHONPATH=code python3 -m precise_matching compare \
  --images local/precise/metadata/approved_images.csv \
  --pairs local/precise/manifests/approved_pairs.csv \
  --method frozen_dinov2 local/precise/runs/frozen/scores.csv \
  --method adapted_dinov2 local/precise/runs/adapted/scores.csv \
  --target-far 0.001 \
  --output reports/precise-comparison-YYYYMMDD/comparison.json
```

Repeat `--method ARM SCORES` for available SDK arms. `--scores` is reserved for
single-method evaluation. Output must be a new JSON path under `reports/`;
existing files and resolved symlink escapes are refused. Every input's byte
digest is checked before and after processing; a change prevents report output.
Inputs are read only. Reports omit private identifiers, filenames, paths and
hashes, but still require manual publication review. Exit codes remain 0 for
successful aggregates, 1 for an aggregate metadata-blocker report without
scoring, and 2 for input/output failure without a successful report.

## Synthetic validation

`tests/test_precise_comparison.py` checks mismatched/duplicate score coverage,
review flags, finite-score contracts, independent score scales, strict unchanged
pooled cutoffs, opposing paired decisions, differing failures, reduced valid
intersections, missing hand/all-failed support, all six four-arm comparisons,
order invariance, CLI round-trip/privacy, duplicate/unknown arm rejection,
immutable input files, overwrite refusal and mutation detection. The regular
CI unit-test discovery includes this suite without extra dependencies or private
inputs. These fixtures provide implementation evidence, not fingerprint results.
