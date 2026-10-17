# Precise offline matcher tooling — 2026-10-17

The user requested progress toward the approved plan and confirmed that Konain's
last update was the testing-file links. No new filename/capture/session
definitions or exposure records have been supplied. The next implemented piece
is offline protocol validation and score evaluation in `code/precise_matching/`.
It uses only the Python standard library and has no model, GPU or network calls.

## What is implemented

`data.py` reads private, externally prepared image metadata and comparison
tables. It checks subject separation across all listed images, including unused
images and both hands; capture ownership; supplied raw/decoded content hashes;
genuine/impostor labels; matching hand types; self-pairs; and duplicate unordered
image/capture comparisons. All three partitions need images, and development
and test need both comparison classes. Training-pair class requirements remain
dependent on the future objective.

Missing capture IDs/hashes, unverified captures and unreviewed exposure are
explicit blockers. Only `left` and `right` four-finger slaps are accepted.
The tool compares supplied metadata; it does not establish acquisition truth,
read image files, check near duplicates, verify historical exposure, allocate
subjects or construct pairs. A passing result is a metadata-consistency check,
not training approval or proof of an untouched test cohort.

`evaluation.py` accepts one system's already generated scores, joined to the
comparison table by pair ID. Every development/test pair must appear exactly
once, including failed attempts; training or unknown pairs are rejected. Labels
and partitions come from the validated tables, not from the score file.

The cutoff is selected **only from valid development impostor scores**. Higher
scores mean more similarity. The explicit decision rule is **accept if score
is strictly greater than the cutoff**. The integer false-accept budget is
`floor(target_far × valid_development_impostors)`, computed with exact rational
arithmetic from the supplied rate string. The cutoff is the next rejected
impostor score in descending order; a tie crossing the budget is rejected as a
whole. Strict comparison handles arbitrary finite score scales without rounding
or adding an arbitrary score increment. No default FAR target is chosen.

Apply that same cutoff to test without retuning. Reports contain acceptance
counts, denominators, TAR/FAR, per-class coverage, failure-status counts and
descriptive valid-score ROC/DET rates, AUC and linearly interpolated EER. The
curve's initial point rejects everything; subsequent points admit complete
score ties. Curves do not select the operating cutoff. Interpolated EER need
not be an attainable operating point. No uncertainty model or population
low-FAR guarantee is implemented.

Failures have empty scores and appear separately. Reports include both a
valid-score view and an all-attempt failure-as-reject view; failed impostors do
not relax the development error budget. Zero is a valid score. If an entire
test class fails, its valid-score rates/curves are unavailable rather than
fabricated. Development requires at least one valid score from each class.
Checkpoint/configuration provenance still requires a separate frozen run record.

## Private CSV contracts

These are schemas for future approved inputs, not generated Precise manifests.
Keep their records and any mapping back to original images under ignored
`local/precise/`. Tests create artificial records in temporary directories only.
Use UTF-8 CSV with exactly these columns, in any order. Repeated/unknown headers,
ragged rows, surrounding whitespace, empty tables and repeated record IDs are
rejected. Subject IDs must consistently denote the same person across hands
and partitions. Identifiers and source hashes are never exported by the CLI.

| Table | Exact header |
|---|---|
| Images | `image_id,subject_id,hand,capture_id,partition,sha256,decoded_sha256,capture_verified,exposure_reviewed` |
| Comparisons | `pair_id,image_a,image_b,label` |
| Scores | `pair_id,status,score` |

Image `partition` is `train`, `development` or `test`; `hand` is `left` or `right`.
`capture_id` denotes the globally unique parent whole-slap acquisition, shared
by all alternative encodings of that capture. Distinct capture IDs must reflect
verified independent acquisitions, not guessed filename-token meanings.
Capture IDs and hashes may be empty for preflight diagnosis, but then evaluation
is blocked. Hashes must be lowercase SHA-256. `sha256` describes original file
bytes. `decoded_sha256` must use one documented, consistent decoding convention
throughout the input. The existing image inspection uses SHA-256 of
`str(gray.size).encode() + gray.tobytes()` after Pillow conversion to `L`;
decoder/version provenance belongs in the private run record. This tool does
not calculate those hashes or validate that their convention was followed.

`capture_verified` and `exposure_reviewed` accept `true`, `false` or `unknown`.
Only `true` passes. A review flag records a human-reviewed assertion; it is
not evidence created by this tool. Exposure review must document prior uses
and permitted claim scope, not relabel previously inspected subjects untouched.
`label` is `genuine` or `impostor`, consistent with subject equality.

Score `status` is `ok`, `decode_error`, `encoder_error`, `missing_input` or
`error`. `ok` requires a finite numeric score; every other status requires an
empty score. Choose one frozen system/configuration per score table. The CLI
does not choose models, train an encoder or generate missing scores.

## Commands for future approved inputs

From the repository root, use `PYTHONPATH=code` to import the package. The input
paths below are placeholders; this task has not created those tables. Choose
new output filenames for each invocation. Reports are aggregate JSON only and
must be under `reports/`; existing files and resolved paths outside that tree
are refused before inputs are read.

```bash
PYTHONPATH=code python3 -m precise_matching validate \
  --images local/precise/metadata/approved_images.csv \
  --pairs local/precise/manifests/approved_pairs.csv \
  --output reports/precise-protocol-YYYYMMDD/preflight.json

# Example rate only; the actual target still requires protocol review.
PYTHONPATH=code python3 -m precise_matching evaluate \
  --images local/precise/metadata/approved_images.csv \
  --pairs local/precise/manifests/approved_pairs.csv \
  --scores local/precise/runs/approved-run/scores.csv \
  --target-far 0.001 \
  --output reports/precise-evaluation-YYYYMMDD/evaluation.json
```

Exit codes: `0` means checks passed and the requested aggregate was written;
`1` means a blocker summary was written and no scores were evaluated;
`2` means an input/output error, with no successful report. Input-byte hashes
are checked before and after processing but remain private. The CLI does not
echo private paths, record IDs, subject IDs or arbitrary error text. Public
outputs still require review before publication.

## Validation and next steps

Synthetic tests exercise the CSV contracts, cross-partition/hand/capture/content
conflicts, duplicate pairs, unknown metadata, threshold ties and exact FAR
budgets, arbitrary score scales, score alignment, failures and aggregate-only
outputs. An end-to-end test writes private synthetic CSVs, runs evaluation,
verifies the fixed test cutoff behavior and checks source bytes are unchanged.
They run through the existing CI unittest discovery; CI syntax checks now
include nested Python packages.

Next: obtain the acquisition definitions and exposure history, agree feasible
claim scope, then review the concrete split and baseline configuration. Encoder
and loss modules remain future implementation. This task has not created real
subject assignments, pairs, crops or model outputs, and makes no learned-matcher
performance claim. The [central plan](precise_first_milestone_proposal_20261011.md)
and [protocol sketch](precise_protocol_sketch_20261010.md) retain the full study
requirements.

## Short message for Konain

> Thanks for sharing the testing files; we have audited the scores. As we set up
> the learned matcher, could you clarify what the middle filename tokens mean,
> whether genuine pairs are independent acquisitions or processed versions of
> one capture, and what session/sensor/resolution metadata exists? Please also
> share the training/development subject or image mappings through the private
> channel so we can check exposure and define the split. Your adapted script
> and SDK settings would help reproduction, but the dataset interpretation is
> our immediate priority.

This is a draft for the user. No external message was sent.
