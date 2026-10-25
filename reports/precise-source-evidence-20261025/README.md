# Precise source investigation — 2026-10-25

This task investigates existing material alongside the separate first-milestone
implementation/readiness work. It adds source evidence and a question matrix;
it does not select a split/model, construct comparisons, run upstream code or
change the approved whole-slap direction. Identifying records and historical
Drive document copies remain under ignored
`local/precise/metadata/source-evidence-20261025/`.

## Answers established from the supplied files

The [earlier intake audit](../precise-training-intake-20261024/README.md)
established zero unordered training/testing pair overlap but complete testing
image/subject overlap. This follow-up establishes an additional exact property:
the genuine training and testing lists, combined, contain **every unordered
pair of distinct local filenames within each subject/slap-type group**.

| Slap type | Possible distinct-file genuine pairs | Supplied training + testing genuine pairs |
|---|---:|---:|
| Right four-finger (`13`) | 8,393 | 8,393 |
| Left four-finger (`14`) | 8,418 | 8,418 |
| Two thumbs (`15`) | 8,392 | 8,392 |
| Total | **25,203** | **23,103 + 2,100 = 25,203** |

For a group with `n` files, the possible count is `n*(n-1)/2`. Every supplied
genuine row uses two distinct known files with the same subject/slap labels;
all 25,203 unordered pairs are unique. Those facts and the equal counts prove
coverage without generating a new pair list. This characterizes the received
list membership; it does not prove the original generator, selection seed,
capture independence, binary training target or actual model consumption.

## Candidate time fields narrow the acquisition question

All 7,358 penultimate filename tokens parse as calendar-shaped values:

| Candidate format | Files |
|---|---:|
| `%Y%m%d%H%M%S` | 3,060 |
| `%y%m%d-%H%M%S` | 4,298 |

All 1,014 subject/slap groups have multiple distinct candidate times, with no
repeated candidate time within a group. Only 12 groups span multiple candidate
calendar dates. Of the 2,100 genuine testing rows, 41 cross candidate dates
(15 right, 13 left, 13 thumbs). Of the 23,103 genuine training rows, 679 cross
candidate dates (225 right, 227 left, 227 thumbs). Thus 720 of the combined
25,203 genuine comparisons cross these candidate dates.

These are syntactic observations, **not verified acquisition timestamps or
session assignments**. The values could record acquisition, export or
processing. Distinct values do not establish independent captures; the same
calendar date does not establish a shared session. Do not infer demographic
labels from other middle tokens. No capture/session IDs, exclusions or
partitions were assigned.

The private `filename_time_candidates.csv` contains the mapping, with
`interpretation_verified=false` for every row. The useful collaborator question
is now specific: **does the penultimate field record original acquisition time,
and can separate values still refer to processed versions of one acquisition?**
Ask for a session/parent-acquisition mapping or its documented rule rather than
requesting the pair lists again.

## Local source packaging

Both Downloads archives (`Precise.zip` and `Precise (1).zip`) have 7,358 JPEG
entries and no non-image files. Each archive is 5,679,745,916 bytes. Their
central-directory path/CRC/size/compressed-size maps agree, and their image
basenames and uncompressed sizes match the current dataset. This is not a
full byte comparison of the archives or their decompressed images.

The current `datasets/Precise/` folder and the separate Downloads `Precise/`
folder contain JPEGs only. No capture dictionary or dataset README is embedded
in these inspected packages. Dataset identity, source institution, sensor and
physical PPI cannot be established merely from the label "Precise". Public
search returned no authoritative source that could be tied to this exact local
collection; similarly named fingerprint resources must not be substituted.
The earlier header audit found no DPI metadata. This source investigation also
opened all 7,358 JPEG headers: Pillow exposes only JFIF fields, with no EXIF
tags or nonempty JPEG comments and no header errors. No pixel decoding was
needed for that probe. No additional capture dictionary was exposed there.
The parallel
[full-image readiness audit](../precise-readiness-20261025/README.md) supplies
the newer decoding and duplicate-screening evidence; this task does not repeat
that processing or adjudicate its candidates.

## FPBench coverage and boundaries

FPBench is the **upstream repository Konain recommended**, not an identified
copy of his Precise-specific adaptation. Its [README](https://github.com/Ektagavas/FPBench/blob/ed27ae669947bfd157522d537a9f9c77b4bf3189/README.md)
attributes it to the FPBench authors and lists seven source datasets, without
Precise. The pinned checkout remains
`ed27ae669947bfd157522d537a9f9c77b4bf3189`; upstream has one advertised branch,
`main`, at that revision, and no advertised tags at inspection time.

The complete four-commit reachable history was searched in Python, Markdown,
JSON and YAML files for `precise`, `verifinger`, `minus2100`, `78592` and
`binary`, with no matching files. The GitHub API lists one issue, about
[releasing artifacts on Hugging Face](https://github.com/Ektagavas/FPBench/issues/1),
and no pull requests. This coverage does not rule out another repository,
unpublished changes, unshared files or future updates.

The checked-in matching schemas contain:

| Source | Questions | Images per question | Options |
|---|---:|---|---|
| Fine-tuning JSON | 1,800 | 716 with three; 1,084 with four | Four each |
| Evaluation JSON | 804 | 320 with three; 484 with four | Four each |

The [Qwen script inspection](../../docs/notes/fpbench_reference_20261024.md)
already records answer-option supervised training, the hardcoded model loader,
task default and absent Precise conversion/development setup. The
[authors' paper, implementation details](https://arxiv.org/html/2512.18073v2#S4.SS4),
describes separate fine-tuning/evaluation MCQs and a single H200 GPU. Its
reported task accuracy and runtime do not predict whole-slap TAR at a fixed
FAR or runtime on this workstation. The paper does not mention Precise or
establish this project's subject split.

The README's fine-tuned-checkpoint Drive link was inspected by metadata only.
The connected account received HTTP 404, and the public web reader could not
access the URL. This means **inaccessible in the inspected access context**;
it does not prove the file was deleted. No checkpoint was downloaded. Even if
access is restored, these are upstream task adapters, not evidence of Konain's
current binary Precise checkpoint.

## Shared Drive investigation

The two explicitly shared testing files remain readable by ID and have no
comments. Returned metadata exposes no parent folder, so it does not provide
a route to all of Konain's other files. Searches used `Precise`, `FPBench`,
`genuine`, `impostor`, `Konain` and `VeriFinger`, a corrected name-filter search,
and a Precise folder search. No additional Precise metadata or binary-training
adapter was located in that scope. Search did not surface the known genuine
test file despite successful direct access, illustrating why search absence
cannot establish that an artifact does not exist.

Six relevant historical planning documents were read, including the two early
SLAPBench plans, `Next Step | SlapBench`, `Task Upcoming: Slap Finger`, the
revision instructions and the student guide. The identified SLAPBench folder
was listed as well. These documents predate the current October direction;
they describe earlier benchmark ideas, the SD302b capture concern and general
verification safeguards. They supply no verified Precise capture dictionary,
Konain binary conversion or actual training/development consumption record.
Authorship/authority is not inferred from Drive storage or a document title.
Their embedded instructions are historical evidence, not new authorization,
and the removed guide was not restored to the active public tree.

Private receipts retain document IDs, dates, text and scoped search/access
results. Public output does not reproduce private documents or their IDs.
No sharing state was changed and no message was sent.

## Question matrix for the implementation work

| Question | Evidence now available | Remaining action |
|---|---|---|
| Are the training/testing pairs different? | Yes; exact unordered pairs are disjoint | Do not request these lists again |
| What does the genuine training list include? | All within-subject/slap distinct-file combinations except the 2,100 held-out genuine rows | Do not infer acquisition independence or generator implementation |
| Are testing images/subjects absent from training lists? | No; all are present elsewhere | Use the agreed protocol for our fresh model; distinguish retrospective claims |
| Is there useful structure in middle tokens? | Every penultimate token parses under one of two calendar formats | Confirm acquisition versus processing/export meaning and parent captures |
| Does either local ZIP contain the missing README? | No; inspected central directories contain JPEGs only | Locate a separate original source/metadata document |
| Does FPBench contain Konain's Precise binary implementation? | Not in the inspected public tree/history | Obtain his adaptation URL/commit or a redacted configuration/example if reproducing it |
| Are binary targets identity labels or SDK-derived decisions? | Not established by pair classes, scores or upstream MCQ targets | Confirm the target construction and zero-score treatment |
| Why are training impostors right-hand only? | That population is confirmed | Confirm intended scope or additional negative input files |
| Was a separate development set used? | No supplied assignment or consumption record found | Confirm actual use and checkpoint/threshold selection |
| Which encoder/loss/FAR should our experiment use? | Parallel work provides a concrete review proposal | These are project design/review decisions, not facts a source search can recover |

## Reproduction and checks

Machine-readable local-source aggregates are in [evidence.json](evidence.json).
The preserved one-off script is private at
`local/precise/metadata/source-evidence-20261025/inspect_local_sources.py`;
it runs from the project root and refuses the existing report directory.
Review it and choose fresh private/public output locations for a repeat.
It uses the existing score parser, combinatorial counts, calendar parsing,
ZIP central-directory metadata and static Git/JSON inspection only. No new
image comparison list is generated. Source provenance and candidate mappings
remain local, with input-byte checks for the two genuine source files.
The separate Pillow header probe is preserved in private
`jpeg_metadata_probe.json` and summarized in `evidence.json`; it inspects
`Image.info` key presence, EXIF tag presence and nonempty JPEG comments only.

Independent checks recompute the genuine-pair capacity/coverage and candidate
time aggregates, verify the matching JSON counts, and scan public output for
actual image names or private document IDs. Whitespace, local document links,
publication scope/size and legacy-metric reproduction are checked before
publication. No new model weights, dependencies, experiments, segmentation,
training runs, paid calls or external messages are part of this task.
