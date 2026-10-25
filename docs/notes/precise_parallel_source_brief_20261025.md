# Parallel source investigation and implementation handoff — 2026-10-25

The user assigned the ten-task first milestone to another worktree and asked
for useful parallel work. This task investigates the material already available
so implementation need not wait for questions that files can answer.
The [source-evidence report](../../reports/precise-source-evidence-20261025/README.md)
records sources, findings, access limitations and a question matrix.

## Findings to carry into the milestone

- The 23,103 training and 2,100 testing genuine rows together exhaust all
  25,203 unordered distinct-file combinations within subject/slap groups.
  They are separated by exact pairs, while their images/subjects overlap.
- All 7,358 penultimate filename fields parse as candidate date/time values.
  A private mapping is available, with every interpretation explicitly
  unverified. Only 41 genuine test rows cross candidate calendar dates;
  these are not verified cross-session labels.
- Both local Precise archives contain JPEGs only. No embedded acquisition
  dictionary was overlooked in their central directories.
- The upstream FPBench repository recommended by Konain has no Precise binary
  conversion in the inspected four-commit history. Its checkpoint link is
  inaccessible in the inspected account. Six historical Drive plans do not
  resolve the missing acquisition or adaptation facts.

The private mapping is
`local/precise/metadata/source-evidence-20261025/filename_time_candidates.csv`.
Use it to focus provenance review; do not promote candidate times into
verified capture IDs or populate `capture_verified=true` from this evidence.
The [current model/protocol draft](precise_milestone_review_20261025.md) remains
a proposal for review. Its design decisions cannot be recovered by searching
Konain's material.

## Useful parallel assignments

| Assignment | Concrete output | Separation from matcher implementation |
|---|---|---|
| Source provenance and artifact recovery | Maintain source/URL/revision receipts; inspect any newly available original metadata or adapted script against the question matrix | Read-only sources and dedicated notes/private receipts |
| Acquisition evidence review | Validate candidate time meanings against source documentation; reconcile reviewed parent captures and the readiness audit's suspicious relationships | Private reviewed evidence; no automatic relabeling, exclusions or split changes |
| Reproduction specification for Konain's binary arm | Record actual base model, binary targets, consumption, development selection, right-only negative scope and zero-score handling when evidence arrives | Separate comparison arm/configuration; no unapproved training |
| Related-work and contribution matrix | Compare our whole-slap continuous-score protocol with FPBench's MCQ setup and other verified primary literature | Bibliography/research notes; no unsupported novelty or performance claims |

Nico could take source curation or help review private evidence against a clear
rubric if authorized to access the data. Acquisition independence and ambiguous
duplicate/label cases still need appropriate source or expert confirmation.
Access and consent to share private inputs are not implied by this assignment.

Keep one owner for the central plan, CLI, dependencies and integration. This
source task adds its own report/brief and a dated STATUS entry, without changing
encoder/training code, subject assignments, existing results or the parallel
readiness audit. New discoveries should update evidence, not silently change
an approved experimental population.

No collaborator or professor message was sent. No new worktree was created and
no training or benchmark experiment was run by this task.
