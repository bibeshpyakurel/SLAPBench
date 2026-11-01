# Precise score-status reconciliation — 2026-11-01

The user supplied collaborator explanations of actual training counts, a 1:4
class ratio and a template-extraction-failure-to-zero convention, plus the
professor's offer of ground-truth metadata. The
[clarification note](../../docs/notes/precise_clarification_20261101.md) separates
those statements from independently verified facts and records remaining gates.

This is a targeted read-only count of the four preserved October 24 source
files using their existing parser. Source SHA-256 values match the original
private audit receipt before processing and are checked again afterward.
No new list, split, eligibility filter, inference, score or threshold is created.
The machine-readable aggregate is [score_status_inventory.json](score_status_inventory.json).

| Source | All rows | Nonzero fused | Zero fused | Zero missing components | Zero with components | Zero with positive component sum |
|---|---:|---:|---:|---:|---:|---:|
| Training genuine | 23,103 | 19,594 | 3,509 | 3,455 | 54 | 51 |
| Training impostor sample | 78,592 | 73,538 | 5,054 | 0 | 5,054 | 5,017 |
| Testing genuine | 2,100 | 2,094 | 6 | 0 | 6 | 6 |
| Testing impostor | 2,100 | 1,956 | 144 | 0 | 144 | 136 |

The reported 19,648 genuine count equals the supplied genuine count after
excluding the 3,455 missing-component rows; it does not equal literal nonzero
fused filtering. The reported 78,592 impostors include 5,054 fused-zero rows in
the sampled file. Many fused-zero records have positive component sums, so
template-conversion status cannot be recovered from the fused value alone.
The collaborator's exact consumed files/filtering code and conversion status
records are still needed. No automatic failure relabeling follows from this
report, and no historical score or metric is changed.

The reported 8,992,383 impostor pool has not been downloaded or counted. The
shared folder was not accessible to the connected Drive account or web fallback;
this does not establish that it is absent or empty. Receipt/provenance and
identifying inputs remain ignored under
`local/precise/metadata/clarification-20261101/`.
