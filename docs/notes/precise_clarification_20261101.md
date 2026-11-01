# Precise collaborator clarification and metadata offer — 2026-11-01

The user supplied new messages from Konain and the professor in the milestone
chat. This is new evidence, not a new training run or approval of a concrete
split. Original message timestamps were not supplied; the date above is the
receipt date. A normalized correspondence receipt, source hashes and access
results remain under ignored `local/precise/metadata/clarification-20261101/`.
The [aggregate reconciliation](../../reports/precise-clarification-20261101/README.md)
checks the new statements against the preserved October 24 files without
changing any score, dataset, pair list or prior report.

## What the messages establish, and their limits

| Topic | New source statement | Evidence boundary |
|---|---|---|
| Filename identity | Konain confirms subject before the first underscore and hand between the last underscore and extension | Consistent with existing parsing; middle tokens and independent capture/session relationships remain unresolved |
| Evaluation selection | He reports randomly selecting 2,100 genuine and 2,100 impostor evaluation pairs and excluding them from the remaining pool | The existing intake independently verifies exact-pair exclusion, with shared images/subjects; this does not establish subject separation or development selection |
| Reported training use | He reports using approximately 19,648 genuine and 78,592 impostor comparisons on an RTX 3090 | Record as actual use reported by the collaborator, upgrading the previous list-membership-only evidence; no consumed-row manifest, original filtering code or checkpoint/run log was provided |
| Class ratio | He explains increasing impostors to four times genuines after lower impostor accuracy | `78,592 = 4 × 19,648`; the quoted accuracy is an unverified collaborator observation, not a new result of our matcher |
| Hand selection | He acknowledges that his selected comparisons might be right-hand only | The preserved sampled impostor list is definitively right-only; the supplied genuine list includes both hands and thumbs; the exact consumed genuine subset is still unverified |
| SDK zero convention | He says failed image-to-`.dat` template conversion causes a pair's exported score to be zero when either endpoint fails | Explains a failure-to-zero convention; it does not establish that every fused zero is a template failure, or identify each failed endpoint |
| Available pool | He gives 23,103 genuine and 8,992,383 impostor pairs and recommends nonzero comparisons after excluding evaluation pairs | The 23,103 genuine file is already local; 8,992,383 is a reported pool, not the 78,592-row sampled file or a locally verified count; do not subtract evaluation rows again from an already excluded pool |
| Ground-truth metadata | The professor says a dataframe containing resolution, subject ID and other ground truth will be shared | The dataframe itself has not yet been received; its schema, units, coverage and capture/session fields remain unknown |

The shared Drive folder could not be inspected: connected metadata returned
`NOT_FOUND`/404, the folder query was empty and the web fallback was inaccessible.
This is an access limitation, not proof that the folder is empty or deleted.
No large pool or new source file was downloaded.

## A precise zero/filtering ambiguity remains

The targeted read-only check uses the exact preserved source bytes, whose
SHA-256 values still match the October 24 intake. No full audit or model
experiment was rerun. Its aggregate counts are:

| Supplied list | All rows | Fused score > 0 | Fused score = 0 | Zero with missing components | Zero with component fields | Zero with positive component sum |
|---|---:|---:|---:|---:|---:|---:|
| Training genuine | 23,103 | 19,594 | 3,509 | 3,455 | 54 | 51 |
| Training impostor sample | 78,592 | 73,538 | 5,054 | 0 | 5,054 | 5,017 |
| Testing genuine | 2,100 | 2,094 | 6 | 0 | 6 | 6 |
| Testing impostor | 2,100 | 1,956 | 144 | 0 | 144 | 136 |

`23,103 − 3,455 = 19,648`, matching his reported genuine count and the count of
records with component fields. However, literal `fused_score > 0` filtering of
our preserved files gives 19,594 genuine and 73,538 sampled impostor rows.
This distinguishes missing-component rows from all fused-zero rows. It is
consistent with a different meaning of "non-zero," a different input version,
or counts stated before a later filter; none of those possibilities is proven.

Positive component sums coexist with fused zero in both training and evaluation
exports. The fused value alone therefore cannot determine per-image template
success or a complete/partial failure status. Do not infer that every zero is
an extraction failure, that every component-bearing zero is a valid completed
comparison, or that an impostor zero is biometric rejection evidence. Obtain
the conversion/error log or failed-image mapping and the exact consumed files
or filtering rule. The 3,455 missing-component rows are plausible candidates
for the reported failures, but their individual status has not been verified.

## Implications for the approved milestone

- Keep the original SDK scores and historical metrics unchanged. Future derived
  reports can distinguish known extraction failures from valid comparisons only
  when the corresponding status evidence is available; missing fields are not
  reconstructed and raw zeros are not silently repaired.
- Record reported prior training use and its remaining uncertainty separately
  from supplied-list membership. Exact-pair exclusion does not provide our
  subject-disjoint train/development/test boundary, and reported counts alone
  do not identify a fresh untouched cohort.
- The proposed DINOv2 objective uses identity/capture supervision, not SDK score
  distillation. Konain's recommendation to use nonzero pairs is not an approved
  SDK-based eligibility rule for our arm. Filtering by a comparator's success
  changes the study population and requires protocol review.
- The selected first milestone remains whole four-finger slaps, same-hand
  comparisons, both eligible left/right hands and thumb exclusion. Konain's
  suggestion to mix thumbs is an option for his binary arm, not a change to
  our reviewed scope. A 1:4 binary-list ratio does not replace our implemented
  within-batch cosine supervision/sampling design.
- The professor's request for a progress update and promise of metadata do not
  waive capture/exposure or results/split review. Receive and inspect the
  dataframe before treating any acquisition/PPI fields as verified. Resolution
  could describe pixel dimensions or physical sampling; confirm the schema/units.
- Frozen/adapted matcher and common-list reporting tooling is implemented with
  synthetic checks. Real Precise fine-tuning has not started. The fresh NVIDIA
  check still fails, so GPU repair and full-resolution resource verification
  remain separate requirements.

## Focused next request and professor update

The useful follow-up is now narrower: obtain the promised dataframe, the exact
post-filter training inputs or filtering script, and template-conversion status
records. Confirm whether "non-zero" means a positive fused value or a successful
template conversion, given the component-bearing zeros above. Development/
checkpoint-selection history and independent parent captures remain open.
Do not request the four already supplied comparison lists again.

An accurate progress reply can say that the current work is data validation and
preparation of the subject-disjoint protocol; the DINOv2 whole-slap matcher,
frozen baseline and training/evaluation pipeline have synthetic validation;
real-data fine-tuning has not begun. The forthcoming dataframe will help review
identity, resolution and any acquisition metadata, after which the concrete
protocol/split can be presented for review. No external reply has been sent.
