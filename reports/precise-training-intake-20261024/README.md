# Precise collaborator training-list intake — 2026-10-24

The user supplied two local files identified by Konain as training pairs and
reported that he is fine-tuning Qwen3 for binary matching. The files are now
preserved byte-for-byte with the previously supplied testing inputs under ignored
`local/precise/metadata/konain-intake-20261024/`. This is an offline intake audit,
not a training run, a new split, or verification of which rows his executable
training pipeline actually consumed.

## Received material

| Role | Private local file | Records | Right / left / thumbs |
|---|---|---:|---:|
| Training genuine | `genuine_filtered_minus2100.txt` | 23,103 | 7,644 / 7,727 / 7,732 |
| Training impostor | `imposter_filtered_minus2100_78592.txt` | 78,592 | 78,592 / 0 / 0 |
| Testing genuine | `testing_genuine.txt` | 2,100 | 749 / 691 / 660 |
| Testing impostor | `testing_impostor.txt` | 2,100 | 756 / 671 / 673 |

The training files contain 101,695 comparisons altogether. The new Drive link
identifies the same impostor testing file received on 2026-10-11. A fresh
authenticated download is byte-identical to that original (372,832 bytes).
The genuine testing input was copied from the preserved earlier intake.
Original Downloads files and prior intake files remain unchanged. Private
`source_provenance.json` records paths, canonical source links, byte sizes and
SHA-256 values; temporary download credentials were not retained.

## What separation actually holds

Pair identity is the unordered pair of exact image basenames after removing
Windows/POSIX path prefixes. Reversing the image order cannot hide a duplicate.

| Scope | Training records | Testing records | Shared exact pairs | Shared image basenames | Shared filename subjects |
|---|---:|---:|---:|---:|---:|
| All slap types | 101,695 | 4,200 | **0** | **4,869 of 4,869 testing images** | **337 of 337 testing subjects** |
| Four-finger only | 93,963 | 2,867 | **0** | **3,283 of 3,283 testing images** | **330 of 330 testing subjects** |

Konain's statement that the training and testing pairs differ is supported at
the exact-pair level. Image and subject separation do not hold for these lists.
The training genuine list references all 7,358 local image basenames and all
338 filename subjects. This establishes membership in supplied lists, not proof
of actual training execution or byte identity with his source images.

These lists therefore describe a different protocol from our approved
subject-disjoint learned-matcher milestone. Preserve them as collaborator
evidence and for separately scoped reproduction; do not relabel them as our
approved train/development/test split. A fresh model can still be trained on a
reviewed subject partition of Precise, with prior evaluation exposure and claim
scope explicitly documented. No new partitions or filtered training files were
created by this audit.

## Format and population findings

- All 105,895 training/testing records parse under the two observed formats.
  Every referenced basename exists locally; filename-derived class and hand
  checks pass. No self-image pairs, repeated unordered pairs within a source,
  or genuine/impostor pair overlaps within training or testing were found.
- **3,455 genuine training rows contain only two image references and fused
  score zero.** Their per-finger scores, sum and finger count are unavailable.
  The other 19,648 genuine rows and all impostor/testing rows have complete
  component fields with consistent sums/counts. Missing components are not
  imputed; zero is not automatically treated as failure or a nonmatch label.
- The supplied training impostors are **right-hand only**. Genuine training
  includes both hands and thumbs. Using the lists unchanged for binary training
  could associate hand/slap type with the class; whether this is deliberate or
  whether additional negative files exist remains a collaborator question.
- No separate development assignment, binary answer construction, capture
  mapping, SDK configuration or trained checkpoint was supplied in these inputs.
  File class, fused score and the eventual binary training target are distinct
  concepts; the target-construction rule remains to be documented.

The machine-readable aggregates are in [audit.json](audit.json). Source hashes
and overlap identities stay in the private intake's `audit/audit_details.json`.
No identifying record or original image filename is exported in this report.

## FPBench reference

The full public FPBench repository is preserved locally at
`local/precise/references/FPBench/`, detached at
`ed27ae669947bfd157522d537a9f9c77b4bf3189`. It contains code, task definitions,
documentation and upstream demo assets. No dependencies, external checkpoints,
linked datasets or model services were installed or executed.

The [reference guide](../../docs/notes/fpbench_reference_20261024.md) explains
the useful modules, restoration command, upstream limitations and differences
from both Konain's reported binary adaptation and our planned embedding matcher.
The reference does not independently supply his Precise-specific changes.

## Reproduce this audit

From the project root, with the four private source files restored:

```bash
python3 code/audit_precise_training_pairs.py \
  --training-genuine local/precise/metadata/konain-intake-20261024/genuine_filtered_minus2100.txt \
  --training-impostor local/precise/metadata/konain-intake-20261024/imposter_filtered_minus2100_78592.txt \
  --testing-genuine local/precise/metadata/konain-intake-20261024/testing_genuine.txt \
  --testing-impostor local/precise/metadata/konain-intake-20261024/testing_impostor.txt \
  --output-dir reports/precise-training-intake-YYYYMMDD \
  --private-dir local/precise/metadata/konain-intake-20261024/audit-YYYYMMDD
```

Both output directories must be new. The command uses only the Python standard
library, reads filenames without decoding images, checks source-byte stability,
and creates aggregate public output plus private overlap details. Exit 0 means
the audit completed; it does not certify training readiness or subject separation.

Eight synthetic tests cover incomplete component records, reversed/path-remapped
pairs, image/subject overlap despite pair exclusion, invalid labels/scores,
source preservation/change detection, privacy and output boundaries. Publication
validation and the final test results are recorded in STATUS.md.
