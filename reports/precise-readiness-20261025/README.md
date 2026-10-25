# Precise acquisition and compute readiness — 2026-10-25

**All 7,358 local JPEGs decode strictly; capture/exposure readiness is still
unresolved.** This is an offline file/provenance audit, not a matcher experiment
or independent-acquisition certificate. The user supplied the ten-task milestone
roadmap and confirmed that no new clarification from Konain is available.
The [review proposal](../../docs/notes/precise_milestone_review_20261025.md)
makes protocol/model choices concrete without approving or executing them.

## Full-image results

| Check | Observed aggregate |
|---|---|
| Population | 7,358 JPEGs, 338 filename subjects; 4,907 four-finger slaps |
| Strict complete decode | 7,358 successes, zero failures; truncated-image loading disabled |
| Dimensions/mode | All 1600×1500, Pillow `L` grayscale |
| Constant grayscale images | Zero; this is not a biometric quality measure |
| Exact source-byte duplicate groups | Zero |
| Native-mode decoded-pixel duplicate groups | Zero |
| Decoded-grayscale duplicate groups | Zero |
| Source-byte stability | All 7,358 image streams and six metadata inputs rehashed unchanged |
| Uninspected subdirectories/other top-level files | Zero / zero |
| Parent captures, independent acquisitions and reviewed exposure | Unknown for all images |

[readiness.json](readiness.json) contains the generated aggregate results.
The native hash includes mode, dimensions and decoded bytes. The grayscale
hash uses the earlier inspection convention:
`SHA256(str(gray.size).encode() + gray.tobytes())`, Pillow conversion to `L`,
without EXIF transformation. Pillow version was 12.3.0. Strict decode success
does not validate physical PPI, capture independence, anatomical labels or
biometric image quality. No image was removed or altered.

## Heuristic derivative screening and targeted review

A deterministic global screen computes horizontal 64-bit dHash after 9×8
LANCZOS resize, retrieves all relationships within Hamming distance two, and
retains nonidentical images whose 32×32 LANCZOS grayscale thumbnail MAE is at
most five grayscale levels. Three hash blocks guarantee retrieval for that
specific radius; they do not guarantee detection of transformed derivatives.
There are 4,549 relationships within the hash radius and **2,685 final
screening candidates**, including 2,497 spanning filename subject labels.
No candidate spans hand-position labels.

| Filename-derived position | All candidates | Cross-subject-label candidates |
|---|---:|---:|
| Right four-finger | 30 | 10 |
| Left four-finger | 38 | 20 |
| Two thumbs | 2,617 | 2,467 |

The concentration among thumbs illustrates the screen's sensitivity to white
canvas and coarse layout. Sixteen targeted relationships were viewed privately:
the four lowest thumbnail-MAE same-subject and four lowest cross-subject
candidates separately for thumbs and four-finger slaps. Visible placement,
contrast and impression differences occur despite the low coarse distances.
These views neither adjudicate identity nor prove acquisition independence.
This targeted sample is not representative and the remaining candidates have
not been individually adjudicated. **2,685 candidates does not mean 2,685
duplicates.** No automatic exclusions or corrected labels were created.
The screen can miss crops, shifts, rotations and other processing variants.

[candidate_review_summary.json](candidate_review_summary.json) records reviewed
aggregates. Original filenames, hashes, candidate relationships, full-resolution
difference statistics and aliased contact sheets stay under ignored
`local/precise/metadata/readiness-20261025/`. Manual source/capture review remains
necessary before split creation; neither this heuristic nor distinct pixels
replaces Konain's acquisition facts.

## Private acquisition and exposure record

The private `readiness_records.json` has one record per image: original name and
tokens, filename-derived subject/position, file/native/grayscale hashes, decode
status, supplied-list and subject-list membership, and historical-manifest
membership. Parent capture, session, verified acquisition, actual training use,
actual development use and exposure-review assertions remain unknown rather
than guessed from filename middle tokens. These are preflight records, not
the validator's approved partitioned image CSV.

The genuine training source references every local image and all 338 subjects;
the right-only impostor training source references 2,299 local images and 332
subjects. The two historical Precise manifests together reference all 4,907
four-finger images and all 338 subjects. Manifest/list membership is evidence
of a proposed or recorded comparison population, not proof of every model's
actual consumption. Original model pretraining exposure is also unknown.
No sizeable untouched cohort can currently be certified.

The [preceding intake audit](../precise-training-intake-20261024/README.md)
already established no exact training/testing pair overlap, complete testing
image/subject overlap, right-only training impostors and 3,455 genuine rows
with zero fused score but no component fields. The files do not explain the
missing fields or right-only policy. They remain explicit collaborator questions;
zero is not recoded as a failure/negative and missing components are not imputed.
No new collaborator facts were supplied in this task.

## Compute and next gates

The [environment report](environment.md) records unavailable CUDA/NVIDIA driver
access and unavailable external model storage. No system repair, weight download,
real model instantiation or forward/backward was performed. The
[review proposal](../../docs/notes/precise_milestone_review_20261025.md)
includes concrete candidate subject counts/seed, comparison sampling, 0.1% FAR,
development selection, failure/uncertainty rules, a pinned DINOv2-small matcher,
verification loss, preprocessing and resource estimates. All are proposed
settings requiring review, not implemented experiments.

Proceed to split creation only after acquisition/exposure resolution and
protocol approval. Retain results/split review before fine-tuning. Loader,
encoder/loss/training implementation and synthetic/real model execution remain
behind their applicable configuration and task-authorization gates. This report
does not complete all ten roadmap tasks or provide new performance evidence.

## Reproduction

Use the existing local verification environment (Pillow and the standard
library suffice for this audit). Restore the authorized source images and four
private collaborator lists using [reproduction notes](../../docs/REPRODUCIBILITY.md).
Choose **new** output directories; the CLI refuses overwrites and public/private
output paths outside `reports/` and `local/`, respectively.

```bash
.venv/bin/python code/audit_precise_readiness.py \
  --training-genuine local/precise/metadata/konain-intake-20261024/genuine_filtered_minus2100.txt \
  --training-impostor local/precise/metadata/konain-intake-20261024/imposter_filtered_minus2100_78592.txt \
  --testing-genuine local/precise/metadata/konain-intake-20261024/testing_genuine.txt \
  --testing-impostor local/precise/metadata/konain-intake-20261024/testing_impostor.txt \
  --historical-manifest results/precise/pairs_precise_eval.csv \
  --historical-manifest results/precise/pairs_precise_full.csv \
  --output-dir reports/precise-readiness-YYYYMMDD \
  --private-dir local/precise/metadata/readiness-YYYYMMDD
```

This command recreates `readiness.json` and private `readiness_records.json`.
The targeted visual-review summary and narrative are separately reviewed outputs,
not automatic adjudications by the CLI. Private review receipts document their
selection. Original dataset/list/manifest bytes remain untouched. Public output
contains aggregates only; all identifying and image artifacts stay ignored.
Exit zero indicates completed auditing, not readiness approval.

Validation: 82 offline tests pass, including seven new synthetic readiness tests
for complete decoding/errors, duplicate conventions, exhaustive hash-radius
retrieval, source change detection, private unknowns and output boundaries.
All 15 legacy published metrics reproduce; correctness lint passes. Final
syntax/link/whitespace and publication review is recorded in STATUS.md.
