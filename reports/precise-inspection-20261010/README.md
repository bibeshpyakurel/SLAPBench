# Precise image inspection — 2026-10-10

This completes the requested local sample inspection and reports aggregate
observations. It is not a matcher experiment, quality benchmark, subject split,
pair-generation step or validation of acquisition independence.

## Coverage and findings

| Check | Observed result |
|---|---|
| Header inspection | All 7,358 top-level JPEG headers opened; zero header failures |
| Pixel dimensions | All 7,358 are 1600×1500 |
| Image mode | All 7,358 are grayscale (`L`) |
| DPI metadata | Missing in all 7,358; actual acquisition PPI remains unknown |
| Recognized EXIF orientation | No recognized tag; this does not prove a standardized capture orientation |
| Full decoding sample | 35 images decoded successfully; zero sample failures |
| Main coverage | Six seeded subjects, first/last sorted file for each hand: 24 slaps |
| Additional coverage | Two thumbs plus all nine three-token filename variants |
| Total sample by filename position | 15 right slaps, 15 left slaps, 5 thumb slaps |
| Source byte preservation | All 35 sampled SHA-256 before/after checks agree |
| Exact canonical grayscale duplicates in sample | Zero; no whole-dataset decoded-duplicate claim |

The 24-image main sample uses seed `20261010` among subjects with both hands and
at least two files per hand; within each subject/hand, first/last means filename
sort order, not chronological acquisition. The extra three-token files cover a
known naming variant. This selection is for inspection, not representative
estimation of dataset-wide quality or a matching pair manifest.

The four-token naming variant has 2,700 distinct first-middle values and 4,615
distinct second-middle values. The three-token variant has four distinct middle
values. Those cardinalities alone do not explain what any token means. The
first/last subject/slap interpretation agrees with the existing parser and
the collaborator's message; session/sensor/capture-token meanings are unverified.

## Visual review

All 35 anonymous thumbnails were reviewed across three local contact sheets;
three images were also reviewed at full pixel dimensions. The slap samples
show four visible fingertip/contact regions, while the thumb samples show two.
Most displayed fingertips point toward the top of the image; finger placement,
separation, tilt and amount of lower finger contact vary. This describes the
reviewed sample and does not establish finger-position labels or a universal
orientation rule.

Visible variation includes faint/broken ridge impressions, darker impressions
with some locally indistinct ridges, white gaps in contact regions, creases and
partial coverage near an image boundary. Many frames contain substantial white
background. Some locally visible ridge lines look clear while other regions
are fragmented. These are qualitative observations, not measurements of
extractor suitability, identity, physical pressure, matcher accuracy or causes
of poor performance. No sample was excluded or given a biometric quality score.

The sample's white-pixel fraction ranges from 0.080152 to 0.951975 and mean
grayscale intensity from 233.214 to 252.350. These describe background/intensity
content, not comparable NFIQ or SDK quality measurements. JPEG noise and white
levels can affect such summaries. Different placement/contrast across files
does not prove independent captures or cross-session acquisition.

## Reproduction and private artifacts

```bash
venv/bin/python code/inspect_precise_images.py --output-dir reports/precise-inspection-20261010 --private-dir local/precise/metadata/inspection-20261010
```

Pillow is required; it is already a project dependency and is included in the
lightweight verification requirements for synthetic-image tests. This run used
Pillow 12.2.0. Both output directories must be new; the CLI requires aggregate
output under `reports/` and private output under ignored `local/`.

Machine-readable aggregates are in [inspection.json](inspection.json).
Original filename mappings, subject labels, per-image source/pixel hashes,
image-level measurements, three contact sheets, full-resolution review copies
and review notes stay in `local/precise/metadata/inspection-20261010/`.
None of those private images/records is staged or published. Source JPEGs are
opened read-only and no preprocessing was written back to them.

All-image header checks do not prove all-image decoding. The sample checks do
not establish dataset-wide decoded/near-duplicate absence, native PPI, capture
independence, session separation, valid minutiae or training/test independence.
The [brief](../../docs/notes/precise_research_brief_20261010.md),
[primer](../../docs/notes/fingerprint_matching_primer_20261010.md) and
[protocol sketch](../../docs/notes/precise_protocol_sketch_20261010.md) explain
how these findings inform the proposed next study.
