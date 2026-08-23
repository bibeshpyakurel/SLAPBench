# SLAPBench — ECCV 2026 FoundGen-Bio submission (Overleaf-ready)

This folder is the ECCV 2026 (LNCS + `eccv.sty`) version of the manuscript,
ported from the original IEEEtran source.

## Upload to Overleaf
Two options:
1. **Zip import (easiest).** In Overleaf: `New Project → Upload Project` and
   select `eccv2026_submission.zip` (in the parent folder).
2. **Manual.** Create a blank project and upload every file here, keeping the
   `figures/` subfolder.

Set the main document to `main.tex` and compiler to **pdfLaTeX**.

## Before you submit — 2 required edits (marked `TODO` in `main.tex`)
1. **Paper ID.** Replace `*****` in
   `\usepackage[review,year=2026,ID=*****]{eccv}` with your OpenReview
   submission number (you get it after creating the submission).
2. **Camera-ready only** (after acceptance): switch the two `\usepackage`
   lines as noted, and add author ORCIDs.

Nothing else needs changing for the review submission — the `review` option
anonymizes the author block, running heads, and institute automatically, so
the real names kept in the source never appear in the compiled PDF.

## ⚠️ Page limit — action needed
ECCV allows **max 14 pages excluding references**. This faithful port is
currently **~20 pages** of body. It must be trimmed before submission, or it
is rejected without review. The workshop permits **supplementary material**
(extra figures, tables, detailed analyses), so the recommended fix is to move
secondary content there rather than delete it:

- Keep in main: the collapse result, the scoring result, the perfect-score /
  matched-resolution analysis, Table 1, the score-distribution figure, the ROC
  figure, limitations, conclusion.
- Move to supplementary: the qualitative-examples figure, the SLAP-anatomy
  figure, the fairness figure, per-FRGP breakdowns, and the long per-model
  prose.

## Two PDFs to submit
- `main.tex` -> the paper (must be <=14 pages excluding references)
- `supplementary.tex` -> supplementary material (separate PDF on OpenReview).
  Holds Figs. S1--S3 (SLAP anatomy, qualitative examples, fairness by
  subgroup), which the main paper points to. It is a separate file, not an
  appendix, as the CfP requires.

Compile each the same way (pdfLaTeX). On OpenReview, upload the main PDF as
the paper and the supplementary PDF in the supplementary-material slot.

## Files
- `main.tex` — the paper
- `supplementary.tex` — supplementary material (Figs. S1--S3)
- `references.bib` — bibliography
- `figures/` — all figure PDFs
- `eccv.sty`, `eccvabbrv.sty`, `llncs.cls`, `splncs04.bst` — official template
