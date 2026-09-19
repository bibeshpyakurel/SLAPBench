# Offline audit artifacts — 2026-09-19

Source repository baseline: `ceb43ca`. See ../../STATUS.md for interpretation.
No experiments were rerun. No raw images or individual subject IDs are included here.

`coverage.json` records exact pair-ID membership for each embedding CSV and each
prompted CSV under the Precise/RidgeBase model `latest/` directories. Expected IDs
come from each dataset's `pairs_<dataset>_eval.csv`. Missing categories come from
that manifest; extra IDs are stored IDs outside it; duplicate counts sum occurrences
beyond the first. Missing scores mean empty similarity-score cells; errors mean
raw text beginning `__ERROR__`; invalid binary means stored `llm_answer == INVALID`.
It does not classify semantic refusals or repair IDs. The split counts use
`code/ridgebase_pairs_full.py:collect` on Task2 Train/Test contactless PNGs.

`precise/` and `ridgebase/` are new provisional summaries generated with:

```bash
python code/analyze_ridgebase.py --dir results/precise --output-dir NEW_PRECISE_DIRECTORY
python code/analyze_ridgebase.py --dir results/ridgebase --output-dir NEW_RIDGEBASE_DIRECTORY
```

Use unused output paths. Missing-score exclusion, parser validity and pooled binary
impostor populations limit interpretation; these are not corrected paper results.
Original result and summary files remain preserved in `results/`.
