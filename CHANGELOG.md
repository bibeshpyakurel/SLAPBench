# Changelog

Notable changes to SLAPBench. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Entries from v0.1.0 onward are generated from Conventional Commits by
`scripts/build-release-notes.sh`.

## [Unreleased]

### Changed

- Repository layout consolidated. Paper sources moved to `paper/`
  (`manuscript/`, `eccv2026_submission/`, `eccv2026_template/`); root analysis
  scripts to `code/paper/`; `CONTRIBUTING.md` and `SECURITY.md` to `docs/`;
  planning notes to `docs/notes/`; `results_current/` and `results_matched*/`
  to `results/sd302b/{current,matched,matched_reversed}/`. Scripts write figures
  to `paper/manuscript/figures/`.

### Removed

- Duplicate copies of the v0.1.0 release: `github/` and the flat
  `results/<model>/` + `results/task8_pairs*.csv`. The same SD302b results live
  under `results/sd302b/`; the originals remain at tag `v0.1.0` and in history.
  `code/verify_metrics.py` now reads `results/sd302b/task8_pairs_all.csv`.

## [0.1.0] — 2026-09-15

First tagged release, matching the state of the code behind the arXiv preprint
[arXiv:2607.15517](https://arxiv.org/abs/2607.15517). A citable tag exists so a
reader can fetch exactly the code that produced the published numbers rather
than whatever `main` happens to be.

### Added

- The benchmark: 7,832 exhaustive pairs (176 genuine, 7,656 impostor) built on
  NIST SD302b, across four open-source models and Claude Opus 4.8, under three
  prompting regimes
- `code/verify_metrics.py`, which recomputes AUC, EER and pair counts from the
  committed per-pair CSVs and fails if any published `.metrics.json` disagrees
- `CITATION.cff`

### Fixed

- `generate_results_table.py` found no results. It globbed only
  `results/<model>/latest/` and `results/<model>/previous/*/`, but the results
  have been flat in `results/<model>/` since the June re-upload, so it matched
  0 of 15 metrics files and rendered every model as "running" — the published
  results table was blank. The flat layout is now searched as a fallback.
- Cleared 8 correctness findings in `run_verification.py`: two unused imports,
  two dead locals, an annotation-only name now guarded behind `TYPE_CHECKING`,
  and explicit `strict=False` on three `zip()` calls. All behaviour-preserving.

### Infrastructure

- CI verifies the published metrics reproduce, on Python 3.11 and 3.12
- A correctness-only ruff gate
- `requirements-verify.txt`, so CI installs pandas/numpy/scikit-learn rather
  than the full torch stack

[Unreleased]: https://github.com/bibeshpyakurel/SLAPBench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bibeshpyakurel/SLAPBench/releases/tag/v0.1.0
