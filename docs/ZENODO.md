# Archiving SLAPBench on Zenodo

A Zenodo DOI makes this repository citable as a versioned, archived artifact
rather than as a moving branch, and Zenodo mints a new DOI for every GitHub
release plus one "concept DOI" that always resolves to the newest version.

**These steps cannot be automated from this repository.** Linking Zenodo to
GitHub is an OAuth authorization performed in a browser by the account owner,
and cutting a release is a publication act that repository policy reserves for
an explicit request ([`AGENTS.md`](../AGENTS.md): *never publish tags or
releases without explicit user instructions*). Everything below is therefore
written for **Bibesh Pyakurel** to do by hand, once. The draft release notes in
[section 4](#4-draft-release-notes-for-v020) are ready to paste.

---

## 1. Connect Zenodo to GitHub

1. Go to <https://zenodo.org> and sign in **with GitHub** (top right → *Sign in* →
   *Sign in with GitHub*). Using the same GitHub account is what lets Zenodo see
   the repository.
2. Authorize the Zenodo application when GitHub asks. It requests
   `admin:repo_hook` and `read:org` — it needs the webhook scope to notice new
   releases.
3. Open <https://zenodo.org/account/settings/github/>.

## 2. Enable archiving for this repository

1. On that page, find **`bibeshpyakurel/SLAPBench`** in the repository list. If
   it is not there, click **Sync now** (top right) and wait a few seconds.
2. Flip the toggle next to it to **ON**.

   This installs a release webhook. **It is not retroactive** — the existing
   `v0.1.0` tag will *not* be archived. Only releases published *after* the
   toggle is on get a DOI. If `v0.1.0` must also be archived, delete and
   re-create that GitHub release after enabling the toggle, or accept that the
   Zenodo record starts at the next version.
3. Optional but recommended: click the repository name and use
   **Get the badge** later — the page will show the concept DOI once the first
   release is archived.

## 3. Cut the release

The repository already generates its own release notes from Conventional
Commits ([`.github/workflows/release.yml`](../.github/workflows/release.yml)
runs [`scripts/build-release-notes.sh`](../scripts/build-release-notes.sh) on
any pushed `v*` tag). Pushing the tag is enough; the workflow creates the
GitHub Release, and Zenodo archives it.

Because the layout consolidation since `v0.1.0` was committed as
`refactor!: consolidate repository layout and drop duplicate release copies`,
the next version is **`v0.2.0`**.

```bash
git switch main
git pull --ff-only
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

Then, at <https://github.com/bibeshpyakurel/SLAPBench/releases>, open the
`v0.2.0` release the workflow created and replace or prepend the generated
notes with the draft below.

> **Heads up on the generated notes.** `build-release-notes.sh` only picks up
> subjects that parse as Conventional Commits. Roughly half the commits between
> `v0.1.0` and `main` predate that convention (`Add Pixtral-12B-2409…`,
> `RidgeBase results complete…`, and others), so the auto-generated notes will
> be noticeably thinner than the real change. That is why the draft below is
> written by hand.

## 4. Draft release notes for `v0.2.0`

Paste everything between the rules into the release body.

---

SLAPBench `v0.2.0` — the v2 reorganization and the independent-capture check.

`v0.1.0` was the state of the code behind the preprint
[arXiv:2607.15517](https://arxiv.org/abs/2607.15517). This release keeps those
SD302b results byte-identical and adds the evidence that qualifies them.

**Reproducibility**

- [`REPRODUCE.md`](https://github.com/bibeshpyakurel/SLAPBench/blob/main/REPRODUCE.md)
  maps every paper table and headline number to the file in `results/` that
  holds it and the command that produces it, records the sampling settings,
  seeds and model identifiers of the published run, and states plainly what
  cannot be reproduced from a clone.
- The metric check runs with no GPU, no dataset and no API key:
  `pip install -r requirements-verify.txt && python code/verify_metrics.py`.
  It recomputes AUC, EER and the pair counts from the committed per-pair CSVs
  and fails if any published `.metrics.json` disagrees.
- `CITATION.cff` now carries the preprint DOI `10.48550/arXiv.2607.15517`.

**The v2 extension**

- **RidgeBase** (University at Buffalo CUBS, contactless four-finger): genuine
  pairs are two independent photographs, not two resolutions of one capture.
  Prompted verification and prompt-free embedding matching, with a
  same-device / cross-device split and a diagnostic same-subject-different-hand
  category.
- **Precise**: a four-finger SLAP *livescan* collection with multiple
  independent impressions per position, bridging the gap between SD302b and
  RidgeBase.
- The capture-divergence gradient (SD302b → Precise → RidgeBase) and its
  figure.
- Pixtral-12B-2409 added as a fifth open-weight model on the v2 datasets.
- **Finding:** on independent captures no open-weight model separates genuine
  from impostor pairs above chance. Qwen3-VL-8B falls from AUC 1.000 on SD302b
  to 0.469 (embedding) and 0.558 (prompted) on RidgeBase. The SD302b
  similarity-scoring columns index near-duplicate sensitivity, not verification
  across independent impressions; the manuscript has been reframed accordingly.

**Repository**

- Layout consolidated: results are grouped by dataset
  (`results/{sd302b,ridgebase,precise}/`), paper sources under `paper/`,
  paper-only analyses under `code/paper/`, contributor and security docs under
  `docs/`. The flat `results/<model>/` layout of `v0.1.0` is preserved at that
  tag and its SD302b files are byte-identical to those now under
  `results/sd302b/`.
- `scripts/check_publication_scope.py` enforces the local-only boundary
  (datasets, weights, environments, credentials, files over 50 MiB) in CI and
  in the pre-commit hook.
- A dated response audit (`reports/response-audit-20260919/`) validates the
  exports and explains the parsing failures.

**Not included, by design:** NIST SD302b, RidgeBase and Precise images, model
weights, and credentials. See
[`docs/REPRODUCIBILITY.md`](https://github.com/bibeshpyakurel/SLAPBench/blob/main/docs/REPRODUCIBILITY.md).

---

## 5. After the release — wire the DOI back in

Within a minute or two the Zenodo record appears at
<https://zenodo.org/account/settings/github/> next to the repository, and the
badge markdown is on the record page. Zenodo gives you **two** DOIs:

- a **concept DOI** (`10.5281/zenodo.<CONCEPT>`) — always resolves to the latest
  version. **Use this one** in the README badge and in `CITATION.cff`.
- a **version DOI** (`10.5281/zenodo.<VERSION>`) — pins `v0.2.0` specifically.
  Use it when a paper must cite one exact snapshot.

### 5a. README badge

Add it to the badge block at the top of `README.md`, after the arXiv badge:

```markdown
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.CONCEPT.svg)](https://doi.org/10.5281/zenodo.CONCEPT)
```

### 5b. `CITATION.cff`

Add a top-level `doi:` — this is the field that identifies the archived
*software*, as distinct from `preferred-citation.doi`, which is the preprint.
The placeholder comment is already in the file:

```yaml
repository-code: "https://github.com/bibeshpyakurel/SLAPBench"
url: "https://github.com/bibeshpyakurel/SLAPBench"
type: software
doi: "10.5281/zenodo.CONCEPT"          # ← add this (concept DOI)
version: "0.2.0"                       # ← and these two
date-released: "YYYY-MM-DD"
license: MIT
```

Validate before committing:

```bash
pipx run cffconvert --validate -i CITATION.cff
```

### 5c. `CHANGELOG.md`

Promote the `[Unreleased]` section to `## [0.2.0] — YYYY-MM-DD`, add a new
empty `[Unreleased]`, and update the link definitions at the bottom.

### 5d. Commit

```bash
git switch -c docs/zenodo-doi
git add README.md CITATION.cff CHANGELOG.md
git commit -m "docs: record the Zenodo DOI for v0.2.0"
gh pr create --base main --title "docs: record the Zenodo DOI for v0.2.0"
```

Conventional Commits are enforced on pull requests to `main`
([`.github/workflows/commits.yml`](../.github/workflows/commits.yml)), and
`main` is protected — changes go in through a pull request, never a direct
push.

## Checklist

- [ ] Signed in to Zenodo with GitHub and authorized the app
- [ ] `bibeshpyakurel/SLAPBench` toggled **ON** at <https://zenodo.org/account/settings/github/>
- [ ] Decided whether to re-cut `v0.1.0` so it is archived too (optional)
- [ ] Tagged and pushed `v0.2.0`
- [ ] Release notes pasted
- [ ] Zenodo record appeared; concept and version DOIs noted
- [ ] Zenodo record metadata checked (authors, ORCIDs, licence, title)
- [ ] Concept DOI badge added to `README.md`
- [ ] `doi:`, `version:` and `date-released:` added to `CITATION.cff`, validated
- [ ] `CHANGELOG.md` `[Unreleased]` promoted to `[0.2.0]`
