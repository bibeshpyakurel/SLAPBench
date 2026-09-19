# Contributing

## Branch and publication policy

All routine changes are committed and pushed to `dev`. Never push directly to
`main`, including through an admin bypass. Publication to `main` requires a
separately authorized pull request and the repository's required checks.
Do not force-push, delete remote branches, or publish tags as part of routine work.

For a new clone:

```bash
git switch dev
git config core.hooksPath .githooks
git config push.default simple
git config remote.origin.push refs/heads/dev:refs/heads/dev
```

The versioned pre-commit and pre-push hooks reject commits outside `dev` and
pushes to any destination other than `refs/heads/dev`. Hooks are local safeguards:
new clones must enable them, and GitHub branch protection is separate.

Before committing, inspect `git status --short`, stage explicit paths, and review
`git diff --cached --stat` and `git diff --cached`. Keep datasets, model weights,
secrets, caches, logs and recovery backups local. Keep reproducibility code,
small textual results, manifests, documentation and paper sources in Git.
Review any file above 25 MiB before staging; never add files above 50 MiB.
Existing paper illustrations are retained; new dataset images need an explicit
publication request and a redistribution-rights review.

```bash
python code/verify_metrics.py
python code/generate_results_table.py
ruff check code/ --select F,E9,B
git diff --check
git commit -m "docs: describe the change"
git push origin dev:dev
```

Install verification dependencies from `requirements-verify.txt` to run the
metric check. This check covers the legacy published `.metrics.json` files;
it does not validate every newer Precise or RidgeBase result. Do not rerun GPU
inference or paid API experiments merely to check a documentation change.

## Commit messages

This repository uses [Conventional Commits](https://www.conventionalcommits.org/).
The format is machine-read to generate the changelog, so it is enforced rather
than encouraged:

```
<type>(<optional scope>): <summary in the imperative>
```

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

Append `!` after the type (or add a `BREAKING CHANGE:` footer) for a change
that breaks an existing contract.

Good:

```
feat(log): accept workouts pasted as free text
fix(auth): stop refreshing an expired session on the server
ci: run the Playwright suite on pull requests
```

Not good: `Update`, `changes`, `deployable`, `fixed stuff`.

## Tests

A bug fix comes with a test that fails before the fix and passes after it. A
feature comes with tests for its contract, not its implementation details.

Tests that exist but never run in CI are worse than no tests, because they
imply a guarantee nothing enforces. If you add a suite, add it to the pipeline
in the same pull request.

## Dependencies

Dependabot proposes updates. Review them like any other change — a version bump
that fails CI is a finding, not an inconvenience.
