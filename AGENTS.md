# Repository instructions for agents

## Git workflow (required)
- Work on `dev` and push completed, reviewed changes to `origin` branch `dev`.
- Before edits, inspect `git status`, the active branch, and remotes. Preserve
  unrelated user changes. Fetch before synchronizing; never discard local work.
- If on `main`, switch to existing `dev` (or create it if absent) before editing.
- Never push to `main`, force-push, delete remote branches, publish tags or releases,
  or change repository visibility. A main release requires separate user instructions.
- Remote: https://github.com/bibeshpyakurel/SLAPBench.git
- Enable `git config core.hooksPath .githooks` in each clone. Use explicit pushes:
  `git push origin dev:dev`. Do not bypass hooks.
- Stage explicit paths, review the staged diff and sizes, run relevant checks,
  commit with a Conventional Commit subject, then push. Do not use blind `git add .`.

## Required completion step for every change
- The user has given standing authorization to commit and push completed repository
  changes to `origin/dev`. Do this for every change task, including documentation
  and agent-instruction updates, without asking for push permission again, unless
  the user explicitly asks to keep that task local or uncommitted.
- Before declaring a change task complete, run relevant checks, review and commit
  the task's publishable changes, and run `git push origin dev:dev`. A local commit
  alone does not complete the task. Continue to respect the exclusions below and
  preserve unrelated user work; this does not authorize publishing ignored files.
- Verify publication with `git rev-parse HEAD` and
  `git ls-remote --heads origin dev`. The remote SHA must match the completed local
  commit. If the remote advances concurrently, fetch and reconcile without force
  pushing or discarding work, then verify again.
- If authentication, connectivity, or a repository rule prevents pushing, keep the
  local commit, report the exact blocker and unpushed commit, and state clearly
  that publication is incomplete. Never claim a push succeeded without verification.

## Publication boundaries
- Never commit `datasets/`, `dataset/`, `models/`, model weights, downloaded archives,
  virtual environments, caches, credentials (`.env` and variants), private keys,
  logs, or interrupted-run backups. Do not force-add ignored files.
- Keep code, dependencies, prompts, textual scores, pair manifests, metrics,
  documentation, paper sources and existing paper figures for reproducibility.
- Review files over 25 MiB; do not commit files over 50 MiB. Check for credentials
  without printing their values. Gitignore does not remove already tracked content.
- Dataset images can also occur outside dataset directories. Do not add new raw
  images or fingerprint crops without explicit publication scope and checking
  redistribution rights. Existing manuscript illustrations are retained.

## Research integrity and areas to preserve
- Do not modify datasets, model files, existing result CSVs/JSONs/TXT exports,
  pair manifests, paper claims or figures unless the task specifically requires it.
- Never fabricate scores, silently repair results, regenerate fixed pairs, or
  overwrite original runs. Keep provenance, seeds and sample populations intact.
- Current experiment code is in `code/`; results are grouped under `results/sd302b/`,
  `results/ridgebase/` and `results/precise/`. `github/` and flat result folders
  preserve the earlier release; do not silently synchronize or delete them.
- Paper sources live in `manuscript/` and `eccv2026_submission/`. Do not edit
  third-party classes, styles, or templates for routine code changes.
- Do not run GPU jobs, dataset preparation that moves files, recovery scripts,
  model downloads, or paid API calls unless required by the user's task.

## Checks and reporting
- Read CONTRIBUTING.md and relevant scripts before changing behavior.
- Run `git diff --check`; parse changed Python files for syntax errors.
- For analysis changes run `python code/verify_metrics.py` with dependencies from
  requirements-verify.txt, plus relevant task-specific validation. This verifier
  checks legacy published metrics, not all newer experiments.
- CI also runs `ruff check code/ --select F,E9,B` and regenerates the results table.
- Report what changed, checks and any limitations, the commit, and destination branch.
