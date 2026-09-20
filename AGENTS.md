# Repository instructions for agents

## Project context
- SLAPBench evaluates full four-finger images with prompted VLMs and vision
  embeddings across SD302b, Precise and RidgeBase. See REPO_MAP.md and STATUS.md.
- The supervisor's full instructions are now preserved verbatim in
  `docs/Fingerprint_Foundation_Model_Verification_Student_Guide.md` (all 16 sections).
  Read this source before research changes and cite its numbered sections.
  Papers remain background. `docs/EVALUATION_PROTOCOL_DRAFT.md` proposes implementation
  choices; pending decisions are not supervisor approval. Do not rewrite the source.
- No M1 completion is established: a conventional baseline and controlled single-finger
  pilot are still missing. Follow §16; no fine-tuning before results/split review.
- The approved scope is documentation, offline validation and summary tooling.
  New experiments, paid calls, fine-tuning and rerunning existing experiments
  require a separate task authorization. Preserve original scores and raw responses;
  derived reports belong at new dated paths. Append dated findings to STATUS.md.
- Never send subject identifiers, original filenames or ground-truth labels to
  hosted models. Preserve the branch and push requirements below.

## Git workflow (required)
- `main` is GitHub's default and publication branch. `dev` is the working branch
  for agents and all routine changes, now and in future sessions.
- Keep `dev` current with `git fetch --prune origin` and `git pull --ff-only origin dev`.
- Work on `dev` and push completed, reviewed changes to `origin` branch `dev`.
- Before edits, inspect `git status`, the active branch, and remotes. Preserve
  unrelated user changes. Fetch before synchronizing; never discard local work.
- Do not commit or push directly to `main`. Bring work from `dev` to `main` only
  when the user explicitly asks to publish it, through a pull request after
  checking the diff, conflicts and CI results.
- Never force-push, delete remote branches, publish tags or releases, or change
  repository visibility without explicit user instructions.
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
  or third-party PDFs. Do not force-add ignored files.
- Keep code, dependencies, prompts, textual scores, pair manifests, metrics,
  documentation, paper sources, existing paper figures, run logs and recovery
  backups for reproducibility. Scan new logs/backups for secrets and dataset images
  before staging; preserve their original bytes and describe their provenance.
- Local datasets and model weights cannot be reconstructed from Git alone. Keep
  their paths, source IDs, and setup instructions current in docs/REPRODUCIBILITY.md.
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
- Current experiment code is in `code/` (paper-only analyses in `code/paper/`);
  results are grouped under `results/sd302b/`, `results/ridgebase/` and
  `results/precise/`. The earlier flat public release is preserved at tag `v0.1.0`;
  do not recreate top-level copies of it.
- Paper sources live in `paper/manuscript/` and `paper/eccv2026_submission/`;
  `paper/eccv2026_template/` is third-party. Do not edit third-party classes,
  styles, or templates for routine code changes.
- Keep the root minimal: new scripts go in `code/`, notes in `docs/notes/`,
  derived analyses in dated `reports/` paths.
- Do not run GPU jobs, dataset preparation that moves files, recovery scripts,
  model downloads, or paid API calls unless required by the user's task.

## Checks and reporting
- Read docs/CONTRIBUTING.md and relevant scripts before changing behavior.
- Run `git diff --check`; parse changed Python files for syntax errors.
- For analysis changes run `python code/verify_metrics.py` with dependencies from
  requirements-verify.txt, plus relevant task-specific validation. This verifier
  checks legacy published metrics, not all newer experiments.
- CI also runs `python -m unittest discover -s tests -v`,
  `ruff check code/ tests/ --select F,E9,B` and regenerates the results table.
- Report what changed, checks and any limitations, the commit, and destination branch.
