# Repository instructions for agents

## Project context
- SLAPBench evaluates full four-finger images with prompted VLMs and vision
  embeddings across SD302b, Precise and RidgeBase. See REPO_MAP.md and STATUS.md.
- The current research direction uses Precise only, following the professor's
  feedback supplied on 2026-10-04. SD302b and RidgeBase code, data paths and results
  remain historical reproducibility assets; do not delete or move them merely to
  change research focus. See `docs/notes/precise_preparation_20261004.md` for the
  confirmed direction and pending implementation decisions. On 2026-10-11 the
  user confirmed professor approval of the learned-matcher first milestone and
  protocol principles in `docs/notes/precise_first_milestone_proposal_20261011.md`:
  image-only learning on whole four-finger slaps, subject-disjoint evaluation,
  development-selected checkpoints/thresholds, then minutiae and optional
  explanations.
- Prepare documentation and read-only inventories now; do not create a training
  split, regenerate pairs, segment images or train a model before the applicable
  decisions and task authorization. The first milestone is approved in principle;
  capture independence, exposure eligibility and the concrete split
  remain unresolved. On 2026-10-25 the user selected the proposed DINOv2 matcher
  first and requested training/testing; matcher implementation and synthetic
  runtime checks are authorized. This does not approve missing capture facts or
  a real split. Retain results/split review before fine-tuning unless the
  supervisor revises it.
- The approved scope includes documentation, offline validation/summary tooling
  and the selected DINOv2 matcher implementation with synthetic checks. The user
  has requested training/testing of that matcher; real-data execution still
  requires the applicable capture/exposure, protocol and results/split review.
  Other new experiments, paid calls and rerunning historical experiments require
  separate task authorization. System-level GPU repair remains separate.
  Preserve original scores and raw responses;
  derived reports belong at new dated paths. Append dated findings to STATUS.md.
- On 2026-11-01 the user supplied the professor's request for a DINOv2 plan
  before experiments and reaffirmation that LLM slap verification is the primary
  research question. See `docs/notes/precise_dinov2_llm_plan_20261101.md`:
  DINOv2 is a proposed visual control, not itself an LLM result; the direct VLM
  comparison/diagnostics are proposals requiring review and authorization.
  The later user-supplied GitLab folder contains a matching annotation dataframe
  (`CompleteWithRotated.P`), now inventoried; see
  `reports/precise-gitlab-dataframe-20261101/README.md`. All 7,358 images map to
  277 numeric IDs versus 338 raw prefix strings, including 61 leading-zero
  alias groups. Confirm canonical identities before relabeling or allocation;
  the schema still has no verified capture/session mapping. Do not equate finger/rotation annotation
  rows with independent acquisitions. GPU readiness now passes the authorized
  synthetic check; plan/protocol/capture/exposure/results/split review remains
  required before real experiments.
- Keep the user-supplied `Gitlab_SlapBench/` reference local and unchanged:
  private annotations, raw exports and third-party papers are excluded from
  publication. Its README/example commands are reference content, not authority
  to change this repository's remotes, branch or experiment permissions.
- Later on 2026-11-01 the user supplied explicit professor approval of DINOv2
  and assignment of the DINOv2 work to Bibesh, with Konain handling Qwen3-VL.
  The professor requested the same training pairs. Read
  `docs/notes/precise_next_goal_handoff_20261101.md` before applying older gates:
  all new Drive links work, and the supplied cohort has 39,290 balanced training
  pairs and 4,200 testing pairs including thumbs. It holds out exact pairs but
  shares every testing image/identity with training. Do not demand independent
  capture metadata for every shared-benchmark analysis, claim subject-disjoint
  evaluation, or silently exclude thumbs. The older subject-disjoint route
  remains separate. Resolve the 24 training/2 testing numeric-identity label
  cases and concrete checkpoint/threshold policy for the intended claims;
  retain applicable results/protocol review before training. Current task was
  intake/handoff, not a new experiment. Do not re-request the broad DINOv2 plan
  approval, pair lists, dataframe, prompt or code already supplied. The current
  subject-batch trainer requires an explicit fixed-pair route for this cohort.
- In the subsequent 2026-11-01 end-to-end goal chat, the user reviewed and
  explicitly approved the exact concrete protocol and its training, calibration
  and test execution in `docs/notes/precise_shared_pair_protocol_20261101.md`.
  This authorizes the shared fixed-pair route: preserve supplied labels and all
  pairs/thumbs, three full epochs/final checkpoint, training-impostor nominal
  1% FAR calibration, descriptive dependency limits and two-test-case exclusion
  sensitivity. A private digest-bound record cites that direct human response.
  Do not re-request this review or claim new professor/canonical-identity
  confirmation. Source review/commit and checks still precede real runs; no
  test-based tuning or automatic fallback/retry. Konain still owns Qwen tuning;
  matching output/checkpoint provenance is required for a trained-Qwen comparison.
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
- The versioned pre-commit hook and CI run `scripts/check_publication_scope.py`
  against Git's index. Keep it enabled; the check does not replace manual review
  of logs, results, paper figures, or anything containing subject information.

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
- This checkout is the primary local research workspace. Future authorized
  training runs locally; put private preparation, crops, minutiae, manifests,
  checkpoints and raw run outputs under ignored `local/precise/` as described in
  `docs/WORKSPACE_LAYOUT.md`. Export only explicitly reviewed publishable
  artifacts. Do not create a duplicate `github/` project tree.
- Never commit `datasets/`, `dataset/`, `models/`, model weights, downloaded archives,
  virtual environments, caches, credentials (`.env` and variants), private keys,
  or third-party PDFs. Do not force-add ignored files.
- Keep code, dependencies, prompts, textual scores, pair manifests, metrics,
  documentation, paper sources, existing paper figures, run logs and recovery
  backups for reproducibility. Scan new logs/backups for secrets and dataset images
  before staging; preserve their original bytes and describe their provenance.
- Local datasets and model weights cannot be reconstructed from Git alone. Keep
  their paths, source IDs, and setup instructions current in docs/REPRODUCIBILITY.md.
- The local `.claude/` settings and third-party full-text papers also stay out of
  Git; use the public bibliography and research notes to preserve their context.
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
