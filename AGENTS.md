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
- The approved shared-pair DINOv2 training, both training-calibration phases and
  both final tests completed on 2026-11-01. Start from
  `reports/precise-dinov2-shared-20261101/README.md` and its professor report;
  do not repeat these runs as unfinished work. The user confirmed Konain's
  matching fine-tuned outputs are not ready. Preserve that pending comparison,
  historical-Qwen provenance limits and all frozen decisions; additional
  training/test runs or tuning require separate authorization.
- On 2026-11-07 a resumed read-only intake found five November 5 Drive inputs:
  37,194 balanced training pairs and 4,200 changed testing pairs, internally
  disjoint by numeric prefix groups (221 train / 55 test), images and pairs.
  See `reports/precise-revised-cohort-20261107/README.md` and
  `docs/notes/precise_revised_cohort_review_20261107.md`. These are new pair
  inputs, not matching trained-Qwen outputs. The user then explicitly confirmed:
  “Yes—use the new lists and run fresh DINOv2 training, calibration and testing.”
  This replaces the active benchmark and approves the exact November 7 proposal:
  fresh pinned base, all new pairs/thumbs, three full epochs/final checkpoint,
  nominal 1% training calibration and numeric-group-holdout claim limits. Record
  that actual approval against new input/configuration hashes; do not ask again.
  All 1,057 new testing images and 55 groups, including 1,918 exact pairs,
  were exposed to the completed DINO adaptation. Do not score that checkpoint
  as unseen-group evaluation or infer biological/capture independence. New
  scope/review checks and source commit precede authorized execution. Original
  results and approved decisions remain preserved.
- The authorized fresh numeric-group-holdout adaptation, both training
  calibrations and both one-time final tests completed on 2026-11-07. Start from
  `reports/precise-dinov2-numeric-holdout-20261107/README.md` and its professor
  report. Do not repeat these completed runs. Frozen/adapted TAR is
  34.76%/34.62%, FAR 0.95%/0.24%, descriptive EER 22.67%/10.95% at the unchanged
  training-selected cutoffs. Score ordering improves without a TAR gain at
  these operating points. Matching new-cohort trained-Qwen outputs/run evidence
  remain unsupplied; the offline comparison path supports both cohorts. No
  biological subject/capture independence, test tuning or further experiment
  authorization follows from completion. Keep prior results and private assets.
- Never send subject identifiers, original filenames or ground-truth labels to
  hosted models. Preserve the branch and push requirements below.
- On 2026-11-08 the user supplied new Qwen summaries, all four full parsed
  prediction files, matched evaluation code, updated training JSON and adapter
  checkpoint/training metadata. Start from
  `reports/precise-qwen-supplied-results-20261108/README.md`. Exact new-cohort
  pair coverage and the 63.31%/89.67% accuracies reproduce; these outputs are no
  longer missing. Preserve their source-verified outcome status and explicit
  run-provenance limits: actual `binary_training_config.json`, resolved base
  revisions/environment, raw replies and checkpoint-selection/evaluation-digest
  evidence remain incomplete. Do not invent them or equate different operating
  policies to matched test FAR. No duplicate Qwen tuning is required.
- The user also relayed the professor's request for a larger DINOv2 model and
  original high-resolution images. The concrete DINOv2-Large/native proposal is
  `docs/notes/precise_large_native_protocol_20261108.md`; the user explicitly
  approved the exact proposal and its training, calibration and test execution
  in the current chat. The direct response and original proposal bytes are bound
  privately to the new configuration/input hashes. Native preparation, gated
  train/calibrate/test entry point, offline verification and synthetic checks are
  implemented. Pinned large weights are already downloaded/verified locally.
  The exact deterministic/BF16/Flash-only native synthetic check passes; see
  `reports/precise-native-large-capacity-20261108/README.md`. Keep originals and
  completed small-model studies intact. Real execution follows reviewed source
  checks/commit/publication. Keep protected runtime source, inputs and review
  fixed across training/calibration/tests. The native entry point records the
  last commit affecting its protected source paths, so unrelated documentation
  commits do not invalidate later phases. Do not ask for the exact approval
  again or infer additional experiments from it.
- The approved Large/native adaptation has started from source commit
  `06d12c249ec7f5db53124f673b2c6947c4b144b7`, after local checks and successful
  hosted CI/CodeQL. Consult the ignored execution `CONTINUATION.json` before
  acting; monitor that existing run rather than starting another. Reporting
  code is independent of the protected runtime paths. The new report exporter
  requires sealed training, both calibrations and both final tests; no native
  results exist merely because report-layout checks pass. Konain's clarification
  about optional `--prompt_json` is distinguished from the separately saved
  training configuration in `docs/notes/precise_qwen_config_clarification_20261108.md`.
- November 10 inspection recovered terminal exit139/SIGSEGV during epoch3;
  the original Large/native run is unfinished, not a completed training job.
  See `reports/precise-native-large-interruption-20261110/README.md` and the
  verified epoch2 checkpoint/evidence. The user then explicitly answered
  “Approve one unchanged-config resume” to the concrete proposal in
  `docs/notes/precise_native_resume_review_20261110.md`. Record that actual
  response against the preserved proposal/input/run/checkpoint digests; do not
  ask again. This authorizes one resume from epoch2, replaying full epoch3 in
  the same run with unchanged protected source/configuration, then the already
  approved two calibrations and two one-time tests. No automatic further retry,
  fallback or new study is authorized. Check the private continuation record
  for whether that second attempt has already started. Final reporting must
  disclose the interruption/replayed partial work and elapsed wall interval
  including downtime. Original approvals, logs, checkpoints and results stay intact.
- The approved resumed native training completed on November 11: final epoch 3,
  all 3,489 updates, final checkpoint sealed and offline integrity passed. Do
  not train/resume again. See `reports/precise-native-training-completed-20261111/README.md`.
  The first frozen-calibration invocation then failed at CUDA preflight before
  model/image access, with no scores, selection or test claim. A reboot changed
  the host to kernel 7.0.0-43, which lacks the NVIDIA compute module; the prior
  working 7.0.0-42 kernel/modules remain installed. Non-hardware runtime bindings
  still match. See `docs/notes/precise_native_cuda_preflight_20261111.md` and
  private continuation metadata. System GPU repair/reboot requires separate
  user/admin action; existing privileges require interactive authentication.
  Do not alter provenance, silently install drivers, use a fallback, retry the
  failed calibration or start tests without applicable actual restoration/
  retry authorization. Preserve the failed preflight and all completed training.
- The user restored CUDA on kernel 7.0.0-43 with the same RTX 4080 SUPER and
  driver595.99.02. Full comparison finds only the kernel patch and driver-version
  discovery metadata differ; protected source, Python/packages/interpreter and
  numerical settings match. The user then explicitly answered “Approve exact
  scoring bridge” to `docs/notes/precise_native_runtime_transition_20261111.md`.
  This approves the separate scoring-only entry `code/score_precise_native_restored.py`
  and exact review gate `code/precise_native_runtime_review.py` for one frozen
  calibration retry and the remaining approved calibration/tests. Record the
  actual response/proposal/checkpoint/input/source/output bindings privately;
  do not ask again or retrain. Preserve original records; record actual training
  and scoring environments and their distinct source identities. The original
  protected19 files remain unchanged; hold the additional bridge source fixed
  across all scoring phases. Reject any other runtime/configuration change,
  fallback, automatic retry or extra test claim. Both calibrations must be
  sealed before either one-time test. Check private continuation before launching
  a phase; do not duplicate a running or completed phase.
- The Large/native benchmark is complete on November 12. Start from
  `reports/precise-dinov2-large-native-20261112/README.md` and its professor report.
  Training, both training calibrations and both one-time final tests completed;
  both native arms cover all 4,200 attempts with zero failures. Frozen/adapted
  accuracy is 73.26%/83.19%, TAR 48.24%/67.48%, FAR 1.71%/1.10%, descriptive
  EER 21.52%/8.67%. Cutoffs remain training-selected; no test tuning occurred.
  Independent NumPy/scikit-learn verification passes and all seven final PDF
  pages were inspected. Training source 06d12c2 and scoring source 12a6df4 remain
  distinct and truthful under the specifically approved runtime bridge. Do not
  repeat or retry any completed phase. Preserve all original assets/attempts,
  numeric-group/capture/dependence limits and source-verified Qwen run-provenance
  qualifications. Completion authorizes no additional study or tuning.

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
