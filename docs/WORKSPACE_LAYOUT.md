# Local workspace and public repository layout

Updated 2026-10-11. This directory is the primary research workspace and the Git
checkout. The public repository receives explicitly reviewed files from this
checkout; a second `github/` copy is unnecessary. `.github/` contains CI and
repository configuration, not a duplicate research project.

## Working layout

```text
datasets/Precise/                 local source images; current study
datasets/sd302b/                  retained local historical data
datasets/ridgebase/               retained local historical data
models/                          local pretrained weights / existing drive symlink
local/                           ignored local workspace
  precise/
    metadata/                    acquisition notes and collaborator artifacts
    splits/                      future private subject partitions
    manifests/                   future local pair lists
    crops/                       future derived fingerprint images
    minutiae/                    future extracted biometric templates
    checkpoints/                 future trained weights and optimizer state
    runs/                        future dated raw outputs, logs and metrics
  legacy-build/manuscript/        retained old LaTeX build log
code/                            publishable source; existing entry points retained
code/paper/                      historical paper analyses
tests/                           offline verification tests
docs/notes/                      dated preparation and research notes
reports/<topic>-YYYYMMDD/         reviewed derived reports and aggregate metrics
results/precise/                  existing recorded Precise results; preserve
results/sd302b/                   existing recorded historical results; preserve
results/ridgebase/                existing recorded historical results; preserve
paper/                           manuscript, submission and retained template
logs/                            existing reviewed historical run logs
```

The local folders were created empty on 2026-10-04. As of 2026-10-11,
`metadata/` contains private inspection records/contact sheets under
`inspection-20261010/`, identifying discovery records under `discovery-20261011/`
and the supplied VeriFinger testing exports, source provenance and identifying
audit records under `verifinger-intake-20261011/`. The
[completed score audit](../reports/precise-verifinger-audit-20261011/README.md)
publishes reviewed aggregates only. Scores are available and audited; training
mappings/exposure, capture independence and SDK settings remain unresolved.
These preparation tasks generated no new subject splits, matching manifests,
segmented crops, minutiae, training checkpoints or matcher runs.

The [central research plan](notes/precise_first_milestone_proposal_20261011.md)
connects the milestone, architecture, protocol and remaining decisions. This
layout defines storage without selecting a model, subject split or thresholds. Existing
scripts still use their documented paths; no runner has been redirected to the
new workspace. All future training runs execute locally when authorized.

Use dated run directories under `local/precise/runs/` and record source data,
preprocessing, software/model revisions, configuration, seeds, manifest hashes,
checkpoint locations and score/error provenance before an authorized run.
Keep checkpoints and raw biometric material local. Export selected, reviewed
aggregate metrics and documentation to a new dated `reports/` directory; add
publishable code and dependencies explicitly. Pair manifests or per-pair scores
containing identities or filenames require content/data-rights review before
publication, even though historical manifests remain tracked.

## Future matcher package

When implementation is authorized, create `code/precise_matching/` as the
package for the new learned matcher. Planned modules are `data.py` for private
data/manifest loading and provenance checks, `encoders.py` for the shared vision
encoder, `losses.py` for verification objectives, and `evaluation.py` for scoring,
development-threshold selection and frozen-system metrics. This is a location
and responsibility plan; no package or model is implemented by this cleanup.
Existing experiment scripts stay at their current paths for historical reproduction.
Code may be published after review; inputs, identifying mappings, future splits
and manifests, minutiae, checkpoints and raw runs stay under ignored local paths.

## Re-create preparation folders on another authorized local machine

Run this from the repository root; it creates directories only:

```bash
mkdir -p local/precise/{metadata,splits,manifests,crops,minutiae,checkpoints,runs}
mkdir -p local/legacy-build/manuscript
```

These local directories, inspection artifacts and any local README are intentionally absent
from Git. The dataset and pretrained weight setup remains in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md); creating folders does not acquire data
or install a matcher/model. Dataset sources, SDK access and research decisions
are linked from the [central research plan](notes/precise_first_milestone_proposal_20261011.md)
and [Precise preparation note](notes/precise_preparation_20261004.md).

## Publication boundary

Git ignores `local/`, datasets, pretrained weights, environments, credentials
and caches. Common root-level training outputs (`checkpoints/`, `runs/`,
`wandb/`, `mlruns/`, `lightning_logs/`, `tensorboard/`) are also ignored and
rejected by the index guard, including small configuration/text files stored
there. This reduces accidental publication if a future tool uses its defaults.
The guard does not inspect file content: continue to review staged code,
reports, metrics and logs for secrets and subject metadata before pushing.

The current index passes the path/size guard and contains no raw dataset/model
directory entries. Existing paper illustrations are retained; this does not
certify the entire public repository or its Git history as free of biometric
information. Existing subject-bearing results remain under the documented
publication policy. No source images, model weights, original result bytes,
manifests or paper sources were moved in this organization task.
