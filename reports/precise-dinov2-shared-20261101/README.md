# Precise DINOv2 shared-pair results

Completed 2026-11-01 under the user's direct review of the
[concrete protocol](../../docs/notes/precise_shared_pair_protocol_20261101.md).
The frozen and adapted DINOv2 runs, training-only calibration, both final tests,
offline checkpoint check and aggregate reproduction are complete. Konain's
matching fine-tuned Qwen3-VL outputs are **not ready**, as the user confirmed;
that comparison remains pending and no local Qwen fine-tuning was duplicated.

The [professor report (PDF)](professor-report.pdf) and
[editable report (Markdown)](professor-report.md) present the findings,
scope and remaining collaborator evidence. The report has four pages, all
rendered and visually checked. It has not been sent to the professor.

## Main observed results

Both arms scored every supplied testing attempt: 2,100 genuine and 2,100 impostor,
including left, right and two thumbs, with **zero DINO attempt failures**.
Cutoffs were fixed from training impostors before test execution; accept means
strictly greater than the cutoff. The nominal calibration target is 1% training
FAR, not a promised test FAR or independent development estimate.

| DINOv2 arm | Cutoff | Genuine accepts | False accepts | Test TAR | Test FAR | Balanced accuracy | Descriptive AUC | Descriptive EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen | 0.8937185406684875 | 876 / 2,100 | 17 / 2,100 | 41.71% | 0.81% | 70.45% | 0.8776 | 19.76% |
| Adapted | 0.9320146441459656 | 1,182 / 2,100 | 23 / 2,100 | 56.29% | 1.10% | 77.60% | 0.9800 | 6.52% |

Adaptation increases observed TAR by 14.57 percentage points and FAR by 0.29
percentage points. It still rejects 918 of 2,100 genuine attempts at the fixed
cutoff. Improved AUC does not establish high acceptance at a strict operating
point, statistical superiority or unseen-person generalization. No test curve
or result was used to select a different checkpoint, cutoff or configuration.

The adapted arm's left/right/thumb TAR is respectively 59.91% / 60.61% / 47.58%,
using the same pooled adapted cutoff. Full per-type counts and rates,
paired decisions, coverage and label sensitivity are in
[aggregate.json](aggregate.json) and [dino_metrics.csv](dino_metrics.csv).
The [pooled ROC figure](pooled_roc.svg) is descriptive only.

## Exact population, labels and uncertainty

- All 39,290 supplied training pairs were used exactly once per epoch for three
  epochs (117,870 pair presentations, 3,684 optimizer updates). No pairs were
  regenerated, extra within-batch cross-pairs created, thumbs excluded or SDK
  zeros dropped. The source JSON's class balance is preserved.
- Exact train/test pair overlap is zero. All 4,869 testing images and 276 numeric
  groups also occur in training. Testing reuses 2,404 images in multiple pairs,
  with up to seven testing pairs per image. This is a held-out-pair benchmark
  with seen images/identities, not subject-disjoint testing.
- All 24 training and two testing numeric-identity label disagreements remain
  labeled as supplied. Canonical identity semantics are unconfirmed. Both flagged
  testing pairs are rejected by both DINO arms. Excluding only these two rows in
  the secondary sensitivity leaves 2,098 impostor attempts and unchanged false
  accept counts; it does not relabel identity truth or retrain the model.
- Training calibration is optimistic resubstitution. Each arm accepted 196 of
  19,645 training impostor pairs; no independent development cohort was removed
  from Konain's supplied training membership.
- Report exact finite-cohort rates and dependency counts. No independent-pair
  binomial intervals, significance tests or population low-FAR bounds are claimed.
  One pooled false accept is 1/2,100, about 0.048 percentage points. This is
  empirical resolution, not a population guarantee. Training-seed variability
  is unestimated from this single approved run.

## Available comparison evidence

All saved comparator rows were joined to the **exact same 4,200 unordered
testing pairs and supplied labels**, without an intersection-only population.

| Supplied continuous-score export | Parsed scores / attempts | Descriptive AUC | Interpolated EER |
| --- | --- | --- | --- |
| VeriFinger fused | 4,200 / 4,200 | 0.9868 | 2.05% |
| VeriFinger per-finger sum | 4,200 / 4,200 | 0.9863 | 2.05% |
| Historical Qwen score export | 4,200 / 4,200 | 0.7612 | 34.03% |

The SDK exports retain all zeros: fused zero occurs in six genuine and 144
impostor rows; sum zero occurs in eight impostor rows. Parsed-score coverage does
not establish successful template extraction. Extraction status remains unknown;
no new licensed SDK run or zero-score repair occurred. Adapted DINO's descriptive
EER remains above these supplied SDK EERs.

Historical Qwen scores use a four-finger 0-100 confidence prompt even though
thumbs occur in testing. The accompanying script uses native PIL images plus
the model's processor and does not load an adapter; actual model revision,
adapter/run provenance and raw generated replies are not established by the
parsed exports. These are historical descriptive context, **not the result of
Konain's new A/B fine-tuning setup**. No verified non-test cutoff was available
for either SDK or historical Qwen arm, so no test-selected operating-point
comparison is reported. DINO-versus-Qwen differences cannot isolate a language
module defect or answer the project's primary LLM question by themselves.

When available, Konain's actual matching A/B outputs require exact pair coverage,
consumed training membership, run/environment configuration, base/adapter and
checkpoint identity, selection rule, prompt/processor, raw replies, parsing and
failure provenance. A/B decisions supply one operating point, not continuous ROC.
The supplied code, prompts and input lists do not need to be requested again.

## Execution provenance and checks

Actual execution source: `82d02111de2d13378c4be610cbc733923b0637de` on `dev`.
Model: `facebook/dinov2-small`, revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`.
Final adapted checkpoint SHA-256:
`6736864ee8211a74eed8e847499f239d53b98084063618b3c7e35432b8f0095c`.
The remaining exact configuration/model/runtime fields are in the aggregate.

Offline verification confirms all 185 frozen tensors (18,505,344 parameters)
unchanged and all 38 selected trainable tensors (3,551,232 parameters) changed.
Mean pair losses were 0.1494806, 0.1064979 and 0.0947178; these are training
losses, not accuracy. Checkpoints and all private scores/approval/source receipts
remain under ignored `local/precise/`; original audited input hashes are unchanged.

An idle waiting launcher was terminated before it started any calibration/test
phase. The remaining authorized phases were then launched directly. An initial
offline verifier compared legacy safetensors key names with Transformers 5.18's
canonical state keys; the verifier was corrected to use the identical pinned
loader as training. It performed no inference or parameter update. No training,
calibration or test experiment was repeated, and no scientific policy changed.
Private operational records preserve these events.

The report tool replays the frozen cutoff from private training outcomes and
reproduces saved test aggregates and one-time test claims. A separate NumPy /
scikit-learn calculation reproduces DINO confusion counts, AUC and interpolated
EER to numerical precision. All 170 repository tests, 74 matcher CPU contracts,
15 legacy metric checks, syntax, correctness lint and whitespace checks pass.
Public files are code, synthetic tests and reviewed aggregates/documents only;
no source images, identifiers, original filenames, weights or raw scores are
published in this report directory.

## Reproduce the report from authorized private outputs

This does not run a matcher or tune a cutoff. Use **new** report/receipt paths
for any repeated aggregate export; existing outputs are refused.

```bash
.venv/bin/python code/report_precise_shared_pairs.py \
  --bundle local/precise/manifests/shared-pairs-20261101/bundle.json \
  --config code/precise_matching/configs/dinov2_shared_pairs.json \
  --frozen-test local/precise/runs/shared-pairs-20261101/frozen-test \
  --frozen-calibration local/precise/runs/shared-pairs-20261101/frozen-calibration \
  --adapted-test local/precise/runs/shared-pairs-20261101/adapted-test \
  --adapted-calibration local/precise/runs/shared-pairs-20261101/adapted-calibration \
  --checkpoint-verification local/precise/runs/shared-pairs-20261101/checkpoint-verification.json \
  --intake local/precise/collaborator/konain-followup-20261101 \
  --private-receipt local/precise/metadata/shared-pair-execution-NEWDATE/report-receipt.json \
  --output reports/precise-dinov2-shared-NEWDATE/aggregate.json

# Use the bundled workspace Python with ReportLab 4.4.9; no model packages needed.
python code/render_precise_shared_report.py \
  --aggregate reports/precise-dinov2-shared-NEWDATE/aggregate.json \
  --output-dir reports/precise-dinov2-shared-NEWDATE
```

The reporting export renames the shared legacy cutoff helper's private
`development_*` counter keys to `training_*`, matching their actual population;
cutoffs, counts and original private records are unchanged. Its explicit
`selected_on` value remains `training_impostors_resubstitution`.
Rendering uses ReportLab 4.4.9 with DejaVu Sans when available (Helvetica fallback)
and Poppler for visual QA. The aggregate SHA-256 embedded in the PDF binds its
source JSON. See [reproducibility instructions](../../docs/REPRODUCIBILITY.md)
and the [protocol](../../docs/notes/precise_shared_pair_protocol_20261101.md)
for actual model execution; no additional experiment is implied by these commands.
