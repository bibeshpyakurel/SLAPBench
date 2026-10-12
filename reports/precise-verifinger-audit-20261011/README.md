# Precise VeriFinger evaluation-score audit — 2026-10-11

The two Google Drive files supplied by the user contain Konain's reported
2,100 genuine and 2,100 impostor testing comparisons. Each record includes two
image references, a VeriFinger fused score, individual finger scores, their sum
and a finger count. Both scoring methods reproduce the screenshot: 2,057/2,100
genuine accepts (97.9524% TAR) and 0/2,100 impostor accepts at thresholds 48
(SDK fusion) and 79 (sum). This recomputes existing score exports; VeriFinger,
VLM inference and training were not run.

## Operating points and population

Accept when the integer score is greater than or equal to the threshold.
Thresholds below are selected descriptively from the supplied evaluation
impostors. They are not frozen development thresholds or a prospective test.
The false-accept budget is floor(target FAR × impostor count), with tied scores
accepted or rejected together.

| Method, all slap types | Target FAR | Threshold | Genuine accepts | False accepts | Actual FAR |
|---|---:|---:|---:|---:|---:|
| SDK fused | 0.01% | 48 | 2,057/2,100 | 0/2,100 | 0% |
| Per-finger sum | 0.01% | 79 | 2,057/2,100 | 0/2,100 | 0% |
| SDK fused | 0.1% | 39 | 2,057/2,100 | 1/2,100 | 0.047619% |
| Per-finger sum | 0.1% | 72 | 2,057/2,100 | 2/2,100 | 0.095238% |

One false accept changes pooled FAR by 0.047619 percentage points. The zero
observed errors at the screenshot thresholds do not establish a population
FAR of 0.01%. Trial reuse, uncertainty and evaluation-selected thresholds must
be considered before making a generalization claim. The email's 0.1% and the
screenshot's 0.01% are distinct requested operating points; both are reported.

These files mix four-finger slaps and two-thumb slaps:

| Filename-derived slice | Genuine | Impostor | TAR at zero observed false accepts, both methods |
|---|---:|---:|---:|
| All slaps | 2,100 | 2,100 | 2,057/2,100 = 97.9524% |
| Four-finger only | 1,440 | 1,427 | 1,412/1,440 = 98.0556% |
| Right | 749 | 756 | 734/749 = 97.9973% |
| Left | 691 | 671 | 678/691 = 98.1187% |
| Thumbs | 660 | 673 | 645/660 = 97.7273% |

Each slice has its own evaluation-selected threshold, available in
[audit.json](audit.json); the pooled thresholds were not applied to every slice.
The pooled AUC is 0.986768 for fusion and 0.986308 for sum. Both have a
linearly interpolated empirical EER of 2.047619%; interpolation need not
correspond to an attainable integer threshold. JSON also contains slice metrics
and operating points at 1% FAR. [curves.csv](curves.csv) supplies all discrete
ROC/DET rate coordinates and acceptance counts, without identifying records.

## Integrity, mappings and comparison limits

All 4,200 nonblank records parse, contain nonnegative integer scores, and have
correct per-finger sums/counts. Filename-derived subject labels agree with the
supplied genuine/impostor classes. There are no self-image pairs, duplicate
unordered pairs within either class or unordered pair overlaps between classes.
All finger counts agree with the filename-derived slap type.

All referenced image basenames exist in the local Precise folder. Workstation
path prefixes are normalized only in derived comparisons; original source
bytes and paths are preserved. Basename availability does not establish byte
identity with the collaborator's images. Combined coverage is 337 filename
subjects and 4,869 unique image basenames. There are 1,492 images shared between
the genuine and impostor testing classes; this is ordinary testing image reuse,
not evidence about training overlap.

Only 60 genuine pairs and zero impostor pairs overlap the existing primary
676+676 Precise evaluation manifest, after unordered basename normalization.
Diagnostic pairs are excluded from this comparison. The existing VLM results
therefore cannot provide a fair full-population comparison against these SDK
results. A future authorized comparison needs identical evaluation pairs.

The files contain no training pair list, subject partition, pre-export failure
counts, SDK version/settings or complete acquisition metadata. Zero scores are
retained and are not automatically classified as failures. Excluding exact test
pairs from training does not establish image- or subject-disjointness. The
filename seed is not sufficient to verify the sampling population or code.
Independent impressions/session meanings remain unverified.

## Reproduction and private provenance

Raw downloads are stored under ignored
`local/precise/metadata/verifinger-intake-20261011/` as `source_genuine.txt` and
`source_impostor.txt`. The local `source_provenance.json` records canonical Drive
source references, metadata, byte counts and SHA-256 hashes; it contains no
temporary download credentials. Identifying parsed records and source hashes
remain in the local `audit/` directory. No original dataset, scores or manifests
were changed. Raw inputs are not reconstructible from public Git alone and must
be obtained through the collaborator's authorized sharing.

The audit requires only Python's standard library. Choose fresh output paths:

```bash
python3 code/audit_verifinger_scores.py \
  --genuine local/precise/metadata/verifinger-intake-20261011/source_genuine.txt \
  --impostor local/precise/metadata/verifinger-intake-20261011/source_impostor.txt \
  --reference-manifest results/precise/pairs_precise_eval.csv \
  --output-dir reports/precise-verifinger-audit-YYYYMMDD \
  --private-dir local/precise/metadata/verifinger-intake-YYYYMMDD/audit
```

All 40 offline unit tests pass, including eight new synthetic tests covering
schema rejection, tied-score budgets, ROC/AUC/EER, reference normalization,
source preservation, identifying output boundaries and overwrite refusal.
Saved aggregates/private records reproduce exactly. All ten ROC/AUC calculations
agree independently with scikit-learn; all 30 operating-point counts were
independently checked. Source hashes remain unchanged. These checks validate
the supplied export calculations, not the SDK configuration or training history.

## Remaining collaborator questions

> Thanks, Konain. These files reproduce the screenshot result for both fused
> and summed scores. Could you also share the training pair mappings and any
> development/validation lists, your adapted fine-tuning script with an example
> training target, and the SDK version/settings? Were the reported thresholds
> chosen using these testing impostors or a separate development set? Which
> subjects/images were exposed to training, and how were pairs sampled? The
> testing exports include thumbs as well as four-finger slaps, so we will report
> those separately while agreeing the comparison scope with the professor.

This is a draft for the user; no external message was sent. The proposed
[protocol](../../docs/notes/precise_protocol_sketch_20261010.md), architecture
and first experiment remain subject to joint review and task authorization.
