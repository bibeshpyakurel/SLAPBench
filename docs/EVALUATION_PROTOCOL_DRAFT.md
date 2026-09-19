# Evaluation protocol — draft for supervisor review

Status: NOT APPROVED FOR NEW EXPERIMENTS. Prepared 2026-09-19 from the professor's
16-section instructions supplied by the user. This implements the source; it does
not replace it. The source is preserved verbatim in
[Fingerprint_Foundation_Model_Verification_Student_Guide.md](Fingerprint_Foundation_Model_Verification_Student_Guide.md).
The earlier assertion that the specification was unavailable is superseded.
This pasted source has no Appendix A or C1–C7 table; neither has been invented.

## Confirmed direction and proposed decisions

| Item | Confirmed by supplied instructions | Still needs a concrete decision |
|---|---|---|
| Question | Do explicit minutiae improve VLM verification? §§1,5 | Model, extractor configuration and pilot size |
| Dataset | Lab-authorized data with repeated impressions and finger positions; §4 | Which authorized dataset, acquisition sessions and subject inventory |
| Split | Subject-disjoint development/final test, even without training; §§4,11 | Exact subject lists, duplicate checks and supervisor review |
| Baseline | MINDTCT + Bozorth3; VeriFinger only with a valid license; §§3,10 | NBIS availability/build/version and whether VeriFinger is licensed |
| Comparison | Same pairs, image-only and image+text-minutiae; §§3,10,16 | Hosted provider/model permitted for this task, credentials, cost limit |
| Thresholds | Choose only on development; apply frozen threshold once to test; §11 | Freeze record and achievable operating points |
| Low FAR | 1%, 0.1%, 0.01% only where statistically supportable; §§9,12 | Sufficient independent evidence and uncertainty policy |
| Advancement | Reproducible baseline first; results and split reviewed before fine-tuning; §§3,16.8 | Actual reviewer/date/approval record |

No supervisor contact or sign-off is recorded. The quoted email supplies the
research direction, not authorization of a particular dataset or API budget.

## Proposed first pilot (§§4,8,12,16)

1. Establish a single-finger pilot before claiming a slap-fusion result. Select an
   authorized repeated-impression dataset with reliable finger-position labels.
   Existing Precise/RidgeBase hand images could supply verified per-finger crops,
   but segmentation/position mapping would first need validation. This is a
   proposal, not confirmation those crops or licenses are ready. Keep SD302b
   cross-resolution comparisons a separately labeled near-duplicate control.
2. Assign subjects to development and final test before constructing pairs. Keep
   every finger, sensor, session, derivative and parent capture from one subject
   in the same partition. Check hashes/near-duplicates across partitions. Save
   subject lists locally, reproducible seed and manifest hashes. Current RidgeBase
   Train/Test folders have 63/25 observed subjects with zero overlap for parsed
   Task2 contactless PNGs, but that alone does not establish an untouched final
   test: existing results have already been inspected. Obtain a genuinely held-out
   cohort or disclose the analysis as retrospective/exploratory.
3. Build genuine pairs from independent impressions of the same finger and
   primary impostors from different subjects at matching finger positions. Store
   `pair_id,image_A,image_B,subject_A,subject_B,finger_A,finger_B,label` and split,
   capture/session, sensor and quality metadata where available. Exclude known
   annotation problems with a reason. Do not fabricate missing session labels.
4. Run MINDTCT on appropriate native-resolution grayscale images; preserve pixel
   size and spatial transforms. Save extractor-native minutiae and matching scores.
   Do not infer minutia type/quality when the extractor does not supply it. Confirm
   the extractor's angle convention and convert to radians before sine/cosine.
5. Feed Bozorth3, Prompt A and Prompt B the identical manifest. For VLMs neutralize
   presentation names to A/B, omit subject IDs/ground truth, preserve image order
   in provenance, and freeze preprocessing. Normalize minutiae coordinates in
   the same image/crop coordinate frame shown to the VLM (§5). Any native-resolution
   baseline versus resized-VLM difference must be disclosed (§11).
6. Freeze the exact JSON schema and prompt on development data. Preserve raw API
   output before parsing, plus API model/version/date, generation settings, prompt
   hash, input hashes, software versions, preprocessing and latency/cost. Validate
   JSON field types, finite score range and permitted decisions; abstain on malformed,
   ambiguous, refusal or error outputs. Do not manufacture scores or bypass refusals.
7. Evaluate image-only vs minutiae-grounded vs Bozorth3 on frozen test pairs.
   Report ROC/DET/EER, score distributions, coverage and failure categories, and
   TAR at development-selected thresholds with actual test FAR (§§9–12).
8. Submit the §12 deliverables for review, including the split. Only then consider
   overlays (M3), learned fusion (M4) and per-finger slap fusion (M5).

## Scores, failures and uncertainty (§§9,11)

- Keep primary different-subject impostors separate from same-subject/opposite-hand
  diagnostic pairs. Neither pool silently replaces the other.
- Keep an output record for every requested pair, even if the method fails. Report
  attempted, valid, failed, refused and abstained counts per class/method.
- ROC/EER on available continuous scores are conditional on score coverage. Show
  both available-score and shared-valid-pair comparisons as sensitivity analyses;
  disclose selection bias. Never assign a zero/midpoint score to a missing response.
- For decision rates, show accepted/genuine-attempted and accepted/impostor-attempted
  together with coverage and rates conditional on valid outputs. Abstentions are
  a separate outcome; failure to produce a score does not prove discrimination.
- Select thresholds on development only; report achieved test FAR/TAR and numerator/
  denominator counts. Do not interpolate below observed resolution and label it a
  demonstrated operating point. A threshold is a calibration choice, not evidence
  that the VLM has learned minutiae correspondence.
- Existing primary-impostor sets have 592 (RidgeBase) and 676 (Precise) comparisons:
  one error is 0.169%/0.148%. SD302b's 7,656 comparisons give 0.0131%. For zero
  observed errors and independent Bernoulli trials, the one-sided 95% upper bound
  is `1 - 0.05**(1/N)`; about 29,956 independent zero-error trials are needed for
  that bound to fall below 0.01%. This is not a universal sufficient sample size:
  repeated subjects/images create dependence, and a precision/power plan is required.
- Report uncertainty with a subject-aware design. Shared-subject impostor pairs
  induce dependence on both endpoints; use a documented two-way/subject-level
  resampling method or an independently designed comparison set, not a naive
  bootstrap of pair rows. The final method needs supervisor/statistical review.

## Root-cause tests proposed for development (§§1,10,15)

First remove measurement artifacts identified in the saved-response audit. Then
compare explicit image-only judgments with fixed-schema similarity scores; test
whether input resizing loses useful ridge detail, and validate that the model
receives two distinct images with the expected template/tokenization. Change one
factor at a time and preserve all outputs. For minutiae grounding, compare image-only,
true text minutiae and a labeled control with shuffled minutiae on development
only, to distinguish useful correspondence from an effect of extra prompt text.
The shuffled control is an additional proposal, not a requirement in the source.

The current data establish parser mistakes and frequent literal same-person
responses. They do not establish why the model produces those responses or that
adding minutiae will solve them. H2 remains a hypothesis to test.

## Review record

- Dataset/access approval: pending.
- Split and previously inspected subject exclusions: pending.
- NBIS version/installation and optional VeriFinger access: pending.
- Hosted model, task permission and API budget: pending.
- Sample-size/uncertainty plan and final-test freeze: pending.
- Supervisor approval, date, and evidence: pending.

References informing implementation (the professor's instructions remain primary):
[NIST NBIS](https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis)
describes MINDTCT and BOZORTH3;
[NIST exact binomial limits](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm)
describes exact confidence limits. The zero-error calculation above follows from
the probability of observing zero errors, `(1-p)**N`.
