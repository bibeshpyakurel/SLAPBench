# Precise research onboarding and preprocessing plan

Updated 2026-11-07. Start by understanding and reproducing the saved Precise
results. Then develop a small, controlled preprocessing study for review.
The completed DINOv2 runs provide the starting evidence; this plan does not
restart them or authorize additional training or final-test experiments.

## The research question and the model roles

The primary question is whether multimodal language models can verify or match
slap fingerprint images, which limitations prevent reliable matching, and which
changes help. DINOv2 supplies a visual control: its embeddings let us investigate
image representation separately from language generation. DINOv2 results alone
cannot establish how a multimodal language model will perform or why it fails.

The historical [SLAPBench paper](https://arxiv.org/abs/2607.15517) evaluates
NIST SD302b. Its genuine pairs are two resolutions of one capture, so a perfect
score can reflect near-duplicate detection. The later RidgeBase extension and
current Precise study are different populations. Keep their results separate.
Current Precise comparisons include left hands, right hands and two thumbs.

## The preprocessing contribution

The preprocessing contributor studies how changes to image preparation affect
the existing matcher. This work owns a scientific question, a controlled
experiment specification, reproducible transformations and an explanation of
the resulting successes and failures. Its value comes from evidence, whether
or not a transformation improves the scores.

| Kind of setting | Examples | Relationship to the initial study |
| --- | --- | --- |
| Input preprocessing | Contrast method, CLAHE parameters, effective resolution, rotation and padding | Candidate variables for the preprocessing contributor |
| Model and training | Encoder, learned weights, loss, learning rate, epochs, optimizer and trainable blocks | Keep fixed in an initial preprocessing comparison; coordinate later changes with the matcher work |
| Evaluation | Pair version, acceptance threshold, class definitions, failure policy and metrics | Predeclare and keep consistent; changing the cutoff changes the error tradeoff |

For example, compare the same pairs and checkpoint with the current grayscale
pipeline and one specified contrast transformation. A change in false accepts
or genuine acceptance then measures that declared preprocessing intervention
under the fixed conditions. Changing contrast, training settings and the cutoff
together would leave the cause of a result unresolved.

The initial responsibility is to understand the workflow well enough to design
this comparison with the team. Specific extra experiments follow that learning
stage and the applicable protocol review.

## What to tackle first

### Session one Understand one current result

Read the [latest completed report](../../reports/precise-dinov2-numeric-holdout-20261107/README.md)
and its [professor report](../../reports/precise-dinov2-numeric-holdout-20261107/professor-report.md).
Write a short explanation of the task, the two target classes, the model,
the training/testing populations and how the decision threshold was chosen.

Use the [earlier report](../../reports/precise-dinov2-shared-20261101/README.md)
to understand why the often-quoted 70.45% and 77.60% accuracies belong to a
different cohort. Passing this milestone means being able to explain the table
below without mixing populations or calling numeric groups confirmed people.

| Cohort | Training pairs | Testing pairs | Training/testing image or numeric-group overlap | Frozen accuracy | Adapted accuracy |
| --- | ---: | ---: | --- | ---: | ---: |
| November 1 shared-image cohort | 39,290 | 4,200 | Every testing image/group also occurs in training | 70.45% | 77.60% |
| November 7 numeric-group holdout | 37,194 | 4,200 | Zero shared images, exact pairs or numeric groups | 66.90% | 67.19% |

Both cohorts have equal genuine/impostor testing counts, so ordinary and
balanced accuracy coincide. These are different testing populations and cannot
be read as a controlled before/after change between dates. The older adapted
checkpoint saw the newer testing images; the November 7 adaptation started
fresh from the pinned pretrained base.

### Session two Trace one pair through the workflow

Walk through one existing training pair with an experienced contributor. Keep
identifiers and image previews in local storage. Follow this sequence:

1. Identify the two source images, the supplied genuine/impostor label and slap
   type in the existing pair list. The class comes from the supplied record.
2. Follow decoding, grayscale conversion, resizing, padding and normalization.
3. Follow the image tensors through DINOv2 to normalized embedding vectors.
4. Compute their cosine similarity and apply the saved cutoff. A larger score
   means more similar; acceptance uses strictly greater than the cutoff.
5. Place that decision in the confusion matrix and identify any failure status.

| Code | What to learn |
| --- | --- |
| [`images.py`](../../code/precise_matching/images.py) | Verified bytes, complete-image decoding, letterboxing and tensor normalization |
| [`encoders.py`](../../code/precise_matching/encoders.py) | Pinned DINOv2, selected trainable blocks, CLS embeddings and cosine scores |
| [`fixed_pairs.py`](../../code/precise_matching/fixed_pairs.py) | Supplied pairs, labels, numeric-group validation and input bindings |
| [`fixed_training.py`](../../code/precise_matching/fixed_training.py) | Explicit-pair loss, full epochs, final-checkpoint selection and provenance |
| [`fixed_evaluation.py`](../../code/precise_matching/fixed_evaluation.py) | Training calibration, saved cutoffs, test coverage and failure accounting |
| [`run_precise_shared_pairs.py`](../../code/run_precise_shared_pairs.py) | Separate preparation, training, calibration and testing entry points |

### Session three Reproduce saved metrics offline

Use the saved outcomes before doing new inference. The independent verifier
reads private scores and pair membership, replays the training cutoffs and
checks the reported confusion counts, AUC and interpolated EER. It does not
load a model, retrain it or select another threshold.

With a compatible verification environment active, run from the project root:

```bash
python code/verify_precise_fixed_metrics.py \
  --bundle local/precise/manifests/numeric-group-holdout-20261107/bundle.json \
  --run-root local/precise/runs/numeric-group-holdout-20261107 \
  --aggregate reports/precise-dinov2-numeric-holdout-20261107/aggregate.json \
  --output local/precise/metadata/onboarding-NEWDATE/independent-metric-check.json
```

Choose a new output path; the verifier refuses an existing receipt. A Git clone
alone does not contain these private inputs. Use the authorized workstation's
existing assets and an accessible interpreter, following
[reproducibility instructions](../REPRODUCIBILITY.md). Do not reinstall a model
or repeat a completed test to resolve an environment or file-permission issue.

The learning deliverable is a one-page explanation plus a successful offline
verification receipt, or an exact access/environment blocker if the runtime is
unavailable. The scientific result remains the saved result in either case.

### Session four Write the first experiment specification

After the walkthrough, propose one preprocessing comparison with a fixed
checkpoint, fixed pair membership and an explicit threshold policy. Start with
the pretrained DINOv2 control to characterize image effects. A later matched
VLM comparison is needed to connect the finding to the primary LLM question.

Keep the first specification small: the current pipeline and one contrast
variant. Define the transformation order and parameters, the affected pair
endpoint, the development evidence, the metrics and the expected output table.
Review the specification before real execution. The earlier DINO approvals do
not automatically authorize a new preprocessing study.

## What the existing 448 by 448 pipeline does

Precise source JPEGs are 1600 by 1500. The current DINO pipeline converts them
to grayscale, scales the complete image to fit a 448 by 448 white canvas using
Lanczos resampling, centers it and converts the result to three-channel RGB.
It then scales pixel values and applies the fixed channel means and standard
deviations in [`images.py`](../../code/precise_matching/images.py).

For these source dimensions, the image content becomes 448 by 420, with white
padding above and below. This changes the tensors supplied to the model; it
does not overwrite the original JPEGs, stretch them into squares, establish
physical PPI or segment individual fingers. Contrast enhancement is absent
from this DINO baseline. Historical VLM runners have separate preprocessing;
their settings should not be inferred from this function.

## Proposed preprocessing and robustness study

| Question | Candidate comparison | What must remain fixed |
| --- | --- | --- |
| Does contrast processing help verification? | Existing grayscale pipeline versus one global normalization or CLAHE variant | Pair labels, checkpoint, input size, transformation order and decision policy |
| How sensitive is matching to lost image detail? | Native source versus controlled reductions such as 224- or 112-pixel longest side, followed by the unchanged 448-pixel input pipeline | Final model input size, resampling method and all other preprocessing |
| How sensitive is matching to alignment? | Small, predeclared rotations or translations, with border handling and clipping recorded | Checkpoint, pair membership, transformation strength and endpoint policy |

These are candidate designs. Parameter values and execution scope are selected
in the experiment specification. [OpenCV's contrast tutorial](https://docs.opencv.org/4.13.0/d5/daf/tutorial_py_histogram_equalization.html)
explains global histogram equalization and CLAHE, which limits local contrast
amplification. Better-looking images do not establish better verification.

Synthetic whole-image rotation/translation measures alignment sensitivity. It
does not reproduce individual-finger movement, pressure, skin deformation or
independent acquisition. Those questions need appropriate capture evidence or
a later individual-finger study.

Distinguish two analyses: keeping a baseline cutoff measures robustness of an
unchanged operating point; selecting each variant's cutoff on training or
independent development data compares calibrated variants. Predeclare the
primary analysis and report actual test FAR. Do not select cutoffs, parameters
or checkpoints from final-test performance. The existing test results are
already known, so further work on that cohort must describe this exposure.

Use training/development material for exploration. Any new development
allocation changes the supplied training membership and requires its own
declared protocol. Apply deterministic transformations without changing source
images or supplied labels. Treat a pair's endpoint policy explicitly rather
than choosing which image to degrade based on the desired answer.

## Understanding accuracy and biometric errors

Define the positive class as a genuine pair and the positive prediction as
accepting the pair as the same person.

| Count | Meaning |
| --- | --- |
| TP | Genuine pair accepted |
| FN | Genuine pair rejected |
| FP | Impostor pair accepted |
| TN | Impostor pair rejected |

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
TAR      = TP / (TP + FN)
FAR      = FP / (FP + TN)
FRR      = FN / (TP + FN)
Balanced accuracy = (TAR + (1 - FAR)) / 2
```

Multiply these proportions by 100 for percentages. Report undefined metrics
when a denominator is zero. The following counts reproduce the saved results:

| Cohort and arm | TP | TN | FP | FN | Accuracy | TAR | FAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| November 1 frozen | 876 | 2,083 | 17 | 1,224 | 70.45% | 41.71% | 0.81% |
| November 1 adapted | 1,182 | 2,077 | 23 | 918 | 77.60% | 56.29% | 1.10% |
| November 7 frozen | 730 | 2,080 | 20 | 1,370 | 66.90% | 34.76% | 0.95% |
| November 7 adapted | 727 | 2,095 | 5 | 1,373 | 67.19% | 34.62% | 0.24% |

The November 7 adapted model improves descriptive AUC from 0.8495 to 0.9416
and EER from 22.67% to 10.95%, while TAR does not increase at the arm-specific
training-calibrated cutoffs. Each arm's nominal calibration target is 1%
training FAR; neither an independent development estimate nor equal test FAR
is implied.

The [shared accuracy calculator](https://calculator.academy/accuracy-calculator-2/)
can check arithmetic from aggregate TP/TN/FP/FN counts. Save the counts and
derive them reproducibly from predictions; the website does not establish
pair correctness, independence or threshold provenance. The
[scikit-learn confusion-matrix documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
defines true labels on rows and predictions on columns; set the label order
explicitly when extracting counts.

Report attempted comparisons, usable outputs and errors/refusals with a
predeclared failure policy. Binary A/B outputs supply an operating point;
continuous-score ROC/AUC/EER require continuous scores. Reused images and
groups make independent-pair confidence intervals inappropriate without a
suitable dependence model. Numeric-group separation is not independent
confirmation of biological identity or capture/session independence.

## Reading sequence

These public references support onboarding; confirm the exact collaborator
reading assignment separately when titles or attachment links are available.

| Priority | Reference | Question to answer while reading |
| --- | --- | --- |
| First | [SLAPBench](https://arxiv.org/abs/2607.15517) | What is a genuine pair, what collapses under prompting, and what limits the perfect result? |
| Next | [DINOv2](https://arxiv.org/abs/2304.07193v2) | What does the vision encoder produce, and how does that differ from an LLM answer? |
| Next | [FPBench](https://arxiv.org/abs/2512.18073v2) | How do single-finger tasks, prompts, fine-tuning and evaluation differ from this whole-slap protocol? |
| Later | [Deep Age-Invariant Fingerprint Segmentation System](https://arxiv.org/abs/2303.03341) | How does CRFSEG localize and label rotated individual fingers? |

The linked segmentation preprint predates the journal version cited in
[`references.bib`](../../paper/manuscript/references.bib); record which version
supports a claim. Do not transfer accuracy values from those papers directly
to this cohort. First read each abstract, task, data split, evaluation and
limitations; detailed architecture can follow the workflow demonstration.

## Deliverables and coordination

The first learning milestone is complete when a contributor can explain the
current cohort, trace a pair, reproduce the saved confusion counts and explain
why accuracy alone is insufficient. The next deliverable is a reviewed,
versioned preprocessing protocol, followed by an authorized reproducible
comparison with failure counts and per-slap-type results.

Before comparing with fine-tuned Qwen3-VL, obtain its matching actual outputs,
pair-list version, consumed training membership, base/adapter/checkpoint identity,
prompt/processor, selection rule and parsing/failure evidence. Existing source
code and pair lists need not be requested again. Individual-finger/minutiae
work follows the whole-image baseline as a separate extension.

Private data, chats, connection routes, source identifiers, image examples,
derived biometric templates and third-party full-text copies remain in local
storage. Publish reviewed code, protocols and aggregate results. No new model
run, test tuning or result replacement is part of this onboarding document.
