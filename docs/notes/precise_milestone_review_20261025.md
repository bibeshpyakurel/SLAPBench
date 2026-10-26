# Precise milestone: concrete review proposal — 2026-10-25

**Status: DINOv2-first implementation accepted; real-data protocol/split review pending.**
The user's later decision in the “Newer Models” chat selected the proposed
DINOv2 matcher first and requested training/testing. The scheduled resumption
verified that decision and continued the interrupted implementation. See the
[implementation guide](precise_training_implementation_20261025.md) for completed
code and synthetic evidence. Numeric protocol choices and real-data readiness
remain subject to the review conditions below.
The user supplied the ten-task roadmap on 2026-10-25 and confirmed that no new
acquisition/exposure clarification from Konain is available. This document makes
the next decisions reviewable while preserving the [approved principles](precise_first_milestone_proposal_20261011.md).
No subject is allocated and no real matching comparison list is constructed.
The [readiness report](../../reports/precise-readiness-20261025/README.md)
records the accompanying offline checks; original collaborator lists remain intact.

## Proposed experimental protocol

| Item | Proposed setting | Review condition |
|---|---|---|
| Population | Authorized Precise subjects with verified parent captures; at least two independent acquisitions for each included hand | Resolve annotation/duplicate conflicts and acquisition evidence first; never manufacture capture IDs from filenames |
| Unit | Entire four-finger left or right slap; compare only matching hand types | Exclude two-thumb slaps from this milestone, retaining their historical assets |
| Subject boundary | All hands, sessions, derivatives, thumbs and unused images of a subject stay together | Check raw/native/grayscale duplicates and review suspect derivative relationships |
| Target sizes | If all 338 subjects qualify: 202 training, 68 development, 68 test | These are proposed counts, not known eligible counts or actual assignments |
| Smaller eligible population | Train `floor(0.60*N)`; development `floor((N-train)/2)`; test remainder | Return for review if either evaluation partition has fewer than 60 subjects; do not silently accept a weakened low-FAR design |
| Allocation seed | `20261025`; rank subjects by SHA-256 of UTF-8 `precise-subject-allocation-v1\0<seed>\0<canonical opaque subject ID>`; ascending digest then opaque ID breaks ties; take train, development, test in order | `\0` denotes one NUL byte; freeze eligibility and private canonical ID mapping first; save implementation/configuration/digests before any assignment |
| Historical exposure | Record supplied-list membership, verified consumption and prior evaluation separately | This is an internal, subject-disjoint reanalysis of a previously evaluated collection; no untouched or external-validation claim |
| Genuine evaluation comparisons | Within each partition/subject/hand: all unordered independent-capture combinations if at most 10; otherwise choose 10 without replacement | Select one reviewed canonical image per capture first; a derivative is not another impression |
| Impostor evaluation comparisons | For each unordered subject dyad and shared hand, choose one eligible image from each endpoint | At most 4,556 impostor comparisons per 68-subject partition with both hands; repeated subjects/hands remain dependent |
| Sampling implementation | Deterministic SHA-256 ranking over a versioned domain tag, seed and private canonical IDs; take lowest ranks; record tie rule | Separate domains for genuine selection, impostor endpoint selection and allocation; no scores enter sampling |
| Primary arms | Frozen selected encoder and adapted version of exactly that encoder | Same immutable development/test lists and deterministic preprocessing |
| Conditional arms | VeriFinger SDK fusion and sum of per-finger scores on those exact lists | Require SDK access/configuration; historical collaborator aggregates are a separate retrospective reference |
| Development target | FAR `0.001` (0.1%), strict `score > cutoff`, existing evaluator's integer error budget and tie handling | Proposed primary target resolves the earlier 0.1%/0.01% ambiguity for review; 0.01% is exploratory only |
| Checkpoint metric | Highest development all-attempt TAR at its development-selected cutoff, then lower valid-score EER, then earlier epoch | Decode/encoder failures invalidate checkpoint selection until investigated; record every attempted checkpoint |
| Final test | Freeze one checkpoint, preprocessing, method configuration, lists and per-method development cutoff before accessing new test scores | Evaluate once; do not adapt to historical or new test scores; report deviations as exploratory |

For complete 68-subject/two-hand development coverage, 4,556 impostor trials
allow four false accepts at the proposed 0.1% empirical budget. One false accept
is about 0.02195%. A 0.01% budget allows zero. Neither the number of image
comparisons nor zero errors establishes a population FAR guarantee; subjects
are reused. Final denominators depend on eligibility and approved sampling.
The supplied SDK testing lists cover 337 of 338 subjects; supplied training lists
contain all 338. Actual training/development use is still unknown. Starting a
fresh checkpoint and making future partitions subject-disjoint cannot erase
historical inspection. Professor review must accept the resulting claim scope
or require an independently acquired cohort.

Keep decode failures, missing inputs, encoder failures and SDK extraction
failures distinct. Store one outcome per attempted comparison. Report both
valid-score discrimination with coverage and all-attempt failure-as-reject
TAR/FAR, exposing failed impostors separately. Never treat a missing score as
zero or silently remove a method's failed comparisons. Training decode failures
stop the run for investigation rather than changing its population. No new
quality cutoff or automatic near-duplicate exclusion is proposed.

Proposed reporting: fixed-threshold counts/TAR/FAR, ROC/DET, EER, coverage and
failure counts; pooled and left/right results using the **same pooled development
cutoff**. Do not select a different hand-specific cutoff after seeing test.
Method differences must use identical attempted lists; a valid-score intersection
can be supplemental with its reduced denominator explicitly reported.

Proposed uncertainty, subject to statistical review: 2,000 paired subject-cluster
bootstrap replicates, seed `20261025`, retaining each subject's hands/images
together. Resample subjects with replacement; weight genuine trials by endpoint
subject multiplicity and impostor trials by the product of both endpoint
multiplicities. Use the same weights for all arms and keep the original selected
development cutoffs fixed. Report percentile intervals for TAR/FAR and paired
differences, invalid-replicate counts and effective observed support. This is an
approximate subject-resampling analysis under exchangeability, not a validated
rare-event confidence guarantee. Zero-error bootstrap intervals can collapse
to zero and must be marked uninformative about unobserved errors. This proposal
does not yet constitute an agreed or implemented uncertainty method.

## Selected first matcher and implementation settings

The user selected **DINOv2 ViT-S/14**, Hugging Face repository `facebook/dinov2-small`,
immutable revision **`ed25f3a31f01632728cabb09d1542f84ab7b0056`**, as the initial
matcher. Initial public model API/configuration inspection did not download
weights; the later authorized implementation obtained and verified them under
ignored `local/precise/models/` and completed a small synthetic CPU probe.
Its metadata lists 22,056,576 parameters, hidden width
384, 12 blocks, six attention heads and patch size 14.
The [upstream model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
describes class/patch outputs and support for larger patch-multiple inputs.
The [pinned checkpoint configuration](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/config.json)
sets a 518-pixel reference image size. Our proposed 448-pixel input requires
positional interpolation and is a new experimental choice, not an upstream
fingerprint recommendation. These sources supply architecture facts, not
Precise performance evidence.

This compact image encoder implements the frozen/adapted comparison without
a language decoder. The user selected this image-only matcher first; its
relationship to the larger LLM/VLM research theme remains explicit. The three
downloaded VLMs remain separate investigation candidates. No claim is made that
this architecture is optimal for ridge detail or whole-slap verification.

| Component | Concrete proposed configuration |
|---|---|
| Loader | Strict Pillow decode from approved private image mapping; reject truncated/invalid images; preserve original bytes and record structured failure |
| Preprocessing | Decode to `L`, replicate to three channels; aspect-preserving LANCZOS resize with longer side 448, round shorter side to nearest integer; center on 448×448 white canvas with odd extra pixel on right/bottom |
| Present inventory geometry | 1600×1500 becomes 448×420, with 14 white rows above/below; retain all four fingers; no segmentation or crop |
| Scaling | Convert bytes to float in [0,1], then channelwise ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`; freeze these values for both arms |
| Augmentation | None for the first controlled run; no flips, crop, rotation, contrast normalization or quality filtering |
| Representation | Model final normalized class token `last_hidden_state[:,0,:]`, float32 L2 normalization with epsilon `1e-12`; 384 dimensions, no learned projection; nonfinite or zero-norm embeddings are encoder failures |
| Score | Float32 dot product of normalized embeddings (cosine); finite continuous score; never report it as calibrated match probability |
| Frozen control | All weights frozen, evaluation mode, inference without gradients; same revision/representation/preprocessing as adapted arm |
| Adapted arm | Train final two transformer blocks and final layer norm; freeze patch embeddings, positional/class/mask tokens and first ten blocks; verify actual parameter names/counts before execution |
| Supervision | Ground-truth subject/hand and verified independent-capture labels within training only; no SDK-score distillation or language targets |
| Batch | One hand per microbatch, alternating left/right; eight distinct training subjects × two distinct independent captures = 16 images; uniform eligible subject/capture sampling |
| Loss | Within-hand unordered batch pairs: `0.5*mean_positive(1-cosine) + 0.5*mean_negative(max(0, cosine-0.2))`; each class mean separately, no diagonal/self/derivative positives |
| Optimizer | AdamW, LR `1e-5`, weight decay `0.01` on matrices only, betas `(0.9,0.999)`, epsilon `1e-8`; no decay on biases/norm parameters |
| Schedule | 30 epochs × 200 microbatches; accumulate four microbatches per optimizer step; 50 optimizer steps/epoch; linear warmup for first 100 steps then cosine decay to `1e-6` |
| Precision | BF16 autocast if supported, otherwise reviewed FP32 configuration; normalization/loss in FP32; gradient norm clipping 1.0 |
| Selection | Development evaluation after each epoch; no final-test data in loader, sampler, loss, resume/selection or training logs |
| Randomness | Seed `20261025` for Python/NumPy/PyTorch and worker generators; deterministic algorithms with unsupported operations failing explicitly |
| Resume | Save model, optimizer, scheduler, RNG states, sampler position, configuration and code/model/manifest hashes locally; reject changed inputs; never overwrite checkpoints |

The verification loss is an intentionally simple proposed cosine contrastive
objective; its margin and all optimizer/batch settings are choices for review,
not results selected from Precise. Each microbatch has eight positive unordered
image pairs and 112 negative pairs. The separate class means prevent negative
count from setting the loss balance. A subject lacking two verified independent
captures for a hand cannot supply that hand's training batch; eligibility and
the balance rule must be reviewed rather than silently changing the sampler.

At 448 pixels there are 1,024 patch tokens plus the class token. FP32 weights
alone occupy about 84 MiB. A single BF16 attention matrix for 16 images, six
heads and 1,025 tokens is about 192 MiB per block; implementations and retained
activations change actual memory considerably. Plan for an **unmeasured 8–12 GiB
training budget**, reserve at least 2 GiB outside the process, and verify with an
authorized synthetic forward/backward before real loading. These are planning
estimates, not a successful fit measurement. Full VLM training is not assumed
to fit. If the synthetic check fails, revise microbatch/accumulation through a
recorded development design change before any real run.

Dataset membership in the encoder's original pretraining corpus is unknown.
Using a public pretrained checkpoint does not certify that Precise is absent
from that corpus; record this separate exposure limitation alongside local and
collaborator training/evaluation history.

## Implemented model/training boundary

`code/precise_matching/` now implements `images.py`, `encoders.py`, `losses.py`
and `training.py`, with a guarded `code/train_precise_matcher.py` entry point.
The [implementation guide](precise_training_implementation_20261025.md) records
exact contracts and synthetic tests, including exact resume and development-only
selection. The original metadata/score contracts and strict cutoff are retained.
CI uses synthetic images and tiny encoders without checkpoint downloads. The
real selected encoder completed the separately authorized small CPU check.
Training refuses missing review assertions, partition checks or immutable input
digests. A completed selected-checkpoint record is separate from future final-test
scoring. Validation assertions do not independently prove acquisition facts or
replace concrete results/split review.

## Compute inspection and preparation boundary

The [compute report](../../reports/precise-readiness-20261025/environment.md)
records the failed NVIDIA driver check, unavailable weight mount, package
versions and disk/RAM inspection. Existing `.venv/` was inspected without
upgrading it. Private full package/environment/model-metadata receipts are in
`local/precise/metadata/compute-20261025/`.

**Dependency preparation update — 2026-10-25:** roadmap task 4 permits isolated
dependency preparation before model execution. A candidate environment now
exists under ignored `local/precise/environments/matcher-candidate-20261025/`.
The [follow-up report](../../reports/precise-environment-20261025/README.md)
records a 50-wheel hash-pinned lock, successful dependency checks and offline
DINOv2 class imports. The historical environment's package inventory is unchanged.
This prepares the proposed stack without accepting the model/protocol choice.
GPU driver/runtime compatibility, memory fit and the authorized synthetic model
check were initially unverified; a later small synthetic CPU check now passes,
while CUDA and full-resolution GPU memory remain unverified. No real-data
training run was created. System-level
driver repair and restoring the external model drive require their own scoped
work; neither was attempted.

## Decisions requested and roadmap state

1. Konain: supply acquisition token meanings/parent-capture evidence, known
   processing variants and session/sensor/PPI metadata through the private channel.
2. Konain: distinguish supplied-list membership from actual training/development
   consumption; explain right-only training impostors and 3,455 zero-fused-score
   genuine rows without components. Do not request the same four lists again.
3. Professor/user: review internal-reanalysis claim scope versus a new untouched
   collection, eligible population/counts, same-hand sampling, 0.1% target,
   selection/failure policy and proposed subject-resampling uncertainty.
4. The user selected DINOv2 first; review implementation/resource evidence and
   concrete execution settings. Review the actual eligible split **after**
   acquisition/exposure facts are available, before fine-tuning.
5. Authorize subsequent implementation and execution at their applicable gates;
   approve any system repair separately. No external message has been sent.

| Roadmap task | State after this work |
|---|---|
| 1. Provenance/quality | Full decoding and duplicate screening performed; private per-image capture/exposure unknowns recorded; acquisition facts, candidate adjudication and actual exposure still open |
| 2. Final protocol | Concrete review draft above; finalization depends on task 1 and review |
| 3. Model/training design | DINOv2-first selected by the user; configuration, encoder/loss and guarded training implementation available |
| 4. Compute | Isolated dependencies and selected local checkpoint verified; small synthetic CPU update/resume check passes; GPU/full-resolution resource verification pending |
| 5. Split/comparison generation | Gated by provenance/exposure and protocol/split review; no artifacts generated |
| 6–7. Loader/encoder/loss/training | Implemented with synthetic contract/resume/leakage tests; real execution requires reviewed inputs and resource readiness |
| 8. Baseline/training | Gated by reviewed tasks 4–7 and explicit execution authorization |
| 9. Comparative evaluation | Existing offline evaluator available; new common-list SDK/model scores and agreed uncertainty still required |
| 10. Research package | Readiness/review documentation prepared; experimental evidence and authorized manuscript work remain future |

Completion of the first milestone requires the later evidence, even if the
learned system fails to improve. This review package is progress toward the
roadmap, not completion of all ten tasks.
