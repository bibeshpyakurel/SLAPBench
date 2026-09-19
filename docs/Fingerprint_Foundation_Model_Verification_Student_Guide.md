STUDENT RESEARCH INSTRUCTIONS
Evaluating Foundation / Vision-Language Models for Fingerprint Verification
From image-only prompting to minutiae-grounded biometric matching

Research goal	Determine whether a general-purpose multimodal model can perform fingerprint verification and whether explicit minutiae information improves verification performance.
Primary comparison	Foundation/VLM methods vs. a conventional matcher such as NBIS Bozorth3 and, if licensed, VeriFinger.
Primary metrics	TAR at fixed FAR, ROC/DET, EER, score distributions, refusal/error rate, and latency/cost where relevant.
Recommended first milestone	Run a controlled, subject-disjoint pilot comparing image-only VLM, image+minutiae VLM, and Bozorth3 on the same genuine/impostor pairs.

Important: Use only datasets for which the lab has authorization. Do not deploy the experimental system for real-world identity or access-control decisions. If a hosted model refuses biometric comparison, record the refusal as an experimental outcome; do not attempt to bypass provider safeguards.


1. Research Question and Hypotheses
Primary research question: Can a general-purpose multimodal foundation model be adapted or grounded with fingerprint minutiae so that it behaves more like a conventional biometric matcher?
Test the following hypotheses:
    • H1 — An image-only VLM will perform above chance on some fingerprint pairs, but will be less reliable and less calibrated than a conventional fingerprint matcher.
    • H2 — Explicit minutiae information will improve discrimination between genuine and impostor pairs compared with image-only prompting.
    • H3 — Structured minutiae representations (tokens/embeddings or spatial overlays) will outperform a simple text list of minutiae if a trainable open-weight model is used.
    • H4 — Improvements should be evaluated at operationally meaningful low FAR values, not only by overall classification accuracy.
2. Key Concept: Verification, Not Identification
This experiment is a 1:1 verification task: given fingerprint sample A and fingerprint sample B, determine whether they originate from the same enrolled identity/finger. Do not frame the task as searching a gallery to identify an unknown person.
Term	Meaning	Use in this study
Genuine pair	Two samples from the same finger/subject	Positive comparison
Impostor pair	Samples from different fingers/subjects	Negative comparison
Similarity score	Continuous number representing match strength	Preferred model output
Decision threshold	Score cutoff for match/non-match	Chosen from development data
FAR	Fraction of impostor comparisons incorrectly accepted	Security-oriented metric
TAR	Fraction of genuine comparisons correctly accepted	Report at fixed FAR
3. Experimental Stages
Complete the study in stages. Do not begin fine-tuning until the baseline experiment is reproducible.
Stage A — Conventional baseline
    1. Prepare the fingerprint dataset and subject/finger identifiers.
    2. Extract minutiae using a conventional extractor. NBIS MINDTCT is an appropriate open baseline; a licensed SDK may also be used if available.
    3. Match minutiae with NBIS Bozorth3. If VeriFinger is available, run the same comparison list through VeriFinger as an additional benchmark.
    4. Store a continuous matcher score for every pair. Do not store only match/non-match decisions.
Stage B — Hosted VLM image-only baseline
    5. For each pair, present the two fingerprint images to the selected multimodal model using the same prompt template.
    6. Request a machine-readable response containing a confidence/similarity value and a binary decision only if the service permits the task.
    7. Use consistent model settings where available. Record model/version, date, prompt, image preprocessing, and any refusal/error.
    8. Do not interpret natural-language confidence as a calibrated biometric score until validated experimentally.
Stage C — Image + explicit minutiae
    9. Use the exact same pair list as Stage B.
    10. Provide each image together with its minutiae. Start with a textual representation; then test a visual minutiae overlay if appropriate.
    11. Compare whether minutiae information improves TAR at the same FAR and improves score separation.
Stage D — Trainable open-weight model (advanced)
If Stages A–C show useful signal, build a trainable model in which image features and minutiae features are fused before the final fingerprint embedding. This is the stage that can move the work from “prompting a VLM” toward a true foundation-model biometric matcher.
4. Dataset Construction
Use a subject-disjoint protocol. A subject appearing in training must not appear in the final test set. If the study is prompt-only and has no training split, still create separate development and final test sets so thresholds and prompt choices are not tuned on the test data.
    • Preserve the identity of the finger position (e.g., right index vs. right middle) when the dataset provides it.
    • Create genuine pairs from repeated impressions of the same finger.
    • Create impostor pairs from different subjects; for a stricter experiment, compare the same finger position across different subjects.
    • Include hard cases: low quality, partial overlap, rotation, pressure variation, sensor variation, and visually similar ridge flow.
    • Avoid duplicate images across comparison pairs that could cause leakage into train/test partitions.
Suggested pair manifest
pair_id,image_A,image_B,subject_A,subject_B,finger_A,finger_B,label
P000001,A_01.png,A_02.png,S001,S001,R_INDEX,R_INDEX,1
P000002,A_01.png,B_04.png,S001,S027,R_INDEX,R_INDEX,0
5. Minutiae Representation
Represent each detected minutia as a structured record. At minimum, use location and orientation; include type and quality when the extractor supplies reliable values.
m_i = (x_i, y_i, theta_i, type_i, quality_i)
Normalize coordinates so that they are independent of image size:
x_norm = x / image_width
y_norm = y / image_height
angle features = [sin(theta), cos(theta)]
Recommended text format for the prompt-only experiment:
Fingerprint A minutiae (normalized):
1: x=0.271, y=0.344, sin_theta=0.530, cos_theta=0.848, type=ending, q=0.91
2: x=0.433, y=0.571, sin_theta=0.857, cos_theta=-0.515, type=bifurcation, q=0.86
...
Important: Do not give the model subject IDs, filenames that reveal identity, or the ground-truth label. Randomize or neutralize filenames before sending images to the model.

6. Prompt Templates for the Initial Experiment
Keep prompts fixed for the formal test. You may develop the prompt on the development set, but freeze it before evaluating the final test set.
Prompt A — Image only
Research comparison task. Compare fingerprint image A with fingerprint image B.
If the service permits this verification task, return only JSON with:
{"similarity": <0 to 100>, "decision": "same" or "different", "quality_A": <0 to 100>, "quality_B": <0 to 100>}
Treat similarity as visual evidence only. Do not use filenames or external identity information.
Prompt B — Image + minutiae
Research comparison task. Compare fingerprint A and B using both the images and the supplied minutiae coordinates/orientations.
Consider the spatial consistency of corresponding ridge endings and bifurcations, rotation/translation, image quality, and local ridge appearance.
If the service permits this verification task, return only JSON with:
{"similarity": <0 to 100>, "decision": "same" or "different", "minutiae_consistency": <0 to 100>, "quality_A": <0 to 100>, "quality_B": <0 to 100>}
If the hosted model refuses the task, record: model, pair ID, prompt version, refusal/error category, and response text where permitted. Do not alter prompts specifically to evade the refusal.
7. Advanced Model: Minutiae-Grounded Foundation Matcher
For an open-weight multimodal model that can be trained for research, use a Siamese verification architecture. Both branches must share weights.
Fingerprint A                                  Fingerprint B
   |                                                |
Image -> Vision Encoder                      Image -> Vision Encoder
   |                                                |
Minutiae -> Minutiae Encoder                Minutiae -> Minutiae Encoder
   |                                                |
        Cross-attention / Feature Fusion
              |                       |
             z_A                     z_B
                \                   /
                 cosine / match head
                         |
                    similarity score
Minutiae token for minutia i:
u_i = [x/W, y/H, sin(theta), cos(theta), type, quality]
e_i = MLP(u_i)
The model then fuses visual patch tokens with minutiae tokens, ideally allowing each minutia to attend to the local ridge region around its coordinate.
Recommended training objective
L_total = L_pair + lambda1 * L_contrastive + lambda2 * L_triplet
    • Pair/BCE loss: predicts genuine vs. impostor.
    • Contrastive loss: pulls genuine embeddings together and pushes impostors apart.
    • Triplet loss: enforces d(anchor, positive) + margin < d(anchor, negative).
    • Use hard-negative mining after the initial model is stable.
8. Slap Fingerprint Extension
For four-finger slap images, first establish the single-finger method. Then segment the slap into finger regions and match corresponding positions before score fusion.
Slap A -> [Index A, Middle A, Ring A, Little A]
Slap B -> [Index B, Middle B, Ring B, Little B]
              | per-finger matcher |
              S1, S2, S3, S4
                    |
            learned/quality-aware fusion
                    |
               slap similarity
Compare simple mean fusion, quality-weighted fusion, maximum/minimum-based rules, and a small learned fusion model. Keep a held-out development set for selecting the fusion method and threshold.
9. Evaluation Protocol
Accuracy alone is not sufficient for biometric verification. Generate a continuous score for every genuine and impostor comparison and calculate the following.
Metric	Required reporting
ROC curve	TAR versus FAR across thresholds
DET curve	FNR versus FAR on biometric-oriented axes
EER	Point at which FAR approximately equals FRR
TAR @ FAR = 1%	Basic operating point
TAR @ FAR = 0.1%	Stronger operating point
TAR @ FAR = 0.01%	Primary low-FAR target if enough impostor comparisons exist
Score distributions	Plot genuine and impostor similarity-score distributions
Failure/refusal rate	Required for hosted VLM experiments
95% uncertainty	Bootstrap confidence intervals when feasible
Definitions:
FAR(T) = # impostor scores >= T / # impostor comparisons
TAR(T) = # genuine scores >= T / # genuine comparisons
FRR(T) = 1 - TAR(T)
Important: To credibly estimate very low FAR, you need enough impostor comparisons. For example, a target FAR of 0.01% (1e-4) requires far more than a few hundred impostor comparisons. Report the number of genuine and impostor trials with every result.

10. Required Ablation Study
System	Image	Minutiae	Structured fusion	Trainable	Purpose
Bozorth3	No	Yes	Geometry	No	Open conventional baseline
VeriFinger*	Yes	Yes/SDK	Proprietary	No	Commercial reference
VLM image-only	Yes	No	No	No	Foundation-model baseline
VLM + text minutiae	Yes	Yes	Prompt-level	No	Tests in-context grounding
Proposed model	Yes	Yes	Cross-attention	Yes	Tests learned minutiae grounding
*Only if the laboratory has a valid license/access.
11. Experimental Controls and Reproducibility
    • Use the same image pairs for all methods.
    • Use the same crop/orientation normalization unless the method explicitly requires otherwise.
    • Freeze the final prompt before final evaluation.
    • Record model name/version and API date for hosted models because behavior can change.
    • Record all preprocessing parameters and software versions.
    • Repeat hosted-model comparisons if outputs are nondeterministic; report aggregation strategy.
    • Select thresholds only on a development set, then apply the fixed threshold once to the test set.
    • Keep a complete pair manifest so every score can be reproduced and audited.
    • Report exclusions and failures; do not silently drop difficult samples.
12. Student Deliverables
For the first milestone, submit the following items:
1. Dataset summary: number of subjects, fingers, images, sensors (if known), and train/development/test split.
2. Pair manifest with the exact genuine and impostor comparisons used.
3. Bozorth3 baseline scores and evaluation metrics.
4. VLM image-only results for the same pairs, including refusals/errors.
5. VLM image+minutiae results for the same pairs.
6. One CSV file containing pair_id, ground_truth, every system score, decision, and error/refusal flag.
7. ROC and DET curves plus genuine/impostor score distributions.
8. Table reporting TAR at FAR = 1%, 0.1%, and 0.01% where statistically supportable.
9. Short analysis (1–2 pages) explaining whether minutiae improved the foundation model, where it failed, and what should be tried next.
Recommended results table
Method	Genuine	Impostor	EER	TAR@1% FAR	TAR@0.1% FAR	TAR@0.01% FAR
Bozorth3	—	—	—	—	—	—
VeriFinger (if available)	—	—	—	—	—	—
VLM image-only	—	—	—	—	—	—
VLM + text minutiae	—	—	—	—	—	—
Proposed trained model	—	—	—	—	—	—
13. Suggested Folder Structure
fingerprint_vlm_project/
  data/
    images/
    minutiae/
    splits/
  manifests/
    pairs_dev.csv
    pairs_test.csv
  baselines/
    bozorth3/
    verifinger/
  vlm/
    prompts/
    raw_responses/
    parsed_scores/
  proposed_model/
    src/
    checkpoints/
  evaluation/
    scripts/
    figures/
    tables/
  README.md
14. Suggested Research Progression
Milestone	Work	Go/no-go criterion
M1: Baseline	Bozorth3 + image-only VLM	Pipeline reproducible; scores available for all/most pairs
M2: Prompt grounding	Add textual minutiae	Check whether separation/TAR improves without leakage
M3: Visual grounding	Minutiae overlay/local crops	Determine whether spatial presentation improves over text
M4: Learned grounding	Open-weight VLM + minutiae encoder	Improvement over image-only learned baseline
M5: Slap fusion	Per-finger matching + fusion	Compare against slap-level direct VLM and conventional baseline
15. Questions the Final Report Must Answer
    • Does the image-only foundation model produce stable, useful verification scores?
    • Does supplying minutiae improve genuine/impostor separation?
    • At the same FAR, how much TAR is gained or lost relative to Bozorth3 and VeriFinger?
    • Which image-quality conditions cause the largest performance drop?
    • Does the model fail because of insufficient biometric discrimination, poor calibration, refusals, or unstable output?
    • Does explicit minutiae grounding provide complementary information beyond the image encoder?
    • Can a trainable multimodal model approach conventional matcher performance at low FAR without leaking subject identity across splits?
16. Minimum First Assignment for the Student
Before implementing the full proposed architecture, complete this small controlled experiment:
1. Select a manageable development subset with repeated impressions and clearly defined subject/finger labels.
2. Generate at least one genuine-pair list and one much larger impostor-pair list.
3. Run Bozorth3 on all pairs and preserve raw scores.
4. Run the chosen hosted VLM on the same pairs using Prompt A (image only).
5. Extract minutiae and rerun the VLM using Prompt B (image + minutiae).
6. Create ROC curves and compare TAR at the same FAR thresholds.
7. Report whether minutiae helped, hurt, or had no measurable effect. Include refusal/error rates.
8. Do not start model fine-tuning until these results and the data split have been reviewed.
End of student instructions