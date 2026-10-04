# Precise research preparation — 2026-10-04

Status: background preparation authorized by the user; no new experiment or
training run authorized. This records the professor's feedback supplied in this
chat and distinguishes decisions from proposals.

## Confirmed direction

- Use Precise for the next research study. Do not plan new SD302b or RidgeBase
  experiments for this direction.
- Investigate fingerprint matching through learned visual representations and
  potentially structured minutiae. An exact model and training objective have
  not been selected.
- A collaborator reports approximately 95% VeriFinger performance and ongoing
  fine-tuning. These are reported observations, not independently verified
  repository results or an agreed evaluation target.
- The user authorized repository preparation, documentation and offline data
  validation while discussing the remaining decisions with the professor and
  collaborator.

The earlier 16-section student guide was read before preparation. It was removed
from the active tree at the user's direction on 2026-09-27 and remains in Git
history at `54bf2f8`, under
`docs/Fingerprint_Foundation_Model_Verification_Student_Guide.md`. This task does
not restore that guide or its removed draft protocol. Its relevant methodological
references are §4 (subjects/impressions), §8 (single-finger then slap fusion),
§9 (verification metrics), §11 (shared pairs and development thresholds) and
§16 (results/split review before fine-tuning). Whether the latest feedback
revises the milestone sequence remains a discussion item.

## Preparation completed now

- Make Precise the visible current direction in README, repository instructions,
  map and setup documentation.
- Keep existing dataset, results and paper paths intact. Reorganizing historical
  artifacts would break manifests and reproduction commands without helping the
  new study. New notes belong here; new derived inventories belong at dated
  `reports/` paths. Future experiment output paths will follow an agreed protocol.
- Add `code/audit_precise_dataset.py`: a read-only, standard-library inventory of
  top-level JPEG filenames and SHA-256 byte duplicates. Its report contains
  aggregate counts only. The script refuses to overwrite reports.
- Save the observed inventory at
  `reports/precise-readiness-20261004/inventory.json`. Its README explains the
  method, findings and limitations. Filename patterns do not validate acquisition
  sessions, sensors, independent captures or individual finger positions.

## Decisions to resolve together

| Decision | Information needed | Work that depends on it |
|---|---|---|
| Research contribution | Must the model adapt an LLM/VLM, or may it be a dedicated image/minutiae matcher? Is using a VLM vision encoder sufficient? | Model selection, architecture and paper framing |
| Unit of matching | Individual fingers followed by fusion, or whole four-finger slaps? How does VeriFinger process the data? | Segmentation, finger labels and score fusion |
| Precise capture metadata | Dataset source/version and terms; meanings of middle filename tokens; independent impressions, sessions, resolution and known annotation issues | Genuine-pair rules and preprocessing |
| Existing commercial result | Meaning of 95%; exact pairs, continuous scores, preprocessing, SDK version/settings, thresholds and failures | Comparable baseline and performance target |
| Existing fine-tuning | Base model, trainable components, loss, subjects/captures used for training, validation and repeated evaluation | Reuse of work and selection of uncontaminated subjects |
| Shared evaluation | Subject-disjoint train/development/test protocol; fixed development thresholds; genuine/impostor counts and uncertainty | Split and manifest construction, final-test claims |
| First milestone | Retain the original prompting pilot, or revise it around VeriFinger and learned embeddings? Retain Bozorth3? | Authorized order of baseline runs and training |
| Practical access | Licensed SDK access and permitted outputs; available compute, budget and model/data permissions | Dependencies and execution setup |

## Proposed next preparation, after this inventory

1. Obtain the collaborator's metadata and protocol artifacts; inspect them offline
   without replacing existing scores. No messaging is authorized by this note.
2. Validate image dimensions/readability, capture provenance and decoded/near
   duplicates. Keep source files unchanged and publish aggregate findings only.
3. Draft a revised protocol using the agreed task and subjects already exposed
   to training or evaluation. Do not call a reused test cohort untouched.
4. Specify a common score/error table and reproducible run record before any
   baseline is launched. Record preprocessing, software/checkpoint versions,
   manifest hashes, failure handling and thresholds selected on development.
5. After protocol review and separate run authorization, establish comparable
   conventional and model baselines before adding architectural components.

Different filename bytes are not evidence of independent captures. Many pairs
from the same subjects are correlated; pair count alone cannot establish low-FAR
precision. Same-session independent impressions can support a pilot, but cannot
by themselves establish cross-session generalization (§§4,9,11).

Existing `code/embed_verify.py` already pools general VLM vision features and
compares cosine similarity. It is a historical baseline, not evidence of
fingerprint-specific metric training. Existing `code/precise_pairs.py` builds
whole-hand pairs without a subject split. Neither script was run or modified in
this preparation, and neither fixes the pending research choices.
