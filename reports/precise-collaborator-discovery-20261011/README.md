# Precise collaborator-file discovery — 2026-10-11

The user authorized searching documents on this computer for Konain's reported
VeriFinger testing scores and fine-tuning pair list. This read-only discovery
did not locate those artifacts in the searched scope. Their absence here does
not establish that they were never shared or that they do not exist elsewhere.

## What was checked

| Check | Result |
|---|---|
| Document/archive files inspected | 637 |
| PDF text extraction | All 24 PDFs had extractable text |
| Word documents | Two inspected |
| Archives | Two inspected: one contains 7,358 JPEGs only; the other contains two HTML documents |
| Additional log/configuration/backup/extensionless candidates | 259; 244 readable text files scanned, six credential/history exclusions, nine binary exclusions |
| TXT/CSV/JSON data-file structure checks | 495 files reviewed |
| Permission-restricted locations | Two under media storage |
| Existing Qwen3 score-export copies | Six copies representing two distinct 676-record exports |
| Export equality and source-byte checks | All six match the repository originals; all six byte checks pass |

The search covered the user's home folder and accessible media/mount locations.
Text formats were searched for VeriFinger, fusion, collaborator, training-pair
and reported population terms. CSV headers, numeric-only lines and JSON keys/
array sizes were checked for SDK scoring and 2,100/4,200-record populations.
Recognized image-pair/score lines helped distinguish data tables from prose.
Relevant hits were reviewed against their source and existing repository
provenance. These heuristics are discovery aids, not proof of complete coverage.

Runtime environments, caches, model/checkpoint trees, credentials/history and
other program/security state were excluded. Symlinks were not followed. Other
user accounts, unmounted storage and remote shares were not searched. Raw
fingerprint images and screenshots were not OCRed. This is a scoped local
search, not a claim to have searched every byte on every device.

## Findings and provenance

The two files supplied from Downloads are identical to
`results/precise/qwen3vl/res_precise_676_genuine_qwen3_vl-8b.txt` and
`results/precise/qwen3vl/res_precise_676_impostore_qwen3_vl-8b.txt`. Matching
copies also exist on the Desktop. Each line contains two image filenames and
one Qwen3-VL score. Together they describe 676 genuine and 676 impostor
comparisons. Their provenance is the existing
`code/export_precise_qwen3_scores.py` and September saved Qwen3 responses.

Konain's reported VeriFinger evaluation is a different population: 2,100
genuine and 2,100 impostor comparisons, evaluated through SDK fusion and the
sum of per-finger scores. Neither that score table nor the collaborator's
training pair list or adapted fine-tuning script was located. Existing
Precise evaluation/full manifests do not establish his training population.
Text mentions of VeriFinger, fusion or 2,100 in plans, model responses and logs
are not the missing SDK records.

A local Word copy of the earlier student guide was found. It provides
methodological instructions and examples, not those testing/training records.
It remains local and was not restored to the public repository. Other
documents' instructions were treated as source material, not new task
authorization.

At the user's follow-up request, the actual `datasets/Precise/` folder was
checked directly: it contains 7,358 JPEGs, zero non-image files and zero nested
directories. Filename parsing identifies 338 subjects, 2,452 right-slap images,
2,455 left-slap images and 2,451 thumb-slap images. There is no score table,
training/testing pair list or separate acquisition-metadata file in that folder.
Filename subject/slap labels do not establish session meanings, independent
captures, physical PPI or which comparisons Konain used. The image collection
and his experiment records are separate inputs.

Konain's latest message confirms that VeriFinger scores provide fine-tuning
supervision. His exact target format and score choice remain unknown. The
linked [FPBench reference at the inspected revision](https://github.com/Ektagavas/FPBench/blob/ed27ae669947bfd157522d537a9f9c77b4bf3189/finetuning/ft_qwen3.py)
uses Qwen3-VL LoRA with supervised answer-text training and defaults to pattern
classification. It does not show his Precise score conversion, subject split or
an embedding/contrastive matching objective. The reference was read, not run.

## Ready-to-send request

> Thanks, Konain. I checked the local files: the score lists I have are the
> older 676 genuine and 676 impostor Qwen3-VL exports. Could you point me to
> the 2,100 genuine and 2,100 impostor VeriFinger testing files, including fused
> and summed scores with their image-pair mappings, and share the training pair
> list when ready? A pair list with image mappings will let us check shared
> images and subjects; a total count alone cannot establish that. We can keep
> these identifying files in private storage while auditing them.

This is a draft for the user; no external message was sent.

## Next audit when those files arrive

1. Preserve source bytes and record provenance, score type, schemas and counts.
2. Validate image mappings, genuine/impostor labels, slap type, duplicate/reversed
   pairs and score/error coverage without modifying the submitted records.
3. Compare training/testing image and subject populations. Exact-pair exclusion
   alone does not establish image- or subject-disjoint evaluation.
4. Recompute descriptive SDK ROC/DET and counts from the supplied scores, and
   distinguish test-selected operating points from development-fixed thresholds.
5. Review Konain's adapted script/example targets and settle the first milestone
   and prospective protocol with the professor before authorized experiments.

No unavailable scores were reconstructed. No new matching pairs, subject
assignments, segmentation, minutiae, training or model inference were performed.

Aggregate counts are in [discovery.json](discovery.json). Source paths, keyword
and schema inventories, source hashes and detailed checks are retained only in
ignored `local/precise/metadata/discovery-20261011/`. The existing
[protocol sketch](../../docs/notes/precise_protocol_sketch_20261010.md) remains
proposed and the [research brief](../../docs/notes/precise_research_brief_20261010.md)
remains the discussion entry point.
