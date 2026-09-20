# Saved-response audit — 2026-09-19

Offline analysis of 45 prompted CSVs / 165,087 rows. Original inputs are unchanged;
all 49 source files are fingerprinted in `audit.json`. See `source_sha256` and
`audit_script_sha256` for exact inputs and implementation. No inference/API calls occurred.

## Qwen3 exports: PASS

Both 676-line exports match the manifest image basenames, row order and stored
numeric scores exactly; all scores also match their raw numeric responses.
The two files contain 1,352 unique exported tuples. This establishes saved-file
consistency, not biological correctness, dataset independence or model calibration.

The unmatched score ID at logical CSV data record 1600 has **612 leading NUL
bytes**, followed by the exact manifest ID `G_146_14_10629`. Removing only that
prefix gives an exact ID and label/category/subject/hand match; no guessed identity
is needed. The normalization is confined to derived records. The original exporter
uses pandas and includes a missing-ID recovery step; the audit instead reads raw
CSV bytes and checks the surviving exact ID. The one absent diagnostic comparison
is excluded from both primary export files. Source interruption/storage corruption
is plausible, but the cause of the NUL bytes is not established by saved files.

## Response parsing and root causes

**Confirmed parsing defect:** the production binary parser tests whether A or B
occurs anywhere in the response. It turns “No, different individuals” into A
because INDIVIDUALS contains A. The conservative reparse finds 30 unambiguous A→B
changes, all in Pixtral: 14 primary impostors, 7 diagnostic impostors, 9 genuine
pairs. The genuine changes do not improve accuracy; parsing must follow the reply,
not the ground-truth label. Other withheld replies are NOT declared misread by
human ground truth; they need review.

**Confirmed output-format failure:** Pixtral has 688 missing Precise scores and
604 missing RidgeBase scores. Every stored Pixtral score/blank agrees with the
current historical integer regex applied to its raw text. Thus missing CSV values
are explained at the parser boundary; no unambiguous numeric score can be recovered
from these missing rows under the declared conservative grammar. They include
empty outputs, prompt echoes, unrelated text, uncertainty and fragments with
digits embedded in words or numbers too long for the old regex. They are not all
provider refusals. Generation stop reasons/token counts were not saved, so upstream
causes such as truncation, model/template compatibility or decoding cannot be isolated.

| Dataset | Missing stored scores | Empty | No numeric score | Uncertainty/refusal candidates | Other numeric prose | Clear scores retained from all rows |
|---|---:|---:|---:|---:|---:|---:|
| precise | 688 | 41 | 584 | 43 | 20 | 145/1690 |
| ridgebase | 604 | 24 | 508 | 57 | 15 | 105/1484 |

**Confirmed model-output problem beyond parsing:** Qwen3 task-description outputs
accept 671/676 Precise and 586/592 RidgeBase primary impostors, with 100% valid
output coverage under the conservative grammar. SD302b accepts 7,571/7,656.
The saved explicit answers themselves produce these high acceptance rates; the
letter-search parser does not explain them. The current pipeline provides images
and language prompts but no conventional minutiae extraction/registration. Whether
minutiae grounding solves the behavior remains hypothesis H2 (§1 of the supervisor
instructions), not a conclusion of this audit.

**Hypotheses requiring controlled development experiments:** 448-pixel preprocessing
may remove local ridge detail; global appearance may dominate; prompting/decision
bias and uncalibrated scores may cause false acceptance. Pixtral applies a chat
template, greedy generation capped at 96 tokens, repetition penalty 1.3 and
no-repeat trigram handling (`code/run_verification.py:_infer_pixtral`). Current code
is evidence about implementation, not a complete historical runtime record.
Template/image wiring, quantization and decoding need controlled tests before
attributing malformed outputs to any one cause. No such runs were performed.

## Derived analysis policy

`conservative-v1` accepts full-string A/B, a small documented set of unambiguous
yes/no forms, and full-string numeric scores (optionally labeled or /100). It
rejects out-of-range values rather than clamping, does not extract numbers from
arbitrary prose, and withholds ambiguous/contradictory/uncertain responses.
It does not use the true label. It is an explicitly post-hoc sensitivity analysis,
not a validated universal language parser or a retrospective improvement in model skill.

`parsed/` contains new per-row interpretations, original parsed value, raw-text
hash, source record number and ID-normalization status. Read the matching original
CSV for full raw text; record numbers count CSV data records, not physical lines.
`audit.json` contains per-file coverage, status counts, selected examples, primary
vs diagnostic binary rates and available-score AUC. Both acceptance over all rows
and acceptance conditional on valid responses are shown. Neither hides abstentions.
AUC after exclusions can be selection-biased, especially for Pixtral; do not
compare it as if every method evaluated identical effective pairs. No threshold
calibration, ROC/DET publication claim or model rerun is implied.

## Reproduce

```bash
python code/audit_saved_responses.py --output-dir reports/NEW_UNUSED_AUDIT_DIRECTORY
python -m unittest discover -s tests -v
```

The output directory must not exist. The script verifies source hashes again
after reading and refuses to overwrite a previous report. The inference parsers
and original result files were left unchanged to preserve historical provenance.
Future pilot runs should use the fixed JSON schema and validation proposed in
`docs/EVALUATION_PROTOCOL_DRAFT.md` (§§6,11), once the run protocol is approved.
