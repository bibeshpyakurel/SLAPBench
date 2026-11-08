# Supplied Qwen results: exact common-pair outcome review

2026-11-08. The user supplied Konain's new evaluation and fine-tuning folders,
including both summaries, four complete parsed prediction files, updated
matched evaluation code, the actual adapter checkpoint and training metadata.
These files now replace the earlier missing-output status. No Qwen inference
or duplicate fine-tuning was performed here.

The preserved originals are under ignored
`local/precise/collaborator/konain-results-20261108/`. The source audit at
[aggregate.json](aggregate.json) verifies all 4,200 unordered pairs and labels
against the completed November 7 DINO bundle, with no missing, extra, repeated
or intersection-filtered rows. All four Qwen files contain only parsed A/B
outcomes, with zero INVALID/ERROR tokens. The supplied source summaries reproduce
exactly. This verifies outcome membership and arithmetic; the actual-run
provenance limitations below remain unresolved.

## Results on the same supplied 4,200 attempts

Positive means a genuine match. Each class has 2,100 pairs, including 700 per
left/right/thumb type. Standard and balanced accuracy therefore coincide.
DINO cutoffs remain frozen from training impostors; Qwen uses native A/B outputs.

| Method | TP | TN | FP | FN | Accuracy | TAR | FAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen DINOv2-small, 448 | 730 | 2,080 | 20 | 1,370 | 66.90% | 34.76% | 0.95% |
| Adapted DINOv2-small, 448 | 727 | 2,095 | 5 | 1,373 | 67.19% | 34.62% | 0.24% |
| Qwen base, supplied | 1,927 | 732 | 1,368 | 173 | 63.31% | 91.76% | 65.14% |
| Qwen fine-tuned, supplied | 2,040 | 1,726 | 374 | 60 | 89.67% | 97.14% | 17.81% |

The Qwen adaptation increases observed accuracy by 26.36 percentage points and
reduces false accepts from 1,368 to 374. Its remaining FAR is 17.81%, despite
high aggregate accuracy. DINO's much stricter operating policy rejects more
impostors and more genuine attempts. This is a descriptive comparison of native
operating points, **not matched test FAR**, a significance claim or a conclusion
that one language/vision component causes the differences. Binary Qwen outputs
supply no confidence scores and no meaningful continuous-score ROC/AUC/EER.

Per-type results are in [operating_points.csv](operating_points.csv). Paired
DINO-to-fine-tuned-Qwen counts are in the aggregate. All methods use the supplied
labels and exact same test membership; rows reuse images/groups, so no
independent-pair confidence intervals or biological subject/capture independence
are claimed. The [completed DINO study](../precise-dinov2-numeric-holdout-20261107/README.md)
and original November 1 study remain unchanged.

## What the new source evidence establishes

- The newly supplied training JSON is byte-identical to the 37,194-pair JSON
  bound to the DINO bundle. The training script is byte-identical to the earlier
  supplied fine-tuning script.
- Both summaries carry the exact same system/user prompt as that JSON. Their
  own prompt hash convention (a JSON list of the two strings) reproduces.
- The supplied evaluation script uses native loaded RGB images, the model's
  processor, greedy generation, five new tokens, four-bit Unsloth loading and
  strict A/B parsing. It does not manually resize images; its processor does
  resize. The saved adapter processor configuration explicitly has
  `do_resize: true`. It cannot be described as untouched pixels at the network.
- The adapter contains 324 FP32 tensors / 33,025,536 parameters. Its SHA-256 is
  `56c79b37af8abe1a5b290c7cc990b5e22b4824ccd6ea945af7721e253ba6d0b7`.
  Consult the aggregate/source receipts for the authoritative digest; no weights
  or raw pair outputs are published.
- `trainer_state.json` records final epoch 3, step/max-step 27,897 and no best
  checkpoint. Static inspection of `training_args.bin` (without unpickling or
  executing it) records batch size one, accumulation four, three epochs, LR
  1e-4, warmup five, seed 3407 and BF16. That step count is consistent with three
  complete epochs of 37,194 examples at accumulation four, but is not itself a
  consumed-record manifest.

## Remaining provenance limits

The evaluation summary references `binary_training_config.json`, but that file
is absent from the supplied folders/checkpoint. Resolved base-model revisions,
actual package/runtime records and raw generated-response logs are also absent.
The base summary names `unsloth/Qwen3-VL-8B-Instruct`; the adapter names
`unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit`, with a null revision. Exact
resolved base-byte equivalence is therefore unconfirmed.

The adapter is hashed upon intake, but the evaluation summaries do not bind its
hash at execution time. The supplied final state is consistent with final-epoch
selection; an actual selection rule/history and confirmation that test results
were not used for selection are still needed. Paths in summaries and source-code
defaults do not establish these missing facts. No approval, environment version,
base revision, raw reply or model-consumption receipt is invented to fill them.
The earlier strict fully reviewed comparison CLI retains these requirements.

## Reproduce offline

```bash
local/precise/environments/verification-compat-20261107/bin/python code/audit_precise_qwen_results.py \
  --intake local/precise/collaborator/konain-results-20261108 \
  --bundle local/precise/manifests/numeric-group-holdout-20261107/bundle.json \
  --config code/precise_matching/configs/dinov2_numeric_group_holdout.json \
  --dino-run-root local/precise/runs/numeric-group-holdout-20261107 \
  --output reports/precise-qwen-supplied-results-NEWDATE/aggregate.json \
  --private-receipt local/precise/metadata/qwen-results-NEWDATE/audit-receipt.json
```

The exporter refuses overwrite, checks input hashes before/after processing,
replays DINO calibration/test outcomes and preserves all original bytes. Public
files contain only aggregate outcomes, generic source names and digests.
