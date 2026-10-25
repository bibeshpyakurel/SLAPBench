# FPBench reference and Konain intake — 2026-10-24

Konain directed the user to FPBench's Qwen3 fine-tuning script and reports that
he is currently fine-tuning Qwen3 for a binary matching task. Preserve that as
collaborator-reported progress; no resulting checkpoint or binary-target adapter
was supplied or executed in this intake.

## Pinned local reference

- Upstream: <https://github.com/Ektagavas/FPBench>
- Revision: `ed27ae669947bfd157522d537a9f9c77b4bf3189` (upstream `main` observed
  on 2026-10-24; the same revision inspected during the earlier audit).
- Local checkout: `local/precise/references/FPBench/`, detached at that revision.
- [Pinned Qwen3 entry point](https://github.com/Ektagavas/FPBench/blob/ed27ae669947bfd157522d537a9f9c77b4bf3189/finetuning/ft_qwen3.py).
- Entry-point SHA-256: `538546a9cf6ac069ca70929fb720db42074f48d6a2f48e5a82f6db47fc178ade`.
- Local receipt: `local/precise/references/FPBench-provenance-20261024.json`.

The complete checked-out repository remains ignored, including its own upstream
demo images. The public SLAPBench repository records links and provenance rather
than republishing upstream assets. No standalone LICENSE file was found at this
revision; the upstream README describes academic/research usage and requests
citation. Review those terms before redistribution or incorporation. Our project's
license does not automatically apply to this third-party reference.

To restore on another authorized machine, when the destination does not exist:

```bash
mkdir -p local/precise/references
git clone https://github.com/Ektagavas/FPBench.git local/precise/references/FPBench
git -C local/precise/references/FPBench checkout --detach ed27ae669947bfd157522d537a9f9c77b4bf3189
git -C local/precise/references/FPBench rev-parse HEAD
```

Keep this checkout unchanged as a reference. Future adaptations belong in our
own reviewed code with attribution and an explicit environment. Reading it does
not require running its scripts or installing its dependencies. Do not copy its
files into the project root as the upstream VLMEvalKit tutorial suggests.

## Useful entry points and observed behavior

| Reference file | What it provides |
|---|---|
| `README.md` | Benchmark scope, dataset sources, setup context and citations |
| `finetuning/ft_qwen3.py` | Unsloth/TRL supervised answer-text fine-tuning with LoRA and optional vision/language adaptation |
| `finetuning/ft_gemma3.py` | Parallel Gemma fine-tuning reference |
| `finetuning/ft_json/` | Upstream task question/answer schemas; not Konain's Precise training conversion |
| `evaluate_qwen3.py`, `aggregate_results.py` | Prompted evaluation and answer aggregation references |
| `misc/` | Minutiae/orientation overlay reference utilities |
| `requirements.yaml` | Upstream environment reference; not a validated environment for this project |

Static inspection of the pinned Qwen script establishes:

1. It constructs assistant answer-text targets and uses `SFTTrainer`, rather
   than a contrastive/metric-learning verification objective.
2. The active default task is `pattern_mixed_all_data.json`; the matching task
   is commented out. Its supplied matching JSON has 1,800 questions, four
   answer options per question and three or four images per question. This is
   not the supplied Precise two-image binary pair format.
3. It loads `unsloth/Qwen3-VL-8B-Instruct` directly. Although a `--model` argument
   exists, that argument does not select the model passed to the loader.
4. It reads relative `./ft_json` paths, invokes CUDA inspection and training,
   and calls `Image.open` without importing `Image` in this file. The unmodified
   entry point is not a verified runnable Precise training command.
5. There is no explicit development dataset or subject-disjoint partition
   construction in the script. It contains no Precise TXT parser, VeriFinger
   score-to-target conversion or Konain-specific preprocessing rule.

These findings describe the published reference, not defects proven in
Konain's private adaptation. No upstream file was patched, imported or run.
Its useful patterns can inform a later separately reviewed binary VLM arm;
the approved whole-slap embedding-matcher milestone remains unchanged.

## What the new collaborator files resolve

The [training/testing intake audit](../../reports/precise-training-intake-20261024/README.md)
now establishes the supplied training populations and exact-pair exclusion.
There are 23,103 genuine and 78,592 impostor training records, with no exact
testing-pair overlap. All testing image basenames and subjects appear elsewhere
in the training lists. The lists are therefore not an implementation of our
subject-disjoint protocol. They do not establish which records were actually
consumed by Konain's current binary fine-tuning job.

The highest-value remaining reproduction details are his adapted script or a
small redacted two-image training example, the binary target rule (identity
class versus a VeriFinger cutoff), development/checkpoint selection, whether
left-hand negative pairs are supplied elsewhere, and the meaning of genuine
zero-score rows without component fields. Capture definitions remain a separate
data question. We no longer need to ask for the two training lists just received.

No message was sent to the collaborator and no model experiment was launched.
