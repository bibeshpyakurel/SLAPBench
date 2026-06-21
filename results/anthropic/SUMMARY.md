# Claude Opus 4.8 — Results Summary

**Model:** claude-opus-4-8 (Anthropic, proprietary)
**Access:** Anthropic API (`--model anthropic --anthropic-model claude-opus-4-8`)
**Pairs:** 7,832 total (176 genuine + 7,656 impostor — all C(88,2) combinations)
**Preprocessing:** grayscale → autocontrast → square-pad (white) → 448×448 → RGB → base64 PNG
**Run date:** 2026-06-20 / 2026-06-21
**Notes:** Images sent as base64 PNG; exponential backoff handles API rate limits.

---

## Zero-Shot (ZS)

| Metric | Value |
|---|---|
| Overall Accuracy | 80.3% |
| Genuine Accuracy (TAR) | 99.4% |
| Impostor Accuracy (TNR) | 79.8% |
| FAR | 20.2% |
| FRR | 0.57% |
| Collapsed | **No** |
| Answer dist | A: 22.0%, B: 78.0% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 82.6% | 17.8% | 1.14% | 3,916 |
| 14 | Left  | 78.0% | 22.5% | 0.0%  | 3,916 |

---

## Task Description (TD)

| Metric | Value |
|---|---|
| Overall Accuracy | 50.2% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 49.1% |
| FAR | 50.9% |
| FRR | 0.0% |
| Collapsed | **No** |
| Answer dist | A: 52.0%, B: 48.0% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 52.0% | 49.1% | 0.0% | 3,916 |
| 14 | Left  | 48.5% | 52.7% | 0.0% | 3,916 |

---

## Similarity Score (SS)

| Metric | Value |
|---|---|
| Genuine Mean Score | 91.3 |
| Impostor Mean Score | 48.4 |
| Score Separation (Δ) | +42.9 pts |
| Best Threshold | 86 |
| Best Accuracy | 99.5% |
| EER | 11.75% |
| EER Threshold | 71 |
| AUC | 0.9533 |
| TAR @ FAR=0.1% | 56.8% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Genuine Mean | Impostor Mean | Δ | n |
|---|---|---|---|---|---|
| 13 | Right | 89.8 | 48.4 | +41.4 | 3,916 |
| 14 | Left  | 92.8 | 48.4 | +44.4 | 3,916 |

---

## Summary Table

| Prompt | Acc | FAR | FRR | AUC | EER | TAR@FAR=0.1% | Collapsed |
|---|---|---|---|---|---|---|---|
| Zero-Shot     | 80.3% | 20.2% | 0.57% | — | — | — | No |
| Task Desc.    | 50.2% | 50.9% | 0.0%  | — | — | — | No |
| Similarity SS | 99.5% (best) | — | — | 0.9533 | 11.75% | 56.8% | No |

---

## Notes

- **Unique among all models:** Claude Opus 4.8 is the only model that resists positive-bias collapse under *both* binary prompts. Every open-source model collapses to ≥96.4% FAR under task-description prompting; Claude stays balanced (TD FAR 50.9%, answering "same person" on only 52% of pairs).
- **Best binary verifier overall:** zero-shot FAR 20.2% is the lowest of any model under either prompt (beats Qwen3-VL's 26.5%).
- **Second-best similarity scoring:** AUC 0.953, EER 11.75%, behind only Qwen3-VL-8B (AUC 1.0) and ahead of Gemma-3-12B (0.837). Genuine mean 91.3 vs impostor mean 48.4 — the widest genuine/impostor separation (Δ = +42.9) of any model.
- Per-hand behavior is mildly asymmetric under zero-shot (FRGP 13 FAR 17.8% vs FRGP 14 FAR 22.5%) and symmetric under scoring.
