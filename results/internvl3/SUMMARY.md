# InternVL3-8B-Instruct — Results Summary

**Model:** InternVL3-8B-Instruct  
**Precision:** bfloat16 (~16 GB VRAM)  
**Pairs:** 7,832 total (176 genuine + 7,656 impostor — all C(88,2) combinations)  
**Preprocessing:** grayscale → autocontrast → square-pad (white) → 448×448 → RGB  
**Run date:** 2026-06-14  
**transformers:** 4.57.6 | **torch:** 2.5.1+cu124

---

## Zero-Shot (ZS)

| Metric | Value |
|---|---|
| Overall Accuracy | 29.3% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 27.7% |
| FAR | 72.3% |
| FRR | 0.0% |
| Collapsed | **No** |
| Answer dist | A: 72.9%, B: 27.1% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 28.8% | 72.9% | 0.0% | 3,916 |
| 14 | Left  | 29.8% | 71.8% | 0.0% | 3,916 |

---

## Task Description (TD)

| Metric | Value |
|---|---|
| Overall Accuracy | 2.2% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 0.0% |
| FAR | 100.0% |
| FRR | 0.0% |
| Collapsed | **Yes** (all A) |
| Answer dist | A: 100% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 2.2% | 100.0% | 0.0% | 3,916 |
| 14 | Left  | 2.2% | 100.0% | 0.0% | 3,916 |

---

## Similarity Score (SS)

| Metric | Value |
|---|---|
| Genuine Mean Score | 57.7 |
| Impostor Mean Score | 79.1 |
| Score Separation (Δ) | **-21.4 pts (INVERTED)** |
| Best Threshold | 86 |
| Best Accuracy | 98.4%* |
| EER | 48.09% |
| EER Threshold | 76 |
| AUC | 0.5895 |
| TAR @ FAR=0.1% | 0.0% |
| Invalid | 0 |

*Best accuracy of 98.4% is misleading — achieved by rejecting nearly all pairs at threshold 86. AUC and EER are the honest indicators.

**Per-FRGP:**

| FRGP | Hand | Genuine Mean | Impostor Mean | Δ | n |
|---|---|---|---|---|---|
| 13 | Right | 55.5 | 79.7 | -24.2 | 3,916 |
| 14 | Left  | 60.0 | 78.5 | -18.5 | 3,916 |

---

## Summary Table

| Prompt | Acc | FAR | FRR | AUC | EER | TAR@FAR=0.1% | Collapsed |
|---|---|---|---|---|---|---|---|
| Zero-Shot     | 29.3% | 72.3% | 0.0% | — | — | — | No |
| Task Desc.    | 2.2%  | 100%  | 0.0% | — | — | — | Yes |
| Similarity SS | 98.4%* | — | — | 0.5895 | 48.09% | 0.0% | No |

---

## Notes

- ZS: Not collapsed, but high FAR (72.3%). Genuine accuracy 100% — never rejects a genuine pair. Per-FRGP is nearly symmetric (FRGP13 72.9% FAR, FRGP14 71.8% FAR).
- TD: Full collapse to 100% FAR — consistent with all other models under task description.
- SS: **Inverted calibration** — assigns higher scores to impostor pairs (79.1%) than genuine pairs (57.7%). AUC=0.589 ≈ random, EER=48.09% ≈ random. This was hidden in the 352-pair run (AUC appeared as 0.823). The exhaustive 7,656-impostor run exposes that InternVL3 similarity scoring is not reliably calibrated for fingerprint verification.
