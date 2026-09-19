# Qwen2.5-VL-7B-Instruct — Results Summary

**Model:** Qwen2.5-VL-7B-Instruct  
**Precision:** 4-bit NF4 quantization (bitsandbytes) (~6 GB VRAM)  
**Pairs:** 7,832 total (176 genuine + 7,656 impostor — all C(88,2) combinations)  
**Preprocessing:** grayscale → autocontrast → square-pad (white) → 448×448 → RGB  
**Run date:** 2026-06-14  
**transformers:** 4.57.6 | **torch:** 2.5.1+cu124

---

## Zero-Shot (ZS)

| Metric | Value |
|---|---|
| Overall Accuracy | 32.0% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 30.4% |
| FAR | 69.6% |
| FRR | 0.0% |
| Collapsed | **No** |
| Answer dist | A: 70.3%, B: 29.7% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 30.2% | 71.4% | 0.0% | 3,916 |
| 14 | Left  | 33.7% | 67.8% | 0.0% | 3,916 |

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
| Genuine Mean Score | 95.0 |
| Impostor Mean Score | 90.1 |
| Score Separation (Δ) | +4.9 pts |
| Best Threshold | 96 |
| Best Accuracy | 97.8%* |
| EER | 43.34% |
| EER Threshold | 76 |
| AUC | 0.5666 |
| TAR @ FAR=0.1% | 0.0% |
| Invalid | 0 |

*Best accuracy of 97.8% is misleading — achieved by high threshold that rejects most pairs. AUC and EER are the honest indicators.

**Per-FRGP:**

| FRGP | Hand | Genuine Mean | Impostor Mean | Δ | n |
|---|---|---|---|---|---|
| 13 | Right | 95.0 | 90.2 | +4.8 | 3,916 |
| 14 | Left  | 95.0 | 90.0 | +5.0 | 3,916 |

---

## Summary Table

| Prompt | Acc | FAR | FRR | AUC | EER | TAR@FAR=0.1% | Collapsed |
|---|---|---|---|---|---|---|---|
| Zero-Shot     | 32.0% | 69.6% | 0.0% | — | — | — | No |
| Task Desc.    | 2.2%  | 100%  | 0.0% | — | — | — | Yes |
| Similarity SS | 97.8%* | — | — | 0.5666 | 43.34% | 0.0% | No |

---

## Notes

- ZS: Not collapsed in the exhaustive run (FAR 69.6%). This is a significant change from the 352-pair run, where this model was marked as collapsed (90.9% FAR). With 7,656 impostors, the model shows some discrimination — it is not answering identically for all pairs. FRGP 14 (left, 67.8% FAR) is slightly better than FRGP 13 (right, 71.4% FAR).
- TD: Full collapse to 100% FAR — model answers "A" (same person) for every single pair.
- SS: Low discrimination. Genuine mean (95.0%) and impostor mean (90.1%) are both compressed into the high end of the scale, leaving only 4.9 points of separation. AUC=0.567 ≈ random, EER=43.34% ≈ random. Both are consistent with a model that is nearly certain all SLAP pairs are the same person regardless of identity.
