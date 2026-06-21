# Gemma-3-12B-IT — Results Summary

**Model:** Gemma-3-12B-IT  
**Precision:** 4-bit NF4 quantization (bitsandbytes) (~8 GB VRAM)  
**Pairs:** 7,832 total (176 genuine + 7,656 impostor — all C(88,2) combinations)  
**Preprocessing:** grayscale → autocontrast → square-pad (white) → 448×448 → RGB  
**Run date:** 2026-06-20  
**transformers:** 4.57.6 | **torch:** 2.5.1+cu124

---

## Zero-Shot (ZS)

| Metric | Value |
|---|---|
| Overall Accuracy | 5.7% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 3.6% |
| FAR | 96.4% |
| FRR | 0.0% |
| Collapsed | **Yes** |
| Answer dist | A: 96.5%, B: 3.5% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 4.4% | 97.8% | 0.0% | 3,916 |
| 14 | Left  | 7.1% | 95.1% | 0.0% | 3,916 |

---

## Task Description (TD)

| Metric | Value |
|---|---|
| Overall Accuracy | 2.3% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 0.0% |
| FAR | 100.0% |
| FRR | 0.0% |
| Collapsed | **Yes** (near-all A) |
| Answer dist | A: 99.97%, B: 0.03% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 2.2% | 100.0% | 0.0% | 3,916 |
| 14 | Left  | 2.3% | 99.9%  | 0.0% | 3,916 |

---

## Similarity Score (SS)

| Metric | Value |
|---|---|
| Genuine Mean Score | 75.8 |
| Impostor Mean Score | 63.3 |
| Score Separation (Δ) | +12.5 pts |
| Best Threshold | 76 |
| Best Accuracy | 98.3% |
| EER | 15.1% |
| EER Threshold | 66 |
| AUC | 0.8372 |
| TAR @ FAR=0.1% | 23.3% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Genuine Mean | Impostor Mean | Δ | n |
|---|---|---|---|---|---|
| 13 | Right | 75.2 | 63.3 | +11.9 | 3,916 |
| 14 | Left  | 76.4 | 63.3 | +13.1 | 3,916 |

---

## Summary Table

| Prompt | Acc | FAR | FRR | AUC | EER | TAR@FAR=0.1% | Collapsed |
|---|---|---|---|---|---|---|---|
| Zero-Shot     | 5.7%  | 96.4% | 0.0% | — | — | — | Yes |
| Task Desc.    | 2.3%  | 100%  | 0.0% | — | — | — | Yes |
| Similarity SS | 98.3% (best) | — | — | 0.8372 | 15.1% | 23.3% | No |

---

## Notes

- ZS collapses to 96.4% FAR — Gemma-3 answers "same person" (A) on 96.5% of all pairs, accepting nearly every impostor. Unlike Qwen3-VL and InternVL3, Gemma-3 does not remain functional under zero-shot binary prompting.
- TD collapses further to ~100% FAR — consistent with all other models. Domain context pushes the model to answer "same" almost universally.
- SS is where Gemma-3 becomes useful: AUC=0.8372, EER=15.1%, with a 12.5-point genuine/impostor separation (genuine mean 75.8 vs impostor mean 63.3). This is the **second-best similarity-scoring result** across all four local models, behind Qwen3-VL (AUC=1.0) and well ahead of InternVL3 (AUC=0.589) and Qwen2.5-VL (AUC=0.567).
- The best-accuracy figure (98.3%) is inflated by the 43:1 impostor-to-genuine class imbalance; AUC, EER, and TAR@FAR=0.1% (23.3%) are the honest measures.
- Per-FRGP behavior is symmetric: both hands show ~12-pt separation with near-identical impostor means (63.3).
