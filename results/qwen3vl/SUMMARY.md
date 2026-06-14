# Qwen3-VL-8B-Instruct — Results Summary

**Model:** Qwen3-VL-8B-Instruct  
**Precision:** 4-bit NF4 quantization (bitsandbytes) (~6 GB VRAM)  
**Pairs:** 7,832 total (176 genuine + 7,656 impostor — all C(88,2) combinations)  
**Preprocessing:** grayscale → autocontrast → square-pad (white) → 448×448 → RGB  
**Run date:** 2026-06-13  
**transformers:** 4.57.6 | **torch:** 2.5.1+cu124

---

## Zero-Shot (ZS)

| Metric | Value |
|---|---|
| Overall Accuracy | 74.1% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 73.5% |
| FAR | 26.5% |
| FRR | 0.0% |
| Collapsed | **No** |
| Answer dist | A: 28.1%, B: 71.9% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 71.7% | 29.0% | 0.0% | 3,916 |
| 14 | Left  | 76.5% | 24.0% | 0.0% | 3,916 |

---

## Task Description (TD)

| Metric | Value |
|---|---|
| Overall Accuracy | 3.3% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 1.1% |
| FAR | 98.9% |
| FRR | 0.0% |
| Collapsed | **Yes** (all A) |
| Answer dist | A: 98.9%, B: 1.1% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 2.5% | 99.7% | 0.0% | 3,916 |
| 14 | Left  | 4.2% | 98.0% | 0.0% | 3,916 |

---

## Similarity Score (SS)

| Metric | Value |
|---|---|
| Genuine Mean Score | 100.0 |
| Impostor Mean Score | 63.8 |
| Score Separation (Δ) | +36.2 pts |
| Best Threshold | 96 |
| Best Accuracy | 100.0% |
| EER | 0.0% |
| EER Threshold | 96 |
| AUC | 1.0000 |
| TAR @ FAR=0.1% | 100.0% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Genuine Mean | Impostor Mean | Δ | n |
|---|---|---|---|---|---|
| 13 | Right | 100.0 | 65.6 | +34.4 | 3,916 |
| 14 | Left  | 100.0 | 61.9 | +38.1 | 3,916 |

---

## Summary Table

| Prompt | Acc | FAR | FRR | AUC | EER | TAR@FAR=0.1% | Collapsed |
|---|---|---|---|---|---|---|---|
| Zero-Shot     | 74.1% | 26.5% | 0.0% | — | — | — | No |
| Task Desc.    | 3.3%  | 98.9% | 0.0% | — | — | — | Yes |
| Similarity SS | 100.0% | — | — | 1.0000 | 0.0% | 100.0% | No |

---

## Notes

- ZS: Best binary result across all three models. FAR dropped from 53.4% (352-pair run) to 26.5% with exhaustive impostors. FRGP 14 (left, 76.5%) slightly outperforms FRGP 13 (right, 71.7%).
- TD: Collapses to near-100% FAR — consistent with all other models. Domain context increases collapse.
- SS: Perfect discrimination. Every genuine pair scores 100; impostor mean is 63.8. AUC=1.0, EER=0.0%, TAR@FAR=0.1%=100%. Best SS result across all models by a wide margin. Notably improved from the 352-pair run (AUC was 0.990, EER 5.11%).
