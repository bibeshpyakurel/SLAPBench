# Qwen3-VL-8B-Instruct — Results Summary

**Model:** Qwen3-VL-8B-Instruct  
**Precision:** 4-bit NF4 quantization (bitsandbytes) (~6 GB VRAM)  
**Pairs:** 352 total (176 genuine × 176 impostor)  
**Preprocessing:** grayscale → autocontrast → square-pad (white) → 448×448 → RGB  
**Run date:** 2026-05-24  
**transformers:** 5.9.0 | **torch:** 2.5.1+cu121

---

## Zero-Shot (ZS)

| Metric | Value |
|---|---|
| Overall Accuracy | 73.3% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 46.6% |
| FAR | 53.4% |
| FRR | 0.0% |
| Collapsed | **No** |
| Answer dist | A: 76.7%, B: 23.3% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 79.5% | 40.9% | 0.0% | 176 |
| 14 | Left  | 67.0% | 65.9% | 0.0% | 176 |

---

## Task Description (TD)

| Metric | Value |
|---|---|
| Overall Accuracy | 50.0% |
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
| 13 | Right | 50.0% | 100.0% | 0.0% | 176 |
| 14 | Left  | 50.0% | 100.0% | 0.0% | 176 |

---

## Similarity Score (SS)

| Metric | Value |
|---|---|
| Genuine Mean Score | 96.4 |
| Impostor Mean Score | 65.3 |
| Score Separation (Δ) | 31.1 pts |
| Best Threshold | 76 |
| Best Accuracy | 94.9% |
| EER | 5.11% |
| EER Threshold | 76 |
| AUC | 0.9904 |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Genuine Mean | Impostor Mean | n |
|---|---|---|---|---|
| 13 | Right | 95.9 | 64.1 | 176 |
| 14 | Left  | 96.8 | 66.5 | 176 |

---

## Summary Table

| Prompt | Acc | FAR | FRR | AUC | EER | Collapsed |
|---|---|---|---|---|---|---|
| Zero-Shot     | 73.3% | 53.4% | 0.0% | — | — | No |
| Task Desc.    | 50.0% | 100%  | 0.0% | — | — | Yes |
| Similarity SS | 94.9% (best) | — | — | 0.9904 | 5.11% | No |

---

## Notes

- ZS is the best binary result across all three models: 73.3% accuracy, FAR 53.4%, not collapsed. FRGP 13 (right, 79.5%) significantly outperforms FRGP 14 (left, 67.0%) — opposite direction from InternVL3.
- TD collapses to 100% FAR — consistent with all other models.
- SS is outstanding: AUC 0.9904, EER 5.11%, 31.1-point genuine/impostor separation. Genuine mean 96.4 vs impostor mean 65.3 — the model clearly distinguishes same-person vs different-person pairs. Best SS result across all models by a wide margin.
