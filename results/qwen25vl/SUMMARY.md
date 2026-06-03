# Qwen2.5-VL-7B-Instruct — Results Summary

**Model:** Qwen2.5-VL-7B-Instruct  
**Precision:** 4-bit NF4 quantization (bitsandbytes) (~6 GB VRAM)  
**Pairs:** 352 total (176 genuine × 176 impostor)  
**Preprocessing:** grayscale → autocontrast → square-pad (white) → 448×448 → RGB  
**Run date:** 2026-05-24

---

## Zero-Shot (ZS)

| Metric | Value |
|---|---|
| Overall Accuracy | 54.5% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 9.1% |
| FAR | 90.9% |
| FRR | 0.0% |
| Collapsed | **Yes** |
| Answer dist | A: 95.5%, B: 4.5% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 55.7% | 88.6% | 0.0% | 176 |
| 14 | Left  | 53.4% | 93.2% | 0.0% | 176 |

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
| Genuine Mean Score | 95.0 |
| Impostor Mean Score | 91.8 |
| Score Separation (Δ) | 3.2 pts |
| Best Threshold | 91 |
| Best Accuracy | 77.6% |
| EER | 22.44% |
| EER Threshold | 91 |
| AUC | 0.7756 |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Genuine Mean | Impostor Mean | n |
|---|---|---|---|---|
| 13 | Right | 95.0 | 93.5 | 176 |
| 14 | Left  | 95.0 | 90.1 | 176 |

---

## Summary Table

| Prompt | Acc | FAR | FRR | AUC | EER | Collapsed |
|---|---|---|---|---|---|---|
| Zero-Shot     | 54.5% | 90.9% | 0.0% | — | — | Yes |
| Task Desc.    | 50.0% | 100%  | 0.0% | — | — | Yes |
| Similarity SS | 77.6% (best) | — | — | 0.7756 | 22.44% | No |

---

## Notes

- All four binary runs (both models, both strategies) now collapse. Qwen collapses under ZS and TD; InternVL3 collapses under TD only.
- SS score scale is compressed high: genuine mean 95.0, impostor mean 91.8 — only 3.2-point separation. The model is over-confident across all pairs regardless of identity.
- FRGP 14 impostor mean (90.1) is lower than FRGP 13 (93.5), giving slightly better separation on left-hand pairs.
- AUC 0.776 is weaker than InternVL3 (0.823), reversing the previous relationship. The prior run (without autocontrast) showed AUC 0.960 with genuine mean 89.7 / impostor mean 84.6. The autocontrast preprocessing appears to have pushed Qwen toward high uniform scores, compressing the distribution and reducing discriminability.
