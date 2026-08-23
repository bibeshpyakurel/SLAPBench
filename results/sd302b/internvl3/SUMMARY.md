# InternVL3-8B-Instruct — Results Summary

**Model:** InternVL3-8B-Instruct  
**Precision:** bfloat16, `device_map="auto"` (~16 GB VRAM)  
**Pairs:** 352 total (176 genuine × 176 impostor)  
**Preprocessing:** grayscale → autocontrast → square-pad (white) → 448×448 → RGB  
**Run date:** 2026-05-24

---

## Zero-Shot (ZS)

| Metric | Value |
|---|---|
| Overall Accuracy | 61.6% |
| Genuine Accuracy (TAR) | 100.0% |
| Impostor Accuracy (TNR) | 23.3% |
| FAR | 76.7% |
| FRR | 0.0% |
| Collapsed | **No** |
| Answer dist | A: 88.4%, B: 11.6% |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Acc | FAR | FRR | n |
|---|---|---|---|---|---|
| 13 | Right | 56.8% | 86.4% | 0.0% | 176 |
| 14 | Left  | 66.5% | 67.0% | 0.0% | 176 |

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
| Genuine Mean Score | 73.0 |
| Impostor Mean Score | 64.1 |
| Score Separation (Δ) | 8.9 pts |
| Best Threshold | 66 |
| Best Accuracy | 83.0% |
| EER | 17.05% |
| EER Threshold | 66 |
| AUC | 0.8228 |
| Invalid | 0 |

**Per-FRGP:**

| FRGP | Hand | Genuine Mean | Impostor Mean | n |
|---|---|---|---|---|
| 13 | Right | 72.5 | 64.1 | 176 |
| 14 | Left  | 73.5 | 64.2 | 176 |

---

## Summary Table

| Prompt | Acc | FAR | FRR | AUC | EER | Collapsed |
|---|---|---|---|---|---|---|
| Zero-Shot     | 61.6% | 76.7% | 0.0% | — | — | No |
| Task Desc.    | 50.0% | 100%  | 0.0% | — | — | Yes |
| Similarity SS | 83.0% (best) | — | — | 0.8228 | 17.05% | No |

---

## Notes

- ZS is the only binary run that does not fully collapse; FRGP 14 (left hand) shows substantially lower FAR (67.0%) than FRGP 13 (right hand, 86.4%).
- TD collapses to 100% FAR — adding domain context worsens discrimination compared to ZS.
- SS achieves AUC 0.8228 with 8.9-point genuine/impostor separation. Score scale shifted lower relative to pre-preprocessing runs (genuine mean 73 vs. 84.5 previously) due to autocontrast making images look more different from training data, but separation improved from 6.6 → 8.9 pts.
