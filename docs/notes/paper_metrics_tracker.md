# Paper Metrics Tracker — SLAPBench
**Purpose:** Cross-reference between what is written in the paper and the actual exhaustive experiment numbers (7,832 pairs). Use this before touching any number in the manuscript.

---

## Dataset Numbers

| Item | In Paper Now | Correct (Exhaustive) | Status |
|---|---|---|---|
| Total participants in SD302b | 201 | 201 | ✅ Correct |
| Clean subjects after filtering | 88 | 88 | ✅ Correct |
| Total SLAP images | 584 | 584 | ✅ Correct |
| Genuine pairs | 176 | 176 | ✅ Correct |
| Impostor pairs | 176 (balanced) | **7,656** (all C(88,2) per FRGP) | ⚠️ NEEDS UPDATE |
| Total pairs | 352 | **7,832** | ⚠️ NEEDS UPDATE |
| Pairs per FRGP | 176 | **3,916** | ⚠️ NEEDS UPDATE |

---

## Table 1 — Binary Verification Results

### In Paper (OLD — 352 pairs, 176G + 176I)

| Model | Prompt | Acc | FAR | FRR | Collapsed |
|---|---|---|---|---|---|
| InternVL3-8B | ZS | 61.6% | 76.7% | 0.0% | No |
| InternVL3-8B | TD | 50.0% | 100% | 0.0% | Yes |
| Qwen2.5-VL-7B | ZS | 54.5% | 90.9% | 0.0% | Yes |
| Qwen2.5-VL-7B | TD | 50.0% | 100% | 0.0% | Yes |
| Qwen3-VL-8B | ZS | 73.3% | 53.4% | 0.0% | No |
| Qwen3-VL-8B | TD | 50.0% | 100% | 0.0% | Yes |

### NEW Numbers (7,832 pairs, 176G + 7,656I)

| Model | Prompt | Acc | FAR | FRR | Collapsed | Changed? |
|---|---|---|---|---|---|---|
| InternVL3-8B | ZS | 29.3% | 72.3% | 0.0% | No | ⚠️ Yes |
| InternVL3-8B | TD | 2.2% | 100.0% | 0.0% | Yes | ⚠️ Yes (acc dropped) |
| Qwen2.5-VL-7B | ZS | 32.0% | 69.6% | 0.0% | **No** | 🔴 Big change — was collapsed, now NOT |
| Qwen2.5-VL-7B | TD | 2.2% | 100.0% | 0.0% | Yes | ⚠️ Yes |
| Qwen3-VL-8B | ZS | 74.1% | 26.5% | 0.0% | No | ⚠️ Yes |
| Qwen3-VL-8B | TD | 3.3% | 98.9% | 0.0% | Yes | ⚠️ Yes |
| **Gemma-3-12B** | **ZS** | **5.7%** | **96.4%** | **0.0%** | **Yes** | 🆕 New model |
| **Gemma-3-12B** | **TD** | **2.3%** | **100.0%** | **0.0%** | **Yes** | 🆕 New model |

**Key narrative change:** The paper originally stated "four of six binary configurations collapse." With 7,832 pairs across **five** models (4 open-source + Claude Opus 4.8), **five of ten** binary configurations collapse (InternVL3-TD, Qwen2.5-TD, Qwen3-TD, Gemma3-ZS, Gemma3-TD). Qwen2.5-VL zero-shot was previously collapsed but is NOT collapsed in the exhaustive run. Gemma-3-12B also collapses under zero-shot (96.4% FAR). **Claude Opus 4.8 resists collapse under both prompts** — the only model to do so. All paper statements now use "5 of 10" across five models.

### Per-FRGP Binary (NEW)

| Model | Prompt | FRGP 13 Acc | FRGP 13 FAR | FRGP 14 Acc | FRGP 14 FAR |
|---|---|---|---|---|---|
| Qwen3-VL-8B | ZS | 71.7% | 29.0% | 76.5% | 24.0% |
| Qwen3-VL-8B | TD | 2.5% | 99.7% | 4.2% | 98.0% |
| InternVL3-8B | ZS | 28.8% | 72.9% | 29.8% | 71.8% |
| InternVL3-8B | TD | 2.2% | 100.0% | 2.2% | 100.0% |
| Qwen2.5-VL-7B | ZS | 30.2% | 71.4% | 33.7% | 67.8% |
| Qwen2.5-VL-7B | TD | 2.2% | 100.0% | 2.2% | 100.0% |
| Gemma-3-12B | ZS | 4.4% | 97.8% | 7.1% | 95.1% |
| Gemma-3-12B | TD | 2.2% | 100.0% | 2.3% | 99.9% |

---

## Table 2 — Similarity Scoring Results

### In Paper (OLD — 352 pairs)

| Model | Gen% | Imp% | Δ | Best Acc | EER | AUC |
|---|---|---|---|---|---|---|
| InternVL3-8B | 73.0 | 64.1 | 8.9 | 83.0% | 17.05% | 0.823 |
| Qwen2.5-VL-7B | 95.0 | 91.8 | 3.2 | 77.6% | 22.44% | 0.776 |
| Qwen3-VL-8B | 96.4 | 65.3 | 31.1 | 94.9% | 5.11% | 0.990 |

### NEW Numbers (7,832 pairs) — ALL COMPLETE ✅

| Model | Gen% | Imp% | Δ | Best Acc | EER | AUC | Changed? |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | **100.0** | **63.8** | **+36.2** | **100.0%** | **0.0%** | **1.0000** | ⚠️ Improved significantly |
| **Gemma-3-12B** | **75.8** | **63.3** | **+12.5** | **98.3%*** | **15.1%** | **0.8372** | 🆕 New model — 2nd best |
| InternVL3-8B | **57.7** | **79.1** | **-21.4 (INVERTED)** | 98.4%* | **48.09%** | **0.5895** | 🔴 Completely different |
| Qwen2.5-VL-7B | **95.0** | **90.1** | **+4.9** | **97.8%*** | **43.34%** | **0.5666** | 🔴 Significantly worse |

*Best_acc figures (Gemma3 98.3%, InternVL3 98.4%, Qwen2.5-VL 97.8%) are misleading — all are achieved by setting a high threshold that rejects almost all pairs, which works trivially because there are 7,656 impostors vs only 176 genuine. AUC and EER are the honest measures.

**Critical findings:**
- **Qwen3-VL** dramatically improved: perfect AUC=1.0, EER=0.0%, compared to AUC=0.990 before.
- **Gemma-3-12B** (new) is the **second-best** model: AUC=0.837, EER=15.1%, with a 12.5-pt genuine/impostor separation (genuine 75.8 vs impostor 63.3). Clearly functional, well ahead of InternVL3 and Qwen2.5-VL but far below Qwen3-VL.
- **InternVL3** assigns HIGHER scores to impostor pairs (79.1%) than genuine pairs (57.7%) — completely inverted. Hidden with only 176 impostors, exposed with 7,656.
- **Qwen2.5-VL** has tiny separation (4.9 pts) and AUC=0.567 — essentially random. Worse than the 352-pair result suggested (AUC was 0.776 before).

### Per-FRGP Similarity Score (NEW — ALL COMPLETE)

| Model | FRGP 13 Gen Mean | FRGP 13 Imp Mean | FRGP 14 Gen Mean | FRGP 14 Imp Mean |
|---|---|---|---|---|
| Qwen3-VL-8B | 100.0% | 65.6% | 100.0% | 61.9% |
| Gemma-3-12B | 75.2% | 63.3% | 76.4% | 63.3% |
| InternVL3-8B | 55.5% | 79.7% | 60.0% | 78.5% |
| Qwen2.5-VL-7B | 95.0% | 90.2% | 95.0% | 90.0% |

---

## Table 3 — TAR @ FAR=0.1%

### In Paper (OLD — 352 pairs)

| Model | TAR@FAR=0.1% | Threshold |
|---|---|---|
| Qwen3-VL-8B | 81.25% | 85.5 |
| InternVL3-8B | 65.91% | 65.5 |
| Qwen2.5-VL-7B | 0.00% | 95.5 |

### NEW Numbers (7,832 pairs) — ALL COMPLETE ✅

| Model | TAR@FAR=0.1% | Threshold | Changed? |
|---|---|---|---|
| Qwen3-VL-8B | **100.00%** | **100.0** | ⚠️ Improved (81.25% → 100%) |
| **Gemma-3-12B** | **23.30%** | **75.5** | 🆕 New model |
| InternVL3-8B | **0.00%** | ∞ | 🔴 Dropped drastically (65.91% → 0%) |
| Qwen2.5-VL-7B | **0.00%** | ∞ | ✅ Unchanged (was already 0%) |

**Note on 0.00% for InternVL3 and Qwen2.5-VL:** Because their impostor scores cluster so close to genuine scores, there is no threshold that brings FAR below 0.1% while still accepting any genuine pairs. This directly reflects their near-random AUC values (0.589 and 0.567). Gemma-3-12B achieves a non-trivial 23.30% TAR@FAR=0.1%, consistent with its AUC of 0.837.

---

## Figure References in Paper

| Figure | Description | Status |
|---|---|---|
| fig_score_distributions.pdf | Score distribution histograms for 4 models | ⚠️ Not generated yet; will look different with new numbers |
| fig_roc_curves.pdf | ROC curves for all 4 models | ⚠️ Not generated yet |

**Thresholds mentioned in figure caption — ALL UPDATED:**
- Qwen3-VL-8B best threshold: 76 → NEW: **96**
- InternVL3-8B best threshold: 66 → NEW: **86**
- Qwen2.5-VL-7B best threshold: 91 → NEW: **96**
- Gemma-3-12B best threshold: **76** (new model)

---

## Key Paper Statements That Need Updating

| Location | Current Text | What To Change |
|---|---|---|
| Abstract | "352 balanced verification pairs (176 genuine, 176 impostor)" | → 7,832 pairs (176 genuine, 7,656 impostor) |
| Abstract | "four of six binary prompting configurations collapse" | → five of ten (five models) ✅ done |
| Abstract | "three models" | → five models (Gemma-3-12B + Claude Opus 4.8) ✅ done |
| Abstract | "AUC=0.990 (EER=5.11%)" for Qwen3 | → AUC=1.0, EER=0.0% |
| Abstract | "AUC=0.823 (EER=17.05%)" for InternVL3 | → AUC=0.589, EER=48.09% |
| Abstract | "AUC=0.776 (EER=22.44%)" for Qwen2.5 | → AUC=0.567, EER=43.34% |
| Abstract | (Gemma + Claude not mentioned) | → Gemma AUC=0.837; Claude AUC=0.953 (2nd) ✅ done |
| Intro contribution #2 | "352 fixed pairs (176 genuine + 176 impostor)" | → update pair counts |
| Intro contribution #3 | "four of six binary prompting configurations collapse" | → five of ten (five models) ✅ done |
| Section 4.1 Pair Construction | Impostor sampling description (random 88) | → update to exhaustive C(88,2) |
| Table 1 | All binary numbers | → update all rows |
| Table 2 | All similarity score numbers | → update all rows |
| Table 3 | TAR@FAR=0.1% numbers | → recompute |
| Section 5.1 | "FAR ranging from 91% to 100%" | → update range |
| Section 5.1 | Hand asymmetry numbers (FRGP 13 vs 14) | → verify with new numbers |
| Section 5.2 | All per-model score values | → update |
| Discussion | All specific metric values | → update |
| Conclusion | AUC values "0.990, 0.823, 0.776" | → update |
| Limitations | "GPT-4o" mentioned | → change to "ChatGPT model (not yet specified)" |
| Future Work | "GPT-4o" mentioned | → change to "ChatGPT" |

---

## Open Notes in Manuscript (\note{} items)

| Location | Note | Action Needed |
|---|---|---|
| Abstract | "is this number calculation right? (201 participants)" | Verify: 201 is in NIST docs — correct |
| Abstract | "AUC, EER? Haven't we described it?" | Add brief definition in Section 4.4 |
| Abstract | "forced binary framing in simple words" | Rephrase as: the model must choose Yes or No with no middle option |
| Related Work | Multiple "citation?" notes | Add missing citations (FingerNet, DeepPrint, VeriFinger, transformer papers) |
| Related Work | "are these numbers correct?" (FaceXBench) | 70.28% human, 84.50% specialized — verify against FaceXBench paper |
| Related Work | "CoT = ?" | Define as Chain-of-Thought |
| Section 4.2 | "mention the paper (cite again)" | Add \cite for FPBench next to InternVL3 description |
| Section 4.3 | Three "UI card" notes for prompts | Create a figure/box showing the three prompts visually |
| Section 4.4 | "creative way to explain" binary/continuous metrics | Consider adding a small diagram |
| Section 5.1 (pair construction) | "now we have all pairs 3838 impostor pair" | Update text to reflect C(88,2)=3,828 per FRGP, 7,656 total |
| Section 5.2 | "as result keeps changing" | ✅ All results now final — update all numbers |
| Discussion | Three hypotheses — "Is this great for the paper?" | Keep H1 and H2; consider removing or softening H3 |
| Limitations | "proprietary results unknown" | Update once ChatGPT/Claude experiments run |
| Future Work | "Tasks 1-7 — reader doesn't know" | Add a sentence explaining the full SLAPBench task list |
| Conclusion | "make sure all metrics are correct" | ✅ All metrics now final — ready to update |
| Section 4.1 SHA-256 | "what is SHA-256 for a reader" | Add brief footnote: SHA-256 is a cryptographic checksum for verifying file integrity |

---

## What Is Still Pending

| Item | Status |
|---|---|
| All 9 core experiments (3 local models × 3 prompting) | ✅ Complete |
| Gemma3-12B experiments (3 prompting) | ✅ Complete (2026-06-20) |
| Claude Opus 4.8 experiments (3 prompting) | ✅ Complete (2026-06-21) |
| TAR@FAR=0.1% recomputed for all models | ✅ Complete (Gemma3=23.30%, Claude=56.8%) |
| Score distribution figures (5 models) | ✅ Generated (incl. Gemma3 + Claude) |
| ROC curve figures (5 models) | ✅ Generated (incl. Gemma3 + Claude) |
| Paper numbers updated throughout manuscript | ✅ Done — 5 models, unified color table |
| ChatGPT / Gemini API experiments | ❌ Awaiting API keys from professor |

## Claude Opus 4.8 — Complete Results (2026-06-21)

| Prompt | Acc | FAR | FRR | Collapsed | Notes |
|---|---|---|---|---|---|
| Zero-Shot | 80.3% | **20.2%** | 0.57% | **No** | Best binary result of ANY model (beats Qwen3 26.5%) |
| Task Desc. | 50.2% | 50.9% | 0.0% | **No** | 🔴 ONLY model that resists TD collapse (answers A 52% / B 48%) |
| Sim. Score | — | — | — | No | Gen 91.3, Imp 48.4, Δ +42.9, EER 11.75%, **AUC 0.953**, TAR@0.1% 56.8% |

**Headline:** Claude is the sole collapse-resistant model (both prompts) and ranks #2 on similarity AUC (0.953), behind only Qwen3-VL (1.0). Binary collapse count is now **5 of 10** configurations (was 5 of 8 across 4 models).
