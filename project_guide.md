# SLAPBench — Complete Project Guide
**For:** Bibesh Pyakurel
**Purpose:** Everything you need to know to explain this project to anyone — your professor, a conference reviewer, or a classmate.

---

## What Is This Project In One Sentence?

We are testing whether AI vision models (multimodal LLMs) can tell if two fingerprint images belong to the same person, using a special type of fingerprint image called a SLAP that captures four fingers at once.

---

## The Problem We Are Solving

At every US border entry point (airports, land crossings), travelers have their four-finger SLAP fingerprint taken and matched against a government database. This has always been done by specialized software. Nobody has ever tested whether modern AI vision models (the same kind used in ChatGPT-4o, Claude, etc.) can do this task — specifically for SLAP fingerprints.

**Why SLAP specifically?** SLAP images are harder than regular fingerprint images because:
- Four fingers appear in a single image (not one finger per image)
- The model has to reason about all four fingers together
- Fingers overlap and the spacing between them varies by person
- All SLAP images look somewhat similar at first glance (same hand shape, similar skin)

**The gap we are filling:** No benchmark exists for this. Every prior fingerprint AI paper tests single rolled fingers, not SLAP images. We are the first.

---

## The Dataset — NIST SD302b

**Full name:** NIST Special Database 302b (part of the Nail-to-Nail Fingerprint Challenge)

**Who collected it:** IARPA (US intelligence research agency) for border security research

**What it contains:**
- Fingerprint images from **201 participants** (diverse age, gender, race, occupation)
- Two types of capture: **FRGP 13** (right-hand four-finger SLAP) and **FRGP 14** (left-hand four-finger SLAP)
- Two resolutions: **500 PPI** (standard operational quality) and **1000 PPI** (high research quality)
- Captured with multiple devices — we use **Device R** only (the only device with both resolutions)

**After cleaning (removing errata subjects and incomplete records):**
- **88 usable subjects** with complete data
- **584 total SLAP images** used in our benchmark

**Why cross-resolution genuine pairs?** We pair each subject's 500 PPI image with their 1000 PPI image. This simulates real-world conditions where a traveler's enrollment scan and verification scan may come from different devices.

---

## Our Benchmark — What We Built (SLAPBench)

### The Pairs

We built **7,832 comparison pairs** (updated from 352 in the original paper):

**Genuine pairs (176 total, 88 per FRGP):**
- Same person, 500 PPI vs 1000 PPI
- Question: "Are these the same person?" → Answer: Yes

**Impostor pairs (7,656 total, 3,828 per FRGP):**
- All possible combinations of two DIFFERENT people from the 88 subjects
- Formula: C(88,2) = 88×87/2 = 3,828 unique pairs per FRGP × 2 FRGPs = 7,656
- Question: "Are these the same person?" → Answer: No

**Why we went from 176 to 7,656 impostors:** Originally we used 176 random impostor pairs (balanced). We expanded to ALL possible impostor pairs because:
- More pairs = more reliable statistics
- Reveals model behavior patterns that are hidden in small samples
- Standard in biometric benchmarking

---

## The Five Models We Test

Four are open-source (free, run locally on GPU without internet); the fifth, Claude Opus 4.8, is a proprietary model accessed via the Anthropic API.

### 1. Qwen3-VL-8B-Instruct (by Alibaba)
- 8 billion parameters
- Latest generation of the Qwen vision-language series
- Uses "DeepStack" multi-level visual feature fusion
- Loaded in 4-bit compressed format → uses ~8 GB of GPU memory
- Scored 49.42% on FPBench (a prior fingerprint AI benchmark — the only one that tested fingerprints)

### 2. InternVL3-8B (by OpenGVLab/Shanghai AI Lab)
- 8 billion parameters
- Uses InternViT vision encoder + InternLM language model
- Loaded in bfloat16 format → uses ~16 GB GPU memory (our full GPU)
- Competitive on fingerprint benchmark, notable for most demographically fair results

### 3. Qwen2.5-VL-7B-Instruct (by Alibaba)
- 7 billion parameters
- Previous generation before Qwen3-VL
- Loaded in 4-bit compressed format → uses ~6 GB GPU memory
- Qwen2-VL-7B (similar predecessor) scored 81.10% on face verification benchmarks

### 4. Gemma-3-12B-IT (by Google DeepMind)
- 12 billion parameters — the largest of our four local models
- Multimodal Gemma 3 generation with a SigLIP vision encoder
- Loaded in 4-bit compressed format → uses ~8 GB GPU memory
- Third-best similarity-scoring result in our benchmark (AUC=0.837)

### 5. Claude Opus 4.8 (by Anthropic) — proprietary API model
- Frontier proprietary model, accessed via the Anthropic API (not run locally)
- The **only** model that resists positive-bias collapse under *both* binary prompts
- Best binary verifier overall (zero-shot FAR 20.2%) and second-best on similarity scoring (AUC=0.953)

**Hardware:** The four open-source models run on an NVIDIA RTX 4080 SUPER (16.9 GB VRAM), one at a time. Claude Opus 4.8 runs in Anthropic's cloud via API.

---

## The Three Prompting Strategies

This is one of our main research questions: **Does how you ask the AI matter?**

### Strategy 1: Zero-Shot (ZS)
We show the model two fingerprint images and ask the simplest possible question:
> "Do these two images belong to the same person? (A) Yes (B) No"

No extra context, no explanation, no instructions about what to look for. Just the images and the question. This matches what all prior fingerprint AI benchmarks do.

### Strategy 2: Task Description (TD)
Same binary Yes/No question, but we first explain what SLAP fingerprints are:
> "These are four-finger SLAP fingerprint images. The same person's fingers produce slightly different images each time due to pressure variation and finger placement. Do these two images belong to the same person? (A) Yes (B) No"

This tests: does giving context help the model or hurt it?

**Finding:** It hurts. Adding context made things worse, not better. All four models fully collapsed under TD.

### Strategy 3: Similarity Scoring (SS)
Instead of Yes/No, we ask for a number:
> "Return a single integer from 0 to 100 representing your confidence that the two images belong to the same person. 0 = completely certain different people. 100 = completely certain same person. Reply with the number only."

This is our novel contribution. No prior fingerprint AI benchmark has used this approach.

**Finding:** It works dramatically better for some models. It eliminates the collapse problem entirely.

---

## The Metrics — What We Measure and What They Mean

### For Binary Prompting (ZS and TD)

**Accuracy (Acc):** Percentage of all 7,832 pairs the model got right (correctly said same or different)

**FAR — False Accept Rate:** Percentage of impostor pairs the model incorrectly said "same person." This is the dangerous metric in security — a high FAR means real impostors are being accepted.
- FAR = 100% → model says "same person" for every single pair, genuine or impostor → complete failure
- FAR = 0% → model never falsely accepts an impostor → perfect (but may miss genuine pairs)

**FRR — False Reject Rate:** Percentage of genuine pairs the model incorrectly said "different person."
- FRR = 0% in all our results → the model NEVER rejects a genuine pair

**Collapse / Positive-Bias Collapse:** When a model answers the same way (almost always "A" / same person) for ≥95% of pairs regardless of what the images show. A collapsed model is useless — it's not actually doing fingerprint verification, it's just always saying "yes, same person." Accuracy lands near 50% on a balanced test because it gets all genuine pairs right but all impostor pairs wrong.

**Answer Distribution:** What percentage of responses were "A" (same) vs "B" (different). A healthy model should answer "B" roughly 98% of the time on our dataset (since we have 7,656 impostors vs 176 genuine pairs).

### For Similarity Scoring (SS)

**Genuine Mean Score:** Average confidence score the model gives to genuine pairs (same person). Should be HIGH.

**Impostor Mean Score:** Average confidence score the model gives to impostor pairs (different people). Should be LOW.

**Separation (Δ):** Genuine mean minus impostor mean. A larger gap = the model discriminates better between real matches and fakes.

**EER — Equal Error Rate:** The threshold point where FAR equals FRR. A lower EER is better. EER=0% means perfect separation. EER=50% means the model is no better than random guessing.

**AUC — Area Under the ROC Curve:** Ranges from 0 to 1. AUC=1.0 means perfect discrimination. AUC=0.5 means random (model can't tell genuine from impostor at all). AUC=0.9+ is excellent.

**Best Accuracy:** The highest accuracy achievable by choosing the optimal score threshold. Example: if we decide "score ≥ 80 = same person, score < 80 = different person," what accuracy does that give?

**TAR@FAR=0.1%:** True Accept Rate when FAR is held below 0.1%. This is the operational standard for border security — you can only accept 1 in 1000 impostors. What percentage of real matches are you still catching at that strictness?

---

## What We Found (Key Results)

### Binary Prompting: Mostly Fails

With 7,832 pairs (the exhaustive run):

| Model | Strategy | FAR | Collapsed? |
|---|---|---|---|
| Claude Opus 4.8 | Zero-Shot | 20.2% | No |
| Claude Opus 4.8 | Task Description | 50.9% | No |
| Qwen3-VL-8B | Zero-Shot | 26.5% | No |
| Qwen3-VL-8B | Task Description | 98.9% | Yes |
| InternVL3-8B | Zero-Shot | 72.3% | No |
| InternVL3-8B | Task Description | 100% | Yes |
| Qwen2.5-VL-7B | Zero-Shot | 69.6% | No |
| Qwen2.5-VL-7B | Task Description | 100% | Yes |
| Gemma-3-12B | Zero-Shot | 96.4% | Yes |
| Gemma-3-12B | Task Description | 100% | Yes |

5 of 10 configurations collapse completely. Task description makes all four **open-source** models collapse — adding context hurts. Zero-shot keeps the three Qwen/InternVL models somewhat functional, but even at 26.5% FAR (Qwen3), 1 in 4 impostors still gets through. Gemma-3-12B collapses even under zero-shot (96.4% FAR). **Claude Opus 4.8 is the standout exception:** it resists collapse under both prompts and has the best binary FAR of any model (20.2% zero-shot, 50.9% task-description).

### Similarity Scoring: Reveals Hidden Capability

| Model | Gen Score | Imp Score | Δ | EER | AUC |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 100.0% | 63.8% | 36.2 | 0.0% | 1.0 |
| Claude Opus 4.8 | 91.3% | 48.4% | 42.9 | 11.75% | 0.953 |
| Gemma-3-12B | 75.8% | 63.3% | 12.5 | 15.1% | 0.837 |
| InternVL3-8B | 57.7% | 79.1% | -21.4 (inverted) | 48.09% | 0.589 |
| Qwen2.5-VL-7B | 95.0% | 90.1% | 4.9 | 43.34% | 0.567 |

**Qwen3-VL-8B:** Perfect scores. Assigns 100% confidence to every genuine pair and averages 63.8% to impostors. AUC=1.0, EER=0.0%. This is remarkable for a zero-shot model.

**Gemma-3-12B:** Functional and second-best. Genuine mean 75.8 vs impostor mean 63.3, a 12.5-point separation. AUC=0.837, EER=15.1% — clearly discriminates same vs different, well ahead of InternVL3 and Qwen2.5 but far below Qwen3-VL.

**InternVL3-8B:** Inverted. Assigns higher scores to impostors (79.1%) than genuine (57.7%). AUC≈0.59 means it's barely better than random. This is a model that learned to score but got the calibration backwards.

**Qwen2.5-VL-7B:** Near-random. Tiny 4.9-point separation (genuine 95.0 vs impostor 90.1), AUC=0.567, EER=43.34%. It rates almost everything highly and cannot distinguish genuine from impostor.

### The Core Finding

The models don't fail because they can't understand fingerprints. They fail because the binary Yes/No format forces them into a prior ("same person") they can't escape. When asked to score instead, Qwen3-VL immediately demonstrates near-perfect discrimination. The prompt format is the bottleneck, not the model.

---

## The Paper Structure

| Section | What It Says |
|---|---|
| Abstract | 1-paragraph summary of the whole paper |
| Introduction | Why SLAP fingerprints matter, what gap we fill, 5 contributions |
| Related Work | 3 subsections: (1) traditional fingerprint recognition, (2) MLLM biometric benchmarks, (3) collapse in binary verification |
| Dataset | What SD302b is, how we filtered it, what we kept |
| Methodology | How we built pairs, which models we use, the 3 prompting strategies, how we parse responses, what metrics we compute |
| Results | Table 1 (binary results), Table 2 (similarity scoring), Table 3 (TAR@FAR=0.1%), figures |
| Discussion | Why binary prompts cause collapse, why scoring works, the left-right hand asymmetry, limitations |
| Conclusion | Summary of findings |

**Paper title:** SLAPBench: Benchmarking Multimodal Large Language Models for Four-Finger SLAP Fingerprint Verification

**Target venue:** IEEE conference (using IEEEtran format)

**GitHub:** https://github.com/bibeshpyakurel/SLAPBench (mentioned in paper, not yet public)

---

## The Code

All experiments run from one script: `code/run_verification.py`

Key flags:
- `--model` → which model (qwen3vl, internvl3, qwen25vl, gemma3, anthropic, openai)
- `--prompting` → which strategy (zero_shot, task_description, similarity_score)
- `--all-impostors` → use all 7,832 pairs (not just 352)
- `--run` → actually run (without this it just shows setup info)
- `--resume path/to/file.csv` → continue an interrupted run
- `--anthropic-model` / `--openai-model` → API model name (e.g. claude-opus-4-8)

Results saved in: `results/<model>/latest/task8_<model>_<prompting>_<date>.csv`
Metrics saved in: same folder as `.metrics.json`

---

## What Is Still Left To Do

1. ~~Qwen2.5-VL similarity_score~~ — ✅ complete (AUC 0.567)
2. ~~Recompute TAR@FAR=0.1%~~ — ✅ complete for all 5 models
3. ~~Gemma3-12B experiments~~ — ✅ complete (all 3 strategies; AUC 0.837)
4. ~~Claude Opus 4.8 experiments~~ — ✅ complete (all 3 strategies; AUC 0.953, collapse-resistant)
5. ~~Generate figures~~ — ✅ score distribution + ROC curves regenerated (5 models)
6. ~~Update all paper numbers~~ — ✅ unified color table + 5-model narrative
7. **ChatGPT + Gemini API experiments** — awaiting API keys from professor
8. **Resolve all \note{} comments** in the manuscript
9. **Add missing citations** (FingerNet, DeepPrint, VeriFinger, etc.)

---

## Quick Reference: Numbers to Know

| Fact | Number |
|---|---|
| Subjects in dataset | 88 (clean) / 201 (total in DB) |
| SLAP images used | 584 |
| Genuine pairs | 176 |
| Impostor pairs | 7,656 |
| Total pairs | 7,832 |
| Pairs per FRGP | 3,916 |
| GPU used | NVIDIA RTX 4080 SUPER (16.9 GB VRAM) |
| Models tested | 5 (Qwen3-VL-8B, InternVL3-8B, Qwen2.5-VL-7B, Gemma-3-12B, Claude Opus 4.8) |
| Prompting strategies | 3 (ZS, TD, SS) |
| Total experiments | 15 (5 models × 3 strategies) |
| Best AUC | 1.0 — Qwen3-VL-8B, similarity scoring |
| 2nd best AUC | 0.953 — Claude Opus 4.8, similarity scoring |
| Best binary verifier | Claude Opus 4.8 — zero-shot FAR 20.2%, collapse-resistant |
| Only collapse-resistant model | Claude Opus 4.8 (both binary prompts) |
| Worst result | Qwen2.5-VL-7B SS: AUC 0.567, EER 43.34% (near-random) |
