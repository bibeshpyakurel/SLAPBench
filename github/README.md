# SLAPBench

**SLAPBench** is the first benchmark for evaluating multimodal large language models (MLLMs) on four-finger SLAP fingerprint verification, built on [NIST Special Database 302b (SD302b)](https://www.nist.gov/srd/nist-special-database-302).

> Pyakurel, B. and Murshed, M. G. S. "SLAPBench: Benchmarking Multimodal Large Language Models for Four-Finger SLAP Fingerprint Verification." *IEEE*, 2026.

---

## Overview

Four-finger SLAP (Slap Livescan Acquisition Protocol) images are the standard biometric capture format at US border entry points, yet no benchmark existed to assess whether MLLMs can reason about them. SLAPBench fills this gap.

**Key findings (7,832 exhaustive pairs — 176 genuine + 7,656 impostor; 4 open-source models + Claude Opus 4.8):**
- Task-description prompting collapses all four open-source models to ≥98.9% FAR — but **Claude Opus 4.8 resists collapse under every prompt** (the only model that does)
- Zero-shot prompting: Claude is the strongest binary verifier overall (FAR 20.2%); 3 of 4 open-source models remain functional with high FAR (26.5%–72.3%); Gemma-3-12B collapses (96.4% FAR)
- Similarity-scoring eliminates collapse and reveals dramatically different model capabilities:
  - Qwen3-VL-8B: **AUC = 1.0, EER = 0.0%, TAR@FAR=0.1% = 100%** — perfect discrimination
  - Claude Opus 4.8: AUC = 0.953, EER = 11.75%, TAR@FAR=0.1% = 56.8% — second-best
  - Gemma-3-12B: AUC = 0.837, EER = 15.1%, TAR@FAR=0.1% = 23.3% — third, functional
  - InternVL3-8B: AUC = 0.589, EER = 48.09% — inverted calibration (scores impostors higher than genuine)
  - Qwen2.5-VL-7B: AUC = 0.567, EER = 43.34% — near-random

---

## Dataset

SLAPBench uses **NIST SD302b** — an operational nail-to-nail fingerprint dataset of 201 participants captured at 500 PPI and 1000 PPI.

**Images are not included in this repository.** NIST SD302b must be obtained directly from NIST:
https://www.nist.gov/srd/nist-special-database-302

### Evaluation Pairs

`results/task8_pairs_all.csv` is the exhaustive pair manifest used in all experiments:
- **7,832 total pairs** (176 genuine + 7,656 impostor)
- **Genuine pairs:** same subject, R-500 vs R-1000 (cross-resolution)
- **Impostor pairs:** all C(88,2) = 3,828 unique pairs per FRGP × 2 FRGPs = 7,656 total
- Covers FRGP 13 (right four-finger) and FRGP 14 (left four-finger)

---

## Models

### Open-Source (Local GPU)

| Model | Precision | VRAM |
|---|---|---|
| InternVL3-8B-Instruct | bfloat16 | ~16 GB |
| Qwen2.5-VL-7B-Instruct | 4-bit NF4 | ~6 GB |
| Qwen3-VL-8B-Instruct | 4-bit NF4 | ~6 GB |
| Gemma-3-12B-IT | 4-bit NF4 | ~8 GB |

All local models are loaded via HuggingFace `transformers`. No vLLM or serving layer required.

### API Models (Cloud)

| Provider | Model Flag | Notes |
|---|---|---|
| OpenAI | `--model openai` | Set `OPENAI_API_KEY` in `.env`; specify model with `--openai-model gpt-4o` |
| Anthropic | `--model anthropic` | Set `ANTHROPIC_API_KEY` in `.env`; specify model with `--anthropic-model claude-opus-4-8` |

API models send images as base64-encoded PNG. Exponential backoff handles rate limits automatically.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/bibeshpyakurel/SLAPBench.git
cd SLAPBench

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA-capable GPU (16 GB VRAM for InternVL3, 8 GB for Qwen models)

For API models, create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

---

## Usage

```bash
# Dry run — inspect pair plan without loading any model
python code/run_verification.py --dry-run

# Save pair manifest to CSV
python code/run_verification.py --pairs-only

# Run evaluation (local GPU models, all 7,832 pairs)
python code/run_verification.py --model internvl3  --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model internvl3  --prompting task_description --all-impostors --run
python code/run_verification.py --model internvl3  --prompting similarity_score --all-impostors --run

python code/run_verification.py --model qwen25vl   --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model qwen25vl   --prompting task_description --all-impostors --run
python code/run_verification.py --model qwen25vl   --prompting similarity_score --all-impostors --run

python code/run_verification.py --model qwen3vl    --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model qwen3vl    --prompting task_description --all-impostors --run
python code/run_verification.py --model qwen3vl    --prompting similarity_score --all-impostors --run

python code/run_verification.py --model gemma3     --prompting zero_shot        --all-impostors --run
python code/run_verification.py --model gemma3     --prompting task_description --all-impostors --run
python code/run_verification.py --model gemma3     --prompting similarity_score --all-impostors --run

# Run evaluation (API models)
python code/run_verification.py --model openai     --prompting zero_shot        --all-impostors --run --openai-model gpt-4o
python code/run_verification.py --model anthropic  --prompting zero_shot        --all-impostors --run --anthropic-model claude-opus-4-8

# Resume an interrupted run
python code/run_verification.py --model internvl3 --prompting zero_shot --all-impostors --run \
    --resume results/internvl3/latest/task8_internvl3_zero_shot_<timestamp>.csv

# Print metrics from a completed results file
python code/run_verification.py --metrics results/internvl3/latest/<file>.csv
```

---

## Results

Pre-computed results for all four local models plus the proprietary Claude Opus 4.8, across all three prompting strategies, are in `results/`.

```
results/
├── task8_pairs_all.csv                     # Exhaustive pair manifest (7,832 pairs)
├── internvl3/
│   ├── SUMMARY.md                          # Full results summary
│   ├── task8_internvl3_zero_shot_*.csv
│   ├── task8_internvl3_zero_shot_*.metrics.json
│   ├── task8_internvl3_task_description_*.csv
│   ├── task8_internvl3_task_description_*.metrics.json
│   ├── task8_internvl3_similarity_score_*.csv
│   └── task8_internvl3_similarity_score_*.metrics.json
├── qwen25vl/
│   └── ...
├── qwen3vl/
│   └── ...
├── gemma3/
│   └── ...
└── anthropic/                              # Claude Opus 4.8 (proprietary API)
    └── ...
```

### Summary (7,832 pairs — 176 genuine + 7,656 impostor)

| Model | Prompt | Acc | FAR | AUC | EER | TAR@FAR=0.1% | Collapsed |
|---|---|---|---|---|---|---|---|
| InternVL3-8B | Zero-Shot | 29.3% | 72.3% | — | — | — | No |
| InternVL3-8B | Task Desc. | 2.2% | 100% | — | — | — | Yes |
| InternVL3-8B | Sim. Score | — | — | 0.589 | 48.09% | 0.0% | No |
| Qwen2.5-VL-7B | Zero-Shot | 32.0% | 69.6% | — | — | — | No |
| Qwen2.5-VL-7B | Task Desc. | 2.2% | 100% | — | — | — | Yes |
| Qwen2.5-VL-7B | Sim. Score | — | — | 0.567 | 43.34% | 0.0% | No |
| Qwen3-VL-8B | Zero-Shot | 74.1% | 26.5% | — | — | — | No |
| Qwen3-VL-8B | Task Desc. | 3.3% | 98.9% | — | — | — | Yes |
| **Qwen3-VL-8B** | **Sim. Score** | **100%** | **—** | **1.000** | **0.0%** | **100%** | No |
| Gemma-3-12B | Zero-Shot | 5.7% | 96.4% | — | — | — | Yes |
| Gemma-3-12B | Task Desc. | 2.3% | 100% | — | — | — | Yes |
| **Gemma-3-12B** | **Sim. Score** | **—** | **—** | **0.837** | **15.1%** | **23.3%** | No |
| **Claude Opus 4.8** | **Zero-Shot** | **80.3%** | **20.2%** | — | — | — | **No** |
| **Claude Opus 4.8** | **Task Desc.** | **50.2%** | **50.9%** | — | — | — | **No** |
| **Claude Opus 4.8** | **Sim. Score** | — | — | **0.953** | **11.75%** | **56.8%** | No |

> **Claude Opus 4.8** (proprietary) is the only model that resists collapse under *both* binary prompts, posts the best binary FAR (20.2%), and ranks #2 on similarity AUC.

---

## Prompting Strategies

**Zero-Shot (ZS):** Minimal binary question — "Do these two images belong to the same person?" (A) Yes / (B) No.

**Task Description (TD):** Same binary question with added domain context explaining SLAP images and capture variation.

**Similarity Score (SS):** Continuous integer 0–100 with calibrated forensic anchors (0–30: almost certainly different; 31–60: uncertain; 61–85: likely same; 86–100: highly confident same).

---

## Image Preprocessing

All images preprocessed identically before model inference:
1. Convert to grayscale
2. Apply autocontrast
3. Square-pad with white background
4. Resize to 448×448
5. Convert to RGB

---

## Citation

```bibtex
@article{pyakurel2026slapbench,
  title     = {SLAPBench: Benchmarking Multimodal Large Language Models
               for Four-Finger SLAP Fingerprint Verification},
  author    = {Pyakurel, Bibesh and Murshed, M. G. Sarwar},
  journal   = {IEEE},
  year      = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

The NIST SD302b dataset is subject to its own terms of use. This repository contains no NIST imagery.
