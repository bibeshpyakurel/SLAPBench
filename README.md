# SLAPBench

**SLAPBench** is the first benchmark for evaluating multimodal large language models (MLLMs) on four-finger SLAP fingerprint verification, built on [NIST Special Database 302b (SD302b)](https://www.nist.gov/srd/nist-special-database-302).

> Pyakurel, B. and Murshed, M. G. S. "SLAPBench: Benchmarking Multimodal Large Language Models for Four-Finger SLAP Fingerprint Verification." *IEEE*, 2026.

---

## Overview

Four-finger SLAP (Slap Livescan Acquisition Protocol) images are the standard biometric capture format at US border entry points, yet no benchmark existed to assess whether MLLMs can reason about them. SLAPBench fills this gap.

**Key findings:**
- Task-description prompting collapses all three models to 100% FAR
- Zero-shot prompting collapses 1 of 3 models
- Similarity-scoring prompt eliminates collapse entirely:
  - Qwen3-VL-8B: AUC = 0.990, EER = 5.11%
  - InternVL3-8B: AUC = 0.823, EER = 17.05%
  - Qwen2.5-VL-7B: AUC = 0.776, EER = 22.44%

---

## Dataset

SLAPBench uses **NIST SD302b** — an operational nail-to-nail fingerprint dataset of 201 participants captured at 500 PPI and 1000 PPI.

**Images are not included in this repository.** NIST SD302b must be obtained directly from NIST:
https://www.nist.gov/srd/nist-special-database-302

### Evaluation Pairs

`results/task8_pairs.csv` is the fixed pair manifest used in all experiments:
- **352 total pairs** (176 genuine + 176 impostor)
- **Genuine pairs:** same subject, R-500 vs R-1000 (cross-resolution)
- **Impostor pairs:** different subjects, same device, same hand (FRGP)
- Covers FRGP 13 (right four-finger) and FRGP 14 (left four-finger)
- Seed = 42 for reproducibility

---

## Models

| Model | Precision | VRAM |
|---|---|---|
| InternVL3-8B-Instruct | bfloat16 | ~16 GB |
| Qwen2.5-VL-7B-Instruct | 4-bit NF4 | ~6 GB |
| Qwen3-VL-8B-Instruct | 4-bit NF4 | ~6 GB |

All models are loaded via HuggingFace `transformers`. No vLLM or serving layer required.

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

---

## Usage

```bash
# Dry run — inspect pair plan without loading any model
python code/run_verification.py --dry-run

# Save pair manifest to CSV
python code/run_verification.py --pairs-only

# Run evaluation
python code/run_verification.py --model internvl3  --prompting zero_shot        --run
python code/run_verification.py --model internvl3  --prompting task_description --run
python code/run_verification.py --model internvl3  --prompting similarity_score --run

python code/run_verification.py --model qwen25vl   --prompting zero_shot        --run
python code/run_verification.py --model qwen25vl   --prompting task_description --run
python code/run_verification.py --model qwen25vl   --prompting similarity_score --run

python code/run_verification.py --model qwen3vl    --prompting zero_shot        --run
python code/run_verification.py --model qwen3vl    --prompting task_description --run
python code/run_verification.py --model qwen3vl    --prompting similarity_score --run

# Resume an interrupted run
python code/run_verification.py --model internvl3 --prompting zero_shot --run \
    --resume results/internvl3/latest/task8_internvl3_zero_shot_<timestamp>.csv

# Print metrics from a completed results file
python code/run_verification.py --metrics results/internvl3/latest/<file>.csv
```

---

## Results

Pre-computed results for all three models and all three prompting strategies are in `results/`.

```
results/
├── task8_pairs.csv                         # Fixed pair manifest
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
└── qwen3vl/
    └── ...
```

### Summary

| Model | Prompt | Acc | FAR | AUC | EER | Collapsed |
|---|---|---|---|---|---|---|
| InternVL3-8B | Zero-Shot | 61.6% | 76.7% | — | — | No |
| InternVL3-8B | Task Desc. | 50.0% | 100% | — | — | Yes |
| InternVL3-8B | Sim. Score | 83.0% | — | 0.823 | 17.05% | No |
| Qwen2.5-VL-7B | Zero-Shot | 54.5% | 90.9% | — | — | Yes |
| Qwen2.5-VL-7B | Task Desc. | 50.0% | 100% | — | — | Yes |
| Qwen2.5-VL-7B | Sim. Score | 77.6% | — | 0.776 | 22.44% | No |
| Qwen3-VL-8B | Zero-Shot | 73.3% | 53.4% | — | — | No |
| Qwen3-VL-8B | Task Desc. | 50.0% | 100% | — | — | Yes |
| Qwen3-VL-8B | Sim. Score | 94.9% | — | 0.990 | 5.11% | No |

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
