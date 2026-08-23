# SLAPBench

Benchmarking multimodal large language models on four-finger SLAP fingerprint verification using the NIST Special Database 302b dataset.

---

## What This Project Does

When you enter the United States at a border crossing, an agent asks you to press your right four fingers flat on a scanner, then your left four fingers. That single flat press — all four fingers captured simultaneously in one image — is called a **SLAP** (Simultaneous Latent Acquisition Press) image. It is the standard biometric capture format used at every US border entry point and at enrollment stations worldwide.

This project asks a simple question: **can a multimodal large language model look at two SLAP fingerprint images and determine whether they belong to the same person?**

No benchmark existed for this before. The closest prior work (FPBench) tested LLMs on individual finger crops in a multiple-choice format. SLAPBench is the first benchmark to test LLMs on full four-finger SLAP images in a verification setting.

---

## v2 — The RidgeBase Extension (in progress)

The first version of this benchmark was built entirely on SD302b, where each
finger position has only **one** capture, so a mated pair could only be formed
by comparing the 500 PPI and 1000 PPI versions of the *same* image. Because the
500 PPI image is an exact downscale of the 1000 PPI one, those mated pairs are
near-duplicates — a shortcut that can produce a perfect score without any real
fingerprint matching. Reviewers correctly flagged this.

**v2 fixes it with a second dataset, [RidgeBase](https://www.buffalo.edu/cubs/research/datasets.html).**
RidgeBase provides *multiple independent captures* of each four-finger hand
(smartphone photos across Apple and Google devices), so a genuine pair is two
truly separate photographs. On this dataset we evaluate three comparison
categories and two methods:

| Category | Meaning | Purpose |
|---|---|---|
| **Genuine** | same subject + hand, two independent captures | real mated pairs (no near-duplicates) |
| **Primary impostor** | different subjects, same hand (device-matched) | standard non-mated comparison |
| **Diagnostic impostor** | same subject, *different* hand | mechanism probe: can the model tell a person's own two hands apart? |

Both **prompted** verification (zero-shot / task-description / similarity-scoring)
and **prompt-free embedding matching** (cosine on the vision encoder) are run,
with genuine pairs split into **same-device** vs **cross-device** for a
cross-sensor analysis. Early result: on independent captures, embedding matching
falls to roughly chance (AUC ≈ 0.49) — the perfect SD302b separation does not
replicate, indicating the models match global appearance rather than ridge
structure.

### Repository layout

```
datasets/            # not committed — see download notes below
  sd302b/            #   NIST SD302b (contact livescan slap)
  ridgebase/         #   RidgeBase Task2 (contactless four-finger)
code/                # inference + pair-construction + evaluation
results/
  sd302b/            # v1 results (per model)
  ridgebase/         # v2 results (per model) + pair manifests
manuscript/          # the paper (LaTeX)
```

Datasets are not committed (22 GB). SD302b is available from NIST; RidgeBase
requires a signed license agreement from the University at Buffalo CUBS lab.

---

## The Dataset — NIST Special Database 302b

NIST SD302b was collected by IARPA for the Nail-to-Nail Fingerprint Challenge. 201 participants had their fingerprints captured on multiple scanner devices under controlled conditions.

### What a SLAP image looks like

A single SLAP image contains all four fingers of one hand pressed simultaneously against the scanner plate. The image shows the index, middle, ring, and little fingers side by side. Every participant was captured for both hands — right hand and left hand — producing two SLAP images per session.

### FRGP — Finger/Position Group

NIST uses a numbering system called FRGP to identify what fingers are in an image:

- **FRGP 13** — right hand, four fingers (index + middle + ring + little)
- **FRGP 14** — left hand, four fingers (index + middle + ring + little)
- FRGP 15 — both thumbs simultaneously (excluded from this benchmark)

Every image filename encodes this directly. For example:

```
00002303_R_500_slap_13.png
│        │  │        │
│        │  │        └── FRGP 13 = right hand 4-finger SLAP
│        │  └─────────── 500 PPI resolution
│        └────────────── Device R
└─────────────────────── Subject ID 00002303
```

### Devices and Resolutions

The dataset contains images from four scanner devices. This project uses two:

**Device R** — high-quality government-grade scanner
- Captures at both **500 PPI** and **1000 PPI**
- 92 subjects
- Image size at 500 PPI: 2,496 × 2,560 pixels
- Image size at 1,000 PPI: 4,992 × 5,120 pixels (4× the pixels of 500 PPI)

**Device S** — different scanner brand
- Captures at **500 PPI only**
- 108 subjects (completely different people from Device R — zero overlap)
- Image size: 1,600 × 1,500 pixels

Devices U and V capture individual rolled fingers (one finger at a time) — these are archived and not used in this benchmark.

### Dataset Files

After cleaning:

| File | Description |
|---|---|
| `dataset/slap_images.csv` | One row per SLAP image — 584 images, 21 columns |
| `dataset/master_dataframe.csv` | One row per finger per image — 2,336 rows, 33 columns (includes bounding boxes and rotation angles for each finger, used by future tasks) |
| `dataset/participants.csv` | Demographics: age, gender, race, work type per subject |
| `dataset/archive/` | 11,219 non-SLAP files moved here (rolls, thumbs, segmented crops) |

**slap_images.csv columns:**

| Column | What it contains |
|---|---|
| `subject_id` | 8-digit subject identifier |
| `image_name` | Filename |
| `file_path` | Path relative to `dataset/` |
| `which_hand` | right or left |
| `device` | R or S |
| `resolution_ppi` | 500 or 1000 |
| `frgp_slap` | 13 (right) or 14 (left) |
| `image_width_px` / `image_height_px` | Pixel dimensions |
| `file_size_bytes` / `sha256` | Integrity verification |
| `age`, `yob`, `gender`, `race`, `work_type`, `collection_day` | Participant demographics |
| `has_errata` | True if subject has a known annotation error |
| `errata_note` | Description of the error |
| `role` | primary_eval / task8_genuine_pair / excluded_errata |
| `genuine_pair_available` | True if subject has both R_500 and R_1000 images |

### Errata

Seven subjects in NIST SD302b have documented annotation errors — swapped labels, mislabeled fingers, or unusable regions. These subjects have `has_errata = True` and are excluded from all pair construction. They are never shown to the models.

---

## Task 8 — Fingerprint Verification

The benchmark task: given two SLAP fingerprint images, determine whether they belong to the same person.

This is a **verification** task (1:1 matching), not identification (1:N search). The model sees exactly two images and answers same person or different person.

### Genuine Pairs — Same Person

A genuine pair compares two impressions of the same person's hand.

```
Image 1:  Subject 2303 — Device R — 500 PPI — right hand (FRGP 13)
Image 2:  Subject 2303 — Device R — 1,000 PPI — right hand (FRGP 13)
Label: SAME PERSON  (ground truth = A)
```

The two images look visually different — the 1,000 PPI image is much larger and sharper, captured at a different press with slightly different finger placement and pressure. But the underlying ridge patterns are the same person's biology.

**Why cross-resolution?** Each subject was captured once at 500 PPI and once at 1,000 PPI. These are the only two impressions available for Device R subjects, so R_500 vs R_1,000 is the only way to create a genuine pair. This also makes the task harder: the model must recognize the same person despite a substantial appearance difference caused by resolution.

One subject produces two genuine pairs — one for the right hand and one for the left hand. After excluding the 4 errata subjects on Device R, 88 clean subjects remain, giving **176 genuine pairs** (88 right hand + 88 left hand).

### Impostor Pairs — Different People

An impostor pair compares two images from different people.

```
Image 1:  Subject 2303 — Device R — 500 PPI — right hand (FRGP 13)
Image 2:  Subject 2315 — Device R — 500 PPI — right hand (FRGP 13)
Label: DIFFERENT PEOPLE  (ground truth = B)
```

**Critical design choices:**
- Both images are the same hand (FRGP 13 vs FRGP 13, or FRGP 14 vs FRGP 14). If we compared a right hand to a left hand, the model could answer correctly just by noticing the hand shape. Keeping the same FRGP forces the model to look at ridge patterns.
- Same device and resolution where possible. This removes the easy visual cue of "these look different because one is blurrier."
- Errata subjects are excluded.

The current benchmark uses **176 impostor pairs** (88 right hand + 88 left hand), sampled to match the genuine count for a balanced 50/50 test set.

### The Complete Test Set

```
352 pairs total
├── 176 genuine   (88 FRGP 13 + 88 FRGP 14)
└── 176 impostor  (88 FRGP 13 + 88 FRGP 14)
```

The pairs are fixed with `seed=42` and saved to `results/task8_pairs.csv`. Every model and every prompting strategy is evaluated on the exact same 352 pairs, making results directly comparable.

---

## Models

### InternVL3-8B-Instruct
- Source: `OpenGVLab/InternVL3-8B-Instruct`
- Local path: `models/internvl3-8b/`
- Loaded in bfloat16
- VRAM: ~16 GB
- Backend: `transformers` AutoModel with `device_map="auto"`

### Qwen2.5-VL-7B-Instruct
- Source: `Qwen/Qwen2.5-VL-7B-Instruct`
- Local path: `models/qwen25vl-7b/`
- Loaded in 4-bit NF4 quantization (bitsandbytes)
- VRAM: ~6 GB
- Backend: `transformers` Qwen2_5_VLForConditionalGeneration
- Loading fix: `device_map={"": "cpu"}` during quantization, then `.to("cuda")` — prevents CUDA OOM during the quantization kernel on GPUs with limited free VRAM

### Qwen3-VL-8B-Instruct
- Source: `Qwen/Qwen3-VL-8B-Instruct`
- Local path: `models/qwen3vl-8b/`
- Loaded in 4-bit NF4 quantization (bitsandbytes)
- VRAM: ~6 GB
- Backend: `transformers` Qwen3VLForConditionalGeneration
- Best model in the benchmark: similarity-scoring AUC = 1.0, EER = 0.0%

### Gemma-3-12B-IT
- Source: `google/gemma-3-12b-it`
- Local path: `models/gemma3-12b/`
- Loaded in 4-bit NF4 quantization (bitsandbytes)
- VRAM: ~8 GB
- Backend: `transformers` Gemma3ForConditionalGeneration
- Third-best in the benchmark: similarity-scoring AUC = 0.837, EER = 15.1%

All four models above are loaded directly via `transformers` — no vLLM server required.

### Claude Opus 4.8 (proprietary, API)
- Source: Anthropic API (`--model anthropic --anthropic-model claude-opus-4-8`)
- Not run locally — requires `ANTHROPIC_API_KEY` in `.env`
- The only model that resists positive-bias collapse under **both** binary prompts
- Best binary verifier overall (zero-shot FAR = 20.2%); second-best similarity scoring (AUC = 0.953, EER = 11.75%)

---

## Image Preprocessing

Before any image is sent to a model, three preprocessing steps are applied:

**1. Grayscale conversion**
Fingerprint ridge patterns are structural, not color-based. Converting to grayscale (luminance channel) removes three nearly identical color channels and gives the model one clean signal.

**2. Autocontrast**
Stretches the pixel histogram so the darkest pixel becomes 0 and the brightest becomes 255. This normalizes brightness and contrast differences between devices and between 500 PPI and 1,000 PPI captures — directly addressing the cross-resolution appearance gap in genuine pairs.

**3. Square padding and resize**
The shorter dimension is padded with white pixels to make the image square before resizing to 448×448. This preserves the true aspect ratio of the fingerprint rather than squashing it, which would distort finger proportions and potentially make the same person look different.

```python
img = Image.open(path).convert("L")        # grayscale
img = ImageOps.autocontrast(img)            # normalize contrast
w, h = img.size
s = max(w, h)
canvas = Image.new("L", (s, s), 255)        # white square canvas
canvas.paste(img, ((s - w) // 2, (s - h) // 2))
canvas = canvas.resize((448, 448))
return canvas.convert("RGB")               # 3-channel for model input
```

---

## Prompting Strategies

All four strategies show the model the same two preprocessed images. The only variable is the text prompt.

### zero_shot
No context, no guidance. Just the question.

```
Below are two SLAP fingerprint images.
A SLAP image captures four fingers from one hand pressed simultaneously on a scanner.
Do these two images belong to the same person?
(A) Yes, same person
(B) No, different people
Reply with only the letter A or B.
```

### task_description
Adds an explanation that ridge patterns persist across impressions, and that variation in pressure and placement is normal.

```
A SLAP fingerprint image captures four fingers from one hand simultaneously.
The same person's fingers produce slightly different images each time they press the scanner
(different pressure, slight movement), but the ridge patterns remain the same.
...
```

### chain_of_thought
Asks the model to reason step by step before answering. The model produces a paragraph of analysis examining ridges, shape, and proportions, then answers.

```
Think step by step: examine the ridge patterns, general shape, and relative finger proportions.
Then answer: do these two images belong to the same person?
...
End your response with the letter A or B on the final line.
```

### similarity_score
Changes the output format entirely. Instead of a binary A/B answer, the model returns a continuous score from 0 to 100 with calibrated anchors.

```
Your task: Estimate the probability that both images were captured from the same person's hand.

Return your response in this exact format:
SCORE: <integer 0-100>
REASON: <one sentence>

Where SCORE means:
  0-30  = almost certainly different people
  31-60 = uncertain, notable differences
  61-85 = likely the same person
  86-100 = highly confident same person
```

The model returns something like:
```
SCORE: 78
REASON: The ridge flow patterns appear consistent across both images but the resolution difference makes detailed comparison difficult.
```

---

## The Collapse Problem

The central finding of this benchmark is **positive-bias collapse**.

Under binary prompting (zero_shot, task_description, chain_of_thought), both models overwhelmingly answer A (same person) for every pair. On a balanced 50/50 test set, this gives:

- Accuracy: ~50% (random chance)
- FAR (False Accept Rate): ~100% — every impostor pair is incorrectly accepted
- FRR (False Reject Rate): ~0% — every genuine pair is correctly accepted (for the wrong reason)

The model is not performing fingerprint verification. It is defaulting to "same person" for everything.

**Why does collapse happen?** SLAP images of the same hand look grossly similar regardless of who they belong to — four fingers, similar ridge density, similar overall shape. The model's prior is "these look like the same hand" before it examines any fine detail.

**Why does similarity scoring break collapse?** When forced into a binary choice, the model commits to "same person" because it cannot confidently say "different." When asked for a number, it expresses genuine uncertainty — genuine pairs score higher than impostor pairs even if the model never explicitly decides "same" or "different." This converts a collapsed binary classifier into a working discriminator.

---

## Results

All 8 evaluations completed. Results are in `results/`.

### Binary prompting (zero_shot, task_description, chain_of_thought)

| Model | Prompting | Accuracy | FAR | FRR | Collapsed |
|---|---|---|---|---|---|
| InternVL3-8B | zero_shot | 52.8% | 94.3% | 0.0% | Yes |
| InternVL3-8B | task_description | 50.0% | 100.0% | 0.0% | Yes |
| InternVL3-8B | chain_of_thought | 50.0% | 100.0% | 0.0% | Yes |
| Qwen2.5-VL-7B | zero_shot | 61.9% | 76.1% | 0.0% | No |
| Qwen2.5-VL-7B | task_description | 50.0% | 100.0% | 0.0% | Yes |
| Qwen2.5-VL-7B | chain_of_thought | 50.0% | 100.0% | 0.0% | Yes |

### Similarity scoring

| Model | Genuine Mean | Impostor Mean | Best Accuracy | EER | AUC |
|---|---|---|---|---|---|
| InternVL3-8B | 84.5 | 77.9 | 84.9% | 15.06% | 0.847 |
| Qwen2.5-VL-7B | 89.7 | 84.6 | 96.0% | 3.98% | 0.960 |

**AUC interpretation:** AUC measures the probability that a randomly chosen genuine pair receives a higher similarity score than a randomly chosen impostor pair. AUC 0.960 means Qwen2.5-VL correctly ranks a genuine pair above an impostor pair 96% of the time.

**EER** (Equal Error Rate) is the threshold where the false accept rate equals the false reject rate. Lower is better.

---

## Code

### `code/archive_non_slap.py`
Cleans the raw NIST SD302b dataset. Moves three categories of files to `dataset/archive/`:
- Roll images (devices U, V) — single-finger sequential captures
- FRGP 15 thumb slaps — two-thumb captures
- Slap-segmented crops — pre-cut individual finger images

Run once before anything else. Default is a dry run; use `--run` to execute.

```bash
python code/archive_non_slap.py          # dry run
python code/archive_non_slap.py --run    # execute
```

### `code/build_master_df.py`
Builds `dataset/master_dataframe.csv` — one row per finger per image (2,336 rows × 33 columns). Loads segmentation bounding boxes and rotation angles for each finger. Used for future tasks (finger localization, rotation estimation, quality scoring).

```bash
python code/build_master_df.py
```

### `code/build_slap_images_df.py`
Builds `dataset/slap_images.csv` — one row per SLAP image (584 rows × 21 columns). The active dataframe used by the Task 8 pipeline. Clean, flat, no per-finger expansion.

```bash
python code/build_slap_images_df.py
```

### `code/setup_models.py`
System check and model download utility. Verifies GPU VRAM, disk space, and required packages. Optionally downloads both models from HuggingFace.

```bash
python code/setup_models.py --check-only    # check system only
python code/setup_models.py --download      # download both models
```

### `code/run_verification.py`
The main evaluation pipeline. Loads a model once, runs all 352 pairs, checkpoints after every pair, computes metrics.

```bash
# Dry run — confirm pairs load correctly, no model needed
python code/run_verification.py --dry-run

# Save pair manifest
python code/run_verification.py --pairs-only

# Run evaluation
python code/run_verification.py --model internvl3 --prompting zero_shot --run
python code/run_verification.py --model internvl3 --prompting task_description --run
python code/run_verification.py --model internvl3 --prompting chain_of_thought --run
python code/run_verification.py --model internvl3 --prompting similarity_score --run

python code/run_verification.py --model qwen25vl  --prompting zero_shot --run
python code/run_verification.py --model qwen25vl  --prompting task_description --run
python code/run_verification.py --model qwen25vl  --prompting chain_of_thought --run
python code/run_verification.py --model qwen25vl  --prompting similarity_score --run

# Resume an interrupted run
python code/run_verification.py --model internvl3 --prompting zero_shot --run \
    --resume results/task8_internvl3_zero_shot_YYYYMMDD_HHMM.csv

# Print metrics from a completed results file
python code/run_verification.py --metrics results/task8_internvl3_zero_shot_YYYYMMDD_HHMM.csv
```

---

## Project Structure

```
MultiModel LLMs/
├── code/
│   ├── run_verification.py       # main pipeline — pairs, inference, metrics
│   ├── build_slap_images_df.py   # builds dataset/slap_images.csv
│   ├── build_master_df.py        # builds dataset/master_dataframe.csv (per-finger)
│   ├── archive_non_slap.py       # moves non-SLAP files to archive
│   └── setup_models.py           # system check and model download
├── dataset/
│   ├── slap_images.csv           # 584 rows — one per SLAP image (active dataframe)
│   ├── master_dataframe.csv      # 2,336 rows — one per finger per image
│   ├── participants.csv          # subject demographics
│   └── sd302b/
│       └── images/baseline/
│           ├── R/500/slap/png/   # 184 right+left hand images at 500 PPI
│           ├── R/1000/slap/png/  # 184 right+left hand images at 1,000 PPI
│           └── S/500/slap/png/   # 216 right+left hand images at 500 PPI
├── models/
│   ├── internvl3-8b/             # InternVL3-8B-Instruct weights (~16 GB)
│   └── qwen25vl-7b/              # Qwen2.5-VL-7B-Instruct weights (~17 GB)
├── results/
│   ├── task8_pairs.csv           # fixed 352-pair manifest (seed=42)
│   ├── task8_internvl3_*.csv     # per-pair results for InternVL3
│   ├── task8_qwen25vl_*.csv      # per-pair results for Qwen2.5-VL
│   └── *.metrics.json            # computed metrics for each run
├── manuscript/
│   ├── main.tex                  # IEEE journal paper
│   └── references.bib            # all citations
└── venv/                         # Python virtual environment
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA GPU with at least 6 GB VRAM (16 GB recommended for InternVL3)
- ~53 GB free disk space for both models

### Install packages

```bash
pip install 'transformers>=4.49.0' huggingface_hub accelerate bitsandbytes \
            qwen-vl-utils pandas Pillow torch torchvision
```

### HuggingFace login (once)

```bash
huggingface-cli login
```

### Download models

```bash
python code/setup_models.py --download
```

### Verify setup

```bash
python code/setup_models.py --check-only
python code/run_verification.py --dry-run
```

---

## Paper

The results of this benchmark are written up as an IEEE journal paper in `manuscript/main.tex`.

Title: *SLAPBench: Benchmarking Multimodal Large Language Models for Four-Finger SLAP Fingerprint Verification*

Authors: Bibesh Pyakurel, M. G. Sarwar Murshed — University of Wisconsin-Green Bay

To compile: upload `manuscript/` to [Overleaf](https://overleaf.com) and compile with pdfLaTeX. The `IEEEtran.cls` file must be present in the same directory as `main.tex`.
