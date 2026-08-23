"""
SLAPBench — Task 8: Genuine vs. Impostor Verification Pipeline
===============================================================
Reads master_dataframe.csv, builds balanced genuine/impostor pairs,
queries a multimodal LLM for each pair, and saves results to CSV.

Genuine pair:   same subject, R_500 vs R_1000 (different impressions)
Impostor pair:  different subjects, R_500 vs R_500, same FRGP

Models (loaded directly via transformers — no vLLM required):
    InternVL3-8B-Instruct    — bfloat16, ~16 GB VRAM
    Qwen2.5-VL-7B-Instruct  — 4-bit NF4 (bitsandbytes), ~6 GB VRAM
    Qwen3-VL-8B-Instruct    — 4-bit NF4 (bitsandbytes), ~6 GB VRAM
    Gemma-3-12B-IT           — 4-bit NF4 (bitsandbytes), ~8 GB VRAM

Usage
-----
# Step 1 — Inspect the pair plan (no model loaded):
    python code/run_verification.py --dry-run

# Step 2 — Save pair manifest to CSV:
    python code/run_verification.py --pairs-only

# Step 3 — Run evaluation:
    python code/run_verification.py --model internvl3 --prompting zero_shot --run
    python code/run_verification.py --model qwen25vl  --prompting zero_shot --run

# Step 4 — Resume an interrupted run:
    python code/run_verification.py --model internvl3 --prompting zero_shot --run --resume path/to/existing.csv

# Step 5 — Print metrics from a completed results file:
    python code/run_verification.py --metrics results/task8_internvl3_zero_shot_YYYYMMDD_HHMM.csv
"""

import argparse
import base64
import csv
import io
import itertools
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Load API keys from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; keys must be set as environment variables

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
SLAP_DF      = DATASET_ROOT / "slap_images.csv"
RESULTS_DIR    = PROJECT_ROOT / "results"
PAIRS_CSV      = RESULTS_DIR / "task8_pairs.csv"       # balanced (176G + 176I)
PAIRS_ALL_CSV  = RESULTS_DIR / "task8_pairs_all.csv"   # exhaustive (176G + 7,656I)

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "internvl3": {
        "display":   "InternVL3-8B-Instruct",
        "hf_id":     str(PROJECT_ROOT / "models" / "internvl3-8b"),
        "backend":   "internvl3",
        "dtype":     "bfloat16",
        "load_4bit": False,
    },
    "qwen25vl": {
        "display":   "Qwen2.5-VL-7B-Instruct",
        "hf_id":     str(PROJECT_ROOT / "models" / "qwen25vl-7b"),
        "backend":   "qwen25vl",
        "dtype":     "float16",
        "load_4bit": True,
    },
    "qwen3vl": {
        "display":   "Qwen3-VL-8B-Instruct",
        "hf_id":     str(PROJECT_ROOT / "models" / "qwen3vl-8b"),
        "backend":   "qwen3vl",
        "dtype":     "bfloat16",
        "load_4bit": True,
    },
    "gemma3": {
        "display":   "Gemma-3-12B-IT",
        "hf_id":     str(PROJECT_ROOT / "models" / "gemma3-12b"),
        "backend":   "gemma3",
        "dtype":     "bfloat16",
        "load_4bit": True,
    },
    "openai": {
        "display":   "gpt-4o",          # overwritten at runtime by --openai-model
        "backend":   "openai",
        "api_model": "gpt-4o",          # overwritten at runtime by --openai-model
    },
    "anthropic": {
        "display":   "claude-sonnet-4-6",   # overwritten at runtime by --anthropic-model
        "backend":   "anthropic",
        "api_model": "claude-sonnet-4-6",   # overwritten at runtime by --anthropic-model
    },
}

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are an expert fingerprint examiner."

PROMPTS = {
    "zero_shot": (
        "Below are two SLAP fingerprint images. "
        "A SLAP image captures four fingers from one hand pressed simultaneously on a scanner.\n"
        "Do these two images belong to the same person?\n"
        "(A) Yes, same person\n"
        "(B) No, different people\n"
        "Reply with only the letter A or B."
    ),
    "task_description": (
        "A SLAP fingerprint image captures four fingers from one hand simultaneously. "
        "The same person's fingers produce slightly different images each time they press the scanner "
        "(different pressure, slight movement), but the ridge patterns remain the same.\n\n"
        "Below are two SLAP fingerprint images. "
        "Do these two images belong to the same person?\n"
        "(A) Yes, same person\n"
        "(B) No, different people\n"
        "Reply with only the letter A or B."
    ),
    "similarity_score": (
        "You are a forensic fingerprint examiner analyzing two SLAP fingerprint images.\n"
        "A SLAP image captures four fingers (index, middle, ring, little) pressed simultaneously on a scanner.\n\n"
        "Your task: Examine the ridge flow patterns, ridge endings, bifurcations, and the relative shape and spacing "
        "of each finger across both images. Then return a single integer between 0 and 100 representing your "
        "confidence that both images belong to the same person, where:\n"
        "  0   = 0%% confident (certain they are different people)\n"
        "  50  = 50%% confident (completely uncertain)\n"
        "  100 = 100%% confident (certain they are the same person)\n\n"
        "A score of 65 means you are 65%% confident the images are from the same person.\n"
        "Reply with the number only. Do not include any text, explanation, or label."
    ),
}

# Prompts that return A/B answers vs. numeric score
SCORE_PROMPTS = {"similarity_score"}

# ── Image helper ───────────────────────────────────────────────────────────────

def image_to_base64(path: str, size: int = 448) -> str:
    """Convert image to base64-encoded PNG string for API submission."""
    img = load_pil(path, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def load_pil(path: str, size: int = 448) -> "Image":
    from PIL import Image, ImageOps
    img = Image.open(path).convert("L")       # grayscale — ridges are structural, not color
    img = ImageOps.autocontrast(img)           # normalize contrast across devices/resolutions
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), 255)       # square pad with white to preserve aspect ratio
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    canvas = canvas.resize((size, size))
    return canvas.convert("RGB")               # models expect 3-channel input

# ── Pair generation ────────────────────────────────────────────────────────────

def build_pairs(n_impostor_per_frgp: int | None = 88,
                n_genuine_per_frgp: int | None = None,
                seed: int = 42) -> list[dict]:
    """
    Returns a list of pair dicts.

    Genuine pairs:  same subject, R_500 vs R_1000.
                    n_genuine_per_frgp=None → all 88 per FRGP (default, used for full runs).
                    n_genuine_per_frgp=int  → random sample of that many per FRGP.
    Impostor pairs: different subjects, R_500 vs R_500, same FRGP.
                    n_impostor_per_frgp=None → all C(88,2) = 3,828 per FRGP (7,656 total).
                    n_impostor_per_frgp=int  → random sample of that many per FRGP.
    """
    random.seed(seed)
    df = pd.read_csv(SLAP_DF)

    r500  = df[(df.device == "R") & (df.resolution_ppi == 500)  & (~df.has_errata)]
    r1000 = df[(df.device == "R") & (df.resolution_ppi == 1000) & (~df.has_errata)]

    pairs = []

    for frgp in [13, 14]:
        hand = "right" if frgp == 13 else "left"

        pool500  = r500[r500.frgp_slap == frgp].set_index("subject_id")
        pool1000 = r1000[r1000.frgp_slap == frgp].set_index("subject_id")
        common   = list(set(pool500.index) & set(pool1000.index))
        random.shuffle(common)

        # ── Genuine ───────────────────────────────────────────────────────────
        genuine_sids = common if n_genuine_per_frgp is None else common[:n_genuine_per_frgp]
        for sid in genuine_sids:
            row500  = pool500.loc[sid]
            row1000 = pool1000.loc[sid]
            pairs.append(_make_pair(
                pair_id   = f"G_{sid}_frgp{frgp}",
                label     = "genuine",
                frgp      = frgp,
                hand      = hand,
                s1        = str(sid).zfill(8),
                s2        = str(sid).zfill(8),
                img1_path = str(DATASET_ROOT / row500["file_path"]),
                img2_path = str(DATASET_ROOT / row1000["file_path"]),
                r1        = row500,
                r2        = row1000,
            ))

        # ── Impostor ──────────────────────────────────────────────────────────
        subs = list(pool500.index)

        if n_impostor_per_frgp is None:
            # All C(n,2) pairs — deterministic order, no random sampling
            for s1_id, s2_id in itertools.combinations(sorted(subs), 2):
                row1 = pool500.loc[s1_id]
                row2 = pool500.loc[s2_id]
                pairs.append(_make_pair(
                    pair_id   = f"I_{str(s1_id).zfill(8)}_{str(s2_id).zfill(8)}_frgp{frgp}",
                    label     = "impostor",
                    frgp      = frgp,
                    hand      = hand,
                    s1        = str(s1_id).zfill(8),
                    s2        = str(s2_id).zfill(8),
                    img1_path = str(DATASET_ROOT / row1["file_path"]),
                    img2_path = str(DATASET_ROOT / row2["file_path"]),
                    r1        = row1,
                    r2        = row2,
                ))
        else:
            random.shuffle(subs)
            count = 0
            used  = set()
            for i in range(len(subs)):
                if count >= n_impostor_per_frgp:
                    break
                for j in range(i + 1, len(subs)):
                    if count >= n_impostor_per_frgp:
                        break
                    s1, s2 = str(subs[i]).zfill(8), str(subs[j]).zfill(8)
                    key = tuple(sorted([s1, s2]))
                    if key in used:
                        continue
                    used.add(key)
                    row1 = pool500.loc[subs[i]]
                    row2 = pool500.loc[subs[j]]
                    pairs.append(_make_pair(
                        pair_id   = f"I_{s1}_{s2}_frgp{frgp}",
                        label     = "impostor",
                        frgp      = frgp,
                        hand      = hand,
                        s1        = s1,
                        s2        = s2,
                        img1_path = str(DATASET_ROOT / row1["file_path"]),
                        img2_path = str(DATASET_ROOT / row2["file_path"]),
                        r1        = row1,
                        r2        = row2,
                    ))
                    count += 1

    random.shuffle(pairs)
    return pairs


def _make_pair(pair_id, label, frgp, hand, s1, s2,
               img1_path, img2_path, r1, r2) -> dict:
    return {
        "pair_id":        pair_id,
        "label":          label,
        "ground_truth":   "A" if label == "genuine" else "B",
        "frgp":           frgp,
        "which_hand":     hand,
        "subject1":       s1,
        "subject2":       s2,
        "img1_path":      img1_path,
        "img2_path":      img2_path,
        # Demographics for stratified analysis
        "s1_age":         r1.get("age"),
        "s1_gender":      r1.get("gender"),
        "s1_race":        r1.get("race"),
        "s2_age":         r2.get("age"),
        "s2_gender":      r2.get("gender"),
        "s2_race":        r2.get("race"),
    }


def save_pairs_csv(pairs: list[dict], path: Path | None = None) -> None:
    p = path or PAIRS_CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(pairs).to_csv(p, index=False)
    print(f"Pair manifest saved → {p.name}  ({len(pairs)} pairs)")

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_key: str):
    """
    Load model + processor/tokenizer into GPU memory.
    Returns (model, processor_or_tokenizer, backend_tag).
    Call once before the evaluation loop; keep in memory for all pairs.
    """
    cfg     = MODELS[model_key]
    backend = cfg["backend"]

    if backend not in ("openai", "anthropic"):
        import torch
        hf_id     = cfg["hf_id"]
        load_4bit = cfg["load_4bit"]
        dtype     = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
        print(f"Loading {cfg['display']} ...")

    if backend == "internvl3":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        ).eval()
        return model, tokenizer, backend

    if backend == "qwen25vl":
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_storage=torch.uint8,
        )

        # Load entirely on CPU first so bitsandbytes quantizes on CPU (avoids
        # GPU OOM during the quantize_4bit kernel when VRAM is limited).
        # After .eval(), inference still dispatches to GPU via device_map hooks.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id,
            quantization_config=bnb_cfg,
            device_map={"": "cpu"},   # quantize every layer on CPU
            low_cpu_mem_usage=True,
        )
        # Move quantized model to GPU now that VRAM is only needed for the
        # already-compressed weights (~6 GB), not the uncompressed shards.
        model = model.to("cuda").eval()

        processor = AutoProcessor.from_pretrained(hf_id)
        return model, processor, backend

    if backend == "qwen3vl":
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_storage=torch.uint8,
        )

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_id,
            quantization_config=bnb_cfg,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
        model = model.to("cuda").eval()

        processor = AutoProcessor.from_pretrained(hf_id)
        return model, processor, backend

    if backend == "gemma3":
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

        model = Gemma3ForConditionalGeneration.from_pretrained(
            hf_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()

        processor = AutoProcessor.from_pretrained(hf_id)
        return model, processor, backend

    if backend == "openai":
        import os
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to .env or export it as an environment variable."
            )
        client = OpenAI(api_key=api_key)
        model_name = cfg["api_model"]
        print(f"OpenAI client ready — model: {model_name}")
        return client, model_name, backend

    if backend == "anthropic":
        import os
        import anthropic as ant
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Add it to .env or export it as an environment variable."
            )
        client = ant.Anthropic(api_key=api_key)
        model_name = cfg["api_model"]
        print(f"Anthropic client ready — model: {model_name}")
        return client, model_name, backend

    raise ValueError(f"Unknown backend: {backend}")


# ── Inference backends ─────────────────────────────────────────────────────────

def _infer_internvl3(model, tokenizer, img1_path: str, img2_path: str,
                     prompt: str) -> str:
    import torch
    import torchvision.transforms as T

    # Find the device where the vision encoder's first layer lives.
    # device_map="auto" may spread layers across CPU+GPU; pixel values
    # must go to whichever device receives them first in the forward pass.
    try:
        device = next(model.vision_model.parameters()).device
    except AttributeError:
        try:
            device = next(model.vision_encoder.parameters()).device
        except AttributeError:
            device = next(p for p in model.parameters() if p.device.type == "cuda").device

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def load_img(path):
        img = load_pil(path, size=448)   # grayscale → autocontrast → square pad → 448×448
        return transform(img).unsqueeze(0).to(torch.bfloat16).to(device)

    pixel_values = torch.cat([load_img(img1_path), load_img(img2_path)], dim=0)
    question     = f"<image>\n<image>\n{prompt}"
    gen_config   = dict(max_new_tokens=150, do_sample=False)

    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        gen_config,
        num_patches_list=[1, 1],
    )
    return response.strip()


def _infer_qwen25vl(model, processor, img1_path: str, img2_path: str,
                    prompt_text: str, max_new_tokens: int = 48) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    img1 = load_pil(img1_path, size=448)   # grayscale → autocontrast → square pad → 448×448
    img2 = load_pil(img2_path, size=448)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    trimmed = [o[len(i):] for o, i in zip(output_ids, inputs["input_ids"])]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def _infer_qwen3vl(model, processor, img1_path: str, img2_path: str,
                   prompt_text: str, max_new_tokens: int = 48) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    img1 = load_pil(img1_path, size=448)
    img2 = load_pil(img2_path, size=448)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    trimmed = [o[len(i):] for o, i in zip(output_ids, inputs["input_ids"])]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def _infer_gemma3(model, processor, img1_path: str, img2_path: str,
                  prompt_text: str, max_new_tokens: int = 48) -> str:
    import torch

    img1 = load_pil(img1_path, size=448)
    img2 = load_pil(img2_path, size=448)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
            ],
        },
    ]

    # Get formatted text with image placeholders
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process text + images together so pixel_values are created correctly
    raw_inputs = processor(
        text=text,
        images=[img1, img2],
        return_tensors="pt",
    )
    # Cast float tensors to bfloat16 to match quantized model compute dtype
    inputs = {
        k: v.to("cuda", dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to("cuda")
        for k, v in raw_inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generation = output_ids[0][input_len:]
    return processor.decode(generation, skip_special_tokens=True).strip()


def _infer_openai(client, model_name: str, img1_path: str, img2_path: str,
                  prompt_text: str, max_retries: int = 5) -> str:
    img1_b64 = image_to_base64(img1_path)
    img2_b64 = image_to_base64(img2_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
            ],
        },
    ]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=50,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt
                print(f"  Rate limited (OpenAI), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"OpenAI API failed after {max_retries} retries")


def _infer_anthropic(client, model_name: str, img1_path: str, img2_path: str,
                     prompt_text: str, max_retries: int = 5) -> str:
    img1_b64 = image_to_base64(img1_path)
    img2_b64 = image_to_base64(img2_path)
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=50,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img1_b64,
                                },
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img2_b64,
                                },
                            },
                        ],
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "overloaded" in str(e).lower():
                wait = 2 ** attempt
                print(f"  Rate limited (Anthropic), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Anthropic API failed after {max_retries} retries")


def call_model(model, proc_or_tok, backend: str,
               img1: str, img2: str,
               prompt_text: str) -> tuple[str, float]:
    """Unified inference call. Returns (raw_response, latency_seconds)."""
    t0 = time.perf_counter()
    if backend == "internvl3":
        raw = _infer_internvl3(model, proc_or_tok, img1, img2, prompt_text)
    elif backend == "qwen25vl":
        raw = _infer_qwen25vl(model, proc_or_tok, img1, img2, prompt_text)
    elif backend == "qwen3vl":
        raw = _infer_qwen3vl(model, proc_or_tok, img1, img2, prompt_text)
    elif backend == "gemma3":
        raw = _infer_gemma3(model, proc_or_tok, img1, img2, prompt_text)
    elif backend == "openai":
        raw = _infer_openai(model, proc_or_tok, img1, img2, prompt_text)
    elif backend == "anthropic":
        raw = _infer_anthropic(model, proc_or_tok, img1, img2, prompt_text)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return raw, time.perf_counter() - t0


def parse_answer(raw: str) -> str:
    """Extract A or B from model response using 3-step fallback (VLMEvalKit style)."""
    upper = raw.upper().strip()
    # Step 1: starts with the letter
    if upper.startswith("A"):
        return "A"
    if upper.startswith("B"):
        return "B"
    # Step 2: letter appears anywhere
    for letter in ("A", "B"):
        if letter in upper:
            return letter
    # Step 3: response text matches option values
    if any(kw in upper for kw in ("SAME", "YES", "GENUINE")):
        return "A"
    if any(kw in upper for kw in ("DIFFERENT", "NO", "IMPOSTOR")):
        return "B"
    return "INVALID"


def parse_score_response(raw: str) -> int | None:
    """Extract integer 0–100 from similarity_score prompt response."""
    match = re.search(r"\b(\d{1,3})\b", raw.strip())
    if not match:
        return None
    score = int(match.group(1))
    return max(0, min(100, score))   # clamp to [0, 100]

# ── Evaluation loop ────────────────────────────────────────────────────────────

def run_evaluation(pairs: list[dict], model_key: str,
                   prompting: str = "zero_shot",
                   delay_s: float = 0.0,
                   resume_path: str | None = None) -> Path:
    """
    Loads the model once, then queries it for every pair.
    Saves results incrementally (checkpoint after every pair).
    Pass resume_path to skip already-processed pairs.

    Returns path to results CSV.
    """
    model_cfg  = MODELS[model_key]
    prompt_txt = PROMPTS[prompting]
    date_str   = datetime.now().strftime("%Y%m%d_%H%M")

    latest_dir   = RESULTS_DIR / model_key / "latest"
    previous_dir = RESULTS_DIR / model_key / "previous"
    latest_dir.mkdir(parents=True, exist_ok=True)
    previous_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: reuse existing file path so rows append correctly ────────────
    processed = set()
    if resume_path:
        out_path = Path(resume_path)
        if out_path.exists():
            existing  = pd.read_csv(out_path)
            processed = set(existing["pair_id"].tolist())
            print(f"Resuming {out_path.name} — {len(processed)} pairs already done.")
    else:
        # Rotate: move current latest/ contents into previous/<date>/
        existing_files = list(latest_dir.iterdir())
        if existing_files:
            import shutil
            date_folder = datetime.now().strftime("%Y-%m-%d")
            archive_dir = previous_dir / date_folder
            archive_dir.mkdir(parents=True, exist_ok=True)
            for f in existing_files:
                shutil.move(str(f), str(archive_dir / f.name))
            print(f"Rotated previous latest/ → previous/{date_folder}/  ({len(existing_files)} files)")
        out_path = latest_dir / f"task8_{model_key}_{prompting}_{date_str}.csv"

    # ── Load model ────────────────────────────────────────────────────────────
    model, proc_or_tok, backend = load_model(model_key)
    print(f"Model loaded: {model_cfg['display']}\n")

    is_score_mode = prompting in SCORE_PROMPTS

    if is_score_mode:
        fieldnames = [
            "pair_id", "label", "frgp", "which_hand", "subject1", "subject2",
            "ground_truth", "similarity_score", "raw_response",
            "latency_s", "model", "prompting_setting", "timestamp",
        ]
    else:
        fieldnames = [
            "pair_id", "label", "frgp", "which_hand", "subject1", "subject2",
            "ground_truth", "llm_answer", "raw_response", "correct",
            "latency_s", "model", "prompting_setting", "timestamp",
        ]

    RESULTS_DIR.mkdir(exist_ok=True)
    write_header = not out_path.exists() or len(processed) == 0
    f_out = open(out_path, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    todo   = [p for p in pairs if p["pair_id"] not in processed]
    total  = len(pairs)
    done   = len(processed)
    errors = 0

    print(f"\nModel:      {model_cfg['display']}")
    print(f"Prompting:  {prompting}")
    print(f"Pairs:      {len(todo)} to run  ({done} already done, {total} total)")
    print(f"Output:     {out_path.name}\n")

    for p in todo:
        try:
            raw, latency = call_model(
                model, proc_or_tok, backend,
                p["img1_path"],
                p["img2_path"],
                prompt_txt,
            )

            if is_score_mode:
                score = parse_score_response(raw)
                writer.writerow({
                    "pair_id":           p["pair_id"],
                    "label":             p["label"],
                    "frgp":              p["frgp"],
                    "which_hand":        p["which_hand"],
                    "subject1":          p["subject1"],
                    "subject2":          p["subject2"],
                    "ground_truth":      p["ground_truth"],
                    "similarity_score":  score,
                    "raw_response":      raw,
                    "latency_s":         round(latency, 3),
                    "model":             model_cfg["display"],
                    "prompting_setting": prompting,
                    "timestamp":         datetime.now().isoformat(timespec="seconds"),
                })
                f_out.flush()
                done += 1
                score_str = str(score) if score is not None else "INVALID"
                gt_label  = "genuine" if p["ground_truth"] == "A" else "impostor"
                print(f"  [{done:>3}/{total}] {p['label']:>9}  frgp={p['frgp']}"
                      f"  score={score_str:>3}  ({gt_label})  ({latency:.2f}s)")
            else:
                answer  = parse_answer(raw)
                correct = answer == p["ground_truth"]
                writer.writerow({
                    "pair_id":           p["pair_id"],
                    "label":             p["label"],
                    "frgp":              p["frgp"],
                    "which_hand":        p["which_hand"],
                    "subject1":          p["subject1"],
                    "subject2":          p["subject2"],
                    "ground_truth":      p["ground_truth"],
                    "llm_answer":        answer,
                    "raw_response":      raw,
                    "correct":           correct,
                    "latency_s":         round(latency, 3),
                    "model":             model_cfg["display"],
                    "prompting_setting": prompting,
                    "timestamp":         datetime.now().isoformat(timespec="seconds"),
                })
                f_out.flush()   # checkpoint after every pair
                done += 1
                status = "✓" if correct else "✗"
                print(f"  [{done:>3}/{total}] {p['label']:>9}  frgp={p['frgp']}"
                      f"  GT={p['ground_truth']}  LLM={answer}  {status}"
                      f"  ({latency:.2f}s)")

        except Exception as exc:
            errors += 1
            print(f"  [{done+1:>3}/{total}] ERROR on {p['pair_id']}: {exc}")

        if delay_s > 0:
            time.sleep(delay_s)

    f_out.close()
    print(f"\nDone. {done} pairs processed, {errors} errors.")
    print(f"Results → {out_path}")
    return out_path

# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_score_metrics(results_path: Path) -> dict:
    """Metrics for similarity_score prompting: thresholded accuracy, EER, AUC."""
    df = pd.read_csv(results_path)
    df = df.dropna(subset=["similarity_score"])
    df["similarity_score"] = df["similarity_score"].astype(float)

    genuine  = df[df["label"] == "genuine"]["similarity_score"].tolist()
    impostor = df[df["label"] == "impostor"]["similarity_score"].tolist()
    invalid  = int(pd.read_csv(results_path)["similarity_score"].isna().sum())

    # Sweep thresholds to find EER and best accuracy
    thresholds = list(range(0, 101))
    best_acc, best_thresh = 0.0, 50
    far_curve, tar_curve  = [], []

    for t in thresholds:
        # Predict genuine if score >= threshold
        tp = sum(1 for s in genuine  if s >= t)
        fn = sum(1 for s in genuine  if s <  t)
        fp = sum(1 for s in impostor if s >= t)
        tn = sum(1 for s in impostor if s <  t)
        n_g = len(genuine) or 1
        n_i = len(impostor) or 1
        tar = tp / n_g        # True Accept Rate
        far = fp / n_i        # False Accept Rate
        acc = (tp + tn) / (n_g + n_i) * 100
        far_curve.append(far)
        tar_curve.append(tar)
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    # Equal Error Rate — threshold where FAR ≈ FRR
    eer, eer_thresh = 1.0, 50
    for i, t in enumerate(thresholds):
        frr = 1.0 - tar_curve[i]
        diff = abs(far_curve[i] - frr)
        if diff < abs(far_curve[thresholds.index(eer_thresh)] -
                      (1.0 - tar_curve[thresholds.index(eer_thresh)])):
            eer        = (far_curve[i] + frr) / 2
            eer_thresh = t

    # AUC via trapezoidal rule on the ROC curve (FAR on x, TAR on y)
    sorted_pairs = sorted(zip(far_curve, tar_curve))
    auc = sum(
        (sorted_pairs[i+1][0] - sorted_pairs[i][0]) *
        (sorted_pairs[i+1][1] + sorted_pairs[i][1]) / 2
        for i in range(len(sorted_pairs) - 1)
    )

    per_frgp = {}
    for frgp in df["frgp"].unique():
        sub = df[df["frgp"] == frgp]
        g   = sub[sub["label"] == "genuine"]["similarity_score"]
        imp = sub[sub["label"] == "impostor"]["similarity_score"]
        per_frgp[int(frgp)] = {
            "genuine_mean":  round(g.mean(), 1)   if len(g)   else None,
            "impostor_mean": round(imp.mean(), 1) if len(imp) else None,
            "n": len(sub),
        }

    return {
        "model":           df["model"].iloc[0] if len(df) else "unknown",
        "prompting":       df["prompting_setting"].iloc[0] if len(df) else "unknown",
        "total_pairs":     len(df),
        "genuine_count":   len(df[df["label"] == "genuine"]),
        "impostor_count":  len(df[df["label"] == "impostor"]),
        "invalid_count":   invalid,
        "genuine_mean_score":  round(sum(genuine)  / len(genuine)  if genuine  else 0, 1),
        "impostor_mean_score": round(sum(impostor) / len(impostor) if impostor else 0, 1),
        "best_threshold":  best_thresh,
        "best_acc":        round(best_acc, 1),
        "eer":             round(eer * 100, 2),
        "eer_threshold":   eer_thresh,
        "auc":             round(auc, 4),
        "per_frgp":        per_frgp,
    }


def compute_metrics(results_path: Path) -> dict:
    df = pd.read_csv(results_path)
    total   = len(df)
    genuine = df[df["label"] == "genuine"]
    impostor= df[df["label"] == "impostor"]

    overall_acc = df["correct"].mean() * 100
    genuine_acc = genuine["correct"].mean() * 100
    impostor_acc= impostor["correct"].mean() * 100

    # FAR = impostor pairs incorrectly accepted (LLM said A = same person)
    far = (impostor["llm_answer"] == "A").mean() * 100
    # FRR = genuine pairs incorrectly rejected (LLM said B = different)
    frr = (genuine["llm_answer"] == "B").mean() * 100

    # Collapse detection: does the model always pick the same answer?
    answer_dist = df["llm_answer"].value_counts(normalize=True).to_dict()
    collapsed   = max(answer_dist.values(), default=0) > 0.95

    # INVALID responses
    invalid = (df["llm_answer"] == "INVALID").sum()

    # Per-FRGP breakdown
    per_frgp = {}
    for frgp in df["frgp"].unique():
        sub = df[df["frgp"] == frgp]
        per_frgp[frgp] = {
            "acc":   sub["correct"].mean() * 100,
            "far":   (sub[sub.label=="impostor"]["llm_answer"]=="A").mean()*100,
            "frr":   (sub[sub.label=="genuine"]["llm_answer"]=="B").mean()*100,
            "n":     len(sub),
        }

    metrics = {
        "model":         df["model"].iloc[0] if len(df) else "unknown",
        "prompting":     df["prompting_setting"].iloc[0] if len(df) else "unknown",
        "total_pairs":   total,
        "genuine_count": len(genuine),
        "impostor_count":len(impostor),
        "overall_acc":   overall_acc,
        "genuine_acc":   genuine_acc,
        "impostor_acc":  impostor_acc,
        "far":           far,
        "frr":           frr,
        "collapsed":     collapsed,
        "answer_dist":   answer_dist,
        "invalid_count": invalid,
        "per_frgp":      per_frgp,
    }
    return metrics


def print_metrics(m: dict) -> None:
    print("\n" + "=" * 56)
    print(f"  Model:      {m['model']}")
    print(f"  Prompting:  {m['prompting']}")
    print(f"  Pairs:      {m['total_pairs']}  "
          f"(genuine={m['genuine_count']}, impostor={m['impostor_count']})")
    print("=" * 56)
    print(f"  Overall accuracy :  {m['overall_acc']:.1f}%")
    print(f"  Genuine accuracy :  {m['genuine_acc']:.1f}%  "
          f"(True Accept Rate = {100-m['frr']:.1f}%)")
    print(f"  Impostor accuracy:  {m['impostor_acc']:.1f}%  "
          f"(True Reject Rate = {100-m['far']:.1f}%)")
    print(f"  FAR (false accept): {m['far']:.1f}%")
    print(f"  FRR (false reject): {m['frr']:.1f}%")
    print(f"  Answer distribution: {m['answer_dist']}")
    if m["collapsed"]:
        print("  *** MODEL COLLAPSED — always picking same answer ***")
    if m["invalid_count"]:
        print(f"  Invalid responses: {m['invalid_count']}")
    print("─" * 56)
    print("  Per-FRGP:")
    for frgp, v in sorted(m["per_frgp"].items()):
        hand = "right" if frgp == 13 else "left"
        print(f"    FRGP {frgp} ({hand:5s}):  "
              f"acc={v['acc']:.1f}%  FAR={v['far']:.1f}%  FRR={v['frr']:.1f}%  "
              f"(n={v['n']})")
    print("=" * 56)


def print_score_metrics(m: dict) -> None:
    print("\n" + "=" * 56)
    print(f"  Model:      {m['model']}")
    print(f"  Prompting:  {m['prompting']}")
    print(f"  Pairs:      {m['total_pairs']}  "
          f"(genuine={m['genuine_count']}, impostor={m['impostor_count']})")
    if m["invalid_count"]:
        print(f"  Invalid (no score parsed): {m['invalid_count']}")
    print("=" * 56)
    print(f"  Genuine  mean score:  {m['genuine_mean_score']:.1f}")
    print(f"  Impostor mean score:  {m['impostor_mean_score']:.1f}")
    print(f"  Separation (delta):   {m['genuine_mean_score'] - m['impostor_mean_score']:.1f} pts")
    print("─" * 56)
    print(f"  Best threshold:  {m['best_threshold']}  →  accuracy {m['best_acc']:.1f}%")
    print(f"  EER:             {m['eer']:.2f}%  (at threshold {m['eer_threshold']})")
    print(f"  AUC (ROC):       {m['auc']:.4f}")
    print("─" * 56)
    print("  Per-FRGP:")
    for frgp, v in sorted(m["per_frgp"].items()):
        hand = "right" if frgp == 13 else "left"
        print(f"    FRGP {frgp} ({hand:5s}):  "
              f"genuine_mean={v['genuine_mean']}  impostor_mean={v['impostor_mean']}  "
              f"(n={v['n']})")
    print("=" * 56)


def save_metrics_json(m: dict, results_path: Path) -> None:
    import numpy as np

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {convert(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    out = results_path.with_suffix(".metrics.json")
    with open(out, "w") as f:
        json.dump(convert(m), f, indent=2)
    print(f"Metrics saved → {out.name}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SLAPBench Task 8 — Genuine vs. Impostor Verification"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",      action="store_true",
                      help="Print pair plan without calling the API")
    mode.add_argument("--pairs-only",   action="store_true",
                      help="Generate and save pair manifest CSV, no API calls")
    mode.add_argument("--run",          action="store_true",
                      help="Run the LLM evaluation")
    mode.add_argument("--metrics",      type=str, metavar="RESULTS_CSV",
                      help="Print metrics from an existing results CSV")

    parser.add_argument("--model",      choices=list(MODELS.keys()),
                        default="internvl3",
                        help="Which model to use (default: internvl3)")
    parser.add_argument("--openai-model", type=str, default="gpt-4o",
                        help="OpenAI model name (default: gpt-4o). Only used with --model openai.")
    parser.add_argument("--anthropic-model", type=str, default="claude-sonnet-4-6",
                        help="Anthropic model name (default: claude-sonnet-4-6). Only used with --model anthropic.")
    parser.add_argument("--prompting",  choices=list(PROMPTS.keys()),
                        default="zero_shot",
                        help="Prompting strategy (default: zero_shot)")
    parser.add_argument("--n-genuine",      type=int, default=None,
                        help="Genuine pairs per FRGP (default: all 88). Use small numbers for trial runs.")
    parser.add_argument("--n-impostor",     type=int, default=88,
                        help="Impostor pairs per FRGP (default: 88, matches genuine count)")
    parser.add_argument("--all-impostors", action="store_true",
                        help="Use ALL C(88,2)=3,828 impostor pairs per FRGP (7,656 total). "
                             "Saves manifest to task8_pairs_all.csv.")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--delay",      type=float, default=0.0,
                        help="Seconds between inference calls (default: 0, no throttle needed for local model)")
    parser.add_argument("--resume",     type=str, default=None, metavar="RESULTS_CSV",
                        help="Path to existing results CSV to resume from")
    args = parser.parse_args()

    # ── Wire API model names into registry before anything else uses them ────
    if args.model == "openai":
        MODELS["openai"]["display"]   = args.openai_model
        MODELS["openai"]["api_model"] = args.openai_model
    elif args.model == "anthropic":
        MODELS["anthropic"]["display"]   = args.anthropic_model
        MODELS["anthropic"]["api_model"] = args.anthropic_model

    # ── Metrics mode — no pairs needed ────────────────────────────────────────
    if args.metrics:
        p = Path(args.metrics)
        if not p.exists():
            print(f"File not found: {p}")
            return
        # Detect score vs. A/B mode from columns present in the file
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        if "similarity_score" in cols:
            m = compute_score_metrics(p)
            print_score_metrics(m)
        else:
            m = compute_metrics(p)
            print_metrics(m)
        save_metrics_json(m, p)
        return

    # ── Build pairs ───────────────────────────────────────────────────────────
    n_imp = None if args.all_impostors else args.n_impostor
    pairs_csv_path = PAIRS_ALL_CSV if args.all_impostors else PAIRS_CSV

    print("Building pairs...")
    if args.all_impostors:
        print("  Mode: EXHAUSTIVE — all C(88,2)=3,828 impostor pairs per FRGP")
    pairs = build_pairs(n_impostor_per_frgp=n_imp, n_genuine_per_frgp=args.n_genuine, seed=args.seed)
    genuine_n  = sum(1 for p in pairs if p["label"] == "genuine")
    impostor_n = sum(1 for p in pairs if p["label"] == "impostor")

    print(f"  Total pairs:   {len(pairs)}")
    print(f"  Genuine:       {genuine_n}")
    print(f"  Impostor:      {impostor_n}")
    print(f"  FRGP 13 (right): {sum(1 for p in pairs if p['frgp']==13)}")
    print(f"  FRGP 14 (left):  {sum(1 for p in pairs if p['frgp']==14)}")

    if args.dry_run:
        print("\n--- Sample pairs (first 6) ---")
        for p in pairs[:6]:
            i1 = Path(p["img1_path"]).name
            i2 = Path(p["img2_path"]).name
            both_exist = Path(p["img1_path"]).exists() and Path(p["img2_path"]).exists()
            print(f"  [{p['label']:>9}] frgp={p['frgp']}  "
                  f"{i1}  vs  {i2}  files_exist={both_exist}")
        print("\nDry run complete — no API calls made.")
        return

    if args.pairs_only:
        save_pairs_csv(pairs, pairs_csv_path)
        return

    # ── Run evaluation ─────────────────────────────────────────────────────────
    results_path = run_evaluation(
        pairs,
        model_key   = args.model,
        prompting   = args.prompting,
        delay_s     = args.delay,
        resume_path = args.resume,
    )
    if args.prompting in SCORE_PROMPTS:
        m = compute_score_metrics(results_path)
        print_score_metrics(m)
    else:
        m = compute_metrics(results_path)
        print_metrics(m)
    save_metrics_json(m, results_path)


if __name__ == "__main__":
    main()
