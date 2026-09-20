"""Local model locations shared by setup and inference.

Weights live outside Git. ``models/`` may be a directory or a local symlink;
SLAPBENCH_MODELS_DIR can point to another storage location on each machine.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("SLAPBENCH_MODELS_DIR") or PROJECT_ROOT / "models").expanduser()

LOCAL_MODELS = {
    "internvl3": ("OpenGVLab/InternVL3-8B", "internvl3-8b"),
    "qwen25vl": ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen25vl-7b"),
    "qwen3vl": ("Qwen/Qwen3-VL-8B-Instruct", "qwen3vl-8b"),
    "gemma3": ("google/gemma-3-12b-it", "gemma3-12b"),
}

# Inference currently loads this model by repository ID from the Hugging Face cache.
PIXTRAL_ID = "unsloth/Pixtral-12B-2409-bnb-4bit"
