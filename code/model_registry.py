"""Local model locations shared by setup and inference.

Weights live outside Git. ``models/`` may be a directory or a local symlink;
SLAPBENCH_MODELS_DIR can point to another storage location on each machine.
"""

import json
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

# Separately authorized additions. Keep the historical default download set intact.
ADDITIONAL_MODELS = {
    "qwen35": ("Qwen/Qwen3.5-9B", "qwen35-9b",
               "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "gemma4": ("google/gemma-4-12B-it", "gemma4-12b-it",
               "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    "internvl35": ("OpenGVLab/InternVL3_5-8B", "internvl35-8b",
                  "9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79"),
}
STORAGE_CONFIG = PROJECT_ROOT / ".cache" / "local-model-storage.json"


def additional_models_dir(config_path=STORAGE_CONFIG):
    """Resolve storage for new models without changing historical model paths."""
    configured = os.environ.get("SLAPBENCH_MODELS_DIR")
    if configured:
        return Path(configured).expanduser()
    if config_path.exists():
        configured = json.loads(config_path.read_text())["models_dir"]
        return Path(configured).expanduser()
    return MODELS_DIR


ADDITIONAL_MODELS_DIR = additional_models_dir()
