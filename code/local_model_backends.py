"""Local-only loaders for the October 2026 additions (no automatic downloads)."""

from pathlib import Path

from model_registry import ADDITIONAL_MODELS, ADDITIONAL_MODELS_DIR


MODEL_CONFIGS = {
    key: {"display": repo.rsplit("/", 1)[1],
          "hf_id": str(ADDITIONAL_MODELS_DIR / folder),
          "backend": key, "dtype": "bfloat16", "load_4bit": True,
          "revision": revision}
    for key, (repo, folder, revision) in ADDITIONAL_MODELS.items()
}


def load_local_model(key, config):
    import torch
    import transformers

    directory = config["hf_id"]
    if not Path(directory).is_dir():
        raise FileNotFoundError(f"Missing {key} snapshot. Run code/setup_next_models.py first.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Restore NVIDIA driver access before loading weights.")
    if key in ("qwen35", "gemma4") and int(transformers.__version__.split(".")[0]) < 5:
        raise RuntimeError("Use the separate .venv environment from requirements-local-models.txt.")
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    kwargs = dict(quantization_config=quantization, device_map={"": "cuda:0"},
                  low_cpu_mem_usage=True, local_files_only=True)
    if key == "internvl35":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            directory, trust_remote_code=True, local_files_only=True,
            fix_mistral_regex=True)
        model = transformers.AutoModel.from_pretrained(
            directory, trust_remote_code=True, torch_dtype=torch.bfloat16,
            use_flash_attn=False, **kwargs).eval()
        return model, tokenizer, key
    model_class = {
        "qwen35": transformers.Qwen3_5ForConditionalGeneration,
        "gemma4": transformers.Gemma4UnifiedForConditionalGeneration,
    }[key]
    processor = transformers.AutoProcessor.from_pretrained(directory, local_files_only=True)
    model = model_class.from_pretrained(
        directory, dtype=torch.bfloat16, attn_implementation="eager", **kwargs).eval()
    return model, processor, key


def prepare_image_pair(processor, image1, image2, prompt, system_prompt):
    """Use the native multimodal template with thinking explicitly disabled."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": image1},
            {"type": "image", "image": image2},
            {"type": "text", "text": prompt},
        ]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    )
