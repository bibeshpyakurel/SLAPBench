"""Offline setup checks on synthetic images; never load weights or dataset files."""

import argparse
import json
from pathlib import Path

from local_model_backends import MODEL_CONFIGS, prepare_image_pair


def check_model(key):
    import torch
    import transformers
    from accelerate import init_empty_weights
    from PIL import Image

    directory = MODEL_CONFIGS[key]["hf_id"]
    if key == "internvl35":
        config = transformers.AutoConfig.from_pretrained(
            directory, trust_remote_code=True, local_files_only=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            directory, trust_remote_code=True, local_files_only=True,
            fix_mistral_regex=True)
        with init_empty_weights():
            model = transformers.AutoModel.from_config(
                config, trust_remote_code=True, use_flash_attn=False)
        # Check tokenizer and image-token count without generating any response.
        token_count = len(tokenizer.encode("Compare these two synthetic images."))
        result = {"model_class": type(model).__name__, "text_tokens": token_count,
                  "image_tokens_per_patch": model.num_image_token}
    else:
        config = transformers.AutoConfig.from_pretrained(directory, local_files_only=True)
        processor = transformers.AutoProcessor.from_pretrained(directory, local_files_only=True)
        model_class = getattr(transformers, config.architectures[0])
        with init_empty_weights():
            model = model_class(config)
        inputs = prepare_image_pair(
            processor, Image.new("RGB", (448, 448), "white"),
            Image.new("RGB", (448, 448), "black"),
            "Describe these two synthetic images.", "You are a visual assistant.",
        )
        result = {"model_class": type(model).__name__,
                  "synthetic_input_shapes": {name: list(value.shape)
                                             for name, value in inputs.items()
                                             if hasattr(value, "shape")}}
    result.update({"key": key, "transformers": transformers.__version__,
                   "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "parameter_devices": sorted({str(parameter.device)
                                                for parameter in model.parameters()}),
                   "weights_loaded": False, "inference_run": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS),
                        default=list(MODEL_CONFIGS))
    parser.add_argument("--report", type=Path, help="New JSON output; never overwritten")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    results = []
    for key in args.models:
        result = check_model(key)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(results, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
