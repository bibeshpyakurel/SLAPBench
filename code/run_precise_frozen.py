"""Reviewed frozen DINOv2 development or separately authorized final-test phase.

No training, split creation, automatic retries or test-driven threshold fitting.
Source paths and raw scores remain under private local/precise storage.
"""

import argparse
import csv
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

from precise_matching.baseline import MODEL, reviewed_baseline_inputs, run_baseline
from precise_matching.data import InputError
from precise_matching.encoders import load_encoder
from precise_matching.images import image_tensor, prepare_image
from precise_matching.testing import ImageFailure
from precise_matching.training import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def git_output(*args, text=False):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def package_versions():
    return {name: metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow", "safetensors")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "test"))
    for name in ("images", "pairs", "paths", "config", "review", "source-root", "weights", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--baseline-run", type=Path, help="Completed frozen development run; test phase only")
    parser.add_argument("--report", type=Path, help="Optional new aggregate JSON under reports/")
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        private = ROOT / "local/precise/runs"
        if not output.is_relative_to(private) or output.exists():
            raise InputError("Use a new private baseline output under local/precise/runs/.")
        if args.baseline_run is not None:
            args.baseline_run = args.baseline_run.resolve(strict=True)
            if (not args.baseline_run.is_relative_to(private) or output.is_relative_to(args.baseline_run)
                    or args.baseline_run.is_relative_to(output)):
                raise InputError("Baseline development and test outputs require separate private directories.")
        if args.report is not None:
            args.report = args.report.resolve()
            if (not args.report.is_relative_to(ROOT / "reports") or args.report.suffix != ".json"
                    or args.report.exists()):
                raise InputError("Aggregate report must be a new JSON under reports/.")
        images, pairs, paths, config, info, hashes, input_integrity = reviewed_baseline_inputs(
            args.images, args.pairs, args.paths, args.config, args.review, args.source_root,
            args.phase, args.baseline_run)
        sources = sorted((ROOT / "code/precise_matching").glob("*.py")) + [ROOT / "code/run_precise_frozen.py"]
        source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
        changes = git_output("status", "--porcelain", "--untracked-files=all", "--",
                             "code/precise_matching", "code/run_precise_frozen.py")
        if changes.strip():
            raise InputError("Commit and review baseline source before executing a real run.")
        provenance = {**MODEL, "source_sha256": source_hashes, "packages": package_versions(),
            "code_commit": git_output("rev-parse", "HEAD", text=True).strip(),
            "python": platform.python_version(),
            "environment_lock_sha256": digest(ROOT / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"),
            "source_root": str(args.source_root.resolve()), "weights_path": str(args.weights.resolve()),
            "input_paths": {name: str(getattr(args, name).resolve()) for name in
                            ("images", "pairs", "paths", "config", "review")}}
        if info is not None:
            from precise_matching.baseline import COMPATIBILITY
            if any(info["record"]["provenance"].get(k) != provenance.get(k) for k in COMPATIBILITY):
                raise InputError("Frozen development source/environment differs from the current baseline runtime.")

        def integrity():
            input_integrity()
            if any(digest(ROOT / name) != value for name, value in source_hashes.items()):
                raise InputError("Baseline source changed during execution.")
            if digest(ROOT / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock") != provenance["environment_lock_sha256"]:
                raise InputError("Baseline environment lock changed during execution.")

        integrity()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        model = load_encoder(args.weights, adapted=False, device=config.device)
        import torch
        provenance["hardware"] = {"device": config.device, "platform": platform.platform(),
            "torch_runtime": str(torch.__version__), "torch_compiled_cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name() if config.device == "cuda" else None}

        def load(row):
            if row.partition != args.phase:
                raise InputError("Only the reviewed baseline phase may enter the image loader.")
            path = paths[row.image_id]
            try:
                if digest(path) != row.sha256:
                    raise InputError("Baseline source bytes differ from reviewed metadata.")
            except FileNotFoundError:
                raise ImageFailure("missing_input") from None
            except OSError:
                raise ImageFailure("decode_error") from None
            try:
                pixels = image_tensor(prepare_image(path, row.sha256, row.decoded_sha256))
            except InputError:
                if not path.exists():
                    raise ImageFailure("missing_input") from None
                if digest(path) != row.sha256:
                    raise InputError("Baseline image changed during decoding.") from None
                raise ImageFailure("decode_error") from None
            if digest(path) != row.sha256:
                raise InputError("Baseline image changed during decoding.")
            return pixels

        report = run_baseline(model, images, pairs, load, config, args.phase, output, hashes,
                              provenance, info=info, integrity=integrity)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, report)
    except (InputError, OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError,
            ImportError, csv.Error, subprocess.SubprocessError):
        print("Frozen baseline did not complete. Inspect private review/runtime records; no test-driven tuning performed.",
              file=sys.stderr)
        return 1
    print("Frozen baseline phase complete; private outcomes and development-only threshold evidence preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
