"""Final-test scoring from a reviewed completed DINOv2 training selection.

No test-driven checkpoint selection, cutoff fitting, split creation or training.
Raw outcomes remain private. Optional report output contains aggregates only.
"""

import argparse
import csv
from datetime import datetime, timezone
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

from precise_matching.data import InputError
from precise_matching.encoders import load_encoder
from precise_matching.images import image_tensor, prepare_image
from precise_matching.testing import ImageFailure, final_test, reviewed_test_inputs
from precise_matching.training import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def git_output(*args, text=False):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def versions():
    return {name: metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow", "safetensors")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "pairs", "paths", "config", "review", "source-root", "weights",
                 "training-run", "selection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional new aggregate JSON under reports/")
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        run = args.training_run.resolve()
        if (not output.is_relative_to(ROOT / "local/precise/runs")
                or not run.is_relative_to(ROOT / "local/precise/runs")
                or output.is_relative_to(run) or output.exists()):
            raise InputError("Use private training-run storage and a new private evaluation output.")
        if args.report is not None:
            args.report = args.report.resolve()
            if (not args.report.is_relative_to(ROOT / "reports") or args.report.suffix != ".json"
                    or args.report.exists()):
                raise InputError("Aggregate report must be a new JSON under reports/.")
        images, pairs, paths, config, info, hashes, input_integrity = reviewed_test_inputs(
            args.images, args.pairs, args.paths, args.config, args.review, args.source_root, run, args.selection)
        provenance = info["record"]["provenance"]
        sources = provenance.get("source_sha256", {})
        required = {"code/train_precise_matcher.py"} | {
            "code/precise_matching/" + name + ".py" for name in
            ("data", "images", "encoders", "losses", "evaluation", "training")}
        if not isinstance(sources, dict) or not required <= set(sources) or any(
                not (name == "code/train_precise_matcher.py" or
                     (Path(name).parent == Path("code/precise_matching") and Path(name).suffix == ".py"))
                for name in sources):
            raise InputError("Training source inventory is incomplete or invalid.")
        if any(digest(ROOT / name) != value for name, value in sources.items()):
            raise InputError("Training/preprocessing source changed; review compatibility before final test.")
        if (versions() != provenance.get("packages") or platform.python_version() != provenance.get("python")
                or digest(ROOT / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock")
                != provenance.get("environment_lock_sha256")):
            raise InputError("Inference environment differs from the frozen training record.")
        changes = git_output("status", "--porcelain", "--untracked-files=all", "--",
                             "code/precise_matching", "code/test_precise_matcher.py")
        if changes.strip():
            raise InputError("Commit and review evaluation source before a real test.")
        current_sources = {str(p.relative_to(ROOT)): digest(p) for p in
                           [*sorted((ROOT / "code/precise_matching").glob("*.py")),
                            ROOT / "code/test_precise_matcher.py"]}

        def integrity():
            input_integrity()
            if any(digest(ROOT / name) != value for name, value in current_sources.items()):
                raise InputError("Evaluation source changed during execution.")

        integrity()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        if config.device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        model = load_encoder(args.weights, adapted=False, device=config.device)

        def load(row):
            if row.partition != "test":
                raise InputError("Only final-test image pixels may enter this loader.")
            path = paths[row.image_id]
            try:
                if digest(path) != row.sha256:
                    raise InputError("Final-test source bytes differ from reviewed metadata.")
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
                    raise InputError("Final-test image changed during decoding.") from None
                raise ImageFailure("decode_error") from None
            if digest(path) != row.sha256:
                raise InputError("Final-test image changed during decoding.")
            return pixels

        report = final_test(model, images, pairs, load, config, info, output, hashes, integrity=integrity)
        write_json(output / "execution.json", {"code_commit": git_output("rev-parse", "HEAD", text=True).strip(),
            "source_sha256": current_sources, "packages": versions(), "device": config.device,
            "seed": config.seed, "torch_runtime": str(torch.__version__), "cuda_build": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name() if config.device == "cuda" else None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_selection_sha256": hashes["selection"], "source_root": str(args.source_root.resolve())})
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, report)
    except (InputError, OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError,
            ImportError, csv.Error, subprocess.SubprocessError):
        print("Final test did not complete. Inspect private review/runtime records; no threshold was fitted on test.",
              file=sys.stderr)
        return 1
    print("Final-test outcomes saved privately with the frozen development cutoff; no test-driven tuning performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
