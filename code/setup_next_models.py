"""Download and checksum the three authorized local-model additions; no inference.

Usage: python code/setup_next_models.py --download --verify --models-dir PATH
Add --save-location to persist the directory in ignored, machine-local config.
The historical code/setup_models.py download set is deliberately unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

from model_registry import (
    ADDITIONAL_MODELS, ADDITIONAL_MODELS_DIR, STORAGE_CONFIG,
)


def verify_snapshot(directory, files):
    """Verify every remote snapshot file against its size and Hub digest."""
    records = []
    for item in files:
        name = item.rfilename
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing snapshot file: {name}")
        size = path.stat().st_size
        if item.size is None or size != item.size:
            raise ValueError(f"Snapshot size mismatch: {name}")
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(usedforsecurity=False)
        git_blob.update(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                sha256.update(block)
                if not item.lfs:
                    git_blob.update(block)
        actual = sha256.hexdigest()
        expected = item.lfs.sha256 if item.lfs else item.blob_id
        computed = actual if item.lfs else git_blob.hexdigest()
        if not expected or computed != expected:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
        records.append({"file": name, "bytes": size, "sha256": actual})
    if not records:
        raise ValueError("The snapshot contains no files")
    return records


def save_location(directory, config_path=STORAGE_CONFIG):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"models_dir": str(directory.resolve())}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(ADDITIONAL_MODELS),
                        default=list(ADDITIONAL_MODELS))
    parser.add_argument("--models-dir", type=Path, default=ADDITIONAL_MODELS_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Check all bytes against Hub digests")
    parser.add_argument("--save-location", action="store_true")
    parser.add_argument("--report", type=Path, help="New JSON report path; never overwritten")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    if args.report and not args.verify:
        parser.error("--report requires --verify")
    directory = args.models_dir.expanduser().resolve()
    if args.models_dir.is_symlink() and not args.models_dir.exists():
        parser.error("Model directory is an unavailable symlink; select available storage")
    report = {"schema_version": 1, "models": []}
    if args.download or args.verify:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
    for key in dict.fromkeys(args.models):
        repo, folder, revision = ADDITIONAL_MODELS[key]
        destination = directory / folder
        print(f"{key}: {repo}@{revision} -> {destination}", flush=True)
        if args.download:
            snapshot_download(repo_id=repo, revision=revision,
                              local_dir=str(destination), max_workers=3)
        if args.verify:
            metadata = api.model_info(repo, revision=revision, files_metadata=True)
            if metadata.sha != revision:
                raise ValueError(f"Unexpected revision for {repo}")
            files = verify_snapshot(destination, metadata.siblings)
            total = sum(item["bytes"] for item in files)
            print(f"  Verified {len(files)} files, {total:,} bytes", flush=True)
            report["models"].append({"key": key, "repo_id": repo, "revision": revision,
                                     "folder": folder, "bytes": total, "files": files})
    if args.save_location:
        save_location(directory)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
