"""Reject local-only files from the Git index before commit or in CI.

This is a path/size guard, not a substitute for reviewing research artifacts
for dataset images, credentials, redistribution rights, or inaccurate claims.
"""

from pathlib import PurePosixPath
import subprocess
import sys


LOCAL_ONLY_ROOTS = {
    "datasets", "dataset", "models", "venv", ".venv", "env",
    ".cache", ".ruff_cache", "related_papers", ".claude",
}
LOCAL_ONLY_SUFFIXES = {
    ".pem", ".key", ".safetensors", ".pt", ".pth", ".ckpt",
    ".onnx", ".bin", ".zip", ".tar", ".tgz", ".7z",
}
MAX_BYTES = 50 * 1024 * 1024


def exclusion_reason(path: str, size: int) -> str | None:
    """Return a reason when a staged path violates the repository boundary."""
    parts = PurePosixPath(path).parts
    if not parts:
        return None
    if parts[0] in LOCAL_ONLY_ROOTS:
        return "local-only directory"
    name = parts[-1]
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "credential file"
    if name.endswith(".tar.gz"):
        return "archive/model artifact"
    if PurePosixPath(name).suffix.lower() in LOCAL_ONLY_SUFFIXES:
        return "archive, key, or model weight"
    if size > MAX_BYTES:
        return f"file exceeds 50 MiB ({size} bytes)"
    return None


def indexed_files() -> list[tuple[str, str]]:
    """Read staged blob IDs, so the check sees what Git would publish."""
    raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"])
    files = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, path = item.split(b"\t", 1)
        mode, blob_id, stage = metadata.decode().split()
        if stage != "0":
            raise SystemExit("Unmerged Git index entry: " + path.decode(errors="replace"))
        if mode == "160000":
            raise SystemExit("Submodules require separate publication review")
        files.append((path.decode("utf-8", errors="replace"), blob_id))
    return files


def main() -> int:
    files = indexed_files()
    blob_ids = "\n".join(blob_id for _, blob_id in files) + "\n"
    sizes = subprocess.check_output(
        ["git", "cat-file", "--batch-check=%(objectsize)"],
        input=blob_ids.encode(),
    ).decode().splitlines()
    problems = []
    for (path, _), size_text in zip(files, sizes, strict=True):
        reason = exclusion_reason(path, int(size_text))
        if reason:
            problems.append(f"  {path}: {reason}")
    if problems:
        print("Publication scope check failed:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    print(f"Publication scope OK: {len(files)} indexed files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
