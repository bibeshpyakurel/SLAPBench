"""Read-only NVIDIA driver provenance; no device, precision or driver repair."""

from pathlib import Path
import re
import shutil
import subprocess

from .data import InputError


def cuda_driver_metadata(driver_path=Path("/proc/driver/nvidia/version")):
    binary = shutil.which("nvidia-smi")
    if binary:
        value = subprocess.check_output([binary, "--query-gpu=driver_version", "--format=csv,noheader"], text=True)
        version = value.splitlines()[0].strip()
        if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
            raise InputError("NVIDIA management output has no recognized driver version.")
        return {"driver_version": version, "driver_metadata_source": "nvidia-smi"}
    try:
        value = Path(driver_path).read_text()
    except OSError:
        raise InputError("NVIDIA driver metadata is unavailable; no system repair is performed.") from None
    match = re.search(r"^NVRM version:.*?\b(\d+\.\d+(?:\.\d+)?)\b", value, flags=re.MULTILINE)
    if match is None:
        raise InputError("The loaded NVIDIA module has no recognized version record.")
    return {"driver_version": match.group(1), "driver_metadata_source": "proc_driver_nvidia"}
