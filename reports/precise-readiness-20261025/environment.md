# Compute inspection — 2026-10-25

**A working GPU matcher environment is not established.** Inspection used the
existing isolated `.venv/` without changing its packages or running a model.
Private command outputs, full installed package versions and public checkpoint
metadata receipts are under `local/precise/metadata/compute-20261025/`.

| Item | Observed state |
|---|---|
| PCI hardware | NVIDIA AD103 / GeForce RTX 4080 SUPER is listed by `lspci` |
| NVIDIA management | `nvidia-smi` exits 9: cannot communicate with NVIDIA driver |
| Device nodes | No `/dev/nvidia*` entries observed |
| PyTorch CUDA | `torch.cuda.is_available()` false; device count zero; compiled CUDA 12.4 |
| VRAM / driver compatibility | Runtime device memory and usable driver version cannot be verified |
| Host RAM | 62.48 GiB total, 51.49 GiB available at inspection; not a reservation |
| Workspace disk | 234.24 GiB available at inspection; source images remain in place |
| Existing model path | `models` symlink target unavailable; no local weight integrity/revision check possible |
| Model execution | No checkpoint loading, forward/backward, inference or training performed |

| Existing environment component | Inspected version |
|---|---|
| Python | 3.12.3 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 5.18.0 |
| Pillow | 12.3.0 |
| NumPy | 2.5.3 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.33.0 |
| accelerate | 1.14.0 |

Package presence and CUDA build tags do not prove that dependencies, GPU kernels
or the chosen matcher work together. Source inspection finds positional
interpolation support in the installed DINOv2 implementation, but no real model
object was created. Public model metadata/configuration only was retrieved for
the [pinned candidate revision](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056);
weights were not downloaded. That candidate's proposed input/loss/training
settings and memory estimates are in the
[review design](../../docs/notes/precise_milestone_review_20261025.md).

After model/configuration review, prepare a dedicated environment under ignored
`local/precise/environments/`, resolve a full dependency lock and verify imports.
Once separately scoped driver/storage work restores GPU access, record driver,
runtime, device memory and package compatibility. After execution authorization,
perform a small synthetic loading/forward/backward check, record peak allocated
and reserved memory, gradients and output contracts, then review readiness before
real data execution. These setup and execution steps remain pending. No driver,
kernel, CUDA toolkit, mount or system package was repaired or installed here.
