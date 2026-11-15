# InternVL3-38B: fixed-pair runner and mixed CPU/GPU preparation

The user requested the recommended dedicated runner and GPU-plus-CPU offload
assessment on November 15. Scope is implementation, offline input validation,
architecture/capacity checks and synthetic execution. The current 4,200 Precise
test pairs are not an implementation-development or resource-tuning set.
No training, pair generation or real test inference follows from this preparation.

## Implementation

`code/run_precise_internvl38b.py` and `code/internvl38b_runtime.py` provide:

- `probe`: a tiny randomly initialized two-layer Qwen2 fixture, saved only
  privately, then loaded with actual NF4 GPU / unquantized BF16 CPU dispatch.
  Two greedy generations test the dtype, quantization and transfer path. This
  is not InternVL3-38B inference and establishes no full-model capacity.
- `plan`: verify snapshot source/configuration bytes and all file sizes, inspect
  Safetensors headers and construct the downloaded architecture on meta. No
  downloaded checkpoint tensors are loaded. Parameters must match every indexed
  tensor. The fixed system/user prompt and synthetic 1600x1500 input context
  are checked against the local tokenizer limit.
- `synthetic`: require the memory gate, rehash the full snapshot, load the
  real model once and run two deterministic geometric image pairs. Check strict
  A/B format and repeat stability, recording latency and memory. No fingerprint
  images are used. A failed gate does not start a checkpoint load.
- `evaluate`: a future formal entry point. It requires a successful full-model
  synthetic report, unchanged runtime/source/configuration/device-map identity,
  the exact current bundle digest, and actual protocol/execution review bound
  to these bytes. No such test-execution review is manufactured by this task.

All outputs must use a new directory under ignored `local/precise/`. There is
no overwrite, automatic resume, automatic fallback or silent crash skip. A
persistent cohort/configuration claim prevents duplicate formal scoring. Each
pair's started event is fsynced before inference and its result afterwards.
Invalid responses stay invalid; execution errors are recorded and stop the
attempt. Unattempted rows remain explicit. Private raw replies and events are
retained; failures are never credited as correct impostor predictions.

The current bundle is
`local/precise/manifests/numeric-group-holdout-20261107/bundle.json`, SHA-256
`48ec4b7652940aef16f3eb8a843b5f9825752219a75f7280b461be63efe2ac86`.
Supplied labels are preserved with genuine=A, impostor=B. Metadata and path
validation retain all 4,200 rows, including thumbs, with no identity relabeling
or pair regeneration. The old 1,690-pair historical CSV is not this benchmark.
Numeric-group separation is not verified biological identity/capture independence.

## Declared candidate configuration

`code/configs/internvl3_38b_offload.json` specifies the official already-downloaded
snapshot at revision `b99bcba4fe0c51d9e475af25ec000dbfea6284af`, Transformers
4.48.3, NF4 with double quantization on GPU, BF16 compute and unquantized BF16
CPU modules and 16 CPU threads. The offload flag's historical name mentions FP32; the tiny probe
checks actual loaded CPU dtypes rather than assuming what the name implies.
The real loader rejects CPU floating parameters with unexpected dtypes.

The GPU weight budget is 12 GiB, with 3 GiB additional GPU working reserve;
the host reserve is 8 GiB beyond estimated CPU weights. Required components,
including vision, projector, embeddings, output and first/last decoder layers,
are co-located on GPU. Whole intermediate decoder layers are added in a fixed
order up to the declared budget; remaining layers are assigned to CPU. Live
resource availability never changes this map or image settings automatically.
These are planning estimates, not measured full-model peaks or a success claim.

Before the first full-weight attempt, source review identified that dense eager
language attention at 5,231 input tokens can exceed the working reserve. The
declared final adapter therefore selects Qwen2's Torch SDPA path and forces
PyTorch's native Flash attention backend, with no math/eager fallback. It does
not require the separate `flash_attn` package. Vision attention remains the
upstream eager implementation but processes two tiles per encoder call before
concatenating features in their original order. CPU-offloaded weights are stored
in host RAM and Accelerate transfers them to GPU for execution; their resting
meta placeholders are validated against actual BF16 tensors in leaf hook storage.
These adapters are explicit configuration/source changes before any real model
load, not a retry or change to downloaded source. Original 20-tile membership,
prompt, image detail, greedy output and checkpoint bytes stay fixed.

Preprocessing converts native grayscale inputs to RGB without autocontrast or
square padding. The fixed dynamic grid rule allows 12 base tiles plus a
thumbnail. At 1600x1500 this produces a 3x3 grid plus thumbnail per image;
patch tensors are 448x448 with ImageNet normalization. This differs from the
historical single-tile 448 runner and must be reported. The native image-token
template is used with no conversation history. Konain's existing thumb-aware
system/user prompt is copied exactly; greedy output is limited to five tokens.
No test result selects resolution, prompting, generation or quantization.

## Environment and execution

`requirements-internvl38b.txt` pins the direct runtime packages. On this host a
new `local/precise/environments/internvl38b-offload-20261115/` compatibility
environment uses the bundled Python 3.12.14, installs Transformers 4.48.3,
tokenizers 0.21.4, huggingface-hub 0.36.0 and requests 2.32.5 locally, and appends
the existing `.venv/lib/python3.12/site-packages` via a recorded `.pth` for
PyTorch/CUDA and other unchanged packages. This is an overlay, not an independent
copy of every shared package. Earlier environments are not upgraded. Actual
runtime versions, executable digest and protected source hashes are recorded;
any change invalidates reuse of a successful synthetic report for formal testing.
For a clean installation, use Python 3.12 and install the pinned requirements in
a separate environment with an appropriate CUDA-enabled PyTorch build.

Example capacity check, using a new output path each time:

```bash
local/precise/environments/internvl38b-offload-20261115/bin/python \
  code/run_precise_internvl38b.py plan \
  --model-dir /home/bibesh/.local/share/slapbench/models/internvl3-38b \
  --output local/precise/metadata/internvl38b-offload-NEWDATE/plan-01
```

The same command with `probe` exercises the tiny mixed-device fixture; with
`synthetic` it attempts the actual model only after the gate passes. Do not loop
over automatic retries or alter the declared settings until one loads. Preserve
every attempt at a new private path. Actual real-weight loading also requires
clean, committed protected runtime source. No network model fallback is allowed:
the program forces offline loading and uses a private custom-module cache.

Formal evaluation additionally requires `--bundle`, `--image-root`,
`--synthetic-report` and `--review`. Review fields are `protocol_reviewed`,
`execution_authorized`, an actual `human_response_reference`, and `input_sha256`
with `identity`, `bundle`, and `synthetic_report` hashes. This describes the
future evidence contract; it is not an approval record. No test command is
published as if that missing full-model pass already exists.

## Development records and verification

Private development records remain under
`local/precise/metadata/internvl38b-offload-20261115/`. The first architecture
inspection failed because upstream computes drop-path scalars with `.item()`;
placing those constructor buffers on meta is unsupported. The corrected
architecture-only inspection keeps tiny buffers on CPU and all parameters on
meta. The failed record is preserved; no real weights or test pixels were used.

The corrected plan assigns 27 decoder layers to GPU and 37 to CPU. Its estimated
weights are about 11.93 GiB on GPU and 33.61 GiB on CPU. With reserves, it needs
about 14.93 GiB free GPU memory and 41.61 GiB available host RAM. The busy-host
inspection failed both gates. The other process then released the GPU and
available RAM rose to about 50 GiB; the next architecture/resource check passed
both gates. Finishing another process may make the route
possible if it releases enough RAM as well as VRAM; completion alone is not a
guarantee. Do not stop another user's job, repair drivers or silently use SSD
offload to pass this gate.

Offline tests cover malformed replies/error strings, changed model bytes and
index structure, path escape, exact cohort/labels/counts, resource rejection,
fixed device maps, no load on a blocked gate, explicit failure denominators,
durable journal/no retry, output boundaries and review bindings. Regular CI
discovers these tests without model assets, private data or a GPU. Final dated
aggregate evidence is reported separately after source publication.

Sources: [official model](https://huggingface.co/OpenGVLab/InternVL3-38B),
[Transformers quantization](https://huggingface.co/docs/transformers/v4.48.3/quantization/bitsandbytes),
[Accelerate big-model inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling).
