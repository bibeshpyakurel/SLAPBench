# InternVL3-38B prelaunch check

Technical checks passed at 19:01 America/Chicago on November 17, 2026.
The workstation had 15.57 GiB of free GPU memory and 50.85 GiB of available
RAM, above the respective 14.93 and 41.61 GiB requirements. GPU discovery
reported zero compute processes. The Qwen coordinator records user cancellation.
The private output path is unused, its parent tree is writable, and the
filesystem has about 61 GiB available.

All 36 snapshot files, including the 16 weight shards, passed a fresh full
SHA-256 check. With the GPU hidden, the architecture, package and source
identity, device map and tokenizer context matched the successful synthetic
run. GPU identity and resources came from the driver tool. See the
[November 15 synthetic run](../internvl3-38b-readiness-20261115/README.md). The model's pinned
configuration uses PyTorch Flash SDPA for language attention. The upstream
notice about the separate FlashAttention2 package does not affect this route.

The supplied Precise bundle has 4,200 pairs and 1,057 unique images across
55 numeric groups. All image files and decoded hashes match. Every image
measures 1600 × 1500, the shape tested in the synthetic run. With the fixed
tiling and prompt, a pair takes 5,231 input tokens plus a budget of five
generated tokens against the 8,192 limit. Left hands, right hands and thumbs
each have 700 genuine and 700 impostor pairs.

All 231 repository unit tests passed, including the runner's 13 tests.
Ruff passed, all 15 legacy metrics reproduced, and the regenerated results
table matched the tracked copy. See [aggregate evidence](aggregate.json).

## Evaluation for approval

Run the pinned InternVL3-38B checkpoint once, locally, on all 4,200 pairs in
the existing Precise bundle, using the same prompt, tiling and GPU/CPU layout
as the synthetic pass. The model returns A/B decisions. The runner writes a
private journal and reports coverage, errors and results overall and by hand
type. The planned destination is
`local/precise/metadata/internvl38b-evaluation-20261117/attempt-01`.
The latest synthetic generation took 14.89 seconds for one warm call, which
would imply about 17 hours of generation at that rate. Actual full-run time
and stability remain unmeasured.

The runner requires a review record tied to the exact cohort, prior synthetic
report and execution configuration. This preflight created no review or test
claim. Once the user authorizes this run, record that authorization privately
and recheck resource availability before starting. This check opened no CUDA
context, loaded no checkpoint tensors and ran no real pair inference.
