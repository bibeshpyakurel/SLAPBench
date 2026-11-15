# InternVL3-38B mixed CPU/GPU readiness

November 15, 2026. **The full downloaded model loaded successfully and passed
two synthetic image-pair calls on this workstation after another GPU process
finished. No Precise inference or benchmark scores were produced.**

The [verified aggregate](aggregate.json) records the exact configuration,
source/runtime identity, fixed device map, resource gate, synthetic responses,
timings and private-evidence digests. Preparation and future evaluation contracts
are in the [implementation guide](../../docs/notes/internvl38b_offload_preparation_20261115.md).

## What is implemented and checked

- A separate 38B runner uses the verified official snapshot, revision
  `b99bcba4fe0c51d9e475af25ec000dbfea6284af`, without network fallback.
- It resolves the current 4,200 supplied pairs through their existing bundle,
  explicitly maps genuine=A/impostor=B, preserves thumbs and uses Konain's
  exact existing system/user prompts. All 1,057 images pass byte and decoded
  digest checks using the new input path; no predictions are made in that audit.
- The dedicated Python 3.12 / Transformers 4.48.3 overlay preserves earlier
  environments. Actual shared-package versions and executable/source digests
  are recorded. A tiny random Qwen2 fixture proves actual GPU NF4 and BF16 CPU
  storage with native Torch Flash SDPA and repeated greedy generation.
- The fixed real-model configuration places 27 decoder layers on GPU and 37
  in host storage. Accelerate transfers host weights to GPU for execution;
  validated leaf-hook tensors back its normal resting meta placeholders.
- Each 1600x1500 image retains nine grid tiles plus thumbnail. Twenty tiles
  per pair are processed by the vision encoder in batches of two. Language
  attention uses native Torch Flash SDPA with no math/eager fallback; the
  downloaded vision implementation uses eager attention. No upstream source
  or original checkpoint byte is edited.
- Strict parsing, durable started/finished events, explicit invalid/error
  outcomes, complete denominators, source/configuration binding, overwrite
  refusal and a persistent formal-test claim replace legacy unsafe behavior.

## Actual full-model synthetic result

Runtime source commit: `0236f5e4a35ce644a6f22b0275f58d61f094fdd1`.
Later test-fixture/documentation commits do not change protected runtime source.
The full snapshot was rehashed before loading. All 16 shards loaded; actual
BF16 host storage is 36,082,788,352 bytes (33.61 GiB). Initial allocated Torch
GPU memory after loading was 12,930,735,616 bytes (12.04 GiB).

| Observation | Result |
| --- | ---: |
| Synthetic input | Two generated geometric 1600x1500 RGB images |
| Tiles / input tokens per pair | 20 / 5,231 |
| Tokenizer limit / output-token budget | 8,192 / 5 |
| First / second response | `B` / `B` |
| First / second generation latency | 36.88 s / 14.89 s |
| Peak Torch-allocated GPU memory | 15,473,447,424 bytes / 14.41 GiB |
| Peak process RSS | 43,044,683,776 bytes / 40.09 GiB |
| Total check, including verification/loading | 157.44 s |
| Runtime errors / automatic retries | 0 / 0 |
| Precise pairs scored | 0 |

The geometric inputs have no biometric labels. `B` is a preserved format-valid
response, not evidence of correctness, accuracy or fingerprint discrimination.
The second call repeats the same synthetic input and is a limited stability
check, not a long-run endurance result. Latencies exclude preprocessing and
are not promised timings for all 4,200 fingerprint pairs. Torch allocation
does not include every CUDA driver/library allocation; RSS includes mapped
checkpoint pages. Full raw synthetic events remain private and hashed.

## Memory availability and next use

The fixed gate requires **14.93 GiB free GPU memory and 41.61 GiB available
system RAM**, including declared working reserves. It initially rejected the
busy host. After the other process finished, free GPU memory was about
15.32 GiB and available host RAM about 50.20 GiB; the gate passed. No process
was stopped by this task. Actual loading and generation then succeeded.

This demonstrates a viable local route when those resources are available.
It does not guarantee operation alongside another large job. The runner checks
again immediately before loading and rejects insufficient resources instead
of changing resolution, dtype, device map or checkpoint automatically.

The formal test entry point remains unexecuted. It requires the exact current
bundle, this successful full-model synthetic report, unchanged tested identity,
and an actual concrete protocol/execution review bound to them. No review or
test claim is manufactured here. No training, calibration, new pairs, test-based
tuning, hosted inference or changes to completed DINO/Qwen studies occurred.

## Validation and preserved development findings

All 231 local repository tests, lint, Python syntax, 15 legacy metric
reproductions and unchanged legacy table generation pass. The new synthetic
contracts are discovered by regular CI. Hosted CI identified a test fixture
that assumed the local private directory existed; the fixture now creates an
isolated temporary directory. This changes no runtime source or model settings.
The corrected fixture commit `57980a0` passes all four
[hosted CI jobs](https://github.com/bibeshpyakurel/SLAPBench/actions/runs/43773432385)
and [CodeQL](https://github.com/bibeshpyakurel/SLAPBench/actions/runs/43773432464).

The initial meta-buffer inspection failure and busy-host rejection are preserved
privately, alongside development kernel probes and the sole full-model synthetic
attempt. The failed inspection involved no checkpoint tensors or test pixels.
All dataset images, model files, prior runs and raw results remain unchanged.
