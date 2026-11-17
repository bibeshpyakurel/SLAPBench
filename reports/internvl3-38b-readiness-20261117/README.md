# InternVL3-38B readiness recheck — November 17, 2026

Read-only validation of the prepared route; no InternVL checkpoint tensors were
loaded, no fingerprint inference ran, and no formal evaluation review or claim
was created. Use the primary `dev` checkout, not the older Orca worktree:
`/home/bibesh/Desktop/MultiModel LLM's`.

## Prepared route

The pinned 36-file official snapshot, dedicated Python 3.12 / Transformers 4.48.3
environment, fixed configuration and runner remain available. A fresh `plan`
reproduces the exact identity digest of the successful November 15 full-model
synthetic check:
`c9c214f0b11b2aa24714e0fae37f357b1a8337aac3232664a6bf7781739655ba`.
Protected runtime source remains unchanged and clean. The 27-GPU/37-CPU decoder
placement and NF4/BF16 policy remain fixed. See the
[earlier full-model evidence](../internvl3-38b-readiness-20261115/README.md).

The current supplied cohort is **4,200 pairs**, using **1,057 unique images**
and 55 numeric groups, including left hands, right hands and thumbs. Today's
cohort validation checks exact membership, supplied labels, path containment,
train/test numeric-group separation, all referenced image bytes and decoded
pixel hashes. This is input validation, with no model predictions. The bundle
remains SHA-256
`48ec4b7652940aef16f3eb8a843b5f9825752219a75f7280b461be63efe2ac86`.

## Resource observations and Qwen interruption

At 07:53:55 America/Chicago, the fresh architecture-only plan rejected both
memory gates. It observed 11.39 GiB free GPU memory and 32.35 GiB available RAM,
versus required 14.93 GiB GPU and 41.61 GiB RAM including reserves. Qwen3-VL-8B
adapter evaluation was then running under the other local account.

The readiness plan calls Torch CUDA discovery/memory APIs, which briefly create
a CUDA context despite loading no real model weights. The existing Qwen
`evaluate_native.py` checks for *any* second compute PID before each pair and
returns 75 with `state=paused`, `reason=another GPU workload`. Its pause at
07:53:55 coincides with this preflight. The readiness check therefore triggered
the guard; this was an unintended interruption, not successful Qwen completion.
No process was explicitly killed. The Qwen coordinator subsequently attempted
report generation and stopped with a report-phase error. Its evaluator retains
47 parseable outcomes with 47 distinct pair IDs out of 4,200 expected.
No outputs were deleted or overwritten, and no resume was attempted.

At 07:55:34, independent `nvidia-smi` and `/proc/meminfo` readings showed no GPU
compute processes, 15.57 GiB free GPU memory and 51.70 GiB available RAM. Both
fixed resource thresholds then passed. These readings create no CUDA context.
Do not interpret the free GPU as evidence the Qwen test completed. Future
readiness checks alongside this evaluator must avoid opening a CUDA context;
coordinate any separately authorized continuation with the other session.

## Remaining execution boundary

No formal InternVL evaluation has started: there are no persistent test claims
or review records in the existing InternVL preparation directory. The runner
requires an actual execution/protocol review bound to the exact bundle,
unchanged identity and successful synthetic report before formal testing.
The user's current request was readiness verification; no approval was fabricated.
Recheck resource availability immediately before any separately authorized run.
Two prior synthetic calls establish local feasibility, not 4,200-pair endurance,
biometric accuracy or guaranteed runtime. Do not duplicate completed tests,
change numerical settings or automatically retry interrupted evaluations.

## Validation and evidence

A fresh full SHA-256 check passes for all 36 snapshot files (76,790,898,977
bytes), and every retained November 15 evidence digest still matches. The
[aggregate](aggregate.json) records the checks and private receipt hashes.
All 13 InternVL-specific unit tests pass in the dedicated environment without
skips. All 15 legacy published metrics reproduce in that environment; JSON
validation and whitespace checks pass. New private plan/integrity/resource receipts are under
`local/precise/metadata/internvl38b-readiness-20261117/`; original preparation,
models, inputs and outputs are preserved. No dataset identifiers, filenames,
raw replies or credentials are included in this report.
