# InternVL3-38B readiness — November 17 evening

At 18:31 America/Chicago (00:31 UTC November 18), a fresh check finds no GPU
compute processes, **15.57 GiB free GPU memory** and **51.35 GiB available RAM**.
Both fixed requirements pass: 14.93 GiB GPU and 41.61 GiB RAM, including reserves.
Qwen's coordinator is marked `cancelled_by_user`, with the recorded reason
`user explicitly requested no evaluation`; its partial test is not a completed
benchmark. No job was restarted or stopped by this recheck.

The configuration, protected source hashes/clean status, package versions,
Python/interpreter, kernel, GPU model, model receipt and successful November 15
synthetic report remain unchanged. All 36 model file sizes and nonweight file
SHA-256 checks pass; the earlier November 17 full weight checksum verification
remains recorded separately. Weight payloads were not fully rehashed again.
The current 4,200 supplied pairs and all 1,057 referenced image byte/decoded
hashes pass fresh validation, preserving labels and thumbs.

The existing tested route is technically prepared and current resources meet
its requirements. There are zero formal InternVL test claims. No new execution
review was created, checkpoint tensors loaded, fingerprint predictions made,
or experiment started. Formal execution still requires the actual review
binding enforced by the existing runner. Resource availability must be checked
again at launch. The prior two synthetic calls do not guarantee full-run
endurance or fingerprint accuracy.

This check uses only filesystem/package metadata, image decoding,
`nvidia-smi` and `/proc/meminfo`; it does not import Torch or create a CUDA
context. No additional GPU synthetic check or test rerun was needed because
protected code and the tested configuration are unchanged.

See [aggregate evidence](aggregate.json), the
[November 15 synthetic result](../internvl3-38b-readiness-20261115/README.md)
and [earlier November 17 audit](../internvl3-38b-readiness-20261117/README.md).
Private receipt: `local/precise/metadata/internvl38b-readiness-20261117/evening-recheck.json`.
JSON assertions and whitespace checks pass. Dataset identifiers, filenames,
raw responses and credentials are excluded from this report.
