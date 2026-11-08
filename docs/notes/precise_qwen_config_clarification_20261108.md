# Qwen prompt input versus saved training configuration

2026-11-08. The user relayed Konain's clarification that the prompt JSON is
optional because the prompts already appear in the code. Read-only inspection
confirms this for the evaluation script's `--prompt_json` argument. No additional
prompt file is needed to reproduce the already verified system/user strings.
Konain subsequently pointed to the already supplied training-data JSON in the
same fine-tuning folder. That file is present and byte-identical to the DINO
37,194-pair input; it is not being requested again. No new configuration file
or training invocation accompanied these clarifications.

There are two distinct uses of JSON in the supplied source:

| Item | Verified source behavior | Current evidence |
| --- | --- | --- |
| Evaluation `--prompt_json` | Optional prompt input; otherwise uses built-in strings | Both supplied summaries' prompt strings and hashes reproduce against the supplied training JSON |
| Training output `binary_training_config.json` | Automatically written in the output directory from `run_metadata`; includes model, selected class counts, limits, seed, epochs, LR, accumulation, vision/language/attention adaptation flags and prompts | The fine-tuned summary references this file in the checkpoint parent; the file itself was not supplied |
| Evaluation checkpoint-prompt check | Searches the adapter directory and parent for that saved configuration; missing file requires `--skip_checkpoint_prompt_check`; when present, checks prompts and model name | The summary records a configuration path, but that does not supply its contents |

The supplied trainer writes `binary_training_config.json` independently of
whether evaluation receives `--prompt_json`. Its source SHA-256 is
`096be29752921ce756d26987ffc7933c938926a6bb236bdb8bedb4a646b5dd88`;
the matched evaluation source SHA-256 is
`dcf68e11ca2258b69e755af3a65c48b93d6dfb397fa2a615f4f7d48dbea3a089`.
Both bind the preserved files in the
[source review](../../reports/precise-qwen-supplied-results-20261108/README.md)
and its aggregate receipt. No collaborator code or model was executed here.

Konain's explanation supplies neither a new file nor the exact training
invocation. It is not treated as proof of absent settings, checkpoint selection,
resolved model bytes or actual consumed membership. Supplied trainer-state and
statically inspected training-argument evidence remain valid, with the existing
limits. No replacement JSON or guessed run record is generated.

The earlier missing-file request referred to the saved training configuration,
not the optional evaluation prompt input. This is a reproducibility distinction,
not a reason to delay the separately
[approved DINOv2-Large run](precise_large_native_protocol_20261108.md).
No new experiment or review approval is inferred from the clarification.
