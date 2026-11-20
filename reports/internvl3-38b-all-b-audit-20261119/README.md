# Why InternVL3-38B returned B on every Precise pair

The November 18 run was a **zero-shot evaluation of the official
InternVL3-38B checkpoint**, with no local training or fine-tuning adapter.
Every saved raw reply was B before parsing. The model rejected every genuine
pair under the tested configuration. The cause of the uniform decisions is
still open. [Completed evaluation](../internvl3-38b-precise-20261118/README.md)

## What the audit checked

- **Checkpoint and scope.** The run recorded full SHA-256 verification of the
  pinned OpenGVLab revision `b99bcba4fe0c51d9e475af25ec000dbfea6284af` before
  loading. The audit checked the recorded binding, protected source, 36 file
  sizes, nonweight hashes and all BF16 tensor headers. The evaluator loaded
  the checkpoint in eval mode and performed inference without an optimizer
  or adapter. Quantization changes weight representation in memory.
- **Images and pairing.** Original image bytes and decoded hashes verified
  again. The 1,057 images produced 1,057 distinct BF16 tensors after tiling
  and normalization. The two prepared tensors differed in all 4,200 pairs.
  For 12 images spanning all six hand-and-class groups, the preprocessed
  tensors exactly matched the pinned model card's recipe at the same tile
  settings. The run journal preserved the supplied pair order.
- **Image and answer routing.** A CPU replay used the pinned tokenizer and
  original chat and generation-wrapper code with stand-ins for the learned
  encoder and decoder. One genuine and one impostor pair per hand type
  reached the decoder with the two visual token regions in the right order,
  with 2,560 visual tokens in each. The 5,231 total input tokens fit the
  8,192-token limit. Both A and B are valid single-token replies and both
  decoded and parsed correctly when emitted by the stand-in. The decoder
  configuration had no suppressed or forced answer token. The offline
  replay checks routing and tokenization, with learned GPU computation
  outside its scope.
- **Dataset cross-check.** The native DINOv2-Large report uses the identical
  test bundle SHA-256. Its frozen and adapted runs achieved 73.26% and
  83.19% on these 4,200 pairs at their respective prespecified operating
  points. That confirms a separate method could distinguish genuine from
  impostor pairs in this cohort. [DINOv2-Large results](../precise-dinov2-large-native-20261112/README.md)

The earlier readiness check loaded the real model and repeated one synthetic
pair. Both replies were B. Its pass condition checked valid format and
repeatability. It had no known match to establish that the model could accept
a pair. The final run saved hard answers without A/B logits or visual
embeddings, which limits the retrospective diagnosis.
[Prior readiness evidence](../internvl3-38b-readiness-20261115/README.md)

## How this differs from FPBench

| Aspect | FPBench | This Precise run |
| --- | --- | --- |
| InternVL3-38B verification | 56.84% zero-shot and 50.37% with reasoning prompting. 52.86% was its overall zero-shot score across tasks. | 50.00% accuracy with B on all 4,200 pairs, equal to the always-reject baseline on this balanced set. |
| Task and inputs | Questions about same-finger relationships among individual impressions. The released metadata has 804 questions, each with three or four images and four shuffled options. | Two images per comparison. Each shows the four fingers of one hand or both thumbs. A always meant same person and B always meant different people. |
| Verification baseline | 25.00% random choice and 29.23% for always choosing the most frequent correct option. | 50.00% for always predicting either class. |
| Model execution | The paper reports an H200 and VLMEvalKit. The public evaluation entry point delegates loading. The exact historical InternVL precision settings are unspecified. | A 16 GiB RTX 4080 SUPER with mixed NF4 and BF16 weights, CPU offload, and greedy generation with a five-token budget. |

FPBench's fine-tuning table covers two Qwen3 sizes and Gemma3-12B. The
InternVL3-38B row is in the zero-shot and reasoning table. The published
accuracy does not specify an expected true accept rate on Precise.
[FPBench v2, Tables 2 and 3 and sections 3.1 and 4.3–4.4](https://arxiv.org/html/2512.18073v2)
[Released verification questions](https://github.com/Ektagavas/FPBench/blob/ed27ae669947bfd157522d537a9f9c77b4bf3189/benchmark/matching_multiple_all_data.json)
[FPBench evaluation entry point](https://github.com/Ektagavas/FPBench/blob/ed27ae669947bfd157522d537a9f9c77b4bf3189/evaluate.py)

## Plausible causes and a test that separates them

1. **Answer bias under the fixed prompt.** The model could prefer the letter
   B, default to “different people,” or give every genuine pair a low match
   score under the tested conditions. The raw answer establishes its decision
   only. The A/B score margin could still vary across pairs. Swapping the
   letter meanings while holding images fixed would distinguish a preference
   for B from a preference for rejection.
2. **Visual task and prompt transfer.** Matching a person across slap or
   thumb images requires comparing corresponding fingers with local ridge
   evidence. FPBench tests individual impressions with a different question
   format. Distinct image tensors entering the model do not establish that
   the model extracts useful matching features from them.
3. **Precision and runtime.** A weightless replay of the pinned Transformers
   replacement routine found **180 vision linears (5.53B weight elements),
   both projector linears, and 189 decoder linears** configured for 4-bit
   NF4. The calculated GPU and host parameter bytes match the sealed run
   exactly. Offloaded decoder weights stayed BF16 in host storage. A
   precision comparison matters because this setup quantized the visual
   pathway as well as the decoder. The audit establishes the configuration.
   Numerical parity against BF16 remains unmeasured.
   [Official checkpoint and loading examples](https://huggingface.co/OpenGVLab/InternVL3-38B)

The first diagnostic would be **32 local calls** with the original mixed
precision runtime. Use four easy synthetic image controls with known
same/different answers and twelve fixed Precise development pairs outside the
4,200-pair test set, two per hand-and-class group. Run each control with
each A/B mapping, keeping the images and wording fixed apart from the option
mapping. Save the raw answers and the relative log probabilities for both
answer forms. If the output follows the letter B after swapping, investigate
option bias. If it follows the semantic rejection, check the easy controls
and whether the match margin varies by pair. A second, paired precision
comparison on hardware that can hold the BF16 model can test the quantized
vision path. To attribute a difference to precision, hold preprocessing,
prompt and execution settings fixed wherever possible.

The control run is a separate experiment under the repository's authorization
rules. This audit used no CUDA context and produced no new learned model
predictions. The 13 InternVL unit tests and 15 legacy metric reproductions
passed. The replay receipt and script are retained under ignored local
metadata with their hashes recorded in the aggregate. [Audit evidence](aggregate.json)
