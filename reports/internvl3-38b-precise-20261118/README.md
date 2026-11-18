# InternVL3-38B on the fixed Precise test pairs

The local evaluation completed all **4,200 pairs** on November 18 at **11:16 a.m.
Central**, after **16 hours and 11 minutes**. Every pair produced a valid
response, with zero execution errors.

The model returned **B (“different people”) for all 4,200 pairs**. It rejected
all 2,100 genuine pairs and correctly rejected all 2,100 impostor pairs. Overall
accuracy was **50.0%**. The observed false accept rate was **0%** because the
model accepted no pairs. The false reject rate was **100%**.

| Pair type | Genuine rejected | Impostor rejected | Accuracy |
| --- | ---: | ---: | ---: |
| Left hands | 700 / 700 | 700 / 700 | 50.0% |
| Right hands | 700 / 700 | 700 / 700 | 50.0% |
| Thumbs | 700 / 700 | 700 / 700 | 50.0% |
| Overall | 2,100 / 2,100 | 2,100 / 2,100 | 50.0% |

The result describes the pinned model and prompt used in this run. The
evaluation produced one A/B decision per pair, so it measures one operating
point and provides no continuous similarity scores.

An independent offline check matched each pair's start and finish events to
the original pair order and labels. Each saved raw reply was B after trimming
whitespace. Recomputed counts and rates agreed with the runner's final report.
The review, input bundle and tested runtime matched their recorded hashes.
Original inputs and run records stayed unchanged. The aggregate includes the
source hashes and run configuration references. [Aggregate results](aggregate.json)

The full run records, including per-pair model replies, remain in the local
ignored run directory. The configuration and authorization scope are described
in the [prelaunch report](../internvl3-38b-prelaunch-20261117/README.md).
