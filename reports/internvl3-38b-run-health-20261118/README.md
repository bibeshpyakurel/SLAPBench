# InternVL3-38B run health after the Ubuntu reboot

The Precise evaluation finished at **11:16:23 a.m. Central on November 18**.
Ubuntu's next boot started at **5:07:56 p.m.**, 5 hours and 51 minutes later.
The previous boot's journal spans the entire evaluation. All six source files
match their checksums in the results summary.
[Verified results](../internvl3-38b-precise-20261118/README.md)

The evaluation journal contains a start and finish for all 4,200 supplied
pairs in the expected order. Each full clock hour of inference recorded 260
or 261 completions. The median interval between saved results was
13.863 seconds and the longest was 14.948 seconds. The console recorded
all 16 checkpoint shards loaded and a final pass, with no traceback, CUDA
error or termination message.

The accessible Ubuntu logs show SSH connection failures in the morning.
The evaluation continued writing results around each failure. All times in
the table are Central on November 18.

| SSH log message | Time | Result immediately before | Result immediately after |
| --- | --- | --- | --- |
| Timeout | 06:27:27 | 06:27:24 | 06:27:37 |
| Timeout | 07:04:35 | 07:04:30 | 07:04:44 |
| Connection reset | 07:46:12 | 07:46:00 | 07:46:14 |
| Broken pipe | 08:05:33 | 08:05:23 | 08:05:37 |

These timestamps show the evaluator kept working while SSH sessions dropped.
The available logs do not identify the cause of the connection problem or
attribute each failure to a particular client. The account could not read the
previous boot's kernel logs, so this check cannot establish overall system
or desktop health during that window.

The all-B outcome was present in the saved raw model replies. The parser reads
the reply and treats unexpected text as invalid. It has no default B answer.
The image digest check passed again for 1,057 distinct images and all 4,200
paired inputs. The checked code sends separate image tiles for each member of
the pair, and the journal recorded 10 tiles per image and 5,231 input tokens
on every call. These checks confirm the input files and recorded responses.
The logs alone cannot explain why the tested configuration returned one
answer for every pair.

This audit used existing logs, outputs and local input checks. It opened no
CUDA context and performed no new inference.
