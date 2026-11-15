# The routing on the baseline's 100 questions

Run on 2026-11-15. The routing of [the whole-benchmark page](2026-11-15-whole-benchmark.md)
was run end to end through `chatchtg eval` on the same 100 questions as
[the Opus 5.5 baseline](2026-11-14-baseline-opus-5-5.md). Code answered 67 of them. The other
33 go to the LLM; their answers are taken from the earlier Opus 5.5 runs, because no billed
run could be made (the Anthropic API returned "credit balance is too low").

**Result: 90 of 100, against 77 and 81 for the plain model in its two runs. The code part cost
$0 of LLM.** The gain is inflated by plans that the detectors were trained on; see "Limits".

| | Plain Opus 5.5 | Routing | Fixed | Broken | Sign test |
|---|---|---|---|---|---|
| Against `baseline-opus-5-5-100` | 77 | 90 | 15 | 2 | p = 0.002 |
| Against `fix-issues-baseline-opus-5-5-100` | 81 | 90 | 12 | 3 | p = 0.035 |

"Fixed" and "broken" count only the 67 questions that code answered. The 33 other answers are
the same in both columns.

## Setup

| | |
|---|---|
| Questions | `questions-subset.jsonl`, `--sample 100 --seed 0`, the baseline's 100 questions |
| Composer | `combined@1+areas@4+room-conventions@3` |
| Settings | `CHATCHTG_ROOM_COUNTER=confident`, `CHATCHTG_OBJECT_COUNTER=detections` |
| Code | this branch: `whole-benchmark` (which holds the Swiss object counter) merged with `combined-composer` |
| LLM | `fake` for the run; the 33 LLM answers from the earlier Opus 5.5 runs with the `baseline@1` context |
| Scoring | the rule: a count must be exact, an area within 2% |

## By answering path

| Path | Questions | Routing right | Plain Opus right (run 1, run 2) |
|---|---|---|---|
| Area pipeline (Swiss areas) | 32 | 32 | 23, 27 |
| Object counter, Swiss plans | 14 | 13 | 11, 11 |
| Object counter, Finnish plans | 18 | 10 | 8, 8 |
| Room counter (Swiss apartments, confident) | 3 | 3 | 3, 3 |
| LLM: Swiss rooms with a label | 16 | 16 | 16, 16 |
| LLM: Finnish room counts | 16 | 15 | 15, 15 |
| LLM: area question that the pipeline does not answer | 1 | 1 | 1, 1 |
| All | 100 | 90 | 77, 81 |

Of the 32 area answers, 20 rest on measured rooms, 10 on printed areas and 2 on both.

## To repeat

```sh
CHATCHTG_LLM_PROVIDER=fake CHATCHTG_ROOM_COUNTER=confident CHATCHTG_OBJECT_COUNTER=detections \
  uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --sample 100 --seed 0 \
  --composer combined --evidence --out var/eval-runs/routing-fake-100.jsonl
uv run python scripts/replay_routing.py var/eval-runs/routing-fake-100.jsonl \
  var/eval-runs/baseline-opus-5-5-100.jsonl
```

The run needs the stored detections (`var/metadata`, see the object counter pages), the OCR
cache of the room counter (`var/ocr-cache`, macOS) and takes the area analyses from
`var/area-cache` when they are there.

## Limits

- **Training plans.** Of the 32 plans whose object count the counter answered, 24 are plans
  that its detector was trained on (Finnish 13 of 18, Swiss 11 of 14). On held-out plans the
  Finnish counter is right on 70.8% of questions and the Swiss one on 89.9%, so the 23 of 32
  here is not the accuracy to expect on a new plan of the same sources.
- **The area constants are fitted to this renderer**, and some of these 33 plans may be among
  the plans the measurer and the label reader were tuned on. On held-out plans the pipeline is
  right on 91% to 96% of area questions, not on 32 of 32.
- **The LLM part is a replay, not a run.** The 16 Finnish room counts would get the
  conventions context and the one area question the image with the room table; the answers
  used here were given with the baseline context. A fresh run can differ on these 33.
- **100 questions**, balanced over source and question type and not over the benchmark, so 90%
  is not an estimate of the accuracy on the whole benchmark. The whole-benchmark page has that.
- **The code paths answer only the benchmark's own sheets** where they need stored detections.
