# The routing on the baseline's 100 questions: the code part run, the LLM part replayed

Made on 2026-11-15. **This is a replay, not a run of the routing, and it is not a comparison
with the baseline.** The routing of [the whole-benchmark page](2026-11-15-whole-benchmark.md)
was put through `chatchtg eval` with the fake LLM on the same 100 questions as
[the Opus 5.5 baseline](2026-11-14-baseline-opus-5-5.md). That measures the answers of the
code paths. The questions that the routing sends to the LLM take their answers from the
earlier Opus 5.5 runs, because no billed run could be made (the Anthropic API returned
"credit balance is too low").

**What is measured: with the defaults of `main`, code answered 53 of the 100 questions and
was right on 45, at $0 of LLM. 35 of the 53 are on plans that the answering path was not
fitted on; code was right on 33 of those 35, and the plain model on 24 and 29 of the same 35
in its two runs.**

**What is not a result: the totals.** With the replayed LLM answers the routing has 88 of 100
with the defaults of `main`, and 90 with the Swiss object kinds answered in code
(`CHATCHTG_OBJECT_COUNTER_SKIP=`, which needs a decision that has not been made). The plain
model had 77 and 81. These totals are not compared with the baseline
([ADR 0002](../ADR/0002-plain-llm-as-baseline.md)), for three reasons:

- **Fitted plans.** 18 of the 53 code answers (30 of the 67 with the Swiss kinds in code) are on
  plans that the path was fitted on: detector training and valid plans, two plans of the area
  tuning lists, and all 3 answers of the room counter (`tuned` plans of its split). The README
  says that an eval with the object counter on cannot be compared with the baseline.
- **Replayed answers.** Of the 47 questions that go to the LLM by default, 17 would get another
  context from the routing than the one their answer was given with (16 Finnish room counts:
  the conventions; 1 area question: the image with the room table).
- **A small sample.** The two plain runs differ from each other on 6 of the 100 questions.

| Swiss object kinds | Code answered | Of those on held-out plans | Total with replayed LLM answers (inflated) |
|---|---|---|---|
| Left to the LLM (default of `main`) | 53, 45 right | 35, 33 right (plain model: 24 and 29) | 88 |
| Answered in code (`SKIP=`, not decided) | 67, 58 right | 37, 35 right (plain model: 26 and 31) | 90 |

On the held-out plans, code against the plain model on the same questions (default of
`main`): against run 1, 9 answers fixed and none broken (two-sided sign test, p = 0.004);
against run 2, 5 fixed and 1 broken (p = 0.22). The second is no evidence of a difference.
"Held-out" is taken from the split that `chatchtg.evals.code_paths` gives each question.

The tables below are as first written, for the Swiss kinds answered in code and for all
plans, fitted ones included. Their sign tests include the fitted plans.

| | Plain Opus 5.5 | Routing (inflated) | Fixed | Broken | Sign test, fitted plans included |
|---|---|---|---|---|---|
| Against `baseline-opus-5-5-100` | 77 | 90 | 15 | 2 | p = 0.002 |
| Against `fix-issues-baseline-opus-5-5-100` | 81 | 90 | 12 | 3 | p = 0.035 |

With the defaults of `main`: 88; fixed 13, broken 2 (p = 0.007) and fixed 10, broken 3
(p = 0.09).

"Fixed" and "broken" count only the 67 questions that code answered. The 33 other answers are
the same in both columns.

## A fresh run with Kimi K3.1 Pro on Modal

The replay above takes the LLM answers from earlier runs. This run asks a real model: the same
100 questions, the code of `main` with its defaults (Swiss object kinds left to the LLM), and
Kimi K3.1 Pro on a Modal Shared Endpoint for the 47 questions that go to the LLM.

**Result: 88 of 100 for $1.30.** Every answer was made in this run.

| Path | Questions | Right | Cost |
|---|---|---|---|
| Area pipeline | 32 | 32 | $0 |
| Object counter, Finnish plans | 18 | 10 | $0 |
| Room counter (Swiss apartments) | 3 | 3 | $0 |
| Kimi K3.1 Pro | 47 | 43 | $1.30 |
| All | 100 | 88 | $1.30 |

Against the earlier runs of the same model on the same questions:

| Earlier run | Questions in both | Earlier | Routing | Fixed | Broken | Sign test | Cost, earlier and routing |
|---|---|---|---|---|---|---|---|
| `baseline@1` (stopped at its cost limit after 63 questions) | 63 | 39 | 54 | 16 | 1 | p = 0.0003 | $2.76, $0.46 |
| `combined` composer, no counters | 100 | 81 | 88 | 11 | 4 | p = 0.12 | $2.63, $1.30 |

```sh
CHATCHTG_LLM_PROVIDER=modal CHATCHTG_ROOM_COUNTER=confident CHATCHTG_OBJECT_COUNTER=detections \
  uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --sample 100 --seed 0 \
  --composer combined --max-cost 3 --out var/eval-runs/routing-kimi-k3-1-pro-modal-100.jsonl
```

- Kimi is a cheaper and weaker model than Opus 5.5, so this is not the Opus score of the
  routing. No Opus run could be made.
- The limits below hold here too: 13 of the 18 Finnish plans that the object counter answered
  are plans its detector was trained on, and the area constants are fitted to this renderer.
- One run. The gain over the `combined` run without counters is not proven by it.

## Setup

| | |
|---|---|
| Questions | `questions-subset.jsonl`, `--sample 100 --seed 0`, the baseline's 100 questions |
| Composer | `combined@1+areas@4+room-conventions@3` |
| Settings | `CHATCHTG_ROOM_COUNTER=confident`, `CHATCHTG_OBJECT_COUNTER=detections`, `CHATCHTG_OBJECT_COUNTER_SKIP=` |
| Code | `main` at 7896267 (first made on this branch before it was merged with `main`; a reviewer repeated both variants on `main` at 42b0fcb with the same rows) |
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
  CHATCHTG_OBJECT_COUNTER_SKIP= \
  uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --sample 100 --seed 0 \
  --composer combined --evidence --out var/eval-runs/routing-fake-100.jsonl
uv run python scripts/replay_routing.py var/eval-runs/routing-fake-100.jsonl \
  var/eval-runs/baseline-opus-5-5-100.jsonl
```

The run needs the stored detections (`var/metadata`, see the object counter pages), the OCR
cache of the room counter (`var/ocr-cache`, macOS) and takes the area analyses from
`var/area-cache` when they are there.

## Limits

- **Fitted plans.** Of the 32 plans whose object count the counter answered, 25 are plans
  that its detector was fitted on (Finnish 13 of 18 trained on; Swiss 12 of 14: 11 trained on
  and 1 valid plan, where the minimum scores were chosen). With the defaults of `main` the
  counter answers the 18 Finnish ones only, 13 of them on training plans. All 3 answers of
  the room counter are on `tuned` plans of its split. On held-out plans the
  Finnish counter is right on 70.8% of questions and the Swiss one on 89.9%, so the 23 of 32
  here is not the accuracy to expect on a new plan of the same sources.
- **The area constants are fitted to this renderer**, and 2 of the 32 area answers are on
  plans that the measurer or the label reader was tuned on (the other 30 are all right). On held-out plans the pipeline is
  right on 91% to 96% of area questions, not on 32 of 32.
- **The LLM part is a replay, not a run.** The 16 Finnish room counts would get the
  conventions context and the one area question the image with the room table; the answers
  used here were given with the baseline context. A fresh run can differ on these 33.
- **100 questions**, balanced over source and question type and not over the benchmark, so 90%
  is not an estimate of the accuracy on the whole benchmark. The whole-benchmark page has that.
- **The code paths answer only the benchmark's own sheets** where they need stored detections.
