# Combined composer: areas and room conventions together

Run on 2026-11-15. The `combined` composer (`combined@1+areas@2+room-conventions@3`) sends an
area question to the `areas` composer and every other question to `room-conventions`, which
adds the counting conventions to a room count on a Finnish plan and builds the baseline
context for anything else. This page records where the benchmark questions go, and a score
for the 100-question baseline sample made by replay from earlier Opus 5.5 runs, at $0 of
Anthropic cost.

**Result: on the baseline sample the combined composer scores 66 to 82 of 100 by replay
(Opus 5.5). 84 of its 100 contexts are byte-for-byte the contexts of earlier runs; the other
16 are Finnish room counts with the conventions, which no earlier run of this sample answered.
With the baseline's answers in their place it would score 81, as `baseline@1` (81) and
`areas@2` (81) do in the runs it was replayed from, at about 60% of the baseline's cost.
A fresh run with Kimi K3.1 Pro on Modal scored 81 of 100 for $2.63, and 48 against 39 for
`baseline@1` on the 63 questions that both Kimi runs answered.**

## Routing on the whole benchmark

`uv run python scripts/route_questions.py data/floorplans/questions.jsonl` asks the
registered composer for its choice on each of the 123,098 questions. No LLM, no OCR.

| Path | Questions | Share | What the questions are |
|---|---|---|---|
| `areas` | 12,016 | 9.8% | every area question (all Swiss, in English, German, French and Italian) |
| `room-conventions`, with conventions | 41,004 | 33.3% | room counts on Finnish plans, in English and Finnish (7 kinds) |
| `room-conventions`, baseline context | 70,078 | 56.9% | object counts on both sources (doors, windows, toilets, ...), room-label and apartment counts on Swiss plans |

- **No question is claimed by both** the area parser (`areas/questions.py`) and the room
  classifier (`rooms/questions.py`).
- **No gap found.** All 12,016 questions of type `area` are read by the area parser. All 14
  Finnish room-count templates (7 kinds, in English and in Finnish) are read by the room
  classifier. The questions on the baseline path are object counts, which neither composer
  is meant to change, and the Swiss room-label and apartment counts, for which
  `room-conventions` has no conventions (it classifies them, profile `swiss`, and adds
  nothing). The script prints the questions of the baseline path by opening words, so a
  new template that the parsers miss shows up there.

## Replay score on the baseline sample

### Setup

| | |
|---|---|
| Model of the answers | `claude-opus-5-5` (answers made on 2026-11-14 and 2026-11-15, not now) |
| Questions | `questions-subset.jsonl`, `--sample 100 --seed 0` (the sample of [the baseline](2026-11-14-baseline-opus-5-5.md)) |
| Scoring | the rule (`correct` in the rows): a count exact, an area within 2% |
| Anthropic cost of this page | $0 |

```sh
R=var/eval-runs
uv run python scripts/replay_composer.py data/floorplans --composer combined \
  --questions questions-subset.jsonl --sample 100 --seed 0 \
  --runs $R/fix-issues-baseline-opus-5-5-100.jsonl $R/fix-issues-areas-opus-5-5-100.jsonl \
  --extra $R/baseline-opus-5-5-100.jsonl $R/room-counting/rooms-*.jsonl $R/room-counting/rooms2-*.jsonl
```

For each question the script builds the combined composer's context from the drawing (the
area pipeline runs, about 2 minutes for the 33 area plans) and looks for an earlier row with
the same drawing, question and context: the system prompt, every text part and the SHA-256 of
the image, compared exactly. The first of `--runs` that has such a row gives the answer.

### Where each answer came from

| Part that builds the context | Questions | Answer taken from | Same context found in | Right |
|---|---|---|---|---|
| `areas@2` (area questions) | 33 | `fix-issues-areas-opus-5-5-100.jsonl` | that file only | 28 |
| `room-conventions@3`, baseline context | 51 | `fix-issues-baseline-opus-5-5-100.jsonl` | also `fix-issues-areas-…` and `baseline-opus-5-5-100.jsonl` | 38 |
| `room-conventions@3`, with conventions | 16 | none | none | unknown |
| **All** | **100** | | | **66 to 82** |

The 33 area contexts are identical to the rows of the `areas@2` run, so the OCR and the
composer on `main` reproduce that run's contexts exactly. Of these, 32 are text only (the
pipeline's result) and one carries the image and the room table.

On the 51 baseline contexts each of the three files has 38 right. They disagree on two
questions (one is right in two files and wrong in the third, the other the other way round):
the same context does not always get the same verdict.

### The 16 questions without an answer

Room counts on Finnish plans. No earlier run answered them with the `room-conventions@3`
context: the runs of [the room-counting page](2026-11-14-room-counting.md) used other plans
or older versions of the conventions.

| Drawing (`data/cubicasa5k/`) | Question | Baseline answer right |
|---|---|---|
| `colorful-13352.png` | kitchens (EN) | yes |
| `high_quality-1039.png` | storeys (EN) | yes |
| `high_quality-3650.png` | kitchens (FI) | yes |
| `high_quality-4026.png` | bedrooms (FI) | yes |
| `high_quality-5594.png` | storeys (FI) | yes |
| `high_quality_architectural-10216.png` | bathrooms (FI) | yes |
| `high_quality_architectural-10267.png` | storeys (FI) | yes |
| `high_quality_architectural-10401.png` | kitchens (FI) | yes |
| `high_quality_architectural-13345.png` | kitchens (EN) | yes |
| `high_quality_architectural-13349.png` | kitchens (EN) | yes |
| `high_quality_architectural-13802.png` | storeys (EN) | **no** |
| `high_quality_architectural-3573.png` | storeys (FI) | yes |
| `high_quality_architectural-3724.png` | bathrooms (FI) | yes |
| `high_quality_architectural-6282.png` | bedrooms (EN) | yes |
| `high_quality_architectural-755.png` | bedrooms (EN) | yes |
| `high_quality_architectural-8091.png` | bathrooms (EN) | yes |

"Baseline answer right" is the same in `fix-issues-baseline-…` and `baseline-opus-5-5-100.jsonl`.

### Against the single composers

| Composer | Right of 100 | LLM cost | Source |
|---|---|---|---|
| `baseline@1` | 81 | $2.67 | `fix-issues-baseline-opus-5-5-100.jsonl` (measured) |
| `baseline@1`, earlier run | 77 | $2.69 | `baseline-opus-5-5-100.jsonl` (measured) |
| `areas@2` | 81 | $1.61 | `fix-issues-areas-opus-5-5-100.jsonl` (measured) |
| `combined@1`, replay | 66 to 82 | about $1.61 | this page; 81 if the 16 conventions answers score as the baseline's (inferred) |

By part, in the two `fix-issues` runs:

| Questions | `baseline@1` | `areas@2` | LLM cost, baseline → areas |
|---|---|---|---|
| 33 area questions | 28 | 28 | $1.23 → $0.17 |
| 51 other questions, baseline context in both | 38 | 38 | $1.29 → $1.29 |
| 16 Finnish room counts, baseline context in both | 15 | 15 | $0.15 → $0.15 |

What this says:

- **On this sample, the area composer's gain is in cost, not in the count.** The area
  questions score 28 of 33 with either context. In the earlier baseline run they scored 24
  ([area pipeline page](2026-11-14-area-pipeline.md)); the plain model's score on them moves
  by 4 between two runs of the same context. Of the five area answers that `areas@2` got
  wrong, three had the pipeline's right result in the text but not as the last number, which
  the rule reads (found in pull request #20, `state-area-in-code`, which has code state the
  result).
- **The conventions can change at most 16 answers, and the baseline already gets 15 of them
  right.** So on this sample the combined composer can gain at most 1 answer over the
  baseline context there, and could lose up to 15. The room-counting page measured a gain of
  2.8 points on 570 room questions (93.3% → 96.1%); on 16 questions that is less than one
  answer.
- **The combination costs what `areas@2` costs** (inferred): the conventions add about 100
  input tokens to 16 questions, a fraction of a cent at Opus prices.

## A real run with Kimi K3.1 Pro on Modal

One fresh run of the combined composer with an open-weights model on a Modal Shared Endpoint,
billed per token to the Modal workspace (no Anthropic cost).

| | |
|---|---|
| Model | `moonshotai/Kimi-K3.1-Pro`, Modal Shared Endpoint, `CHATCHTG_LLM_PROVIDER=modal` |
| Questions | `questions-subset.jsonl`, `--sample 100 --seed 0` |
| Composer | `combined@1+areas@2+room-conventions@3` |
| Answered | 100 of 100 |
| Cost | $2.63 (price of `llm/pricing.py`: $2 / $10 per million input / output tokens) |
| Results | `var/eval-runs/combined-kimi-k3-1-pro-modal-100.jsonl` |

```sh
CHATCHTG_LLM_PROVIDER=modal uv run chatchtg eval data/floorplans --composer combined \
  --questions questions-subset.jsonl --sample 100 --seed 0 --max-cost 3 \
  --out var/eval-runs/combined-kimi-k3-1-pro-modal-100.jsonl
```

**81 of 100 right** (areas 31 of 33, counts 50 of 67). The earlier `baseline@1` run with the
same model (`baseline-kimi-k3-1-pro-modal-100.jsonl`) stopped at its cost limit after 63
questions, so the two are compared on those 63 only:

| The same 63 questions | `baseline@1` | `combined@1` | Cost, baseline → combined |
|---|---|---|---|
| Area questions (part `areas`) | 8 of 15 | 14 of 15 | $1.05 → $0.05 |
| Finnish room counts (part `room-conventions`, with conventions) | 11 of 16 | 15 of 16 | $0.15 → $0.11 |
| Other questions (baseline context in both) | 20 of 32 | 19 of 32 | $1.56 → $1.63 |
| **All** | **39 of 63** | **48 of 63** | **$2.76 → $1.78** |

- Areas: 7 answers fixed, 1 broken. Conventions: 4 fixed, none broken. On the questions whose
  context did not change, 1 answer was fixed and 2 broken: that is the noise of two runs of
  the same context.
- With Kimi, unlike Opus on this sample, both gains show: the weaker model reads areas and
  counts rooms worse from the image, so the measured table and the conventions help more.
  The area questions cost a twentieth: Kimi spent many reasoning tokens on them with the
  image (up to $0.24 for one question), and almost none on the text-only result.
- 15 and 16 questions per group are small. Neither change is proven by this run alone
  (two-sided sign test: areas 7 to 1, p = 0.07; conventions 4 to 0, p = 0.125); both point
  the same way as the earlier measurements with Opus.

## Limits

- **A replay is not a fresh run.** The answers were made by Opus 5.5 in earlier runs. A run
  of the same contexts today could give other answers: the three files disagree on 2 of the
  51 shared contexts, and the plain model's score on the area questions moved by 4 between
  two runs.
- **16 of 100 contexts have no answer.** The replay gives a range (66 to 82), not a score.
  The figure of 81 assumes the conventions do not change these 16 verdicts; it is an
  inference, not a measurement.
- **100 questions is a small sample.** With 33 area questions and 16 Finnish room counts, a
  difference of a few answers between composers is not a finding (ONBOARDING, "Experiments
  and scores").
- **The Kimi comparison covers 63 questions**, because the earlier baseline run stopped at
  its cost limit; each composer was run once.
- **The routing count reads the questions only.** For a question routed to `areas`, the
  area composer itself still falls back to the baseline context when the drawing is not a
  PNG or JPEG or when its pipeline fails.
