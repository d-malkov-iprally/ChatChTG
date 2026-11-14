# Room counting: what was tried and what works

Run on 2026-11-14. Room counts are 55,814 of the 123,098 benchmark questions. The research
notes proposed a router that answers them without the LLM (OCR first, usual answers for
kitchens and bathrooms, shapes for apartments and storeys). This page records what was built
from those notes and what it scores.

**Result: the plain model was already better than the router. A prompt that states the
counting conventions made the model better still.**

| Method | Sample 1 (360 questions) | Sample 2 (210 questions, Finnish only) | Cost per question |
|---|---|---|---|
| Opus 5.5, `baseline@1` | 340 correct, 94.4% | 192 correct, 91.4% | $0.012 |
| Opus 5.5, `room-conventions@3` | 351 correct, 97.5% | 197 correct, 93.8% | $0.012 |
| Room counter, no LLM (`CHATCHTG_ROOM_COUNTER=always`) | 299 correct, 83.1% | 166 correct, 79.0% | $0 |

- Over both samples the conventions prompt fixed 20 answers and broke 4 (sign test p = 0.002).
- On sample 2 alone, which the prompt was not written on, it fixed 7 and broke 2 (p = 0.18).
  The direction is the same, but that sample alone does not prove the gain.
- The room counter is right where Opus is wrong on only 13 of 570 questions, and wrong where
  Opus is right on 80. Asking it first would lower the score.

## Setup

| | |
|---|---|
| Model | `claude-opus-5-5`, effort `medium` |
| Questions | English room-count questions on plans of the `test` split (see below), one question per plan |
| Sample 1 | 40 questions of each of the 9 kinds (seed 1) |
| Sample 2 | 30 questions of each of the 7 Finnish kinds, on plans not in sample 1 (seed 2) |
| Scoring | the count must be exact (`chatchtg eval`, the rule, no judge) |
| LLM cost of all runs | $12.71 |

The samples are balanced over the kind of question, not drawn in proportion to the benchmark.
So the overall percentages compare methods; they do not estimate the accuracy on the whole benchmark.

**Splits.** The lexicon of the room counter was tuned on 600 Finnish and 400 Swiss plans
(`tuned`). The other plans are divided in two by a hash of their path: `dev` and `test`
(`rooms/split.py`). All LLM samples come from `test`.

## Opus 5.5 with and without the conventions

| Kind of question | n | `baseline@1` | `room-conventions@3` | Fixed | Broken |
|---|---|---|---|---|---|
| Rooms with a given label (Swiss) | 40 | 100% | same context | | |
| Apartments (Swiss) | 40 | 100% | same context | | |
| Bedrooms | 70 | 84.3% | 90.0% | 4 | 0 |
| Kitchens | 70 | 92.9% | 97.1% | 5 | 2 |
| Bathrooms and washrooms | 70 | 91.4% | 94.3% | 2 | 0 |
| Separate WC rooms | 70 | 95.7% | 98.6% | 3 | 1 |
| Balconies | 70 | 92.9% | 95.7% | 2 | 0 |
| Saunas | 70 | 91.4% | 94.3% | 3 | 1 |
| Storeys | 70 | 97.1% | 98.6% | 1 | 0 |
| All | 570 | 93.3% | 96.1% | 20 | 4 |

The composer changes the context only for room counts on Finnish plans. For the Swiss kinds
and for every other question it sends the baseline context, so those rows cannot change.

**Why it helps.** 14 of the 20 fixed answers had counted the rooms of a neighbouring home.
A Finnish sales plan shows one home, and parts of the homes beside it are often visible at
the edge of the sheet. The reference answers count the one home. The plain model counted
every readable label: "2 kitchens, one in each apartment". The conventions tell it to count
the highlighted or complete home only.

**What stays wrong.** Most of the remaining misses are plans without room labels, where the
reference says 0 and the model infers a room from the layout (or the reverse), and reference
answers that contradict the printed labels (three rooms marked MH1 to MH3, reference 0).
The notes estimated a human ceiling of about 95% on Finnish plans, so little is left here.

**Two conventions in the notes were wrong or harmful.**

- The notes said that rooms marked TH or AH count as bedrooms. On the `tuned` and `dev` plans
  read so far, 13 of 14 plans with a TH room have a reference equal to the number of MH labels.
  Version 1 of the prompt had the rule and it broke an answer. Version 2 removed it.
- The lexicon lists TK as a kitchen. On two plans the model counted a TK room (an entrance
  vestibule) because version 2 named TK as a kitchen label. Version 3 says it is not one.

Version 3 was run on the bedroom and kitchen questions only; the other kinds have the same
text in all three versions. The kitchen fix was found on sample 2, so the kitchen row of
sample 2 is not a clean test of it.

## The room counter without the LLM

`rooms/` ports the prototype of the notes: Apple Vision OCR, a lexicon for Finnish
abbreviations and Swiss labels, the usual answer for kitchens and bathrooms, and shape
counting for apartments and storeys. Scores on every room question of the benchmark:

| Kind of question | Questions | Usual answer | Room counter, `test` | Room counter, all | Notes' figure |
|---|---|---|---|---|---|
| Rooms with a given label (Swiss) | 12,730 | 21.4% | 96.0% | 95.8% | 95.0% |
| Apartments (Swiss) | 2,080 | 27.6% | 96.1% | 96.4% | 97.8% |
| Bedrooms | 5,920 | 31.1% | 73.8% | 73.9% | 81.5% |
| Kitchens | 6,524 | 95.8% | 95.3% | 95.8% | 96.5% |
| Bathrooms and washrooms | 4,768 | 88.6% | 88.9% | 88.6% | 88.6% |
| Separate WC rooms | 4,664 | 55.6% | 83.1% | 83.4% | 83.5% |
| Balconies | 2,924 | 51.0% | 86.4% | 85.3% | 82.7% |
| Saunas | 6,630 | 56.9% | 67.2% | 67.5% | 71.0% |
| Storeys | 9,574 | 79.9% | 79.1% | 78.9% | 86.3% |
| All | 55,814 | 55.8% | 84.2% | 85.0% | |

"Usual" is the most frequent answer of that kind. The `test` column is the one to quote.

- The Swiss numbers of the notes hold up: 96% on labelled rooms and on apartments.
- The usual answers for kitchens and bathrooms hold up too (95% and 89%).
- Most Finnish numbers are at or below the notes; balconies are above (86% against 83%). Bedrooms (74% against 82%), saunas (67% against 71%) and storeys (79% against 86%, below the usual answer) fall short of them on the full benchmark.
- Against Opus (`baseline@1`) on the same 570 questions the counter loses on every kind except
  kitchens, where it answers 1 without looking:

| Kind of question | n | Opus | Room counter | Only Opus right | Only counter right |
|---|---|---|---|---|---|
| Rooms with a given label (Swiss) | 40 | 100% | 97.5% | 1 | 0 |
| Apartments (Swiss) | 40 | 100% | 87.5% | 5 | 0 |
| Bedrooms | 70 | 84.3% | 68.6% | 12 | 1 |
| Kitchens | 70 | 92.9% | 95.7% | 3 | 5 |
| Bathrooms and washrooms | 70 | 91.4% | 88.6% | 5 | 3 |
| Separate WC rooms | 70 | 95.7% | 82.9% | 9 | 0 |
| Balconies | 70 | 92.9% | 85.7% | 6 | 1 |
| Saunas | 70 | 91.4% | 65.7% | 20 | 2 |
| Storeys | 70 | 97.1% | 71.4% | 19 | 1 |
| All | 570 | 93.3% | 81.6% | 80 | 13 |

- **Apartments are the exception.** When the unit tags read by OCR and the unit outlines found
  from the red lines give the same number, the count was right on all 1,926 such questions of
  the benchmark. `CHATCHTG_ROOM_COUNTER=confident` answers those in code and saves the most
  expensive kind of room question ($0.025 with Opus). It does not change the score.

### Update of 2026-11-15: three label rules corrected

Measured again on all 55,814 room questions with `rooms.bench` and `rooms.report`, on the
same OCR cache. The other kinds did not change.

| Kind | Split | Before | After |
|---|---|---|---|
| Apartments | tuned / dev / test | 97.8% / 95.1% / 96.1% | 99.0% / 99.0% / 98.2% |
| Balconies | tuned / dev / test | 82.2% / 85.1% / 86.4% | 86.3% / 86.9% / 88.5% |
| Rooms with a given label | tuned / dev / test | 95.0% / 96.5% / 96.0% | 95.8% / 97.0% / 96.5% |

- A unit tag is one capital letter. "Whg. Al", a misread "Whg. A)", was a unit of its own.
- A label read in two overlapping tiles of one pass was counted twice.
- `LASITETTU` (glazed) alone counted as a balcony, also in "LASITETTU TERASSI".

The count that tags and outlines agree on is now right on all 1,970 such questions (before:
1,926). These numbers are for the room counter alone; no LLM run was repeated.

## What was not built

| Direction of the notes | Status |
|---|---|
| OCR and label lexicon | built, measured: below Opus |
| Usual answers for kitchens and bathrooms | built into the room counter |
| Shape counting for apartments and storeys | built; apartments are used, storeys are below Opus |
| Glossary of conventions for the model | built as `room-conventions`: the one change that raised the score |
| Swiss PDF and DXF parsing | not built: ADR 0003 limits the work to raster plans. Opus is at 80 of 80 on the Swiss room questions, so it would save cost, not add correctness |
| Trained segmentation model | not built: the notes measured it below OCR, and OCR is below Opus |
| One inventory call per plan, smaller models, cropped Swiss sheets | not run: these lower cost. None was expected to raise correctness |
| RapidOCR for Linux | not run: the room counter is not on the accuracy path |

## To repeat

```sh
# the two LLM runs (about $2.70 each on sample 1's Finnish questions)
CHATCHTG_LLM_PROVIDER=claude uv run chatchtg eval data/floorplans --questions <sample>.jsonl --composer baseline
CHATCHTG_LLM_PROVIDER=claude uv run chatchtg eval data/floorplans --questions <sample>.jsonl --composer room-conventions

# the room counter on the whole benchmark (no LLM; macOS)
uv run python -m chatchtg.rooms.warm data/floorplans var/ocr-cache
uv run python -m chatchtg.rooms.bench data/floorplans var/ocr-cache var/rooms.jsonl
uv run python -m chatchtg.rooms.report data/floorplans var/rooms.jsonl <llm result files>
```

The sample files and the result files are in `var/eval-runs/room-counting/` of the main
checkout, which Git ignores.

## Limits

- 570 questions, 30 to 70 per kind. A per-kind difference of a few points is not a finding.
- Sample 1 is where the misses were read before the prompt was written. Only sample 2 is a
  held-out test, and on it the gain is 2.4 points with p = 0.18.
- English questions only. Each has a second-language twin with the same answer, which was not run.
- One run per question. The model's answers are not deterministic.
- The conventions describe Finnish sales plans of one home. On a sheet that really shows
  several homes of interest they would undercount.
