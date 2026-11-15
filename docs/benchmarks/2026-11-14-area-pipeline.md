# Area pipeline: area questions answered in code

Run on 2026-11-14. The `areas@1` composer builds a room table from the PNG (local OCR,
membership from the grey fill and the red outlines, measured or printed areas) and adds up
the rooms the question asks for. These numbers score that computed answer directly. No LLM
was called, so they cost $0 and they are deterministic.

**Result: 31 of 33 area questions of the baseline run correct (94%), against 24 of 33 (73%)
for the plain model. On 100 other plans, 1,050 of 1,156 area questions correct (90.8%).**

## The baseline's 33 area questions

The same questions as in [the Opus 5.5 baseline](2026-11-14-baseline-opus-5-5.md)
(`--sample 100 --seed 0`, the 33 rows of type `area`).

| Group | Plain model | Pipeline |
|---|---|---|
| All area questions | 24 of 33 | 31 of 33 |
| Plans without printed areas | 10 of 19 | 18 of 19 |
| Plans with printed areas | 14 of 14 | 13 of 14 |
| Area of one numbered room | 7 of 11 | 10 of 11 |
| Total of rooms with a given label | 5 of 8 | 8 of 8 |
| Area of one apartment | 4 of 6 | 6 of 6 |
| Net floor area of the whole floor | 8 of 8 | 7 of 8 |

With 33 questions the 95% interval of 94% is about 80% to 98%. The two misses are measured
rooms that are off by more than 2%.

```sh
uv run python scripts/eval_areas.py --results var/eval-runs/baseline-opus-5-5-100.jsonl
```

## 100 held-out plans, all their area questions

100 plans drawn at random (`--seed 3`) from the 1,000 plans that were not used to tune the
measurer. 87 are 1:100 and 13 are 1:200; 35 print the room areas.

| Group | Questions | Correct |
|---|---|---|
| All | 1,156 | 90.8% |
| Plans without printed areas | 756 | 88.1% |
| Plans with printed areas | 400 | 96.0% |
| Plans at 1:100 | 1,000 | 92.8% |
| Plans at 1:200 | 156 | 78.2% |
| Area of one numbered room | 396 | 91.9% |
| Total of rooms with a given label | 200 | 95.0% |
| Area of one apartment | 360 | 87.8% |
| Net floor area of the whole floor | 200 | 90.0% |

The interval of the overall figure, counting plans and not questions as independent, is a
few points wide. The 1:200 figure rests on 13 plans.

```sh
uv run python scripts/eval_areas.py --plans 100 --seed 3 --exclude var/measure-dev-stems.txt
```

### Why the wrong answers were wrong

The first stage that explains each of the 106 wrong answers, by comparison with the DXF:

| Cause | Answers |
|---|---|
| A measured room is off by more than 2% | 70 |
| A room label was not read | 14 |
| A room was put in the wrong apartment, or its outdoor flag is wrong | 14 |
| A room has no area | 4 |
| A room number was misread | 2 |
| Other | 2 |

### When the pipeline knows it may be wrong

The label reader reports gaps in the room numbering. A total on a plan with a gap may lack a
room.

| Plans | Numbering | Questions | Correct |
|---|---|---|---|
| Printed areas | no gap that matters | 350 | 98.3% |
| Printed areas | a gap | 50 | 80.0% |
| No printed areas | no gap that matters | 684 | 89.8% |
| No printed areas | a gap | 72 | 72.2% |

The composer uses this: on a plan with printed areas and a gap, the LLM gets the image and
the room table, because the plain model reads printed labels well. On a plan without printed
areas the tool's total is kept, because the plain model was right there about half the time.

## The stages alone

| Stage | Sample | Result |
|---|---|---|
| Question parser | all 12,016 area questions | 100% parsed |
| Aggregation rules, with true areas and membership from the image | all 12,016 | 100% of room, label and floor questions; 99.9% of apartment questions |
| Room in the right apartment | 28,839 rooms, 1,040 plans | 99.9% |
| Room number found by OCR | 60 held-out plans | 98.5% at 1:100, 94.4% at 1:200 |
| Printed area read exactly | same | 97.8% at 1:100, 89.6% at 1:200 |
| Measured room within 2%, label positions from the DXF | 100 held-out plans, 3,539 rooms | 88.0% at 1:100, 66.8% at 1:200 |

The measurer of the research prototype had 62% at 1:100 and 40% at 1:200. The gain comes from
three changes:

- **Lines inside a room no longer cut it.** Fixtures, door swings, stair treads and lift
  crosses are told from room boundaries (walls and light-grey separators) by their darkness.
- **Door and window openings belong to no room.** The pocket in the wall is found by its
  shape: a strip between two wall ends.
- **Edges are placed to a fraction of a pixel.** Each edge of the room polygon is moved to
  the centre of the line under it, read from the anti-aliased grey values. The median error
  of a measured room at 1:100 is 0.3%.

## Cost

- **LLM:** a question that the pipeline answers needs no image. The context is the result as
  text, about 100 to 400 tokens, against about 4,900 for the baseline. Not measured on the
  API in this run.
- **CPU:** the room measurement takes about 0.5 s per plan and the membership 0.1 s. The OCR
  dominates: about 5 to 15 s per plan on a laptop CPU. It was not timed cleanly, because the
  machine was shared with other jobs during the run. The analysis is kept in memory per
  drawing, so the roughly 12 area questions of a plan pay for it once.

## Limits

- **The answer was scored, not the LLM's sentence.** At the time of this run the composer
  handed the LLM the result and asked it to state it. This no longer applies: since
  2026-11-15 code states the result and the text itself is scored, see
  [the update below](#update-of-2026-11-15-the-result-is-stated-in-code).
- **Tuned to this renderer.** The grey levels, line widths, red outline and grey outdoor fill
  are constants of the synthetic Swiss plans. The measurer was tuned on 40 plans and the label
  reader on 40 others; the 100 plans above exclude the first 40 but may include some of the
  second. If the judges' plans are these same 1,040, the constants are fitted to the test set.
- **1:200 plans are the weak spot**: small rooms are a few dozen pixels wide, labels cover
  fixtures, and 1 room number in 18 is not read.
- **Stairs next to a corridor without a wall** are sometimes split wrongly, because the tread
  lines hide the separator.
- **A room polygon that overlaps a wall in the data** cannot be recovered from the image.

## Update of 2026-11-15: rooms measured as one region

When no wall is found between two labelled rooms, the pipeline now counts their region once
in a total, and gives no answer of its own when it cannot tell the rooms apart (the LLM then
gets the image and the room table). Before, a part of the region was counted twice.

Measured with `scripts/eval_areas.py --plans 100` (seed 0, 1,142 area questions, no LLM call),
on the code before and after the change:

| | Before | After |
|---|---|---|
| Answered in code and correct | 1,016 | 1,012 |
| Answered in code and wrong | 120 | 108 |
| Not answered in code (goes to the LLM) | 6 | 22 |
| Correct among the answers the pipeline is confident of | 91.9% | 92.3% |

Of the 16 questions that now go to the LLM and did not before, 12 had a wrong answer and 4 a right one.
No question changed from right to wrong in code. On `site10161_bldg14756_floor39134` (not in
this sample) the apartment total went from 68.14 m² to 57.70 m², against a reference of 57.87.
No LLM run was repeated for this update.

## Update of 2026-11-15: the result is stated in code

**When the pipeline is confident, code now writes the answer and no LLM is called
(`areas@3`, model `area-pipeline`). On the 32 such questions of the baseline sample the
text is correct 30 times at $0, against 27 times at $0.141 when Opus 5.5 stated the same
results.**

Before, the composer gave the LLM the result as text and asked it to state it. The eval rule
reads the last number of an answer, and the LLM's sentence did not always end with the
result. The text that code writes names the rooms that were added up, says for each whether
its area was printed or measured, adds a caution for a measured answer at 1:200 or coarser
and for a total on a plan with a gap in the numbering, and always ends with
`Answer: <value> m²`.

### Run A: the baseline's area questions

The 33 area questions of `--sample 100 --seed 0`. 32 of them get an answer from the pipeline;
the other one (`site1125_bldg2268_floor10441`, printed areas and a gap in the numbering)
goes to the LLM with the image and was not answered in this run.

| The same 32 questions | Correct by the rule | Cost |
|---|---|---|
| Stated by the LLM (`areas@2`, Opus 5.5, `fix-issues-areas-opus-5-5-100.jsonl`) | 27 | $0.141 |
| Stated in code (`areas@3`, no LLM) | 30 | $0 |
| The pipeline's own unrounded value | 30 | $0 |

9 of the 32 answers rest on printed areas (9 correct) and 23 on measured ones (21 correct).
The two wrong answers are wrong in both runs: the measured value is off by more than 2%.
The three answers that the LLM's sentence lost had the right number in the text, but not as
the last number:

- "…beträgt insgesamt **338.17 m²**. Zwei Werte (2.12 Bad und 2.25 Lift) sind aus der
  Zeichnung gemessen…": read as 2.25.
- "The tool reports a net floor area of **3.01 m²** for room 4.39 (Bad).": read as 4.39.
- "…is **112.63 m²**. (The rounded room areas listed add up to 112.62 m². The 0.01 m²
  difference…": read as 0.01.

No answer was correct in the LLM-stated run and wrong in the code-stated one.

### Run B: all area questions of the subset

All 500 questions of `questions-subset.jsonl`; 167 are area questions, each on another plan.

| | Questions |
|---|---|
| Stated in code | 157 of 167 (94%) |
| of those, correct by the rule on the text | 141 (89.8%) |
| of those, correct by the pipeline's own value | 141 |
| Not answered in this run (go to the LLM with the image and the room table) | 10 |

By how the areas were obtained: printed 29 of 29 correct, measured 112 of 128. In both runs
every stated text ends with the total rounded to two decimals, and no answer is wrong by the
rule while its value is right.

### Cost

- **$0.0044 saved for each stated question**: the 32 calls of the `areas@2` run cost $0.141
  (about 430 input and 134 output tokens each, Opus 5.5). A stated question now costs $0 of
  LLM, against $0.038 for an area question of the baseline.
- No network round trip for these answers. The CPU cost of the OCR is unchanged. Run A took
  6.5 minutes and run B 29 minutes of wall time on a laptop that other jobs were using.

### Commands

```sh
CHATCHTG_LLM_PROVIDER=fake uv run chatchtg eval data/floorplans --questions questions-subset.jsonl \
  --sample 100 --seed 0 --composer areas --max-cost 0.01 --out var/eval-runs/areas-stated-fake-100.jsonl
uv run python scripts/eval_area_statement.py --results var/eval-runs/areas-stated-fake-100.jsonl \
  --compare var/eval-runs/fix-issues-areas-opus-5-5-100.jsonl

CHATCHTG_LLM_PROVIDER=fake uv run chatchtg eval data/floorplans --questions questions-subset.jsonl \
  --composer areas --max-cost 0.01 --out var/eval-runs/areas-stated-fake-500.jsonl
uv run python scripts/eval_area_statement.py --results var/eval-runs/areas-stated-fake-500.jsonl
```

Ignore the accuracy that `chatchtg eval` prints for these runs: the count questions and the
area questions that go to the LLM were answered by the fake LLM.

### Limits

- **No run with a real model was made for this update.** The `areas@2` figures come from one
  run of a model that is not deterministic; another run could lose more or fewer sentences.
  With 32 questions, 30 against 27 is not a finding about accuracy by itself. The finding is
  that the code-made text scores exactly as the pipeline's value does, at no cost.
- **The questions that go to the LLM were not answered in these runs** (1 of 33 and 10 of
  167). Their context did not change, so their accuracy is unchanged, and it is not measured
  here. A total over all area questions needs a run with a real model.
- **Two decimals.** The text gives the total with two decimals, as the plans and the
  reference answers do. In rare cases the rounding moves an answer across the 2% line: in a
  replay of 1,028 answers on 100 plans the unrounded value was right 936 times and the
  rounded one 934 times (one quantity, asked in two languages). In runs A and B it changed
  no answer. The LLM was given the rounded value too, so this is not a change.
- **The text is English** for a question in any language; the room names are as printed on
  the plan.
- **A room number that was read twice** is described by its reading that has an area. This
  correction was made after the two runs; no row of either run has such a room.
