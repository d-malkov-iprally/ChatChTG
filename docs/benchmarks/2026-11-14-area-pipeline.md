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

- **The answer was scored, not the LLM's sentence.** The composer hands the LLM the result
  and asks it to state it. A run through `chatchtg eval --composer areas` with a real model
  has not been made.
- **Tuned to this renderer.** The grey levels, line widths, red outline and grey outdoor fill
  are constants of the synthetic Swiss plans. The measurer was tuned on 40 plans and the label
  reader on 40 others; the 100 plans above exclude the first 40 but may include some of the
  second. If the judges' plans are these same 1,040, the constants are fitted to the test set.
- **1:200 plans are the weak spot**: small rooms are a few dozen pixels wide, labels cover
  fixtures, and 1 room number in 18 is not read.
- **Stairs next to a corridor without a wall** are sometimes split wrongly, because the tread
  lines hide the separator.
- **A room polygon that overlaps a wall in the data** cannot be recovered from the image.
