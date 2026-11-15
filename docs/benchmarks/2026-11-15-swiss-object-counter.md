# Object counter on Swiss plans: counts from the Swiss symbol detector

Run on 2026-11-15. The object counter (`src/chatchtg/objects/`) now answers the six Swiss
object-count templates (doors, windows, toilets, washbasins, bathtubs, showers) in code, from
the stored detections of the Swiss symbol detector (run `swiss-medium-1280`,
[record](2026-11-15-swiss-symbol-detector.md)), as it already did for the Finnish plans
([record](2026-11-15-object-counter.md)). No LLM call was made for anything on this page: every
number is scored in code, at $0.

**Result: 89.9% exact on the Swiss test plans (1,518 of 1,688 questions on 163 plans that the
detector was not trained on and whose minimum scores were not chosen on them). The fixtures are
92.6% to 99.4%; doors 79.1% and windows 74.2%. All six templates are answered by default under
`CHATCHTG_OBJECT_COUNTER=detections`.**

## What changed

- **Classifier** (`objects/questions.py`): the six Swiss templates in English, German, French
  and Italian. The Finnish and the Swiss templates about windows, doors and toilets are one kind
  each; `washbasin`, `bathtub` and `shower` are new kinds (the Finnish templates ask for "sinks and
  wash basins" and "showers and bathtubs" together).
- **One counter, two detector profiles** (`objects/counter.py`): `CUBICASA` and `SWISS`. A profile
  names the stored producer and version, the detector classes that each kind counts, and the
  minimum score of each class. The counter asks the profiles in order; the first one that
  answers the kind and has stored detections of the drawing counts. A drawing without stored
  detections goes to the LLM as before.
- **Store** (`objects/importer.py --profile swiss`): the Swiss detections are their own producer,
  `rfdetr-swiss`, version 1 = run `swiss-medium-1280`, with its own minimum scores (door 0.55,
  window 0.40, toilet 0.50, washbasin 0.70, bathtub 0.30, shower 0.80, chosen on the 150 valid
  plans by `modal_swiss::evaluate`). The detections file has no image hashes, so each plan is
  stored under the SHA-256 of its PNG in `data/floorplans/data/swiss_dwellings`, which is the
  document id that the app gives that file. The file names no run; the importer refuses one whose
  minimum scores are not those of the version.
- **Setting** `CHATCHTG_OBJECT_COUNTER_SKIP`: `profile:kind` pairs that the counter leaves to the
  LLM. Not set: `DEFAULT_SKIP` in `objects/counter.py`, which is empty.
- **Report** (`objects/report.py`): one table per profile; the replay replaces the answers of both.

## Classifier, checked on every question of the benchmark

Over all 123,098 questions of `data/floorplans/questions.jsonl` (3,171 distinct texts):

| | Questions claimed | Distinct texts |
|---|---|---|
| Finnish (CubiCasa5K) | 44,510 | 10 (5 templates x English, Finnish) |
| Swiss | 10,758 | 24 (6 templates x 4 languages) |
| any other question (room labels, apartments, areas, Finnish room counts) | 0 | |

10,758 is every Swiss object question: 1,040 plans x 2 questions for doors, windows, toilets and
washbasins, plus 897 x 2 for bathtubs and 322 x 2 for showers. No claimed question is also
claimed by the room classifier. The test `test_classifier_claims_exactly_the_object_questions_of_the_benchmark`
checks the same on `tests/fixtures/benchmark_questions.jsonl` (every distinct count question).

## Exact counts per template and split

`python -m chatchtg.objects.report data/floorplans/questions.jsonl --profile swiss`. Counts are of
questions: the English question and the German, French or Italian question of a plan have the
same answer, so each plan is counted twice.

| Template | train (inflated) | valid (thresholds chosen here) | **test** | Mean abs. error, test |
|---|---|---|---|---|
| doors | 75.0% of 1,454 | 72.0% of 300 | **79.1% of 326** (258) | 0.45 |
| windows | 72.2% of 1,454 | 66.0% of 300 | **74.2% of 326** (242) | 0.68 |
| toilets | 99.3% of 1,454 | 96.7% of 300 | **98.2% of 326** (320) | 0.02 |
| washbasins | 99.2% of 1,454 | 96.7% of 300 | **99.4% of 326** (324) | 0.01 |
| bathtubs | 98.9% of 1,268 | 98.4% of 250 | **99.3% of 276** (274) | 0.04 |
| showers | 98.2% of 442 | 95.7% of 94 | **92.6% of 108** (100) | 0.07 |
| all | 89.2% of 7,526 | 86.1% of 1,544 | **89.9% of 1,688** (1,518) | |

- The split is the one recorded in the detections file (by a hash of the site id: 727 train,
  150 valid, 163 test plans). Each Swiss sheet is one plan; no image is in the benchmark twice.
- The test and valid columns agree with the detector's own record to the plan (for example
  doors 129 of 163 test plans, 108 of 150 valid plans): the counter counts the same boxes with the
  same minimum scores.
- The train column is not higher than test for doors and windows: what limits them is the
  stacked duplicates of the answers, not the fit.
- Without doors and windows, the test column is 98.3% (1,018 of 1,036).

## Which templates the counter answers

**All six, by default.** `DEFAULT_SKIP` is empty; `CHATCHTG_OBJECT_COUNTER_SKIP=swiss:door,swiss:window`
leaves doors and windows to the LLM.

- **Fixtures** (toilets, washbasins, bathtubs, showers): 92.6% to 99.4% exact on test. Clear.
- **Doors and windows**: the alternative is the LLM, and no LLM number on held-out Swiss plans
  exists. The Opus 5.5 baseline has no Swiss door question, and 2 window questions (both right,
  on a train and a valid plan). What is known:
  - Measured ([detector record](2026-11-15-swiss-symbol-detector.md#doors-and-windows-the-answers-count-stacked-duplicates)):
    on 148 of the 163 test plans the door answer differs from the number of doors that the PNG
    shows, because the DXF holds some doors twice at one place; for windows on 37 of 163.
  - Inferred from that: a method that sees only the image and counts every visible door exactly
    would match the answer on 15 of 163 test plans (9%); every visible window exactly, on 126 of
    163 (77.3%). The LLM sees the same image. To beat the counter on windows it would have to
    count the visible windows exactly on at least 121 of 163 sheets with about 17 windows each
    on average (2,741 counted windows on the 163 test plans). Opus 5.5 got 3 of 15 Finnish
    window counts right in the existing runs (a different dataset; most were large plans). On
    doors it could beat 79% only by guessing the hidden duplicates, which the detector learnt
    from its labels.
  - The counter costs $0 and shows every counted box; the LLM costs about $0.027 per Swiss count
    question (baseline) and shows nothing. The judges weigh cost and verifiability as much as
    accuracy.

  So the counter answers doors and windows too. **What would change it:** an Opus (or other LLM)
  run on the Swiss door and window questions of the test plans (326 questions each, about $9 each
  at $0.027) that beats 79.1% or 74.2%. Then add the pair to `DEFAULT_SKIP` or set the variable.

## Replay of the existing Opus runs, at $0

`report --replay` replaces the answer of every question that the counter answers by default
(Finnish and Swiss) in an existing result file, and scores again. No LLM call.

**The "after" totals are inflated and are not a result** ([ADR 0002](../ADR/0002-plain-llm-as-baseline.md)):
24 of the 32 replaced plans are training plans of a detector, and the sample was not drawn to
avoid them.

| Run (100 questions) | Before | After, Finnish replaced | After, Finnish and Swiss replaced |
|---|---|---|---|
| `baseline-opus-5-5-100.jsonl` | 77 | 79 | 81 |
| `fix-issues-baseline-opus-5-5-100.jsonl` | 81 | 83 | 85 |
| `fix-issues-areas-opus-5-5-100.jsonl` | 81 | 83 | 85 |

32 answers are replaced in each run, on the same 32 plans:

| Profile | train | valid | test |
|---|---|---|---|
| Finnish (18 plans) | 13 | 0 | 5 |
| Swiss (14 plans) | 11 | 1 | 2 |

- **Swiss, 14 questions**: the LLM was right on 11, the counter on 13. Both of the counter's fixes
  are on train plans (`site1601_bldg2794_floor12912` toilets 2, Opus 0; `site144_bldg99_floor441`
  washbasins 7, Opus 4). Both are wrong on `site914_bldg1947_floor8548` (bathtubs 4; Opus 3 or 2,
  counter 5; a train plan). On the 3 plans not trained on (2 bathtub questions on test plans, 1
  window question on a valid plan) both are right in all three runs.
- The 14 Swiss questions are 1 per plan: 6 bathtubs, 4 washbasins, 2 toilets, 2 windows, no doors
  and no showers. With doors and windows skipped the Swiss replay gives the same totals (the two
  window answers are right either way).
- On the 8 replaced plans not trained on (both profiles), the LLM was right on 5, 6 and 6 in the
  three runs, the counter on 6. That is no finding in either direction.

## Setup

| | |
|---|---|
| Detector | RF-DETR medium, run `swiss-medium-1280`, `checkpoint_best_total.pth` (PR #28) |
| Detections | `runs/swiss-medium-1280/detections_all.json` on the Modal volume `chatchtg-object-detector`, all 1,040 sheets, score floor 0.05 |
| Stored as | `var/metadata/<sha256>/rfdetr-swiss@1.json`, 1,040 extractions |
| Benchmark | `data/floorplans/questions.jsonl`, all 10,758 Swiss object questions |
| Scoring | the count must be exact |
| LLM cost | $0; no LLM of any provider was called |

## Commands to repeat

```sh
uvx modal volume get chatchtg-object-detector runs/swiss-medium-1280/detections_all.json var/swiss-detections/
uv run python -m chatchtg.objects.importer --profile swiss var/swiss-detections/detections_all.json data/floorplans/data/swiss_dwellings
# The Finnish detections, for the replay of both (see the Finnish record)
uv run python -m chatchtg.objects.importer var/detections/detections.jsonl data/floorplans/data/cubicasa5k
uv run python -m chatchtg.objects.report data/floorplans/questions.jsonl \
    --replay var/eval-runs/baseline-opus-5-5-100.jsonl var/eval-runs/fix-issues-baseline-opus-5-5-100.jsonl \
    var/eval-runs/fix-issues-areas-opus-5-5-100.jsonl
# Finnish only: --profile cubicasa; Swiss only: --profile swiss (the replay then replaces only that profile)
CHATCHTG_OBJECT_COUNTER_SKIP=swiss:door,swiss:window uv run python -m chatchtg.objects.report \
    data/floorplans/questions.jsonl --profile swiss --replay var/eval-runs/baseline-opus-5-5-100.jsonl
```

The classifier check over all questions was a one-off script (claimed per dataset and kind,
none claimed by the room classifier, Swiss claims equal to the 24 template starts); the test
above checks the same on the fixture of distinct count questions.

## Limits

- **Synthetic, one style.** The Swiss sheets are rendered in one constant style, and the detector
  was fitted to it (drawing area, 150 dpi, tile size). A real drawing in another style would need
  new labels; nothing here says how the counter does on one.
- **Only the benchmark's sheets have detections.** The detector does not run in the app (it needs
  torch and a GPU). A new upload, or a benchmark PNG changed by one byte, has no stored detections
  and is answered by the LLM.
- **Stacked doors and windows.** The answers count doors and windows that the DXF holds twice at
  one place and the PNG shows once. The detector matches many door answers because it learnt to
  give more boxes where its labels had stacked doors. The boxes it shows as evidence then do not
  map one to one onto the visible doors: on 148 of 163 test plans at least two counted door boxes
  overlap (IoU 0.05 or more; 6 plans at IoU 0.4 or more). Whether those overlaps are the stacked
  duplicates was not checked by eye. A reviewer who counts the visible doors will disagree with
  the benchmark answer, and with the counter, on most plans.
- **No held-out LLM number.** The choice to answer doors and windows rests on the reasoning above,
  not on a measurement of the LLM.
- The classifier matches the starts of the benchmark's wording. "How many doors are there?" is
  not claimed and goes to the LLM.
- A Finnish-worded window, door or toilet question on a Swiss plan is answered by the Swiss
  detector, and the reverse; the benchmark asks neither, so this was not scored.
