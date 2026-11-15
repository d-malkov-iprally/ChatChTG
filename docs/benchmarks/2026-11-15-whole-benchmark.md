# The whole benchmark: every code path together, and a routing between code and the LLM

Run on 2026-11-15. Four code paths answer benchmark questions without an LLM: the area
pipeline, the room counter, and the object counter with its Finnish and its Swiss detector.
Each one was scored alone before, on its own sample and its own split. This page scores
all of them together, on all 123,098 questions of `data/floorplans/questions.jsonl`, and on
each path's held-out plans. It also estimates the plain LLM on the whole benchmark from the
Opus 5.5 answers that already exist, and gives a routing: which kinds code answers and which
the LLM. **No LLM of any provider was called for this page. Every number in the code columns
is measured, at $0. Every LLM number on the whole benchmark is an estimate from small samples.**

**Results:**

- **Code only: 81.2% on the plans that each path was not fitted on (95% interval over plans
  80.2% to 82.2%), at $0 of LLM.** On all plans, fitted ones included, 83.6%. That figure is
  inflated and is not a result.
- **Plain LLM (Opus 5.5, baseline context), estimate: 72.4%**, at about $0.020 per
  question. 55% of the benchmark is in kinds with fewer than 30 measured questions. With
  those kinds at the ends of their 95% intervals, the estimate is 59.1% to 84.5%.
- **Recommended routing, estimate: 86.4%**, at about $0.006 of LLM per question. Code answers
  object counts, area questions and apartments. The LLM answers Finnish room counts, with the
  conventions prompt, and Swiss labelled-room counts. With the LLM-answered kinds at the ends
  of their 95% intervals, the estimate is 84.2% to 87.3%. The routing can be set with
  existing settings ([below](#the-settings)).

Every question is claimed by exactly one path. No question is claimed by two paths, and no
question is claimed by none.

## The table

"Held-out" is the accuracy on the plans that the path was not fitted on ([which plans](#held-out-plans)).
"Best Opus" is the best measured Opus 5.5 accuracy for the kind, over all existing answers.
It gives the prompt, the number of answers and the number of distinct questions. "Plain" is
the baseline context. "Conv." is the room conventions prompt (`room-conventions@3` for
bedrooms and kitchens, the same text in earlier versions for the other kinds). The cost of
code is $0 of LLM. Its compute cost is [below](#cost).

| Kind | Questions | Code, held-out (plans, questions) | Code, all plans (inflated) | Best Opus 5.5 (answers / questions) | Opus cost per question |
|---|---|---|---|---|---|
| Finnish: windows | 9,574 | **47.9%** (386, 772) | 63.8% | 20.0% plain (15 / 5) | $0.028 |
| Finnish: doors | 7,038 | **70.4%** (277, 554) | 78.0% | 0.0% plain (6 / 2) | $0.022 |
| Finnish: toilets | 9,564 | **82.6%** (385, 770) | 84.5% | 50.0% plain (6 / 2) | $0.010 |
| Finnish: sinks and wash basins | 9,450 | **78.7%** (381, 762) | 83.0% | 60.0% plain (15 / 5) | $0.022 |
| Finnish: showers and bathtubs | 8,884 | **74.7%** (364, 728) | 79.9% | 75.0% plain (12 / 4) | $0.020 |
| Finnish: bedrooms | 5,920 | **73.8%** (1,286, 2,572) | 73.9% | 90.0% conv. (70 / 70) | $0.011 |
| Finnish: kitchens | 6,524 | **95.3%** (1,420, 2,840) | 95.8% | 97.1% conv. (70 / 70) | $0.009 |
| Finnish: bathrooms and washrooms | 4,768 | **88.9%** (1,043, 2,086) | 88.6% | 94.3% conv. (70 / 70) | $0.008 |
| Finnish: separate WC rooms | 4,664 | **83.1%** (1,026, 2,052) | 83.4% | 98.6% conv. (70 / 70) | $0.011 |
| Finnish: balconies | 2,924 | **88.5%** (655, 1,310) | 87.6% | 95.7% conv. (70 / 70) | $0.008 |
| Finnish: saunas | 6,630 | **67.3%** (1,434, 2,868) | 67.6% | 94.3% conv. (70 / 70) | $0.009 |
| Finnish: storeys | 9,574 | **79.1%** (2,080, 4,160) | 78.9% | 98.6% conv. (70 / 70) | $0.009 |
| Swiss: rooms with a given label | 12,730 | **96.5%** (333, 4,060) | 96.4% | 100.0% plain (85 / 55) | $0.023 |
| Swiss: apartments | 2,080 | **98.2%** (333, 666) | 98.8% | 100.0% plain (52 / 44) | $0.025 |
| Swiss: windows | 2,080 | **74.2%** (163, 326) | 71.6% | 100.0% plain (6 / 2) | $0.037 |
| Swiss: doors | 2,080 | **79.1%** (163, 326) | 75.2% | not measured | about $0.033 |
| Swiss: toilets | 2,080 | **98.2%** (163, 326) | 98.8% | 50.0% plain (6 / 2) | $0.033 |
| Swiss: washbasins | 2,080 | **99.4%** (163, 326) | 98.8% | 75.0% plain (12 / 4) | $0.035 |
| Swiss: bathtubs | 1,794 | **99.3%** (138, 276) | 98.9% | 83.3% plain (18 / 6) | $0.028 |
| Swiss: showers | 644 | **92.6%** (54, 108) | 96.9% | not measured | about $0.033 |
| Swiss: area of one numbered room | 4,152 | **91.9%** (963, 3,844) | 91.9% | 72.7% plain (22 / 11) | $0.027 |
| Swiss: total area of rooms with a label | 2,034 | **96.1%** (942, 1,884) | 95.7% | 75.0% plain (16 / 8) | $0.036 |
| Swiss: area of one apartment | 3,750 | **91.4%** (963, 3,460) | 91.0% | 66.7% plain (12 / 6) | $0.045 |
| Swiss: net floor area of the floor | 2,080 | **91.6%** (963, 1,926) | 91.3% | 100.0% plain (16 / 8) | $0.048 |
| **All, question-weighted** | **123,098** | **81.2%** (80.2% to 82.2%) | 83.6% | estimate 72.4% (59.1% to 84.5%) | about $0.020 |

- The Opus answers are repeated runs of the same questions. The baseline sample was run two
  or three times, so 15 answers to Finnish windows are 5 questions. The Wilson intervals of
  the script count answers, so they are too narrow for these kinds.
- "Not measured": no Opus answer exists for Swiss doors and Swiss showers. The cost given for
  them is the mean of the other Swiss object kinds.
- The Opus cost per question is the mean `cost_usd` of those answers (effort `medium`). The
  conventions prompt costs about the same as the plain one.

## Held-out plans

Each path was fitted on other plans. The held-out column uses only the plans that the path
was not fitted on:

| Path | Fitted on (left out) | Held out (in the column) |
|---|---|---|
| Object counter, Finnish | the 4,200 `train` plans of CubiCasa5K (detector training), the 400 `valid` plans (minimum scores), and 5 plans whose image is also a train image | `test`: 386 plans, 3,586 questions |
| Object counter, Swiss | 727 `train` sites (detector training), 150 `valid` (minimum scores) | `test`: 163 plans, 1,688 questions |
| Room counter | `tuned`: the 600 Finnish and 400 Swiss plans that the lexicon was tuned on; `dev`: left for further tuning | `test` of `rooms/split.py`: 22,614 questions |
| Area pipeline | the 40 plans that the measurer was tuned on (`stems/tune-100.txt`, `stems/tune-200.txt`, identical to `var/measure-dev-stems.txt`) and the 40 plans of the label reader's development sample (`stems/label-dev.txt`: the 20 + 20 draw with seed 101 of `scripts/eval_labels.py`). The two lists share 3 plans, so 77 plans are left out | 963 plans, 11,114 questions |

- The membership rules of the area pipeline (grey fill, red outlines) were checked on all
  1,040 plans (`scripts/eval_membership.py`), so no Swiss plan is unseen by that stage. They
  are fixed rules, with 99.9% of rooms in the right apartment, so this changes little. It is
  still a fit to these plans.
- The room counter's lexicon and the constants of every path were made for these renderers
  and these datasets. "Held out" means another plan of the same dataset, not another kind of
  drawing.

## The three scores

### Code only: 81.2%

The question-weighted mean of the 24 held-out accuracies. Each kind counts with its number
of questions in the benchmark. The 95% interval, 80.2% to 82.2%, is from 2,000 bootstrap
draws over plans. In each draw the plans of each dataset are drawn again with replacement,
and every question of a drawn plan comes with it. So the questions of one plan are not
treated as independent, and neither are the two language versions of a question. The
interval does not include the uncertainty of the choice of split. It also does not cover the
weights, which are fixed.

The weakest kinds weigh the most: Finnish windows (47.9%, 7.8% of the benchmark), saunas
(67.3%), doors (70.4%) and bedrooms (73.8%).

### Plain LLM: estimate 72.4%

The question-weighted mean of the Opus 5.5 accuracy per kind, with the baseline context
(`baseline@1`, or a composer that sent the baseline context for that question). This is an
estimate, not a measurement:

- **Kinds with almost no measurement** (fewer than 30 distinct questions): all five Finnish
  object kinds (2 to 5 questions each), all six Swiss object kinds (0 to 6), and all four area
  kinds (6 to 11). Together they are 54.7% of the benchmark. Swiss doors and showers have no
  answer at all. They get the pooled accuracy of the other Swiss object answers (33 of 42)
  and a range of 0 to 100%.
- **Range:** with those kinds at the ends of their 95% intervals (Wilson, over answers) and
  the others at their point estimate, 59.1% to 84.5%. With every kind at the ends of its
  interval, 55.9% to 85.8%.
- The room kinds rest on 52 to 85 answers each, all on `test` plans of the room split. The
  baseline sample of 100 questions is balanced over source and type, not over kinds.
- The 100-question baseline was 77 of 100 and 81 of 100. Its sample has more area and Swiss
  questions, and fewer Finnish object questions, than the benchmark. The objects are where the
  LLM is weakest in the existing answers, so a lower whole-benchmark score is plausible. With
  2 to 5 questions per object kind, it is not shown.

### Recommended routing: estimate 86.4%

| Kind | Route | Evidence | Accuracy used | LLM cost per question |
|---|---|---|---|---|
| Finnish: windows, doors, toilets, sinks, showers and bathtubs | code | LLM side 2 to 5 questions | code, held out | $0 |
| Finnish: bedrooms, kitchens, bathrooms, WC rooms, balconies, saunas, storeys | LLM, conventions prompt | measured on the same 70 questions per kind | Opus, conv. | $0.008 to $0.011 |
| Swiss: rooms with a given label | LLM, baseline context | measured on the same questions (55; 43 on held-out plans) | Opus 100% | $0.023 |
| Swiss: apartments | code when tags and outlines agree (93.1% of held-out questions, 100% right), LLM otherwise | measured on the same questions (44) | 100% | $0.0017 |
| Swiss: windows, toilets, washbasins, bathtubs | code | LLM side 2 to 6 questions | code, held out | $0 |
| Swiss: doors, showers | code | LLM not measured; chosen by reasoning | code, held out | $0 |
| Swiss: the four area kinds | code when the pipeline is confident (90% to 100% of held-out questions, 92.1% to 96.3% right), LLM with the image and the room table otherwise | LLM side 6 to 11 questions | see below | $0.0001 to $0.0044 |

- **Score: 86.4%** question-weighted, against 81.2% for code only and an estimated 72.4% for
  the plain LLM. **LLM cost: about $0.006 per question** on average, against about $0.020 for
  the plain LLM. 43.7% of the questions go to the LLM.
- Where the routing differs from code only, the gain rests on comparisons on the same
  questions, with 70 questions per Finnish room kind and 55 and 44 on Swiss rooms. The routing
  takes the LLM only there. Where it takes code, the comparison with the LLM rests on a few
  questions or on none. **This page does not claim that code beats the LLM on those kinds.**
  The code side is measured on hundreds of held-out plans. The LLM side is the open part
  ([what would settle it](#what-would-settle-the-open-kinds)).
- **Area questions where the pipeline is not confident** go to the LLM with the image and the
  room table (the `areas` composer). That context has never been scored on these questions.
  The estimate uses the plain LLM's accuracy of the kind there. If code answered these too, the
  routing scores the same, 86.4%, at $0.0028 of LLM per question. On the held-out plans the
  pipeline's own value is right on 63% (apartments), 65% (floor) and 93% (labels) of the
  questions where it is not confident.
- **Range:** with each LLM-answered kind at the ends of its 95% interval (Wilson, 70 or 85
  answers), the estimate is 84.2% to 87.3%. Code-answered kinds are held at their measured
  accuracy.
- **Kitchens are a tie.** On the same 70 questions code was right on 67 and the conventions
  prompt on 68. With kitchens answered by code, the routing scores 86.3% and saves $0.009 on
  5% of the questions. Code answers 1 there without looking (`usual`), so it shows no evidence.
  The existing settings also cannot send kitchens alone to the room counter. The routing keeps
  the LLM.

### The settings

The routing needs no new setting:

```sh
CHATCHTG_COMPOSER=combined          # areas composer for area questions, room conventions for Finnish room counts, baseline otherwise
CHATCHTG_ROOM_COUNTER=confident     # apartments in code when tags and outlines agree; every other room count to the LLM
CHATCHTG_OBJECT_COUNTER=detections  # the Finnish object kinds in code
CHATCHTG_OBJECT_COUNTER_SKIP=       # set but empty: the six Swiss object kinds in code too
```

**The Swiss object kinds are off by default** (`DEFAULT_SKIP` in `objects/counter.py`). The
Swiss detector was trained on labels from the DXF twins of the plans, and whether such a
model may answer is not decided: it needs an ADR. Without `CHATCHTG_OBJECT_COUNTER_SKIP=`
the Swiss object questions (10,758, 8.7% of the benchmark) go to the LLM, and the routing's
score for them is the LLM's, which is measured on 0 to 6 questions per kind. The routing
figures on this page are for the routing with the Swiss kinds answered in code.

The defaults are unchanged. The room counter needs Apple Vision (macOS). Without it,
apartments go to the LLM, which raises the cost by about $0.0004 per question and leaves the
score the same. Only drawings with stored detections get object counts from code. A new
upload gets the LLM.

## What would settle the open kinds

Each settling run is an Opus 5.5 run with the baseline context, on held-out plans of the
path, one question per plan, compared with code on the same questions. The costs use the
per-question costs above. None of these runs was made.

| Kinds | Run | Approx. cost | What it decides |
|---|---|---|---|
| Finnish windows, doors, toilets, sinks, showers+bathtubs | 150 questions per kind on `test` plans of CubiCasa5K | $3 to $4.20 per kind, about $17 | whether the LLM beats 48% to 83%; windows and doors first (the most questions and the lowest code accuracy) |
| Swiss windows, doors | all 163 `test` plans, each | about $6 per kind | whether to set `CHATCHTG_OBJECT_COUNTER_SKIP=swiss:door,swiss:window` |
| Swiss toilets, washbasins, bathtubs, showers | 100 questions per kind (showers: the 54 test plans) | about $12 | low priority: code is 92.6% to 99.4% |
| The four area kinds | 150 held-out questions per kind, plain LLM; separately the non-confident questions (about 650 on held-out plans) with the `areas` context | $4 to $7 per kind, about $23; the second run is not costed | the floor area first: the LLM was right on 8 of 8 there, code is 91.6% |
| Finnish kitchens | 300 questions on `test` plans, conventions prompt | about $2.70 | whether code at $0 is as good |

## Cost

| Path | LLM cost | Compute |
|---|---|---|
| Object counter, Finnish | $0 | detector run on Modal: under $0.80 for all 5,000 CubiCasa5K plans, so about $0.00002 per question; the detector does not run in the app |
| Object counter, Swiss | $0 | about 1.1 s of an A10G per sheet, about $0.0003 per sheet, about $0.00003 per question |
| Room counter | $0 | Apple Vision OCR on the laptop, cached per plan |
| Area pipeline | $0 | OCR 5 to 15 s of CPU per plan, about 12 questions per plan. The 743 plans that were missing from the cache took about an hour of wall time with 5 workers, on a laptop shared with other jobs |
| Opus 5.5 | $0.008 to $0.048 per question by kind (above) | |

## Where the numbers come from

| Number | Source |
|---|---|
| Object answers | the stored detections in `var/metadata` (`rfdetr-large-1024@1`: 4,980 plans, `rfdetr-swiss@1`: 1,040 plans), copied from the `swiss-counter` worktree; counted with `objects.counter.count`, split by `objects.report.split_of`. These reproduce the test columns of the two object records to the question |
| Room answers | `var/rooms.jsonl`, made again for this page with `python -m chatchtg.rooms.bench data/floorplans var/ocr-cache var/rooms.jsonl 4` on the Apple Vision cache of the main checkout (copied). The file in `var/eval-runs/room-counting/all.jsonl` dates from before the three label rules of 2026-11-15 were corrected, so it was not used. `rooms.report` on the new file gives the numbers of that update exactly (apartments 99.0 / 99.0 / 98.2%, balconies 86.3 / 86.9 / 88.5%, labels 95.8 / 97.0 / 96.5%) |
| Area answers | the plan analyses in `var/area-cache`, all 1,040 Swiss plans, measurer `areas@4`. 297 come from the cache of the `area-1-200` worktree, written by its final runs after commit `54d77d5`. The code of `areas/` has not changed since, except `crosscheck.py`. The other 743 were made for this page by `scripts/eval_areas.py --stems var/swiss-all-stems.txt --workers 5 --out var/areas-all.jsonl`. The scorer turns each analysis into the answer that the app gives: `context.areas.decide` says whether the pipeline is confident, and the value is read from `statement_text` as the eval rule reads it. On 6 of 12,016 questions the two decimals of the statement make a right raw value wrong. Without them `eval_areas.py` gives 92.2% on all plans |
| Opus answers | the 59 distinct files under `var/eval-runs/` of the main checkout (the 100-question baselines `baseline-opus-5-5-100.jsonl`, `fix-issues-baseline-opus-5-5-100.jsonl`, the non-area rows of `fix-issues-areas-opus-5-5-100.jsonl`, and `room-counting/rooms*-*.jsonl`). The other worktrees hold no further Opus answers. Rows of the `areas@2` composer on area questions were left out, because there the LLM only restated the pipeline's result |

To repeat (from a worktree with the caches and detections above in `var/`, and `data` linked
to the main checkout's `data`):

```sh
ls data/floorplans/data/swiss_dwellings/*.png | xargs -n1 basename | sed 's/\.png$//' > var/swiss-all-stems.txt
uv run python scripts/eval_areas.py --stems var/swiss-all-stems.txt --workers 5 --out var/areas-all.jsonl
uv run python -m chatchtg.rooms.bench data/floorplans var/ocr-cache var/rooms.jsonl 4
uv run python -m chatchtg.evals.code_paths data/floorplans --area-cache var/area-cache --rooms var/rooms.jsonl \
    --area-tuned docs/benchmarks/stems/tune-100.txt docs/benchmarks/stems/tune-200.txt docs/benchmarks/stems/label-dev.txt \
    --llm $(find ../ChatChTG/var/eval-runs -name '*.jsonl') --out var/code-paths.jsonl
```

The last command prints every table of this page except the one about settling runs and the
cost table. The routing variants (areas in code for all questions, kitchens in code, the
range) were computed from `var/code-paths.jsonl` with a few lines of Python that are not in
the repository. Cost: $0, no LLM call.

## Limits

- **Only the benchmark's sheets have detections.** The object counter answers from detections
  stored by a batch run. A new drawing goes to the LLM. The 81.2% is a score on these sheets
  and nothing more.
- **Constants fitted to these renderers.** The detectors, the room lexicon and the area
  measurer were all made for the CubiCasa5K and the Swiss Dwellings renderings. Held-out plans
  are new plans of the same style, not new styles.
- **Swiss door and window answers count duplicates.** The reference answers count doors and
  windows that the DXF holds twice at one place and the PNG shows once. On 148 of 163 test
  plans the door answer differs from the visible doors, and on 37 the window answer does. The
  Swiss detector learnt the doubled doors from its labels. A reviewer who counts what the
  image shows will disagree with the benchmark and with the counter.
- **The LLM column is an estimate.** It rests on 2 to 85 answers per kind, from runs that were
  not drawn for this purpose. The answers are repeated runs of a few questions, and the model
  is not deterministic. Swiss doors and showers have no measurement. No conclusion about
  code against the LLM on the object and area kinds should be drawn from it.
- **The routing estimate mixes measured and estimated parts.** Its code parts are measured on
  held-out plans. Its LLM parts are Opus samples of 44 to 85 questions on `test` plans of the
  room split. The non-confident area questions are an estimate.
- **The conventions prompt was written on sample 1** of the room questions. On sample 2 alone
  its gain was not significant (p = 0.18; [room record](2026-11-14-room-counting.md)).
- **Weights.** Each kind counts with its number of questions in the benchmark, in both
  languages. The English and the translated question of a plan have the same answer, so the
  benchmark counts most plans twice per kind.
