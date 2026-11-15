# Object counter: counts from a trained symbol detector

Run on 2026-11-15. Five object-count templates on CubiCasa5K plans (windows, doors, toilet
bowls, sinks and wash basins, showers and bathtubs together), each in English and Finnish, are
44,510 of the 123,098 benchmark questions (36%). The plain LLM is weakest there. The object
counter (`src/chatchtg/objects/`) answers them in code from the detections of the RF-DETR
detector of `object_detector/` (run `large-1024`), at no LLM cost.

**Result: 70.8% exact on the test split of CubiCasa5K (2,540 of 3,586 questions on plans that
the detector was not trained on and the thresholds were not chosen on). Windows are the weak
kind: 47.9%.**

| Kind | train (inflated) | valid (thresholds chosen here) | **test** | Mean abs. error, test |
|---|---|---|---|---|
| window | 65.7% of 8,046 | 60.6% of 756 | **47.9% of 772** | 1.48 |
| door | 79.0% of 5,922 | 75.1% of 562 | **70.4% of 554** | 0.55 |
| toilet | 84.6% of 8,038 | 85.2% of 756 | **82.6% of 770** | 0.24 |
| sink | 83.6% of 7,942 | 81.2% of 746 | **78.7% of 762** | 0.25 |
| bath_shower | 80.4% of 7,474 | 79.8% of 682 | **74.7% of 728** | 0.30 |
| all | 78.6% of 37,422 | 76.4% of 3,502 | **70.8% of 3,586** | |

- Counts are of questions; the English and the Finnish question of a plan have the same answer,
  so each plan is counted twice. The test column covers 386 plans.
- **The train column is inflated**: the detector was trained on those plans. It is shown only
  because the benchmark contains them; it is not an estimate of accuracy on new drawings.
- The valid column is slightly optimistic: the minimum scores were chosen on it.
- The split is the split of CubiCasa5K. Twenty images are in CubiCasa5K twice, under two plans.
  Five of these pairs join a train plan with a valid (4) or test (1) plan; those plans are
  counted as train here, because the detector saw the image.
- The labels of CubiCasa5K give the benchmark's answer on 99.6% to 99.9% of plans, so almost all
  the error is the detector's.

### Against the LLM, replayed on the 100-question baseline sample

The baseline runs were replayed with the counter's answer in place of the LLM's answer for
every question that the counter claims (18 of 100; `--replay`). No LLM call was made.

**The totals of this table are inflated and are not a result.** 13 of the 18 replaced plans
are in the detector's training split, and the benchmark sample was not drawn to avoid them.
Do not compare "After" with the baseline ([ADR 0002](../ADR/0002-plain-llm-as-baseline.md)).
In the fix-issues run the whole gain of 2 comes from train plans: on the 5 test plans the
counter fixed one answer and broke one. In the baseline run 1 of the 2 comes from a test plan.

| Run | Before | After (inflated) | LLM right on the 18 | Counter right on the 18 |
|---|---|---|---|---|
| `fix-issues-baseline-opus-5-5-100.jsonl` | 81 of 100 | 83 of 100 | 8 | 10 |
| `baseline-opus-5-5-100.jsonl` | 77 of 100 | 79 of 100 | 8 | 10 |

- In the fix-issues run the counter fixed 5 answers and broke 3; in the baseline run it fixed 4
  and broke 2.
- 13 of the 18 replaced plans are in the train split, 5 in test. On the 5 test plans the counter
  was right on 3, the LLM on 3 (fix-issues run) and 2 (baseline run).
- 18 questions are too few to call a difference of two answers a finding. The split table above
  is the measurement. On plans that the detector did not see, the replay has 5 questions:
that is no finding in either direction.
- The counter's mistakes in the replay are large plans with many windows (31 of 33, 22 of 23,
  11 of 12) and bathtubs that the high bathtub threshold drops (expected 1, counted 0, twice).
  The LLM's mistakes on the same plans are larger (12 of 33, 14 or 17 of 23). Most of
  these plans are train plans too.

### Compared with `evaluate` of the detector branch

`object_detector/modal_app.py::evaluate` (the other session, `runs/large-1024/scores.json`)
uses one minimum score for all classes (0.45, chosen on valid) and scored the test split
against the benchmark: windows 47.5%, doors 71.2%, toilets 81.3%, sinks 76.4%, showers and
bathtubs 66.0% (387 test plans; it keeps the one test plan that duplicates a train image). The
per-class minimums here give the same windows and doors and are better on sinks (+2.3 points),
toilets (+1.3) and showers and bathtubs (+8.7, by nearly ignoring bathtub detections).

## Setup

| | |
|---|---|
| Detector | RF-DETR large at 1024 px, 40 epochs, run `large-1024`, `checkpoint_best_total.pth` |
| Prediction | every 5,000 plans of CubiCasa5K (4,200 train, 400 valid, 400 test), `F1_original.png`, all detections with a score of at least 0.05 |
| Benchmark | `data/floorplans/questions.jsonl`, all 44,510 questions of the five templates on 4,787 plans |
| Scoring | the count must be exact |
| Thresholds | per class, chosen on the valid split only (`objects/tune.py`): window 0.46, door 0.53, toilet 0.48, sink 0.34, shower 0.35, bathtub 0.90 |
| LLM cost | $0 |

**Thresholds.** For each template, the minimum score that gives the most exact counts on the
valid plans; of equal values, the middle one. Showers and bathtubs are one question, so their two
minimums were chosen together (grid of 0.05). Bathtubs are rare (38 boxes on the valid plans) and
most bathtub detections there were wrong, so the chosen bathtub minimum (0.90) keeps almost none.
Exact on valid at the chosen values, and at 0.5 for all: window 60.6% (60.1%), door 75.1%
(74.4%), toilet 85.2% (80.7%), sink 81.2% (77.7%), showers and bathtubs 79.8% (66.0%).

**Where the windows go wrong.** On the valid plans the window count is too low on average
(mean error −0.08 on images under 1,500 px, −1.12 on images of 1,500 to 2,500 px). Most plans
are small, and even there only 63% of window counts are exact, so the input size is not the
main cause.

## Cost and time

| Step | GPU | Time | Cost (estimate) |
|---|---|---|---|
| 20-plan check | L4 | 15 s of prediction, about 2 min of container | < $0.05 |
| All 5,000 plans | A10G, 20 containers in parallel | 686 s of prediction plus the start of 20 containers; 2 min wall time once the containers ran | under $0.80 |
| All 5,000 plans again with `checkpoint_best_ema.pth` | A10G | 637 s of prediction plus the start of 20 containers | under $0.80 |

- L4 had no capacity at the time: a first full run waited for an L4 worker and was stopped
  before any container started. The runs above used A10G (`OBJECT_DETECTOR_PREDICT_GPU=A10G`).
- The EMA checkpoint gave byte-identical detections: `checkpoint_best_total.pth` is the EMA
  weights. That second run was not needed.
- Costs are estimated from Modal's list price for A10G ($1.10 per hour) and the container time;
  they are not read from the bill. Container start-up time was not
  measured; it is counted at up to 90 s per container.

## Commands to repeat

```sh
uv sync --group train
OBJECT_DETECTOR_PREDICT_GPU=A10G uv run --group train modal run -m object_detector.modal_predict::predict_all --run-name large-1024
uv run --group train modal volume get chatchtg-object-detector predictions/large-1024/detections.jsonl var/detections/
uv run --group train modal volume get chatchtg-object-detector predictions/large-1024/settings.json var/detections/
uv run python -m chatchtg.objects.importer var/detections/detections.jsonl data/floorplans/data/cubicasa5k
uv run python -m chatchtg.objects.tune data/floorplans/questions.jsonl var/detections/detections.jsonl
uv run python -m chatchtg.objects.report data/floorplans/questions.jsonl \
    --replay var/eval-runs/fix-issues-baseline-opus-5-5-100.jsonl var/eval-runs/baseline-opus-5-5-100.jsonl
```

The detections are on the volume under `predictions/large-1024/` (`detections.jsonl`, 32 MB,
`settings.json`, `chunks/`).

## Limits

- The detector does not run in the app. A drawing gets a counted answer only if its detections
  were stored by the batch run; a new upload is answered by the LLM.
- Only the five CubiCasa5K templates are claimed. The Swiss questions about windows, doors,
  toilets, washbasins, bathtubs and showers are worded differently and are not claimed: the
  detector was not trained on Swiss plans.
- The question classifier matches the starts of the benchmark's wording. A user who asks
  "how many windows are there?" gets the LLM.
- The model returns at most 300 detections per image. 149 plans reach that at a score of 0.05,
  but on each of them the lowest kept score is below every chosen minimum, so no counted object
  was cut off.
- No LLM answer was measured on the test split beyond the 5 test plans of the baseline sample
  (see the update below).

## Update 2026-11-15 (later): why the counts miss, and a 1536 px model

**Result: the errors come mostly from how multi-pane windows are labelled and from true
objects that get a low score, not from objects that are too small at the 1024 px input. A
model fine-tuned at 1536 px is 0.6 points better on valid and exactly as good on the test split
(70.8%), so the app keeps `large-1024`.**

### Diagnosis (valid split only, no GPU)

`object_detector/diagnose.py` matches the stored detections (score floor 0.05) to the label
boxes of `datasets/cubicasa-v1/valid` (399 plans) at the counter's minimum scores. A detection
matches a label at IoU 0.5, or (`--lenient`) when one box lies 70% inside the other: toilets and
showers are often labelled with a smaller box than the drawn symbol, and for a count that is
still one object.

Box-level recall and precision at the chosen minimums (lenient matching):

| Class | Labels | Recall | Precision |
|---|---|---|---|
| window | 3,375 | 89.2% | 90.8% |
| door | 3,652 | 92.9% | 97.1% |
| toilet | 540 | 84.6% | 91.6% |
| sink | 1,033 | 90.0% | 91.4% |
| shower | 481 | 67.2% | 80.0% |
| bathtub | 38 | 13.2% | 100% |

With strict IoU 0.5 the toilet and shower numbers fall to 53%/57% and 44%/52%: most of that gap
is the box convention, not a counting error.

**Windows** (366 labels missed, 306 false detections):

- Missed: 234 had a box at IoU ≥ 0.5 that scored below the minimum (0.46); 63 only a poorly
  placed box; 63 nothing at all.
- False: 140 lie inside a labelled window that another detection already matched (the model
  splits a window into its panes); 99 are on no label (on the overlays several of them are real
  windows that the labels leave out); 38 poorly placed; 21 duplicates of a matched window.
- Size at the 1024 input (shorter side): missed 28% under 6 px (36 labels), 14% of 6 to 10 px,
  12% of 10 to 20 px, 7.6% at 20 px or more. Small windows are missed more often, but 83% of the
  missed windows are 10 px or larger.
- The way a window of several panes is labelled varies: 16.5% of window labels touch another
  window label in line (one box per pane), on 88 of the 399 plans; on other plans the same kind
  of window is one box. On plans with per-pane labels the window count is exact on 38% (1 to 7
  windows) and 36% (8 to 15) of plans; on the other plans on 75% and 61%. A rule that merges
  touching window detections does not help on valid (233 against 231 exact of 382 plans, worse
  with a larger gap), because the convention is not the same on all plans.
- Exact window count by the true count: 1 window 14 of 15, 2 to 3: 59 of 71, 4 to 7: 104 of 166,
  8 to 15: 48 of 90, 16 or more: 17 of 56. Each window is one more chance of an error.
- Against the labels of all 399 valid plans: of the 156 plans with a wrong window count, 65 only
  miss windows, 51 only have extra detections and 40 have both; on 19 more plans misses and
  extras cancel out.

**Doors** (258 missed, 101 false): 146 missed doors scored below the minimum, 56 were not
detected, 53 only poorly placed. Doors under 20 px are missed often (80 of 161), but they are
few; 178 of the 258 missed doors are 20 px or larger. 81 of the 121 plans with a wrong door
count only miss doors.

**Cause.** Per object the detector is right about 9 times in 10 both ways (recall and precision
near 90%). Most mistakes are true objects with a middling score and split or unlabelled windows,
not small objects. The count over 10 or 20 objects then fails often. **What would refute it:**
if a model that sees more pixels per object lowered the window and door misses a lot and raised
the exact counts on test, size would be the cause. That was tested next.

Two ways to count without a hard minimum were cross-validated on valid (two halves by a hash of
the image name): rounding the sum of the scores lost on windows (193 against 224 exact of 378),
doors and sinks and won on toilets (339 against 321); a minimum relative to the plan's top
scores lost on every class. Neither was adopted. The toilet gain also holds on test (328 against
318 of 385 plans), but a count that is not the number of the shown boxes is harder to check, so
it is left as an option, not built.

### The 1536 px model

| | |
|---|---|
| Run | `runs/large-1536-ft`, own app `chatchtg-object-finetune` (`object_detector/modal_finetune.py`) |
| Start | the weights of `large-1024/checkpoint_best_total.pth` |
| Training | 10 epochs at 1536 px, batch 2 × 8 accumulation, lr 5e-5 (encoder 7.5e-5), L40S |
| Why not more epochs at 1024 | `large-1024` had flattened: validation mAP50 0.69 to 0.71 from epoch 26 to 39 |
| Validation (EMA), best epoch | mAP50 0.709, mAP50:95 0.490 (large-1024: 0.706 and 0.481) |
| Minimum scores (valid) | window 0.46, door 0.45, toilet 0.47, sink 0.34, shower 0.25, bathtub 0.95 |
| Detections | `predictions/large-1536-ft/detections.jsonl` on the volume, `var/detections-1536/`, extraction version `rfdetr-large-1024@2` |

Window misses with 1536 px: 40 of 356 under 10 px (was 50), 206 of 1,762 of 10 to 20 px (was
212), 83 of 1,220 at 20 px or more (was 93). They fell at every size, a little.

Exact counts (`objects/report`, `--version 1` and `--version 2`):

| Kind | valid 1024 | valid 1536 | **test 1024** | **test 1536** |
|---|---|---|---|---|
| window | 60.6% | 60.8% | **47.9%** | **51.3%** |
| door | 75.1% | 76.5% | **70.4%** | **70.0%** |
| toilet | 85.2% | 82.5% | **82.6%** | **81.8%** |
| sink | 81.2% | 84.5% | **78.7%** | **78.5%** |
| showers + bathtubs | 79.8% | 80.9% | **74.7%** | **72.3%** |
| all | 76.4% | 77.0% | **70.8%** | **70.8%** |

The train column rose from 78.6% to 81.1%: the extra epochs fit the training plans better. On
test, windows gain 3.4 points and the other kinds lose up to 2.4; the total is the same to the
question. **This is not a gain**, so `CHATCHTG_OBJECT_COUNTER=detections` still reads version 1.
Picking the 1536 model for windows only would be a choice made on the test split, and was not
done.

Fusing the two models' detections (boxes paired at IoU 0.5, mean score) was no better on valid
(1,347 exact against 1,348 for 1536 alone) and was not scored on test.

### The LLM on the same questions

Every object-count answer of Opus 5.5 in the existing result files (`var/eval-runs/**/*.jsonl`
of all worktrees, no new LLM call), against the counter (version 1) on the same question:

| Kind | Split | Answers | LLM right | Counter right | Both | Neither |
|---|---|---|---|---|---|---|
| bath_shower | test | 3 | 2 | 0 | 0 | 1 |
| bath_shower | train | 9 | 7 | 6 | 4 | 0 |
| door | train | 6 | 0 | 0 | 0 | 6 |
| sink | test | 3 | 0 | 3 | 0 | 0 |
| sink | train | 12 | 9 | 9 | 6 | 0 |
| toilet | test | 3 | 3 | 3 | 3 | 0 |
| toilet | train | 3 | 0 | 3 | 0 | 0 |
| window | test | 6 | 3 | 3 | 3 | 3 |
| window | train | 9 | 0 | 3 | 0 | 6 |
| all | | 54 | 24 | 30 | 16 | 16 |

The 54 answers are the same 18 questions on 18 plans, answered in three runs (baseline twice,
`areas` composer once). On the 5 test plans (15 answers) the LLM was right 8 times, the counter
9 times. **This does not settle whether the counter should answer instead of the LLM.** That
needs an Opus run on a few hundred object-count questions of test plans (about $0.03 each).

### Cost

| Step | GPU | Time | Cost (estimate) |
|---|---|---|---|
| Check: 1 epoch on 200 images at 1536 | L40S | 1.1 min training, about 3 min container | $0.10 |
| Fine-tune `large-1536-ft` | L40S | 54.1 min training, about 1 h container | $2.00 |
| Predict 5,000 plans at 1536 | A10G, 20 containers | 923 s of prediction plus container start | under $0.80 |

About 1.6 GPU-hours, under $3, at Modal's list prices ($1.95 per hour for L40S, $1.10 for A10G);
not read from the bill. No LLM cost.

### Commands to repeat

```sh
uv run python -m object_detector.diagnose <valid dir with _annotations.coco.json> var/detections/detections.jsonl --lenient \
    --overlays data/floorplans/data/cubicasa5k var/overlays
uv run --group train modal run --detach -m object_detector.modal_finetune::finetune --run-name large-1536-ft --resolution 1536 --epochs 10
OBJECT_DETECTOR_PREDICT_GPU=A10G uv run --group train modal run -m object_detector.modal_predict::predict_all --run-name large-1536-ft
uv run --group train modal volume get chatchtg-object-detector predictions/large-1536-ft/detections.jsonl var/detections-1536/
uv run --group train modal volume get chatchtg-object-detector predictions/large-1536-ft/settings.json var/detections-1536/
uv run python -m chatchtg.objects.tune data/floorplans/questions.jsonl var/detections-1536/detections.jsonl
uv run python -m chatchtg.objects.importer var/detections-1536/detections.jsonl data/floorplans/data/cubicasa5k --version 2
uv run python -m chatchtg.objects.report data/floorplans/questions.jsonl --version 2
uv run python -m chatchtg.objects.report data/floorplans/questions.jsonl --llm var/eval-runs/*.jsonl var/eval-runs/room-counting/*.jsonl
```

The valid labels are on the volume: `modal volume get chatchtg-object-detector datasets/cubicasa-v1/valid/_annotations.coco.json <dir>/`.
