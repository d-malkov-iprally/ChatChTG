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

| Run | Before | After | LLM right on the 18 | Counter right on the 18 |
|---|---|---|---|---|
| `fix-issues-baseline-opus-5-5-100.jsonl` | 81 of 100 | **83 of 100** | 8 | 10 |
| `baseline-opus-5-5-100.jsonl` | 77 of 100 | **79 of 100** | 8 | 10 |

- In the fix-issues run the counter fixed 5 answers and broke 3; in the baseline run it fixed 4
  and broke 2.
- 13 of the 18 replaced plans are in the train split, 5 in test. On the 5 test plans the counter
  was right on 3, the LLM on 3 (fix-issues run) and 2 (baseline run).
- 18 questions are too few to call a difference of two answers a finding. The split table above
  is the measurement; the replay only shows that the counter is not worse on this sample.
- The counter's mistakes in the replay are large plans with many windows (31 of 33, 22 of 23,
  11 of 12) and bathtubs that the high bathtub threshold drops (expected 1, counted 0, twice).
  The LLM's mistakes on the same plans are larger (12 of 33, 14 or 17 of 23).

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
- No LLM answer was measured on the test split; the comparison with the LLM is the 18-question
  replay only.
