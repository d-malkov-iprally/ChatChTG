# A symbol detector for the Swiss Dwellings plans

Run on 2026-11-15. RF-DETR medium, fine-tuned on Modal, counts the doors, windows, toilets,
washbasins, bathtubs and showers that are drawn on the 1,040 Swiss Dwellings sheets of the
floor-plan benchmark. The detector sees only the PNG.

**Result, on the 163 test plans (never trained on, not used to choose a threshold):**

| Template | Plans | Exact | Rate | Mean abs. error | 1:100 exact | 1:200 exact |
|---|---|---|---|---|---|---|
| doors | 163 | 129 | 79.1% | 0.45 | 114 of 139 (82.0%) | 15 of 24 (62.5%) |
| windows | 163 | 121 | 74.2% | 0.68 | 108 of 139 (77.7%) | 13 of 24 (54.2%) |
| toilets | 163 | 160 | 98.2% | 0.02 | 138 of 139 (99.3%) | 22 of 24 (91.7%) |
| washbasins | 163 | 162 | 99.4% | 0.01 | 138 of 139 (99.3%) | 24 of 24 (100%) |
| bathtubs | 138 | 137 | 99.3% | 0.04 | 116 of 116 (100%) | 21 of 22 (95.5%) |
| showers | 54 | 50 | 92.6% | 0.07 | 38 of 38 (100%) | 12 of 16 (75.0%) |

Run `swiss-medium-1280`. A plan is scored for a template if the benchmark asks that question
about it, as in the benchmark (bathtubs and showers are asked only when there is at least one).

The fixtures are counted almost always right. The doors and windows are not, mostly because the
benchmark answer counts objects that are not visible (see "Doors and windows" below).

## The DXF files are labels, not answers

Each Swiss sheet has a DXF twin. Here it is used for two things only: to make the training
labels, and to score. The detector never reads it, and nothing at answer time does (ADR 0003).
This is the same use that the Finnish detector makes of the SVG labels of CubiCasa5K. The split
keeps it honest: the plans that are scored are never trained on, and the minimum score of each
class is chosen on the valid plans, not on the test plans.

## Labels

`object_detector/swiss_dxf.py` reads the layers `DOOR`, `WINDOW` and `FIXTURE`:

- door: one open 5-point polyline on `DOOR` (the opening in the wall). The box covers the
  opening and the swing arcs whose hinge is a corner of it. A double door has one opening and
  two arcs, and is one door: counting the arcs agrees with the answers on only 597 of 1,040 plans.
- window: one closed 4-point polyline on `WINDOW`. The two lines in it are the glass.
- fixtures: each is a closed 4-point outline on `FIXTURE` with marks inside. Toilet: a 48-point
  bowl without a drain circle. Washbasin: a 48-point bowl with a drain circle. Bathtub: a drain
  circle of radius 0.025 m that is in no bowl (3 narrow bathtubs have no 69-point inner line).
  Shower: a 5-point tray mark. The box is the smallest outline that holds the mark.
- washing machine: an outline whose only mark is a circle of radius 0.15 to 0.3 m. It is not
  asked about. It is a class so that its drum is not taken for a washbasin.

The PNG offset is fitted by a cross-correlation of the wall polygons with the ink, at the
nominal scale (150 dpi). It agrees with the sub-pixel fit of `chatchtg.evals.room_truth`
within 2 pixels on every plan. Centring the drawing from the DXF extents is wrong by up to 25
pixels on about 5% of the plans, so it is not used.

**Agreement of the label counts with the benchmark answers (all plans):**

| Template | Plans with the question | Label count = answer |
|---|---|---|
| doors | 1,040 | 1,040 |
| windows | 1,040 | 1,040 |
| toilets | 1,040 | 1,040 |
| washbasins | 1,040 | 1,040 |
| bathtubs | 897 | 897 |
| showers | 322 | 322 |

None of the 143 plans without a bathtub question has a bathtub label, and none of the 718 plans
without a shower question has a shower label. The boxes were drawn on sheets at 1:100 and
1:200 and checked by eye.

## Doors and windows: the answers count stacked duplicates

On many plans, the DXF holds the same door or window twice, at the same place: same opening,
same arc. The answer counts both, and the PNG shows one symbol.

| | Doors | Windows |
|---|---|---|
| Plans with a stacked duplicate (all 1,040) | 966 | 275 |
| Stacked duplicates in the answers (all plans) | 4,903 | 787 |
| Test plans whose answer differs from the visible count | 148 of 163 | 37 of 163 |

Duplicates are counted by comparing the openings and window outlines of the DXF (overlap of at
least half of the smaller box).

- **Doors.** The detector has learnt to give two boxes where the training labels had two stacked
  doors, so it often matches the answer (129 of 163 test plans). Every test plan with a wrong
  door count has stacked duplicates.
- **Windows.** The detector gives one box for a stacked pair. Against the visible windows (the
  answer minus the stacked duplicates) it is exact on 154 of 163 test plans (94.5%), and on 140
  of 150 valid plans. Against the answer it is exact on 121 of 163. Of the 42 test plans with a
  wrong window count, 36 have stacked duplicates.

No image-only method can see a stacked duplicate. It can only guess them from a pattern. The
LLM baseline has the same limit.

## Compared with the plain LLM

The only Swiss object-count answers of Opus 5.5 are in `var/eval-runs/baseline-opus-5-5-100.jsonl`:
14 questions on 14 plans (the same questions are in the two `fix-issues-*` runs, with the same
right and wrong answers). 11 of these 14 plans are in the train split, 1 in valid,
2 in test.

| Plan | Split | Scale | Template | Answer | Opus 5.5 | Detector |
|---|---|---|---|---|---|---|
| site11137_bldg16371_floor42543 | train | 1:100 | bathtub | 2 | 2 | 2 |
| site1214_bldg2385_floor11077 | train | 1:100 | window | 13 | 13 | 13 |
| site124_bldg2_floor319 | train | 1:100 | washbasin | 0 | 0 | 0 |
| site1257_bldg2450_floor11435 | train | 1:100 | toilet | 2 | 2 | 2 |
| site133_bldg155_floor681 | valid | 1:100 | window | 12 | 12 | 12 |
| site144_bldg99_floor441 | train | 1:200 | washbasin | 7 | 4 | 7 |
| site1601_bldg2794_floor12912 | train | 1:100 | toilet | 2 | 0 | 2 |
| site3525_bldg5407_floor23753 | test | 1:100 | bathtub | 1 | 1 | 1 |
| site359_bldg776_floor3399 | train | 1:100 | washbasin | 6 | 6 | 6 |
| site3814_bldg5849_floor24689 | test | 1:100 | bathtub | 1 | 1 | 1 |
| site461_bldg1116_floor4889 | train | 1:100 | bathtub | 2 | 2 | 2 |
| site534_bldg1225_floor5451 | train | 1:100 | bathtub | 6 | 6 | 6 |
| site914_bldg1947_floor8548 | train | 1:200 | bathtub | 4 | 3 | 5 |
| site92_bldg70_floor377 | train | 1:100 | washbasin | 2 | 2 | 2 |

Opus 5.5 gets 11 of 14 right, the detector 13 of 14. Most of these plans are training plans of
the detector, so this table says little about the detector. On the 3 held-out plans, both get
3 of 3. The sample is too small to compare the two. The fair comparison is the test table
above against a run of the LLM on the same test plans, which was not made (it is billed).

The detector needs about 1.1 seconds of an A10G for one sheet (1,131 s for 1,040 sheets),
about $0.0003 at the list price. Opus 5.5 cost $0.027 per Swiss count question in the baseline.

## Setup

| | |
|---|---|
| Model | RF-DETR medium (`rfdetr[train]==1.12.1`), COCO weights, 7 classes |
| Data | 1,040 Swiss sheets, 2480 x 1753 px, 150 dpi; 821 landscape, 219 portrait |
| Split | by a hash of the site id: 727 train, 150 valid, 163 test plans. Each site has one plan in the benchmark, so the split by site is a split by plan. |
| Input | the drawing area of the sheet (inside the frame, above the title block and its legend, found from the PNG), cut into 640 x 640 px tiles at full resolution with 192 px overlap; tiles with no ink are left out. 9,685 train and 1,978 valid tiles. |
| Joining tiles | each detection is kept by the tile whose centre part holds the centre of its box; then non-maximum suppression per class at IoU 0.5 |
| Thresholds | the minimum score of each class that gives the most exact counts on the 150 valid plans (grid 0.10 to 0.95, step 0.05) |

### Runs

1. `swiss-medium-640`: 15 epochs at resolution 640 (the tile is not scaled), batch 8 x 2, L40S,
   43.4 minutes. Test: doors 55.2%, windows 57.1%, toilets 65.6%, washbasins 66.9%, bathtubs
   87.7%, showers 66.7% exact. The diagnosis on the valid split: at 1:200, only 43% of the
   toilets and 48% of the washbasins were found (median missed size 18 px); a second box on an
   object already found was rare (13 of 284 false positives); the validation F1 moved between
   0.75 and 0.84 over the last 10 epochs. The symbols were too small for the model at 640, so
   the next run scales the tiles up.
2. `swiss-medium-1280`: the same tiles at resolution 1280 (each tile scaled up 2x), started
   from the best checkpoint of run 1, 10 epochs, batch 4 x 4, L40S, 93.2 minutes. The validation
   F1 was 0.965 after the first epoch and 0.966 to 0.969 after it, so more epochs would not help.
   This is the result above.

Thresholds of `swiss-medium-1280`, chosen on valid: door 0.55, window 0.40, toilet 0.50,
washbasin 0.70, bathtub 0.30, shower 0.80.

Valid split of `swiss-medium-1280`, against the answers: doors 108 of 150 (72.0%), windows 99
(66.0%), toilets 145 (96.7%), washbasins 145 (96.7%), bathtubs 123 of 125 (98.4%), showers 45
of 47 (95.7%).

### GPU time and cost

| Step | GPU | Time |
|---|---|---|
| quick check (`swiss-quick`, 200 tiles, 1 epoch, and evaluate on 9 sheets) | L40S, L4 | about 5 min |
| training `swiss-medium-640` | L40S | 43.4 min |
| evaluate `swiss-medium-640` (1,040 sheets) | A10G | about 7 min |
| training `swiss-medium-1280` | L40S | 93.2 min |
| evaluate `swiss-medium-1280` (1,040 sheets) | A10G | about 21 min |

About 2.8 GPU-hours, about $5.50 at Modal's list prices (L40S $1.95, A10G $1.10, L4 $0.80
per hour) plus CPU and memory. This is an estimate from the measured times, not read from the
bill. An evaluate that waited 20 minutes for an L4 never started a container.

### To repeat

```sh
uv sync --group train
uv run python -m object_detector.swiss_dataset labels --out var/swiss-detector/labels.json
uv run --group train modal volume put chatchtg-object-detector data/floorplans/data/swiss_dwellings raw/swiss-dwellings/png
uv run --group train modal volume put chatchtg-object-detector var/swiss-detector/labels.json datasets/swiss-v1/labels.json
uv run --group train modal run -m object_detector.modal_swiss::build_tiles
uv run --group train modal run --detach -m object_detector.modal_swiss::train --run-name swiss-medium-640 --epochs 15
uv run --group train modal run --detach -m object_detector.modal_swiss::train --run-name swiss-medium-1280 \
    --resolution 1280 --epochs 10 --batch-size 4 --grad-accum-steps 4 \
    --pretrained runs/swiss-medium-640/checkpoint_best_total.pth
uv run --group train modal run -m object_detector.modal_swiss::evaluate --run-name swiss-medium-1280
```

### Where the results are

On the Modal volume `chatchtg-object-detector`:

- `runs/swiss-medium-1280/checkpoint_best_total.pth`: the model.
- `runs/swiss-medium-1280/detections_all.json`: every detection with a score of at least 0.05
  on all 1,040 sheets, with the split and the scale of each plan, and the thresholds. Boxes are
  x0, y0, x1, y1 in PNG pixels.
- `runs/swiss-medium-1280/scores.json`: the thresholds and the valid and test scores.
- `runs/swiss-medium-1280/samples/`: 6 test sheets with the kept detections drawn.
- `runs/swiss-medium-640/`: the same for the first run.
- `datasets/swiss-v1/`: `labels.json` (with the benchmark answers of each plan), the tiles, `stats.json`.

## Limits

- The sheets are synthetic, rendered in one constant style. The counting rules, the drawing
  area, the 150 dpi scale and the tile size are fitted to this renderer. A real drawing in
  another style needs new labels.
- The training plans include 11 of the 14 plans with an Opus answer, so those cannot compare the
  two methods.
- The scale (1:100 or 1:200) in the tables comes from the DXF title; the detector does not use it.
- The detector does not answer a question. It is not wired into `AnswerService`.
