# object_detector

Training of a detector for the symbols that the floor-plan benchmark counts: window, door, toilet, sink, bathtub, shower and sauna bench. The detector is RF-DETR, fine-tuned on the labels of CubiCasa5K. Training runs on Modal ([ADR 0005](../docs/ADR/0005-modal-for-detector-training.md)).

## Licence of the data

CubiCasa5K (https://zenodo.org/records/2613548) is CC BY-NC-SA 4.0. A model that is trained on it is for non-commercial use only. No file of the dataset and no trained weight is in this repository.

## Set up, once

```sh
uv sync --group train                      # installs the Modal client
uv run --group train modal setup           # opens a browser to log in to Modal
```

A GPU needs a payment method on the Modal account. Without one, Modal refuses the whole app, also the commands that need no GPU. Set `OBJECT_DETECTOR_GPU=none` to run `download` and `build_dataset` then. The same variable selects the GPU of `train` (default `L40S`; for example `L4` or `A100-40GB`).

To compare the labels and the predictions with the benchmark answers, copy the benchmark questions to the volume (25 MB). Without this step, those comparisons are left out:

```sh
uv run --group train modal volume put chatchtg-object-detector data/floorplans/questions.jsonl benchmark/questions.jsonl
```

## Commands

Run them from the root of the repository.

```sh
uv run --group train modal run -m object_detector.modal_app::smoke          # a GPU works
uv run --group train modal run -m object_detector.modal_app::download       # CubiCasa5K to the volume, 5.5 GB
uv run --group train modal run -m object_detector.modal_app::build_dataset  # labels to a COCO dataset
uv run --group train modal run --detach -m object_detector.modal_app::train --run-name large-1024
uv run --group train modal run -m object_detector.modal_app::evaluate --run-name large-1024
uv run --group train modal run -m object_detector.modal_predict::predict_all --run-name large-1024
uv run --group train modal run --detach -m object_detector.modal_finetune::finetune --run-name large-1536-ft --resolution 1536 --epochs 10
uv run python -m object_detector.diagnose <dir with _annotations.coco.json> var/detections/detections.jsonl --lenient
```

- `download` does nothing if the data is on the volume.
- `build_dataset` prints the number of boxes of each class, the plans that it skipped, and how often the label counts equal the benchmark answers.
- `train` has the options `--size` (nano, small, medium, large), `--resolution` (a multiple of 32), `--epochs`, `--batch-size`, `--grad-accum-steps` and `--limit`. With `--limit 200 --epochs 1` it trains on 200 images, as a quick check of the whole pipeline.
- `train` with the name of a run that was stopped is written to continue that run from its last checkpoint. Modal can stop a GPU container at any time, and it then starts the function again by itself, up to 5 times. **This was not tested**: no run was stopped and continued. If no checkpoint is found, each new start trains from the pretrained weights and is billed again, so watch a run that restarts.
- `--detach` keeps the run alive when the terminal closes. Read its output with `uv run --group train modal app logs -f chatchtg-object-detector`. Stop it with `uv run --group train modal app stop chatchtg-object-detector`.
- `predict_all` runs a checkpoint of a run over every plan of CubiCasa5K (all 5,000, in the split of CubiCasa5K, also the plans that `build_dataset` skipped) and writes every detection with a score of at least `--min-score` (0.05) to `predictions/<run>/detections.jsonl`, with the split and the SHA-256 of each image. It runs in chunks of 250 plans on L4 GPUs; set `OBJECT_DETECTOR_PREDICT_GPU=A10G` (or `T4`, `L40S`) when L4 has no capacity. `--limit 20` is a quick check. The app counts from these detections: see "Object counting" in the main README.
- `finetune` (own app `chatchtg-object-finetune`) continues training from the best weights of `--init-run` (default `large-1024`) at another `--resolution` (a multiple of 64) into a new run; the run that it starts from is only read. It is written to resume from its own `last.ckpt` when stopped, and Modal starts it again up to 5 times. **This was not tested**: the one run was never stopped. Without a checkpoint each new start trains from the weights of `--init-run` again and is billed again. `--limit 200 --epochs 1` is a quick check: give it a `--run-name` of its own, because a full run under the same name would resume from the checkpoint of the 200 images. GPU: `OBJECT_DETECTOR_FINETUNE_GPU` (default L40S).
- `diagnose` needs no GPU: it matches the stored detections of one split to its labels and prints recall and precision per class, why labels were missed and detections were false, the misses by box size at the model input, and the exact counts by the true count. `--overlays <images> <out>` draws the worst plans (labels green, counted detections red, lower scores blue). See `docs/benchmarks/2026-11-15-object-counter.md`.
- `evaluate` selects the minimum score that gives the best counts on the validation split, then scores the test split with that value: for each class, on how many plans the count is exactly right, and the mean error of the count.

## Where the results are

All files are on the Modal volume `chatchtg-object-detector`:

| Path | Content |
|---|---|
| `raw/cubicasa5k/` | the extracted dataset |
| `datasets/cubicasa-v1/` | `train`, `valid`, `test` with `_annotations.coco.json` and the images, `stats.json`, `label_check.json` |
| `runs/<run>/` | `checkpoint_best_total.pth`, `last.ckpt`, `scores.json`, `predictions_test.json`, `samples/` |
| `predictions/<run>/` | `detections.jsonl`, `settings.json`, `chunks/` of `predict_all` |

Download the checkpoint of a run:

```sh
uv run --group train modal volume get chatchtg-object-detector runs/large-1024/checkpoint_best_total.pth var/models/
```

## First run

`large-1024` on 2026-11-15: RF-DETR Large at 1024 px, 40 epochs, 118 minutes on an L40S. Best validation mAP 0.481 (0.706 at IoU 0.5), at epoch 38. Counts on the 400 test plans, with the minimum score 0.45 that was selected on the validation plans:

| Class | Plans with the exact count | Mean error of the count |
|---|---|---|
| window | 48.5% | 1.46 |
| door | 63.2% | 0.69 |
| toilet | 81.2% | 0.25 |
| sink | 76.5% | 0.30 |
| bathtub | 94.5% | 0.06 |
| shower | 65.2% | 0.47 |
| sauna bench | 75.0% | 0.46 |

Against the benchmark answers on the same plans: windows 47.5%, doors 71.2%, toilet bowls 81.3%, sinks 76.4%, showers and bathtubs 66.0%.

Window counts are too high as often as too low. Shower counts are almost always too low. The largest errors are on plans with many objects: one plan has 61 labelled windows and 21 found.

## Do not score the detector on the benchmark as it is

The benchmark of this project (`data/floorplans/questions.jsonl`) asks about 4,787 CubiCasa5K plans, and 4,200 of the 5,000 CubiCasa5K plans are the training split of this detector. So most benchmark plans were seen in training. A count accuracy on the whole benchmark, or on `questions-subset.jsonl`, would be inflated and cannot be compared with the baseline ([ADR 0002](../docs/ADR/0002-plain-llm-as-baseline.md)). Score the detector only on the plans of the CubiCasa5K test split, as `evaluate` does. The Swiss Dwellings plans are not in the training data.

The figures of the first run above are in `scores.json` on the Modal volume of the person who ran it. They are not recorded in `docs/benchmarks`.

## The dataset

- The split is the split of CubiCasa5K: 4,200 plans to train, 400 to validate, 400 to test. The 2019 model of CubiCasa5K has the same test plans.
- The image of a plan is its `F1_original.png`, with the name `<category>-<id>.png`. That is the name and the size of the plan in the benchmark of this project.
- `classes.py` maps the fixture names of CubiCasa5K to the classes. A urinal is not a toilet. A double sink is one sink. A shower screen or a shower cabin is a part of a shower, not a second shower.
- A door box covers the opening in the wall and the drawn swing. An opening without a door leaf is not a door.
- A label file holds all floors of a building. All of them are read, because the image shows them side by side.
- A plan is skipped if more than 20% of its boxes are outside the image.

## Swiss Dwellings

`modal_swiss.py` trains a second detector, for the 1,040 Swiss Dwellings sheets of the benchmark. It is the Modal app `chatchtg-swiss-detector`, a different app from `chatchtg-object-detector`, on the same volume. The result is in [docs/benchmarks/2026-11-15-swiss-symbol-detector.md](../docs/benchmarks/2026-11-15-swiss-symbol-detector.md).

**Labels.** Each sheet has a DXF twin in `data/full/data/swiss_dwellings/`. `swiss_dxf.py` reads the layers `DOOR`, `WINDOW` and `FIXTURE` as boxes for door, window, toilet, washbasin, bathtub, shower and washing machine (the module docstring has the rules). The washing machine is not asked about; it is a class so that its drum is not taken for a washbasin. The counts of the labels equal the benchmark answers on all plans for all six templates (`label_check.json`). The DXF files are used only as training labels and to score: the detector sees only the PNG (ADR 0003). The plans that score the detector are never trained on, and the thresholds are chosen on the valid split.

**Split.** By a hash of the site id: 70% train, 15% valid, 15% test. All floors of a site are in one split. (In the benchmark, each site has one plan.)

**Tiles.** A 1:200 toilet is about 12 x 18 pixels of the 2480 x 1753 sheet: scaled to 1024, it is 5 pixels wide. So the sheets are not scaled: the drawing area (inside the frame, above the title block and its legend) is cut into tiles of 640 x 640 pixels that overlap by 192 pixels. When a sheet is scored, each detection is kept by the tile whose centre part holds the centre of its box. The legend shows one of each symbol, and it is never counted.

```sh
# 1. Labels, on a machine with the data (47 s with 8 processes):
uv run python -m object_detector.swiss_dataset labels --out var/swiss-detector/labels.json
uv run python -m object_detector.swiss_dataset draw --labels var/swiss-detector/labels.json \
    --out var/swiss-detector/drawn site1000_bldg2082_floor9385   # look at the boxes
# 2. Upload the sheets (280 MB) and the labels:
uv run --group train modal volume put chatchtg-object-detector data/floorplans/data/swiss_dwellings raw/swiss-dwellings/png
uv run --group train modal volume put chatchtg-object-detector var/swiss-detector/labels.json datasets/swiss-v1/labels.json
# 3. Tiles, training, scores:
uv run --group train modal run -m object_detector.modal_swiss::build_tiles
uv run --group train modal run --detach -m object_detector.modal_swiss::train --run-name swiss-medium-640 --epochs 15
uv run --group train modal run --detach -m object_detector.modal_swiss::train --run-name swiss-medium-1280 \
    --resolution 1280 --epochs 10 --batch-size 4 --grad-accum-steps 4 \
    --pretrained runs/swiss-medium-640/checkpoint_best_total.pth
uv run --group train modal run -m object_detector.modal_swiss::evaluate --run-name swiss-medium-1280
```

**Result** (`swiss-medium-1280`, 163 test plans, exact counts): doors 79.1%, windows 74.2%, toilets 98.2%, washbasins 99.4%, bathtubs 99.3%, showers 92.6%. At resolution 640 (the tile not scaled) the 1:200 fixtures were too small to be found; scaling each tile up 2x fixed that. The doors and windows are limited by the benchmark: its answers count the doors and windows that the DXF holds twice at the same place, which the PNG shows once (966 and 275 of the 1,040 plans have such duplicates). Against the visible windows the detector is exact on 94.5% of the test plans.

- `train` has the options of `modal_app.train` (default `--size medium --resolution 640`), and `--pretrained <path on the volume>` to start from another checkpoint. A run name must start with `swiss-`.
- `evaluate` (on an A10G; `OBJECT_DETECTOR_PREDICT_GPU` sets another type) finds the objects on all 1,040 sheets, chooses the minimum score of each class on the valid split, and scores the test split per template and per scale (1:100, 1:200). `--limit 3` takes 3 plans of each split, as a quick check.
- Read the output of a detached run with `uv run --group train modal app logs <app id>` (`modal app list` shows the id).

| Path on the volume | Content |
|---|---|
| `raw/swiss-dwellings/png/` | the 1,040 sheets |
| `datasets/swiss-v1/labels.json` | for each plan: split, scale, fitted transform, drawing area, label boxes in PNG pixels, benchmark answers |
| `datasets/swiss-v1/{train,valid,test}/` | the tiles and `_annotations.coco.json` |
| `runs/swiss-<run>/` | `checkpoint_best_total.pth`, `scores.json`, `detections_all.json`, `samples/`. The model to use is `runs/swiss-medium-1280/`. |

`detections_all.json` holds, for every sheet, its split and scale and every detection with a score of at least 0.05: class, score, box in PNG pixels (x0, y0, x1, y1). It also holds the thresholds chosen on the valid split. An answer path that uses it must use the test plans only to quote a score.

## Files

| File | Content |
|---|---|
| `classes.py` | the classes and the name mapping |
| `cubicasa_svg.py` | `model.svg` to boxes |
| `dataset.py` | the COCO dataset |
| `scoring.py` | scores of counts against the labels and against the benchmark |
| `modal_app.py` | the Modal functions |
| `modal_predict.py` | the batch prediction over all plans, in its own Modal app `chatchtg-object-predict` |
| `modal_finetune.py` | more training of a finished run at another resolution, in its own Modal app `chatchtg-object-finetune` |
| `diagnose.py` | box-level analysis of the count errors on one split |
| `swiss_dxf.py` | a Swiss Dwellings DXF to boxes in PNG pixels |
| `swiss_dataset.py` | the Swiss labels and their check, the split, the tiles, the scores |
| `modal_swiss.py` | the Modal functions of the Swiss detector |

The tests are in `tests/test_object_detector.py` and `tests/test_swiss_detector.py`. They need no Modal account and no GPU.
