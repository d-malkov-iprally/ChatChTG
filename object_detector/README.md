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

The tests are in `tests/test_object_detector.py`. They need no Modal account and no GPU.
