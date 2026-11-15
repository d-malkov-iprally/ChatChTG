# ChatChTG
Junction demo of drawing conversational platform

The challenge this answers is recorded in [docs/challenge.md](docs/challenge.md).

New to the project? Read [ONBOARDING.md](ONBOARDING.md) for the rules and conventions.

## Quick start

You need [uv](https://docs.astral.sh/uv/). It installs the Python version given in `.python-version`.

```sh
uv sync                                # install the dependencies into .venv
scripts/download_data.sh preview       # optional: sample drawings, into data/preview
uv run chatchtg ingest data/preview    # optional: show those drawings in the app
uv run chatchtg serve                  # start the app at http://localhost:8501
```

You can also upload your own PNG or JPEG drawing in the app. The answers come from the fake LLM, which needs no key. To get answers from Claude:

```sh
cp .env.example .env      # put your Anthropic key in CHATCHTG_ANTHROPIC_API_KEY
uv run chatchtg serve
```

`.env.example` already sets `CHATCHTG_LLM_PROVIDER=claude`, so after the copy every command uses Claude. Each request is billed to the key's account. To use the fake LLM again, set `CHATCHTG_LLM_PROVIDER=fake`.

The sections below give the details.

## Setup

```sh
uv sync
git config core.hooksPath .githooks   # enable the pre-commit hook
scripts/download_data.sh preview      # optional: the dataset, into data/preview
```

The pre-commit hook (`.githooks/pre-commit`) runs `uv run pytest` and aborts the commit if a test fails. It also refuses commits on `main`: work on a branch and open a PR. The hook tests the files as they are on disk, including changes that are not staged.

## Run

```sh
uv run chatchtg serve                                  # Streamlit app
uv run chatchtg ingest data/preview                    # register the dataset's PNG/JPEG drawings
uv run chatchtg eval data/preview --limit 50           # answer and score dataset questions
uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --sample 100 --max-cost 4
uv run pytest
```

### Detection demo

A separate Streamlit page to try open-vocabulary detectors on a drawing: select an image, name the objects to find ("window, door"), and see the boxes and the count of each label.

```sh
uv sync --extra detect                       # torch, transformers and the image libraries, about 0.8 GB
uv run --extra detect chatchtg detect-demo   # start the demo at http://localhost:8501
```

- The image is an upload, or a PNG/JPEG file of the data folder `CHATCHTG_DATA_ROOT` (`data`) or of one of its subfolders. The page does not read files outside the data folder.
- The demo is served to this machine only (`127.0.0.1`).
- The models are Grounding DINO tiny and base, MM Grounding DINO base, OWLv2 base and large, and SAM 3. They run on this machine: on a CUDA GPU, on Apple MPS, or on the CPU.
- The SAM 3 weights are gated. To use SAM 3, request access at https://huggingface.co/facebook/sam3 with your Hugging Face account, wait for the approval, then run `uv run --extra detect hf auth login`. Without access, the demo shows these steps and the other models still work.
- CubiCasa5K is the floor-plan model of https://github.com/CubiCasa/CubiCasa5k. It is not open-vocabulary: it finds windows, doors, closets, electrical appliances, toilets, sinks, sauna benches, fire places, bathtubs and chimneys, and the rooms kitchen, living room, bed room, bath, entry, storage, garage and outdoor. The prompt selects which of these are shown. It has no class for showers.
  - It was trained on the CubiCasa5K floor plans. Most `cubicasa5k` images of the benchmark are images that it was trained on, so its results there are better than on other plans. Use `swiss_dwellings` to see how well it works on plans that it has not seen.
  - The first use downloads its weights (209 MB) from the Google Drive link of that project into `var/models`. The file is checked against a SHA-256 value.
  - Its code is in `src/chatchtg/detection/cubicasa`. Code and weights are CC BY-NC 4.0 (non-commercial), see the `LICENSE` file of that folder.
- SAM 3.1 is not included. It changes the video tracker of SAM 3, and its weights do not load with `transformers`.
- The first use of a model downloads its weights from the Hugging Face hub. That can take some minutes.
- "Minimum score" removes detections below that score. Scores of different models are not comparable: OWLv2 gives much lower scores than Grounding DINO, so its default is 0.10 and not 0.25.
- The demo does not use the document library or the LLM, and it stores nothing.

### Floor-plan benchmark

Scores are compared on the floor-plan benchmark ([ADR 0003](docs/ADR/0003-floor-plans-only.md)). Build it once from the full dataset:

```sh
scripts/download_data.sh full
uv run chatchtg benchmark data/full                    # writes data/floorplans (about 2.3 GB)
uv run chatchtg eval data/floorplans --questions questions-subset.jsonl   # the 500-question subset
uv run chatchtg eval data/floorplans                   # all 123,098 questions
```

`data/floorplans` holds the PNG floor plans of the `cubicasa5k` and `swiss_dwellings` sources that have questions, their licence files, `questions.jsonl` and `questions-subset.jsonl`. The folder has the layout of a downloaded dataset and contains its own images, so it can be copied to another machine. The subset is balanced over source and question type; the same `--seed` and `--subset-size` always give the same subset. Building again copies only the files that are missing.

### Area questions

The `areas` composer answers area questions on Swiss floor plans from a room table that it builds from the PNG, with no LLM call for the measuring:

```sh
CHATCHTG_COMPOSER=areas uv run chatchtg serve
uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --composer areas
uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --composer areas --evidence
```

- **Labels** (`areas/labels.py`): local OCR reads the scale, the room numbers, names, printed areas and apartment tags.
- **Membership** (`areas/membership.py`): a grey floor marks an outdoor room; the red outlines assign rooms to apartments.
- **Areas** (`areas/measure.py`): a printed area is used where there is one. Otherwise the room is measured: its outline is found from the drawn lines and each edge is placed at the centre of its line, to a fraction of a pixel.
- **Answer** (`areas/questions.py`): the question is parsed (English, German, French, Italian) and the areas are added up in code.

When the pipeline is confident of its result, code states the answer and no LLM is called. The text names the rooms that were added up, says for each whether its area was printed on the plan or measured from the drawn lines, adds a caution when a measured area comes from a plan at 1:200 or coarser or when a total is from a plan whose room numbering has a gap, and ends with `Answer: <value> m²`. It is in English for a question in any language. The response's model is then `area-pipeline`, and the answer costs nothing. The LLM gets the image and the room table when the pipeline cannot answer. Other questions are composed as in the baseline. The first question on a drawing takes several seconds of CPU for the OCR; later questions on the same drawing reuse the analysis.

When the pipeline's result is the answer, the response carries `method` (`printed`, `measured` or `printed+measured`) and `evidence`: the rooms that were added up, with their outlines, and the total. The app draws these rooms on the drawing: blue for an area printed on the drawing, orange for an outline that code measured, violet for rooms measured as one region because no wall was found between them. Each room has a tag with its number and its area. Next to the drawing the app names the method, lists the rooms and the total, and has a control to zoom to one room and a switch that brings back the plain drawing. When the LLM answers with the image, the response has no evidence and the app says so.

To score the pipeline without any LLM call:

```sh
uv sync --dev                                   # ezdxf and shapely, for the ground truth
uv run python scripts/build_room_truth.py       # room areas from the DXF twins, into var/room-truth
uv run python scripts/eval_areas.py --plans 100 # answers the area questions of 100 plans in code
uv run python scripts/eval_areas.py --stems docs/benchmarks/stems/heldout-200.txt --refresh   # of exactly these plans
uv run python scripts/eval_measure.py --stems docs/benchmarks/stems/tune-200.txt   # the measurer alone, with the failure pattern of each wrong room
uv run python scripts/eval_labels.py 20 20      # the label reader alone
uv run python scripts/eval_membership.py        # the aggregation rules and apartment membership alone
```

To score the answers that code states, run the eval with the fake provider and read the result file:

```sh
CHATCHTG_LLM_PROVIDER=fake uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --composer areas --out var/eval-runs/areas-fake.jsonl
uv run python scripts/eval_area_statement.py --results var/eval-runs/areas-fake.jsonl   # add --compare <another run> for the same questions in that run
```

The script counts only the rows with the model `area-pipeline`. In a run with the fake provider every other row is answered by the fake LLM and is not a result: ignore the accuracy that `chatchtg eval` prints for such a run.

`eval_measure.py` takes the label positions from the DXF truth, so every error it reports is an error of the measurer; `--compare OLD.jsonl` lists the rooms that changed against an earlier run. The lists in [docs/benchmarks/stems](docs/benchmarks/stems/) split the plans of each scale into the 20 the measurer was tuned on and the rest. `eval_areas.py` keeps each analysis in `var/area-cache` by the plan name: pass `--refresh` after a change to the pipeline. Both scripts read the plans from `CHATCHTG_DATA_ROOT`.

To measure what a cross-check of measured areas against an LLM would change, without building it (the result is in [docs/benchmarks/2026-11-15-area-cross-check.md](docs/benchmarks/2026-11-15-area-cross-check.md)):

```sh
uv run python scripts/eval_area_check.py signals    # the pipeline's value for every area question, and whether the check would be made; no LLM call
uv run python scripts/eval_area_check.py select     # one such question per plan, into <state dir>/area-check/questions.jsonl
uv run python scripts/eval_area_check.py replay --results <a baseline eval run on those questions>   # what the check would change
```

The script reads `CHATCHTG_DATA_ROOT` and `CHATCHTG_STATE_DIR` and writes under `<state dir>/area-check/` (`var/area-check/` by default). `signals` keeps the analysis of each plan there: pass `--refresh` after a change to the pipeline. The eval run between `select` and `replay` is an ordinary `chatchtg eval ... --composer baseline` with the absolute path of the question file as `--questions`; with a real provider it is billed.

The DXF files are used only to score and to explain errors, never to answer ([ADR 0003](docs/ADR/0003-floor-plans-only.md)). Results are in [docs/benchmarks](docs/benchmarks/).

### Combined composer

The `combined` composer gets the gains of `areas` and `room-conventions` together. It sends an area question (one that the parser in `areas/questions.py` reads) to the `areas` composer and every other question to `room-conventions`, which adds the conventions to a room count on a Finnish plan and builds the baseline context for anything else. It returns the context of the composer it chose unchanged, so it changes when they change. The two change disjoint sets of questions: on the 123,098 benchmark questions no question is read by both parsers.

```sh
CHATCHTG_COMPOSER=combined uv run chatchtg serve
uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --composer combined
uv run python scripts/route_questions.py data/floorplans/questions.jsonl   # where each question goes, no LLM
```

Its id names the versions of its parts, in the form `combined@1+areas@<n>+room-conventions@<n>`, and each response and eval row has `context_composer`: the part that built that context, for example `areas@4`. `scripts/replay_composer.py` scores a composer from earlier eval runs whose rows have the same context, without an LLM call; a context that carries an answer stated in code is scored by that statement, at no cost. The default stays `baseline`, the reference of [ADR 0002](docs/ADR/0002-plain-llm-as-baseline.md). Results: [docs/benchmarks/2026-11-15-combined-composer.md](docs/benchmarks/2026-11-15-combined-composer.md).

### Judge

The eval scores an answer with a rule: it takes the last number in the answer and compares it with the reference answer ([ADR 0002](docs/ADR/0002-plain-llm-as-baseline.md)). The last number is not always the answer: in "13 windows, 2 of them on floor 1" the rule reads 2. Add `--judge` to also have a small model read each answer:

```sh
uv run chatchtg eval data/floorplans --questions questions-subset.jsonl --judge
```

- The judge gets the question, the expected unit and the answer. It reports the number that the answer commits to, or that the answer gives no number. It does not see the reference answer and it does not decide if the answer is correct.
- Code compares the judge's number with the reference answer under the same rules: counts exact, measures within 2%.
- The judge reads only the answers that the rule can score. It makes one more LLM call for each of them.
- `correct` and `accuracy` stay the result of the rule. That is the score to compare between runs. The judge adds the columns `judged`, `judge ok`, `judge acc` and `disagree` (answers where rule and judge give a different result), and the fields `judge_model`, `judge_value`, `judge_unit`, `judge_quote`, `judge_correct`, `judge_error` and `judge_usage` to each result row.
- A judge call that fails sets `judge_error` and leaves `judge_correct` empty. It is not counted as a wrong answer.

With the `fake` provider the judge is fake too and makes no LLM call.

Answers come from a fake LLM unless you select a provider. To use Claude:

```sh
CHATCHTG_LLM_PROVIDER=claude uv run chatchtg serve
CHATCHTG_LLM_PROVIDER=claude CHATCHTG_LLM_MODEL=claude-haiku-5-5 uv run chatchtg eval data/preview --limit 50
```

To use an open-weights model on [Modal](https://modal.com) instead, create a Shared Endpoint and a proxy token in your Modal workspace. Both commands need the Modal CLI, and the endpoint is billed per token to that workspace:

```sh
modal endpoint create --shared --model moonshotai/Kimi-K3.1-Pro
modal workspace proxy-tokens create
```

Put the endpoint's hostname (the Modal dashboard shows it, for example `my-endpoint.us-west.modal.direct`) in `CHATCHTG_MODAL_ENDPOINT`, and the token in `CHATCHTG_MODAL_PROXY_TOKEN` as `wk-<id>.ws-<secret>`. Then:

```sh
CHATCHTG_LLM_PROVIDER=modal uv run chatchtg serve
CHATCHTG_LLM_PROVIDER=modal uv run chatchtg eval data/preview --sample 20 --max-cost 2
```

The results and the cost assume that the endpoint serves Kimi K3.1 Pro. For an endpoint with another model, set `CHATCHTG_LLM_MODEL` to that model's name and add its price to `llm/pricing.py`. `eval --judge` still uses the Claude judge, so it also needs an Anthropic key.

`--sample N` answers N questions, balanced over source and question type; `--seed` changes the pick. `--max-cost` stops the run before its cost passes that many US dollars. Each result line records the token usage and the cost of its answer. If a run is interrupted, run the same command again with the same `--out` file and `--resume`: it keeps the answers and asks only the remaining questions. `--resume` needs `--out`. With `--evidence`, each result line also has `method`, `evidence` (the items that the app draws on the drawing) and `evidence_note` (what a check of the evidence found, or null), so a wrong answer that code made can be inspected later. It is off by default because it makes the result file much larger (about 1 to 3 KB for each area answer); a run resumed with another `--evidence` choice has both kinds of lines. A question whose drawing is missing or not readable is skipped, and the run prints how many it skipped.

Claude needs an Anthropic key, and each request is billed to that key's account. Keep the key in this project:

```sh
cp .env.example .env      # then put the key in CHATCHTG_ANTHROPIC_API_KEY
```

Every `CHATCHTG_*` variable can be set in `.env`. The file is read from the folder you run the command in, and Git ignores it. A variable that is also set in the shell takes its value from the shell. Each Git worktree is a separate folder, so it needs its own `.env`.

If `CHATCHTG_ANTHROPIC_API_KEY` is not set, the Anthropic SDK looks for credentials itself: `ANTHROPIC_API_KEY`, or a login made with `ant auth login`.

State (uploaded files, `chatchtg.db`, extractions, eval results) is written to `var/`.

Set these in the shell or in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `CHATCHTG_DATA_ROOT` | `data` | root of the downloaded datasets |
| `CHATCHTG_STATE_DIR` | `var` | where state is written |
| `CHATCHTG_COMPOSER` | `baseline` | context composer used by the app: `baseline`, `areas`, `room-conventions`, `located` or `combined` (`areas` and `room-conventions` together, see [Combined composer](#combined-composer)) |
| `CHATCHTG_ROOM_COUNTER` | `off` | `confident`: code answers a room-count question when it is as accurate as the LLM; `always`: code answers every room-count question; see [Room counting](#room-counting) |
| `CHATCHTG_OBJECT_COUNTER` | `off` | `detections`: code answers a window, door, toilet, sink or shower-and-bathtub count on a Finnish plan from the detections stored for the drawing; see [Object counting](#object-counting) |
| `CHATCHTG_LLM_PROVIDER` | `fake` | LLM adapter: `fake`, `claude` or `modal` |
| `CHATCHTG_ANTHROPIC_API_KEY` | not set | this project's Anthropic key, used instead of `ANTHROPIC_API_KEY` |
| `CHATCHTG_MODAL_ENDPOINT` | not set | hostname of the Modal Shared Endpoint that the `modal` adapter calls |
| `CHATCHTG_MODAL_PROXY_TOKEN` | not set | Modal proxy token for that endpoint, as `wk-<id>.ws-<secret>` |
| `CHATCHTG_LLM_MODEL` | adapter's default | model the adapter calls; for `claude` the default is `claude-opus-5-5`. For `modal` it is the name of the model that the endpoint serves, used for the results and the price; the default is `moonshotai/Kimi-K3.1-Pro` |
| `CHATCHTG_JUDGE_MODEL` | `claude-haiku-5-5` | model that reads the answers for `eval --judge` |

## Room counting

Almost half of the benchmark questions count rooms: bedrooms, saunas, kitchens, bathrooms, WC rooms, balconies and storeys on Finnish plans, rooms with a given label and apartments on Swiss plans. Two things exist for them. The measurements are in [docs/benchmarks/2026-11-14-room-counting.md](docs/benchmarks/2026-11-14-room-counting.md).

- **The `room-conventions` composer** adds the counting conventions of Finnish plans to a room-count question, for example that the rooms of a neighbouring home on the sheet do not count. It raised the accuracy of Opus 5.5 on room questions. Every other question gets the baseline context.

  ```sh
  CHATCHTG_COMPOSER=room-conventions uv run chatchtg serve
  uv run chatchtg eval data/floorplans --composer room-conventions --questions questions-subset.jsonl
  ```

- **The room counter** (`rooms/`) answers a room-count question in code: it reads the printed labels with OCR, maps them to the question with a lexicon, and counts apartments and storeys from shapes. It costs nothing per question, but it is less accurate than Opus 5.5 on every kind of question except apartments. So `CHATCHTG_ROOM_COUNTER=confident` uses it only for apartments, when the unit tags and the unit outlines agree; `always` uses it for every room-count question and is for runs without an LLM budget. The response then carries `method` and `evidence` (the counted labels with their boxes), and the app draws the counted labels as boxes on the drawing.

  The OCR engine is Apple Vision, so the counter works on macOS only. On another system it steps aside and the LLM answers. OCR results are cached in `var/ocr-cache/`.

  ```sh
  uv run python -m chatchtg.rooms.warm data/floorplans var/ocr-cache          # read every plan once (about an hour)
  uv run python -m chatchtg.rooms.bench data/floorplans var/ocr-cache var/rooms.jsonl
  uv run python -m chatchtg.rooms.report data/floorplans var/rooms.jsonl      # accuracy per kind of question
  ```

## Object counting

About 36% of the benchmark questions count objects on Finnish (CubiCasa5K) plans: windows, doors, toilet bowls, sinks and wash basins, and showers and bathtubs together, in English and Finnish. The object counter (`objects/`) answers them in code from the detections of a symbol detector (RF-DETR, trained on the CubiCasa5K labels, see [object_detector/README.md](object_detector/README.md)). The measurements are in [docs/benchmarks/2026-11-15-object-counter.md](docs/benchmarks/2026-11-15-object-counter.md).

- The detector does not run in the app: it needs torch and a GPU. The detections are made in one batch on Modal and stored as extractions (`var/metadata/<sha256>/rfdetr-large-1024@1.json`). A drawing without stored detections, for example a new upload, is answered by the LLM as before.
- With `CHATCHTG_OBJECT_COUNTER=detections`, `AnswerService` asks the object counter first. The response carries `method` (`detector`) and `evidence`: each counted object with its box and its reference in the extraction store.
- A detection is counted when its score reaches the minimum of its class (`THRESHOLDS` in `objects/counter.py`), chosen on the valid split of CubiCasa5K.
- **An eval with this setting on is inflated.** The detector was trained on 4,200 of the 5,000 CubiCasa5K plans, and most object questions of `questions.jsonl` and `questions-subset.jsonl` are about those plans. The accuracy of `chatchtg eval` with `CHATCHTG_OBJECT_COUNTER=detections` cannot be compared with the baseline. The number to quote is the test split of `objects.report` (70.8% exact counts). An eval row does not say whether its plan was a training plan.

```sh
# Predict every CubiCasa5K plan on Modal (GPU), then download the detections
uv run --group train modal run -m object_detector.modal_predict::predict_all --run-name large-1024
uv run --group train modal volume get chatchtg-object-detector predictions/large-1024/detections.jsonl var/detections/
uv run --group train modal volume get chatchtg-object-detector predictions/large-1024/settings.json var/detections/

# Store them as extractions, under the hash of each benchmark image
uv run python -m chatchtg.objects.importer var/detections/detections.jsonl data/floorplans/data/cubicasa5k

# Choose the minimum scores on the valid split; score per split; replay eval runs
uv run python -m chatchtg.objects.tune data/floorplans/questions.jsonl var/detections/detections.jsonl
uv run python -m chatchtg.objects.report data/floorplans/questions.jsonl --replay var/eval-runs/<run>.jsonl

# Compare with every Opus answer to these questions in existing result files (no LLM call)
uv run python -m chatchtg.objects.report data/floorplans/questions.jsonl --llm var/eval-runs/*.jsonl
```

Each detector run is its own version of the producer `rfdetr-large-1024`, with its own minimum scores (`THRESHOLDS_BY_VERSION`); `PRODUCER_VERSION` in `objects/counter.py` is the one the app reads. `importer --version N` stores a run as version N and `report --version N` scores it. Version 1 is `large-1024`, the one in use; version 2 (`large-1536-ft`) was no better on the test split.

## Answers that show where they came from

With the `located` composer, a count question also asks the model for the position of each item it counted. The app then draws one green mark for each item on the drawing, so a person can see a miss or a double count without counting again. It is an experiment and not a default. Measured with Kimi K3.1 Pro on 42 count questions against the baseline of the same model: the same number of object counts correct (11 of 23), 4 more room counts correct (18 against 14 of 19), 1.36 times the cost per question, and 74 of 81 reviewed marks on an item of the asked kind ([record](docs/benchmarks/2026-11-15-located-answers.md)).

- **Which questions:** those that start with a counting phrase in English, Finnish, German, French or Italian (`How many`, `Kuinka monta`, `Wie viele`, `Combien`, `Quanti`/`Quante`). Every other question gets the baseline context.
- **What the response carries:** `evidence` items `{"text": "<label> <n>", "kind": "llm", "point": [x, y]}` in pixels of the stored image, and the method `llm-located`. These are marks that the LLM claimed. Code did not check them against the drawing, and the app says so.
- **What code checks:** it drops a mark outside the image, and it compares the count that the answer states with the number of marks. A difference, a dropped mark or two marks that are very close together are reported in `evidence_note`, which the app shows as a warning. The answer text and the marks are both kept as the model gave them.
- **Which provider:** only the `modal` adapter asks for positions. It requests a JSON object (`response_format` with a JSON schema), shows only its `answer` field as the text, and takes the positions as pixels of the file as it was sent. That convention was found by trial for Kimi K3.1 Pro: it is not documented. Use `located` with the `modal` provider only. The `claude` adapter ignores the request, so it returns no marks, but the system prompt of `located` still asks the model to list positions: its plain-text answer can then contain coordinates, which a person would have to read past and which the eval rule can take for the answer. That combination was never run.

```sh
CHATCHTG_LLM_PROVIDER=modal CHATCHTG_COMPOSER=located uv run chatchtg serve
CHATCHTG_LLM_PROVIDER=modal uv run chatchtg eval data/floorplans --composer located --evidence \
  --questions questions-subset.jsonl --sample 100 --max-cost 4 --out var/eval-runs/located.jsonl
```

To look at the marks of an eval run, draw them on the drawings. The script takes the rows that asked for positions: every N-th of them (`--every`), at most `--limit`, and the rows whose file name is given with `--paths`:

```sh
uv run python scripts/render_evidence.py var/eval-runs/located.jsonl data/floorplans var/evidence-review --every 3 --limit 20
```

## Architecture

One interaction is `(document, message) -> response`, single turn.

```
ui/app.py (Streamlit)   evals/runner.py (CLI)
            \               /
          answering.AnswerService
          /         |          \
documents.library  context.*   llm.client
  /          \     (composer)  (LLMClient)
blobs.py   metadata.py
```

- **Document storage** (`documents/blobs.py`): uploads are copied to `var/blobs/`; dataset files are referenced in place.
- **Metadata storage** (`documents/metadata.py`): one SQLite record per document, with free-form JSON attributes.
- **Extraction storage** (`extractions/`): the facts that a producer derived from a drawing (rooms, objects, their geometry and measures), one JSON file per drawing and producer version in `var/metadata/`. The format is in [docs/metadata-format.md](docs/metadata-format.md). The object counter reads the detections from it.
- **Object counter** (`objects/`): optional. `AnswerService` asks it first; when the question is not an object count on a Finnish plan, or the drawing has no stored detections, the room counter and the LLM answer. See [Object counting](#object-counting).
- **Room counter** (`rooms/`): optional. `AnswerService` asks it after the object counter; when it is not confident, or the question is not a room count, the LLM answers. See [Room counting](#room-counting).
- **Context composer** (`context/`): a pure function `(document record, bytes, message) -> Context`. It never calls an LLM. To add a strategy, write a composer and list it in `context/registry.py`. A context can carry an answer that code computed and stated (`Context.computed`); `AnswerService` then uses that text and does not call the LLM. The `areas` composer does this, see [Area questions](#area-questions).
- **LLM** (`llm/`): `LLMClient.complete(Context) -> Completion`. To add a provider, write one adapter and add a branch to `build_llm`. A context can ask for the position of each counted item (`Context.locations`); an adapter that supports it returns them in `Completion.locations`, in pixels of the image that was sent, and any other adapter ignores the request.
  - `fake` (`llm/fake.py`): a deterministic stand-in for tests and offline runs.
  - `claude` (`llm/claude.py`): sends the system prompt and one user message (the images and text, in order) to the Anthropic Messages API. The completion records the model that served the request and the token usage. If the model declines a request, the API retries it on a fallback model; if that also declines, the answer says so and the run continues.
  - `modal` (`llm/modal.py`): sends the system prompt and one user message to a Modal Shared Endpoint through its OpenAI-compatible Chat Completions API. The completion records the model that the endpoint is set to serve and the token usage; reasoning tokens count as output tokens. If the model spends its whole output limit on reasoning, the answer says so and the run continues. A failed request is not sent again. For a location request it asks for a JSON object with the answer and the items, and reads both from it; a reply that cannot be read gives an answer that says so.
- **Judge** (`evals/judge.py`): `Judge.read(question, unit, response) -> Judgement`. It is separate from `LLMClient` because it is not the system that the eval measures.
- **Evidence** (`evidence.py`): the format of evidence items (where an answer comes from on the drawing), the words for each method, and the function that draws the items on the drawing. The interface calls it. `evidence_check.py` turns the positions of an LLM into evidence items and checks them against the stated count.
- **Interface** (`ui/app.py`): holds no logic. It shows the answer, how it was made, and its evidence on the drawing. The eval runner calls the same `AnswerService`, not the UI.
- **Schemas** (`schemas.py`): the request and response of an interaction, with `method`, `evidence` and `evidence_note` for where the answer comes from. Add new fields as optional.
- **Detector training** (`object_detector/`, outside the application): fine-tunes a floor-plan symbol detector on Modal. The app does not use it. Its commands need the `train` dependency group (`uv sync --group train`) and a Modal account, and a training run is billed. See [object_detector/README.md](object_detector/README.md) and [ADR 0005](docs/ADR/0005-modal-for-detector-training.md).
