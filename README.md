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

The LLM gets the result as text, without the image, and states it. It gets the image and the room table when the pipeline cannot answer. Other questions are composed as in the baseline. The first question on a drawing takes several seconds of CPU for the OCR; later questions on the same drawing reuse the analysis.

When the pipeline's result is the answer, the response carries `method` (`printed`, `measured` or `printed+measured`) and `evidence`: the rooms that were added up, with their outlines, and the total. The app draws these rooms on the drawing: blue for an area printed on the drawing, orange for an outline that code measured, violet for rooms measured as one region because no wall was found between them. Each room has a tag with its number and its area. Next to the drawing the app names the method, lists the rooms and the total, and has a control to zoom to one room and a switch that brings back the plain drawing. When the LLM answers with the image, the response has no evidence and the app says so.

To score the pipeline without any LLM call:

```sh
uv sync --dev                                   # ezdxf and shapely, for the ground truth
uv run python scripts/build_room_truth.py       # room areas from the DXF twins, into var/room-truth
uv run python scripts/eval_areas.py --plans 100 # answers the area questions of 100 plans in code
uv run python scripts/eval_labels.py 20 20      # the label reader alone
uv run python scripts/eval_membership.py        # the aggregation rules and apartment membership alone
```

The DXF files are used only to score and to explain errors, never to answer ([ADR 0003](docs/ADR/0003-floor-plans-only.md)). Results are in [docs/benchmarks](docs/benchmarks/).

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

`--sample N` answers N questions, balanced over source and question type; `--seed` changes the pick. `--max-cost` stops the run before its cost passes that many US dollars. Each result line records the token usage and the cost of its answer. If a run is interrupted, run the same command again with the same `--out` file and `--resume`: it keeps the answers and asks only the remaining questions. `--resume` needs `--out`. With `--evidence`, each result line also has `method` and `evidence` (the items that the app draws on the drawing), so a wrong answer that code made can be inspected later. It is off by default because it makes the result file much larger (about 1 to 3 KB for each area answer); a run resumed with another `--evidence` choice has both kinds of lines. A question whose drawing is missing or not readable is skipped, and the run prints how many it skipped.

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
| `CHATCHTG_COMPOSER` | `baseline` | context composer used by the app: `baseline`, `areas` or `room-conventions` |
| `CHATCHTG_ROOM_COUNTER` | `off` | `confident`: code answers a room-count question when it is as accurate as the LLM; `always`: code answers every room-count question; see [Room counting](#room-counting) |
| `CHATCHTG_LLM_PROVIDER` | `fake` | LLM adapter: `fake` or `claude` |
| `CHATCHTG_ANTHROPIC_API_KEY` | not set | this project's Anthropic key, used instead of `ANTHROPIC_API_KEY` |
| `CHATCHTG_LLM_MODEL` | adapter's default | model the adapter calls; for `claude` the default is `claude-opus-5-5` |
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
- **Extraction storage** (`extractions/`): the facts that a producer derived from a drawing (rooms, objects, their geometry and measures), one JSON file per drawing and producer version in `var/metadata/`. The format is in [docs/metadata-format.md](docs/metadata-format.md). Nothing reads or writes it yet.
- **Room counter** (`rooms/`): optional. `AnswerService` asks it first; when it is not confident, or the question is not a room count, the LLM answers. See [Room counting](#room-counting).
- **Context composer** (`context/`): a pure function `(document record, bytes, message) -> Context`. It never calls an LLM. To add a strategy, write a composer and list it in `context/registry.py`.
- **LLM** (`llm/`): `LLMClient.complete(Context) -> Completion`. To add a provider, write one adapter and add a branch to `build_llm`.
  - `fake` (`llm/fake.py`): a deterministic stand-in for tests and offline runs.
  - `claude` (`llm/claude.py`): sends the system prompt and one user message (the images and text, in order) to the Anthropic Messages API. The completion records the model that served the request and the token usage. If the model declines a request, the API retries it on a fallback model; if that also declines, the answer says so and the run continues.
- **Judge** (`evals/judge.py`): `Judge.read(question, unit, response) -> Judgement`. It is separate from `LLMClient` because it is not the system that the eval measures.
- **Evidence** (`evidence.py`): the format of evidence items (where an answer comes from on the drawing), the words for each method, and the function that draws the items on the drawing. The interface calls it.
- **Interface** (`ui/app.py`): holds no logic. It shows the answer, how it was made, and its evidence on the drawing. The eval runner calls the same `AnswerService`, not the UI.
- **Schemas** (`schemas.py`): the request and response of an interaction. Add new fields as optional.
