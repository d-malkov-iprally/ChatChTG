# ChatChTG
Junction demo of drawing conversational platform

The challenge this answers is recorded in [docs/challenge.md](docs/challenge.md).

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
uv run pytest
```

### Floor-plan benchmark

Scores are compared on the floor-plan benchmark ([ADR 0003](docs/ADR/0003-floor-plans-only.md)). Build it once from the full dataset:

```sh
scripts/download_data.sh full
uv run chatchtg benchmark data/full                    # writes data/floorplans (about 2.3 GB)
uv run chatchtg eval data/floorplans --questions questions-subset.jsonl   # the 500-question subset
uv run chatchtg eval data/floorplans                   # all 123,098 questions
```

`data/floorplans` holds the PNG floor plans of the `cubicasa5k` and `swiss_dwellings` sources that have questions, their licence files, `questions.jsonl` and `questions-subset.jsonl`. The folder has the layout of a downloaded dataset and contains its own images, so it can be copied to another machine. The subset is balanced over source and question type; the same `--seed` and `--subset-size` always give the same subset. Building again copies only the files that are missing.

Answers come from a fake LLM unless you select a provider. To use Claude:

```sh
CHATCHTG_LLM_PROVIDER=claude uv run chatchtg serve
CHATCHTG_LLM_PROVIDER=claude CHATCHTG_LLM_MODEL=claude-haiku-5-5 uv run chatchtg eval data/preview --limit 50
```

Claude needs an Anthropic key, and each request is billed to that key's account. Keep the key in this project:

```sh
cp .env.example .env      # then put the key in CHATCHTG_ANTHROPIC_API_KEY
```

Every `CHATCHTG_*` variable can be set in `.env`. The file is read from the folder you run the command in, and Git ignores it. A variable that is also set in the shell takes its value from the shell. Each Git worktree is a separate folder, so it needs its own `.env`.

If `CHATCHTG_ANTHROPIC_API_KEY` is not set, the Anthropic SDK looks for credentials itself: `ANTHROPIC_API_KEY`, or a login made with `ant auth login`.

State (uploaded files, `chatchtg.db`, eval results) is written to `var/`.

Set these in the shell or in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `CHATCHTG_DATA_ROOT` | `data` | root of the downloaded datasets |
| `CHATCHTG_STATE_DIR` | `var` | where state is written |
| `CHATCHTG_COMPOSER` | `baseline` | context composer used by the app |
| `CHATCHTG_LLM_PROVIDER` | `fake` | LLM adapter: `fake` or `claude` |
| `CHATCHTG_ANTHROPIC_API_KEY` | not set | this project's Anthropic key, used instead of `ANTHROPIC_API_KEY` |
| `CHATCHTG_LLM_MODEL` | adapter's default | model the adapter calls; for `claude` the default is `claude-opus-5-5` |

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
- **Context composer** (`context/`): a pure function `(document record, bytes, message) -> Context`. It never calls an LLM. To add a strategy, write a composer and list it in `context/registry.py`.
- **LLM** (`llm/`): `LLMClient.complete(Context) -> Completion`. To add a provider, write one adapter and add a branch to `build_llm`.
  - `fake` (`llm/fake.py`): a deterministic stand-in for tests and offline runs.
  - `claude` (`llm/claude.py`): sends the system prompt and one user message (the images and text, in order) to the Anthropic Messages API. The completion records the model that served the request and the token usage. If the model declines a request, the API retries it on a fallback model; if that also declines, the answer says so and the run continues.
- **Interface** (`ui/app.py`): holds no logic. The eval runner calls the same `AnswerService`, not the UI.
- **Schemas** (`schemas.py`): the request and response of an interaction. Add new fields as optional.
