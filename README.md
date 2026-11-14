# ChatChTG
Junction demo of drawing conversational platform

The challenge this answers is recorded in [docs/challenge.md](docs/challenge.md).

## Setup

```sh
uv sync
scripts/download_data.sh preview      # optional: the dataset, into data/preview
```

## Run

```sh
uv run chatchtg serve                                  # Streamlit app
uv run chatchtg ingest data/preview                    # register the dataset's PNG/JPEG drawings
uv run chatchtg eval data/preview --limit 50           # answer and score dataset questions
uv run pytest
```

State (uploaded files, `chatchtg.db`, eval results) is written to `var/`.

| Variable | Default | Meaning |
|---|---|---|
| `CHATCHTG_DATA_ROOT` | `data` | root of the downloaded datasets |
| `CHATCHTG_STATE_DIR` | `var` | where state is written |
| `CHATCHTG_COMPOSER` | `baseline` | context composer used by the app |
| `CHATCHTG_LLM_PROVIDER` | `fake` | LLM adapter |

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
- **LLM** (`llm/`): `LLMClient.complete(Context) -> Completion`. To add a provider, write one adapter and add a branch to `build_llm`. Only a fake adapter exists now.
- **Interface** (`ui/app.py`): holds no logic. The eval runner calls the same `AnswerService`, not the UI.
- **Schemas** (`schemas.py`): the request and response of an interaction. Add new fields as optional.
