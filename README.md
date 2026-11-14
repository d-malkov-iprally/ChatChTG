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

Answers come from a fake LLM unless you select a provider. To use Claude:

```sh
CHATCHTG_LLM_PROVIDER=claude uv run chatchtg serve
CHATCHTG_LLM_PROVIDER=claude CHATCHTG_LLM_MODEL=claude-haiku-5-5 uv run chatchtg eval data/preview --limit 50
```

Claude needs Anthropic credentials: set `ANTHROPIC_API_KEY`, or log in once with `ant auth login`. Each request is billed to that account.

State (uploaded files, `chatchtg.db`, eval results) is written to `var/`.

| Variable | Default | Meaning |
|---|---|---|
| `CHATCHTG_DATA_ROOT` | `data` | root of the downloaded datasets |
| `CHATCHTG_STATE_DIR` | `var` | where state is written |
| `CHATCHTG_COMPOSER` | `baseline` | context composer used by the app |
| `CHATCHTG_LLM_PROVIDER` | `fake` | LLM adapter: `fake` or `claude` |
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
