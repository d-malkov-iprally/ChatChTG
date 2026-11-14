# Onboarding

The rules and conventions of this project on one page. The [README](README.md) has the commands and the architecture in detail; this page tells you how we work.

ChatChTG answers natural-language questions about construction drawings. It is our entry for the Stoa challenge at Junction 2026.

## Read first

1. [docs/challenge.md](docs/challenge.md): the challenge and how it is judged.
2. [README.md](README.md): quick start, commands, settings, architecture.
3. [docs/ADR/](docs/ADR/): the decisions that are hard to reverse. There are three and they are short.
4. [docs/benchmarks/](docs/benchmarks/): the baseline that every experiment is compared with.

## Set up

```sh
uv sync                                # dependencies, into .venv
git config core.hooksPath .githooks    # enable the pre-commit hook
cp .env.example .env                   # then put your Anthropic key in CHATCHTG_ANTHROPIC_API_KEY
scripts/download_data.sh preview       # sample drawings, into data/preview
uv run pytest                          # must pass before you change anything
```

- [uv](https://docs.astral.sh/uv/) manages Python (the version in `.python-version`) and the dependencies. Run every command through `uv run`.
- Without a key, set `CHATCHTG_LLM_PROVIDER=fake`. The fake LLM is enough for the app and the tests.
- Each request to Claude is billed to the key's account.

## Git and pull requests

- **No commits on `main`.** Work on a branch and open a pull request. The pre-commit hook refuses commits on `main`.
- **One branch for one piece of work.** Give it a short name in kebab-case: `llm-judge`, `floorplan-benchmark`.
- **One worktree for each branch**, next to the main checkout:
  ```sh
  git worktree add ../ChatChTG-<branch> -b <branch> main
  ```
  A worktree is a separate folder. It needs its own `uv sync` and its own `.env`.
- **The tests must pass before each commit.** The hook runs `uv run pytest` and aborts the commit if a test fails. It tests the files on disk, including changes that are not staged.
- **Commit messages.** The subject is an imperative sentence with a capital first letter, no prefix and no full stop: `Add an LLM judge to the eval`. If the change needs explanation, the body says what changed and why.
- **Open the pull request when the work is finished** and the tests pass. Do not open one for unfinished work.
- **Pull request description.** Three sections:
  - `What`: what the change does and how to try it.
  - `Checked`: what you ran and what you saw. Say plainly what you did not test.
  - `Not included`: what you left out, and why.
- Pull requests are merged into `main` with a merge commit.

## Code

The architecture section of the README describes the parts. These rules keep them apart:

- **One interaction is `(document, message) -> response`**, single turn. `AnswerService` is the only way to answer. The app and the eval runner both call it.
- **The UI holds no logic** (`ui/`). Put logic where the eval runner can also reach it.
- **A context composer is a pure function** `(document record, bytes, message) -> Context`. It never calls an LLM. To add a strategy, write a composer and list it in `context/registry.py`.
- **One adapter for each LLM provider** (`llm/`). To add a provider, write the adapter and add a branch to `build_llm`. Do not assume that the model runs on the same host as the app (ADR 0001).
- **Only `wiring.py` builds concrete objects.** The entry points (UI, CLI) call it; other modules receive what they need.
- **Settings are `CHATCHTG_*` variables**, read in one place: `Settings.from_env` in `config.py`. When you add one, add it there, to the table in the README and, if a developer must set it, to `.env.example`.
- **Add new schema fields as optional** (`schemas.py`).
- **The judge is not an `LLMClient`.** It is not the system that the eval measures.
- **Heavy dependencies go into an optional extra**, as `detect` does for torch and transformers. The main app and the tests must work with `uv sync` alone.
- **Copied third-party code keeps its licence file** in its folder. `detection/cubicasa` is CC BY-NC 4.0 (non-commercial).

## Tests

- Tests live in `tests/`, one file for each area. Run them with `uv run pytest`.
- Tests use no network, no key and no model weights. Use `FakeLLM`, the fake judge, or a stub of the client.
- Each test runs in an empty temporary folder, so your `.env` cannot change the result (`tests/conftest.py`). Use the `library` and `service` fixtures.

## Files and secrets

- Keys live in `.env`. Git ignores it. Never commit a key.
- Datasets live in `data/`, state and results in `var/` (uploads, `chatchtg.db`, eval runs, model weights). Git ignores both. Do not write state anywhere else.
- Code that reads files for a user must stay inside the data folder. The eval runner and the detection demo refuse paths that leave it.

## Experiments and scores

The judges weigh four things equally: accuracy, how well a human can verify the answer, creativity, and low inference cost. An approach that is more accurate but costs much more is not clearly better, so report both.

- **We work on raster floor plans only** (ADR 0003). Scores are compared on the floor-plan benchmark in `data/floorplans`, built by `chatchtg benchmark`.
- **Every approach must beat the baseline** (ADR 0002): a plain multimodal LLM that gets the drawing and the question. The current baseline is 77 of 100 with Opus 5.5 ([report](docs/benchmarks/2026-11-14-baseline-opus-5-5.md)).
- **Scoring:** a count must be exact; an area must be within 2%.
- **A score is comparable only with its setup.** Record the model, the question file, and for a sample the seed and the size.
- **The rule score is the score to compare** (`correct`, `accuracy`). The `--judge` columns are additional information.
- **Limit the cost of a run** with `--max-cost`, and start with `--sample 100` on `questions-subset.jsonl`. Use `--resume` if a run is interrupted.
- **Do not read much into small differences.** With about 33 questions in a group, a difference of a few points between two runs is not a finding.
- **Record a result that others will compare with** in `docs/benchmarks/YYYY-MM-DD-<slug>.md`: the result, the setup, the command to repeat it, and the cost.

## Decisions

Write an ADR only when the decision is hard to reverse, surprising without context, and a real trade-off. Copy [docs/ADR/TEMPLATE.md](docs/ADR/TEMPLATE.md) to `docs/ADR/NNNN-short-slug.md`. Never rewrite an accepted ADR to change the decision: write a new one and mark the old one as superseded.

## Documentation

When a change adds a command, an option or a setting, update the README in the same pull request.
