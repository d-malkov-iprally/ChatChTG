# AGENTS.md

ChatChTG answers natural-language questions about construction drawings. Python, managed by `uv`: run every command through `uv run`.

## Workflow

1. Start each task in its own worktree on a new branch, named in short kebab-case:
   ```sh
   git worktree add ../ChatChTG-<branch> -b <branch> main
   cd ../ChatChTG-<branch> && uv sync
   ```
   The main checkout stays on `main` and may hold someone's uncommitted work: leave it as it is.
2. Make the change. When it adds a command, an option or a `CHATCHTG_*` setting, update the README in the same branch.
3. Commit on the branch. The pre-commit hook runs `uv run pytest` on the files on disk and aborts on a failure; fix the failure and commit again.
   Subject: an imperative sentence, capital first letter, no prefix, no full stop (`Add an LLM judge to the eval`). Body: what changed and why.
4. Push the branch and open a pull request against `main`. The description has three sections: `What`, `Checked` (what you ran and saw, and plainly what you did not test), `Not included`.

The task is done when the tests pass, the branch is pushed and the pull request is open. A human merges it.

## Cost

Tests and default runs use the fake LLM: no network, no key, no model weights. Keep new tests that way (`FakeLLM`, the fake judge, a stub client; fixtures in `tests/conftest.py`).

A run with `CHATCHTG_LLM_PROVIDER=claude` is billed. Make one only when the task needs a real answer or a score, and give `eval` a `--max-cost` and a `--sample`.

## Read before you act

- **Changing code under `src/`**: [ONBOARDING.md](ONBOARDING.md), section "Code". It has the rules that keep the parts apart (pure composers, one adapter for each provider, only `wiring.py` builds objects, settings only in `config.py`, heavy dependencies in an optional extra).
- **Reading or writing files, keys or state**: [ONBOARDING.md](ONBOARDING.md), section "Files and secrets".
- **Running an eval, reporting a score, or comparing approaches**: [ONBOARDING.md](ONBOARDING.md), section "Experiments and scores", then [ADR 0002](docs/ADR/0002-plain-llm-as-baseline.md) and [ADR 0003](docs/ADR/0003-floor-plans-only.md).
- **Making a choice that is hard to reverse**: [docs/ADR/TEMPLATE.md](docs/ADR/TEMPLATE.md) says when an ADR is needed and how to write one.
- **Commands, settings and architecture**: [README.md](README.md).
