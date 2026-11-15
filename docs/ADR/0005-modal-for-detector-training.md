---
status: accepted
date: 2026-11-14
---

# Modal for detector training

The floor-plan symbol detector in `object_detector/` is trained on Modal. Training needs a GPU for a few hours and nothing in between, and Modal bills by the second, builds the environment from a few lines of Python and keeps the data on a volume. [ADR 0001](0001-verda-as-cloud-provider.md) stays as it is: Verda hosts the project's own infrastructure, and that ADR leaves model work free to run elsewhere.

## Consequences

- The dataset, the checkpoints and the scores are on a Modal volume of one account, not in the repository. Another person must run the pipeline again or get the checkpoint file.
- The training code is tied to Modal's API in one file, `object_detector/modal_app.py`. The label conversion and the scoring have no Modal code and run anywhere.
