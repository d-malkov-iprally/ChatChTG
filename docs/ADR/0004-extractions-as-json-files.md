---
status: accepted
date: 2026-11-14
---

# Extractions as JSON files

The area pipeline, the object detectors and room counting each derive facts from a drawing, and each stored them in its own shape under its own key. Three readers need these facts: the context composer, the interface that shows where an answer came from, and the evals that compare producer versions. We store them in one format, as one JSON file per drawing and producer version: `var/metadata/<document sha256>/<producer>@<version>.json`. The format is in [docs/metadata-format.md](../metadata-format.md).

## Considered Options

- **JSON file per drawing and producer version**: chosen. Versions are side by side without more work. A write is a rename of a file, so batch jobs and Git worktrees can write at the same time. A training job can write the files without this package. The OCR cache already uses this convention.
- **Tables in `chatchtg.db`**: rejected. Queries over all drawings are easy, but the database has no migrations, and each worktree has its own database, so results made in another worktree need an export and an import.
- **One JSONL file per producer run, with an index**: rejected. It fits batch output, but to find the facts of one drawing needs an index, and the index can disagree with the files.
- **In `DocumentRecord.attributes`**: rejected. Room outlines are 20 to 75 KB per plan. One JSON value per document cannot hold two versions, and each change rewrites it under a write lock.

## Consequences

- A query over all drawings ("which plans have a sauna") reads one file per drawing. That is acceptable for the 5,827 drawings of the benchmark. If it becomes slow, build an index from the files; the files stay the source.
- The key is the SHA-256 of the drawing's bytes. A producer that names drawings by file name must hash the file to store its result.
- Coordinates are pixels of the stored image. A producer that works on a resized image must convert its coordinates before it stores them.
