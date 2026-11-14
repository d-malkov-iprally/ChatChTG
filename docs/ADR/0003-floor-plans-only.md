---
status: proposed
date: 2026-11-14
---

# Floor plans only

The dataset mixes floor plans with IFC models, schedules and archive scans, and we do not have the time to answer questions well on all of them. We work on raster floor plans only. Scores are compared on the floor-plan benchmark built by `chatchtg benchmark` (`data/floorplans`), not on the whole `questions.jsonl` that ADR 0002 names. The rest of ADR 0002 still applies.

The dataset has no drawing-type field, so a floor plan is defined by its source folder: `cubicasa5k` and `swiss_dwellings`. Swiss questions point at a PDF; the benchmark uses the PNG of the same drawing.

## Considered Options

- **Floor plans rendered from `ifc_models`** (69 sheets, 2,694 questions): left out. They exist only as PDF and SVG, and rendering them needs a new dependency for 2% more questions.
- **Raster scans without questions** (`finna`, `habs`, `europeana`, `wikimedia`): left out. Without questions they cannot be scored.

## Consequences

- A score is comparable only when it names the question file (`questions.jsonl` or `questions-subset.jsonl`) and, for the subset, the seed and size.
- The benchmark has two question types: count (111,082) and area (12,016). Length, volume and text answers are not measured.
- All CubiCasa questions are counts. Area questions come only from Swiss Dwellings.
