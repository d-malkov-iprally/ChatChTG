---
status: accepted
date: 2026-11-14
---

# Plain LLM as baseline

We need a reference score before investing in drawing parsing, tools, agents or fine-tuning; otherwise we cannot tell whether that work helps. The baseline is a general-purpose multimodal LLM that receives the drawing and the question and returns the answer, with no preprocessing beyond what is needed to make the file readable (for example, rendering a CAD file to an image). Every later approach must beat the baseline's score on `questions.jsonl` under the dataset's scoring rules: counts exact, areas, lengths and volumes within 2%.

## Consequences

- The concrete model is not part of this decision and may change. A baseline score is only comparable to other scores when the model and the question set are recorded with it.
- Format conversion (DWG, DXF, IFC, SVG to something the model can read) belongs to the baseline. It must stay a plain rendering step: extracting quantities or structure from the file is an improvement to be measured, not baseline.
