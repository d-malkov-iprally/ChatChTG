---
status: proposed
date: 2026-11-15
---

# DXF labels train the Swiss detector, and it answers only on opt-in

The Swiss symbol detector was trained on boxes read from the DXF twins of the Swiss plans. The rule so far, stated in the README and in `object_detector/README.md`, was that the DXF files are used only to score and to explain errors, never to answer; no ADR recorded it. We propose a third use: a DXF file may be a training label for a model that sees only the PNG when it answers. The Swiss kinds of the object counter stay off by default (`DEFAULT_SKIP` in `objects/counter.py`) and answer only when `CHATCHTG_OBJECT_COUNTER_SKIP` is set.

Why off by default: 727 of the 1,040 Swiss plans are the detector's training plans, and its minimum scores were chosen on 150 more (valid). On a training plan its answer rests on the DXF of the very plan that is asked about. The counts of the DXF labels equal the benchmark answers on all 1,040 plans for all six templates, so the labels are the answers. Only the 163 test plans are plans that it has not seen.

This is the same use that the Finnish detector makes of the CubiCasa5K labels. Once `CHATCHTG_OBJECT_COUNTER=detections` is set, the Finnish kinds answer with no further setting; the Swiss kinds also need `CHATCHTG_OBJECT_COUNTER_SKIP=`. The two differ in one point: the Finnish labels are a public dataset made for training, with a published split; the Swiss labels were made by us from the files that also hold the reference answers.

Hard to reverse: the default is one setting, but the rule for the data is not. Weights trained on the DXF files exist, and scores made with them are compared with others.

## What accepting this option commits to

- A DXF file may be a training label.
- **When a user turns the Swiss kinds on, the counter answers on every Swiss plan that has stored detections, the 727 training plans and the 150 valid plans included.** `DetectorObjectCounter.find` does not look at the split of a plan. Accepting this option accepts those answers in the app.
- A score of a run with the Swiss kinds on is quoted on the 163 test plans only. No code enforces this: an eval row does not say whether its plan was a training plan. The tools that score by split are `objects.report --profile swiss` and `chatchtg.evals.code_paths`.
- No gain over the baseline is claimed ([ADR 0002](0002-plain-llm-as-baseline.md)). No LLM was run on held-out Swiss plans. The existing LLM answers are 14 Swiss object questions, 11 of them on training plans: too few to compare.

## Considered Options

- **Off by default, on by opt-in, scores on test plans only** (proposed): keeps the counts available at about $0 per question (on the 163 test plans: toilets 98.2%, washbasins 99.4%, bathtubs 99.3%, showers 92.6% exact). The opt-in answers on training plans too (above).
- **Never answer from it**: the strictest reading of the README rule. Against the proposed option it removes only the opt-in. It gives up code answers to the 6,598 Swiss fixture questions, whose accuracy is measured on the 1,036 of them that are on test plans (98.3% exact). No LLM accuracy on held-out Swiss plans is known for comparison.
- **On whenever the object counter is on, as the Finnish kinds are**: simplest for the demo. A run with `CHATCHTG_OBJECT_COUNTER=detections` is already inflated by the Finnish detector (trained on 4,200 of the 5,000 CubiCasa5K plans); this adds the Swiss questions, of which 9,070 of 10,758 are on train or valid plans.
- **Answer on test plans only** (train and valid excluded): the only option under which no answer in the app rests on the DXF of the plan asked about. It needs a check of the stored split in `DetectorObjectCounter.find`, which is not written; the split is stored with each extraction. It would answer 1,688 of the 10,758 Swiss object questions and send the others to the LLM by a rule that a user cannot see, as the app already does for a drawing without stored detections.

## Open: doors and windows

`CHATCHTG_OBJECT_COUNTER_SKIP=` turns on all six kinds, doors and windows included. Whether they may be turned on is a decision of its own, and this ADR must say which before it is accepted: in the opt-in, or kept out of it (`CHATCHTG_OBJECT_COUNTER_SKIP=swiss:door,swiss:window`).

- **Doors** (79.1% exact on the test plans): the reference answers count doors that the DXF holds twice at one place and the PNG shows once. The counter matches the answer on 129 of the 163 test plans, and on at least 114 of them it does so by marking more door boxes than the PNG shows doors. Its evidence then shows boxes that are not visible doors.
- **Windows** (74.2%): the step that joins the tiles keeps one of two boxes at the same place, so the 37 test plans with a window that the DXF holds twice can never get the exact count. Against the visible windows the detector is exact on 94.5% of the test plans.
- No LLM was measured on Swiss doors or windows on held-out plans.

## Consequences

- A benchmark page that reports a run with the Swiss kinds on must give the split of the plans and quote the test plans.
- If this ADR is accepted, the places that say "no ADR covers this" must name it: `README.md` (the paragraph on DXF files and "Object counting"), `object_detector/README.md`, the `DEFAULT_SKIP` docstring, and the records `2026-11-15-swiss-symbol-detector.md`, `2026-11-15-swiss-object-counter.md` and `2026-11-15-whole-benchmark.md`.
- If it is rejected, the opt-in is removed: the `swiss` profile leaves `PROFILES` of the object counter, or its kinds can no longer be turned on, and the README setting changes. `object_detector/modal_swiss.py` and the records stay as a measurement.
