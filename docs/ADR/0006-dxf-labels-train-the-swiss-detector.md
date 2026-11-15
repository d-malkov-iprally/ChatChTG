---
status: proposed
date: 2026-11-15
---

# DXF labels train the Swiss detector, and it answers only on opt-in

The Swiss symbol detector was trained on boxes read from the DXF twins of the Swiss plans. The rule so far was that the DXF files are used "only to score and to explain errors, never to answer" ([ADR 0003](0003-floor-plans-only.md), README). We propose a third use: a DXF file may be a training label for a model that sees only the PNG when it answers. The detector stays off by default (`DEFAULT_SKIP` in `objects/counter.py`), because 727 of the 1,040 benchmark plans are its training plans: on those, its answer rests on the DXF of the very plan that is asked about. A run that turns it on (`CHATCHTG_OBJECT_COUNTER_SKIP=`) must say so, and its score counts only on the 163 test plans.

This is the same use that the Finnish detector makes of the CubiCasa5K labels, which answers by default. The two differ in one point: the Finnish labels are a public dataset made for training, with a published split; the Swiss labels were made by us from the files that also hold the reference answers.

## Considered Options

- **Off by default, on by opt-in, scores on test plans only** (proposed): keeps the measured gain available (toilets, washbasins, bathtubs 98% to 99% exact on the test plans, showers 93%, against a plain model that was right on 11 of 14 such questions) and keeps every default score comparable with the baseline.
- **Never answer from it**: the strictest reading of ADR 0003. It gives up about 6,400 fixture questions on which no other method of this project is near as accurate, for $0 per question.
- **On by default, as the Finnish detector is**: simplest for the demo. A score on `questions-subset.jsonl` would then be inflated without any setting that shows it.
- **On by default for test plans only**: honest scores with no setting. The app would then answer the same question in two ways, by a rule that a user cannot see, on plans that look alike.

## Consequences

- A benchmark page that reports a run with the Swiss kinds on must give the split of the plans and quote the test plans.
- Swiss doors (79.1% exact on the test plans) and windows (74.2%) need a second look before they are turned on: their reference answers count doors and windows that the DXF holds twice at one place and the PNG shows once, and the detector has learnt to mark some of them twice. No model has been measured on them on the test plans.
- If this ADR is rejected, `object_detector/modal_swiss.py` and the `swiss` profile of the object counter stay as a measurement and must not be wired to answer.
