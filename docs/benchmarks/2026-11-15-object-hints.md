# Object counts on test plans: plain LLM, detector count, LLM with detector hints

Status on 2026-11-15: **incomplete**. The sample, the counter's score on it and the
`object-hints` composer are done. The two billed runs (plain LLM and `object-hints`) were not
made: the first request of the 5-question trial was refused by the API with "Your credit
balance is too low". No answer was billed. There is no result yet for the question this
record is about.

## Question

On plans that the detector never saw, are the CubiCasa5K object-count questions answered best
by (a) the plain LLM (`baseline`), (b) the detector's count in code (`CHATCHTG_OBJECT_COUNTER=detections`),
or (c) the LLM given the detector's boxes as a hint (`object-hints`)?

## Sample

`scripts/sample_object_questions.py`, seed 0: the English questions of the five templates
(windows, doors, toilet bowls, sinks and wash basins, showers and bathtubs) on plans of
CubiCasa5K's test split as `objects.report` assigns it (the test plan whose image is also a
train plan counts as train and is left out). The test plans are shuffled with the seed; each
template in turn takes the next unused plan that has its question, until each has 30.
Result: 150 questions on 150 different plans, ordered round-robin over the templates, so that a
run stopped by its cost limit still covers them evenly. The file is
`var/object-hints/questions-test-150.jsonl` (not committed; the command below rebuilds it from
the stored detections and `questions.jsonl`). Image sizes: median long edge 850 px, largest
2,338 px; none is reduced by the API.

## Results so far

| Template | (a) plain LLM | (b) counter | (c) object-hints |
|---|---|---|---|
| windows | not run | 14 of 30 | not run |
| doors | not run | 20 of 30 | not run |
| toilet bowls | not run | 25 of 30 | not run |
| sinks and wash basins | not run | 24 of 30 | not run |
| showers and bathtubs | not run | 23 of 30 | not run |
| all | not run | **106 of 150 (70.7%)** | not run |

The counter run made no LLM call: all 150 rows have the model `object-counter` and a cost of
$0.00. Its 70.7% agrees with its 70.8% on all 3,586 test-split questions
([object counter](2026-11-15-object-counter.md)).

## The `object-hints` composer (version 1)

- The drawing as it is, then a copy with the candidates outlined (2 px, 3 px outside the box
  so that the symbol stays visible) and numbered next to the box. A small drawing is enlarged
  to a long edge of 1,400 px first, so that the numbers are readable; a large one is reduced
  to what the API takes (2,576 px, 3.75 MP), so that the marks are not thinned by the API's
  own reduction. The plain copy is sent as well, so that no mark hides a symbol; on this
  sample it adds about 870 image tokens per question (an estimate from the image sizes).
- Candidates: the detections that the counter counts (magenta) and doubtful ones (blue, `n?`):
  below the counter's minimum of their class but at least half of it. Half was chosen on the
  valid split: there, the true window count lies between the counted and the counted plus the
  doubtful windows on 92% of plans (61% are exact without them), with about 2 doubtful windows
  per plan.
- The text lists each candidate with its class, score, whether it is counted, and its centre;
  the system prompt says that the detector is right about 9 times in 10 per object, misses
  objects, marks wrong ones and can mark one window twice, and asks the model to check each
  candidate, look for unmarked objects, count by the rules of the question, and end with
  `Answer: <n>`.

## Commands

```sh
uv run python -m chatchtg.objects.importer <detections.jsonl of large-1024> data/floorplans/data/cubicasa5k
uv run python scripts/sample_object_questions.py data/floorplans/questions.jsonl var/object-hints/questions-test-150.jsonl --seed 0
Q="$PWD/var/object-hints/questions-test-150.jsonl"
# (b), no LLM call
CHATCHTG_LLM_PROVIDER=claude CHATCHTG_OBJECT_COUNTER=detections uv run chatchtg eval data/floorplans \
  --questions "$Q" --composer baseline --max-cost 3 --out var/eval-runs/object-hints-counter-150.jsonl
# Not run yet: the 5-question trial, then (a) and (c)
CHATCHTG_LLM_PROVIDER=claude uv run chatchtg eval data/floorplans --questions "$Q" --composer object-hints \
  --limit 5 --max-cost 3 --out var/eval-runs/object-hints-trial-5.jsonl
CHATCHTG_LLM_PROVIDER=claude uv run chatchtg eval data/floorplans --questions "$Q" --composer baseline \
  --max-cost 3 --out var/eval-runs/object-hints-baseline-150.jsonl
CHATCHTG_LLM_PROVIDER=claude uv run chatchtg eval data/floorplans --questions "$Q" --composer object-hints \
  --max-cost 3 --out var/eval-runs/object-hints-150.jsonl
```

## Cost to expect (estimate, not measured)

The plain LLM cost $0.022 per object-count question in the earlier 100-question baseline runs
(18 questions, mostly train plans). At that rate (a) costs about $3.3 for 150 questions, so a
run with `--max-cost 3` would stop after about 135 questions; the round-robin order keeps the
templates even. The cost of (c) is unknown until the trial; it sends two images and a list, so
it costs more per question than (a).

## Limits

- 150 questions, 30 per template, one run per method: a difference of a few questions in a
  template is not a finding.
- English questions only. The Finnish twin of each question has the same answer and the
  counter gives it the same count.
