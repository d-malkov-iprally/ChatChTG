# Baseline: Opus 5.5 on 100 floor-plan questions

Run on 2026-11-14. The plain model gets the floor plan image and the question, with the
baseline prompt and nothing else. Later experiments compare against these numbers.

**Result: 77 of 100 correct (77%), for $2.68.**
The 95% interval is 68% to 84%.

## Setup

| | |
|---|---|
| Model | `claude-opus-5-5`, effort `medium` |
| Prompt | `baseline@1` (system prompt, then the image, then the question) |
| Questions | 100 from `questions-subset.jsonl` of the floor-plan benchmark, balanced over source and question type (`--sample 100 --seed 0`), on 98 floor plans |
| Scoring | a count must be exact; an area must be within 2% |
| Cost | $2.68 in total, $0.027 per question |

To repeat the run:

```sh
CHATCHTG_LLM_PROVIDER=claude CHATCHTG_LLM_MODEL=claude-opus-5-5 uv run chatchtg eval data/floorplans \
  --questions questions-subset.jsonl --sample 100 --seed 0 --max-cost 3.90 \
  --out var/eval-runs/baseline-opus-5-5-100.jsonl
```

The model's answers are not deterministic, so a repeat gives slightly different numbers.

## Results by group

| Group | Correct | Accuracy | 95% interval | Cost | Median input tokens | Median output tokens |
|---|---|---|---|---|---|---|
| Finnish plans, counts | 23 of 34 | 68% | 51% to 81% | $0.55 | 1,024 | 360 |
| Swiss plans, counts | 30 of 33 | 91% | 76% to 97% | $0.88 | 4,844 | 232 |
| Swiss plans, areas | 24 of 33 | 73% | 56% to 85% | $1.25 | 4,858 | 872 |

Each group has about 33 questions, so the intervals are wide. A difference of a few points
between two runs of this size is not a finding.

## Results by subject

| Subject | Kind | Correct | Wrong |
|---|---|---|---|
| Rooms with a given label | Rooms | 15 of 15 | 0 |
| Area of one numbered room | Areas | 7 of 11 | 4 |
| Sinks and washbasins | Fixtures | 6 of 9 | 3 |
| Net floor area of the whole floor | Areas | 8 of 8 | 0 |
| Total area of rooms with a given label | Areas | 5 of 8 | 3 |
| Windows | Openings | 3 of 7 | 4 |
| Area of one apartment | Areas | 4 of 6 | 2 |
| Bathtubs | Fixtures | 5 of 6 | 1 |
| Floors (storeys) | Layout | 4 of 5 | 1 |
| Kitchens | Rooms | 5 of 5 | 0 |
| Apartments | Layout | 4 of 4 | 0 |
| Showers and bathtubs together | Fixtures | 3 of 4 | 1 |
| Toilets | Fixtures | 2 of 4 | 2 |
| Bathrooms and washrooms | Rooms | 3 of 3 | 0 |
| Bedrooms | Rooms | 3 of 3 | 0 |
| Doors | Openings | 0 of 2 | 2 |

Most subjects have fewer than ten questions, so read this table as a pointer to where to look,
not as a ranking.

## What the run shows

- **Room counts were all correct.** All 26 questions that count rooms of one kind
  (labelled rooms, kitchens, bedrooms, bathrooms) were answered correctly. No question about saunas,
  separate WC rooms or balconies was in the sample.
- **Doors and windows are the weakest subject.** Windows were correct 3 times of 7 and doors 0 of 2.
  The largest misses are undercounts on big plans: 12 answered for 33 windows, and 14 for 23.
- **Fixture counts are mixed.** Sinks, toilets, bathtubs and showers were correct 16 times of 23,
  and the misses are mostly off by one or two.
- **Areas split by kind.** The area of the whole floor was correct 8 times of 8. The areas of one
  room, one apartment, or all rooms with one label were correct 16 times of 25. Those misses are
  close: between 2% and 8% from the correct value, against a tolerance of 2%.
- **Area questions cost the most.** The model writes about 870 output tokens on an area question
  and 230 to 360 on a count.

## Wrong answers

For a count, "off by" is the answer minus the correct count. For an area, it is the difference in percent.

| # | Group | Subject | Floor plan | Correct | Answered | Off by |
|---|---|---|---|---|---|---|
| 6 | Finnish plans, counts | Windows | `high_quality-9336.png` | 12 | 10 | -2 |
| 7 | Finnish plans, counts | Windows | `high_quality_architectural-10063.png` | 33 | 12 | -21 |
| 11 | Finnish plans, counts | Doors | `high_quality_architectural-10759.png` | 6 | 4 | -2 |
| 15 | Finnish plans, counts | Sinks and washbasins | `high_quality_architectural-12283.png` | 4 | 6 | +2 |
| 18 | Finnish plans, counts | Toilets | `high_quality_architectural-13436.png` | 1 | 2 | +1 |
| 19 | Finnish plans, counts | Floors (storeys) | `high_quality_architectural-13802.png` | 3 | 2 | -1 |
| 20 | Finnish plans, counts | Windows | `high_quality_architectural-14260.png` | 7 | 9 | +2 |
| 21 | Finnish plans, counts | Doors | `high_quality_architectural-2001.png` | 8 | 9 | +1 |
| 22 | Finnish plans, counts | Windows | `high_quality_architectural-3060.png` | 23 | 14 | -9 |
| 28 | Finnish plans, counts | Showers and bathtubs together | `high_quality_architectural-6704.png` | 1 | 2 | +1 |
| 29 | Finnish plans, counts | Sinks and washbasins | `high_quality_architectural-7071.png` | 1 | 2 | +1 |
| 35 | Swiss plans, counts | Toilets | `site1601_bldg2794_floor12912.png` | 2 | 0 | -2 |
| 40 | Swiss plans, areas | Total area of rooms with a given label | `site11696_bldg18244_floor48836.png` | 69.42 | 68 | -2.0% |
| 48 | Swiss plans, areas | Total area of rooms with a given label | `site1938_bldg2928_floor13405.png` | 190.02 | 185 | -2.6% |
| 49 | Swiss plans, areas | Area of one numbered room | `site3656_bldg5636_floor24301.png` | 5.1 | 5.5 | +7.8% |
| 51 | Swiss plans, areas | Total area of rooms with a given label | `site48_bldg293_floor51438.png` | 23.68 | 22 | -7.1% |
| 65 | Swiss plans, areas | Area of one apartment | `site1056_bldg2171_floor9800.png` | 33.11 | 32 | -3.4% |
| 69 | Swiss plans, counts | Sinks and washbasins | `site144_bldg99_floor441.png` | 7 | 4 | -3 |
| 73 | Swiss plans, areas | Area of one numbered room | `site121_bldg34_floor395.png` | 16.37 | 15.9 | -2.9% |
| 77 | Swiss plans, areas | Area of one numbered room | `site326_bldg658_floor2799.png` | 7.56 | 7.3 | -3.4% |
| 84 | Swiss plans, areas | Area of one numbered room | `site1053_bldg2159_floor9758.png` | 6.33 | 6.5 | +2.7% |
| 93 | Swiss plans, counts | Bathtubs | `site914_bldg1947_floor8548.png` | 4 | 3 | -1 |
| 97 | Swiss plans, areas | Area of one apartment | `site11879_bldg18546_floor50616.png` | 107.56 | 105 | -2.4% |

`#` is the line of the answer in the results file.

## Limits

- 100 questions is a small sample. The overall interval is about 16 points wide.
- The sample is balanced over source and question type, not over subject. The benchmark as a whole
  has far more Finnish count questions than this sample suggests, so the overall 77% is not an
  estimate of accuracy on the whole benchmark.
- The run was interrupted twice by connection timeouts and resumed. Every question was
  answered once, and all answers came from `claude-opus-5-5`.
- The results file is in `var/`, which Git ignores. It holds every question, the full answer text,
  the token usage and the cost.
