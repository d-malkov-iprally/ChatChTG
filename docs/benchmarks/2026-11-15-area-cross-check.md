# Area cross-check: a second estimate from the LLM for doubtful measured areas

Measured on 2026-11-15. The idea: where the area pipeline states a result that rests on a
measured room, and the plan gives a reason for doubt, ask an LLM that sees the drawing for
its own estimate, and answer with that estimate when the two are more than 10% apart.

**Result: negative. The idea did not pass its first gate, so the cross-check was not built
and no billed run was made. On all 1,040 Swiss plans only 82 of the 2,218 answers that the
check would look at are grossly wrong (3.7%; the gate asked for 5%). A checker that never
errs could add at most 0.7 correct answers per 100 area questions.**

Nothing here called an LLM. The cost was $0.

## Setup

- Code: branch `area-cross-check`, on top of `state-area-in-code` (pull request #20). The
  pipeline and its confidence rule are those of `areas@3`; `decide()` in
  `context/areas.py` gives the value, the basis and the doubts of each answer.
- Questions: all 12,016 area questions of `data/floorplans/questions.jsonl`, on 1,040 Swiss
  plans. The pipeline analysed every plan.
- Scoring: an area is correct within 2% of the reference (`score_value`). "Gross" means
  more than 10% off. The pipeline's value is scored, not a sentence.
- **Eligibility rule** (fixed before this measurement): the pipeline is confident, at least
  one counted room was measured and not printed, and the plan's scale is 1:200 or coarser
  or its room numbering has a gap.
- **Threshold** (fixed before this measurement): 10% (`DISAGREEMENT` in
  `areas/crosscheck.py`).
- **Gate A** (fixed before this measurement): stop if fewer than 5% of the eligible
  questions are grossly wrong. The check would then mostly fire on answers that it cannot
  improve.
- **Gate B** (not reached): after one billed `baseline@1` run on 100 eligible questions, at
  the threshold of 10%: at least 5 questions flagged, the pipeline wrong on at least 80% of
  them, and at least 3 more answers that become right than become wrong.

## Gate A: where the gross errors are, on all plans

The answers that the pipeline states with confidence and that rest on a measured area:

| Group | Questions | Wrong (more than 2% off) | Gross (more than 10% off) | Plans |
|---|---|---|---|---|
| All | 8,936 | 924 (10.3%) | 138 (1.5%) | 877 |
| Eligible: 1:200 or coarser, or a numbering gap | 2,218 | 456 (20.6%) | 82 (3.7%) | 204 |
| Not eligible | 6,718 | 468 (7.0%) | 56 (0.8%) | 673 |

The eligible questions are 18.5% of all area questions. **Gate A: 3.7% is below 5%, so the
work stops here.**

The eligible answers by their doubt, and by the kind of question:

| Group | Questions | Wrong | Gross |
|---|---|---|---|
| 1:200 or coarser, no gap | 450 | 116 | 14 (3.1%) |
| 1:200 or coarser, and a gap | 978 | 256 | 50 (5.1%) |
| A gap, at 1:100 | 790 | 84 | 18 (2.3%) |
| Area of one numbered room | 750 | 196 | 50 (6.7%) |
| Total of rooms with a given label | 364 | 36 | 6 (1.6%) |
| Area of one apartment | 724 | 150 | 22 (3.0%) |
| Net floor area of the whole floor | 380 | 74 | 4 (1.1%) |

34 of the 204 eligible plans have at least one grossly wrong eligible answer. These
sub-groups were looked at after the gate was applied. They are a description, not a new
rule: a rule picked from this table would be fitted to the plans it is measured on.

### Why the estimate from 100 plans was wrong

The rule and the gate were chosen on a sample of 100 plans (`scripts/eval_areas.py --plans
100 --seed 0`). This measurement reproduces that sample exactly and then covers the rest:

| | Eligible share of the area questions | Gross among the eligible |
|---|---|---|
| The 100 plans of the sample | 118 of 1,142 (10.3%), on 11 plans | 12 of 118 (10.2%) |
| The other 940 plans | 2,100 of 10,874 (19.3%), on 193 plans | 70 of 2,100 (3.3%) |
| All 1,040 plans | 2,218 of 12,016 (18.5%), on 204 plans | 82 of 2,218 (3.7%) |

The 12 gross answers of the sample sat on a few of its 11 eligible plans. Eleven plans were
too few to estimate a rate of a few percent. On all plans the check would fire nearly twice
as often as expected and find a gross error about a third as often.

## What the check could be worth at most

Suppose a checker that flags every grossly wrong eligible answer, gives the right value for
each, and never flags a good answer:

| | Per 100 checks | Per 100 area questions |
|---|---|---|
| Checks made | 100 | 18.5 |
| Answers that become right, at most | 3.7 | 0.68 |

Against that stands the cost. The pipeline's answer costs $0 today. Each check is one
`baseline@1` call with the image. With Kimi K3.1 Pro on Modal a baseline answer cost $0.003
to $0.20 in another session's run on this benchmark (45 questions, mean about $0.04; the 4
area questions among them $0.05). At $0.05 a check, the cross-check adds about $0.009 to
every area question, or about $1.35 for each answer it could repair at best. A real model
does worse than this bound: it misses some gross errors and flags some good answers.

## The billed run that was not made

One billed run was approved: `baseline@1` with Kimi K3.1 Pro on Modal
(`CHATCHTG_LLM_PROVIDER=modal`, model `moonshotai/Kimi-K3.1-Pro`) on the eligible
questions, under a cap of $4.50 in total. It belongs to Gate B and was not started, because
Gate A said stop first. $0.00 was spent.

The question file was still prepared, with no LLM call (`select --n 100 --seed 0`: one
eligible question per plan, picked at random, for 100 of the 204 eligible plans picked at
random; 67 plans at 1:200, 33 at 1:100). What is known about it without a model:

| | Correct | Cost per question |
|---|---|---|
| Pipeline alone (the value that code states) | 80 of 100 | $0 |
| LLM alone (`baseline@1`, Kimi K3.1 Pro) | not measured | not measured |
| Cross-check | not measured | not measured |

Of the 20 wrong answers, 4 are gross (3 among the first 90 questions, which is about what
the cap would have paid for). Gate B needs at least 5 flags with the pipeline wrong on 80%
of them. With 4 gross errors in the file, the run could pass only if the model caught every
one of them and also flagged a small error of the pipeline without one false flag. So the
run would very probably have ended in "failed" or "too few flags to tell", for $4.50.

**Not measured, and still unknown: how well Kimi K3.1 Pro estimates an area on these
plans.** Nothing here says whether its error stays below the 10% threshold.

## Background: the offline analysis with Opus 5.5

These numbers are from before this measurement, describe Claude Opus 5.5 and not Kimi, and
are the reason for the threshold and the eligibility rule. They were not repeated here.

- On the 33 area questions of the baseline sample, in two Opus 5.5 runs, 23 answers rest on
  a measured area. The pipeline was right on 21 of them, Opus on 14 and 18.
- Opus's error on the 19 questions on plans without printed areas (38 answers): median
  1.3%, 90th percentile 3.4%, maximum 9.2%. Hence the threshold of 10%.
- Where Opus and the pipeline were 2% to 10% apart, Opus was the one that was wrong. So an
  LLM can detect a gross error at best; it cannot arbitrate a small one.
- On the 100-plan sample, 12 of the 20 gross answers were on plans at 1:200 or with a
  numbering gap. Hence the eligibility rule. A simulation with Opus's errors predicted 6.5
  more correct answers per 100 checks at 10%. The table above shows why that was too high:
  on all plans a perfect checker gets 3.7 at most.

## Limits

- The gate was applied to the pipeline's own errors. No model was measured, neither Kimi
  nor Opus, on the eligible questions.
- The eligible questions are not independent: 2,218 questions sit on 204 plans, and each
  quantity is asked in two languages. Counting plans, 34 of 204 have a gross eligible
  answer. The 95% interval of 3.7%, with plans as the unit (a bootstrap over the 204
  plans), is 2.4% to 5.2%: its upper end is just above the gate, and 4% of the resamples
  reach 5%.
- The eligibility rule was chosen on 100 plans that are part of the 1,040 measured here.
  Without them the figure is 3.3%.
- The reference answers come from the DXF twins of the plans. An answer that is "gross"
  against the reference is taken to be an error of the pipeline.
- A different rule (for example single rooms on 1:200 plans with a gap) was not tested as a
  gate, on purpose: it would be chosen after seeing the data.

## Repeat it

```sh
export CHATCHTG_DATA_ROOT=/absolute/path/to/data    # the folder that holds floorplans/
uv run python scripts/eval_area_check.py signals --workers 7   # about an hour of CPU; no LLM call
uv run python scripts/eval_area_check.py select --n 100 --seed 0
```

The billed step, not run. `replay` prints the numbers of Gate B and the size of the model's
error:

```sh
CHATCHTG_LLM_PROVIDER=modal uv run chatchtg eval "$CHATCHTG_DATA_ROOT/floorplans" \
  --questions "$PWD/var/area-check/questions.jsonl" --composer baseline \
  --sample 100 --seed 0 --max-cost 4.50 --out var/eval-runs/area-check-baseline-kimi-k3-1-pro.jsonl
uv run python scripts/eval_area_check.py replay --results var/eval-runs/area-check-baseline-kimi-k3-1-pro.jsonl
```

Cost of this record: $0.
