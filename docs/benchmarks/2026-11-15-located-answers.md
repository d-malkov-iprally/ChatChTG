# Located answers: Kimi K3.1 Pro marks what it counted

Run on 2026-11-15. With the `located@1` composer, a count question also asks the model for
the position of each item it counted. This record compares that with the plain baseline of
the same model on the same questions, and reports where the marks landed.

**Result: no clear difference in accuracy on object counts (11 of 23 correct in both runs),
4 more room counts correct (18 against 14 of 19), for 1.36 times the cost per count question.
In a first review by an agent, 74 of 81 marks (91%) sat on an item of the asked kind.**
The cost limit stopped the run after 52 of the 100 questions, so the comparison covers 42 of
the 67 count questions of the sample. `located` is an experiment and stays off by default.

## Setup

| | |
|---|---|
| Model | `moonshotai/Kimi-K3.1-Pro` on a Modal Shared Endpoint (`CHATCHTG_LLM_PROVIDER=modal`), the model's own reasoning level, output limit 32,000 tokens |
| Prompt | `located@1`: for a question that starts with a counting phrase, the baseline system prompt plus the request to list every counted item with its position in absolute pixels. Every other question: the `baseline@1` context |
| Request | `response_format` of type `json_schema` (`strict`), schema `{items: [{label, x, y}], answer}`, and one sentence about that shape added to the system prompt by the adapter (`LOCATED_FORMAT_PROMPT` in `llm/modal.py`; it is not in the `context` of the result rows) |
| Questions | the first 52 of the 100 questions of `questions-subset.jsonl` picked by `--sample 100 --seed 0`: all 34 questions on Finnish plans and 18 on Swiss plans |
| Compared with | `baseline@1` of the same model, provider, sample and seed (`var/eval-runs/baseline-kimi-k3-1-pro-modal-100.jsonl`, written by another session; 63 questions answered when this was written). All 52 questions of this run are in it, so the comparison has 52 pairs |
| Scoring | a count must be exact; an area must be within 2% |
| Cost | $3.00 for the 52 answers, $0.058 per question. See "Cost of the whole experiment" for the calls outside this file |

The commands (in a worktree whose `.env` has the endpoint and the token):

```sh
MAIN=/path/to/ChatChTG
CHATCHTG_LLM_PROVIDER=modal CHATCHTG_DATA_ROOT=$MAIN/data uv run chatchtg eval $MAIN/data/floorplans \
  --questions questions-subset.jsonl --sample 100 --seed 0 --max-cost 3.60 --composer located --evidence \
  --out var/eval-runs/located-kimi-k3-1-pro-modal-100.jsonl
# The connection was reset at question 20. Resumed, with a lower limit:
#   the same command with --max-cost 3.30 --resume
uv run python scripts/render_evidence.py var/eval-runs/located-kimi-k3-1-pro-modal-100.jsonl $MAIN/data/floorplans \
  var/evidence-review/located-kimi-k3-1-pro-modal-100 --every 2 --limit 20 \
  --paths high_quality_architectural-10063.png high_quality_architectural-3060.png
```

The run printed `Stopped early: the next question could pass the $3.30 limit.` The limit was
not raised and no further run was started. The model's answers are not deterministic, so a
repeat gives different numbers.

## Accuracy, cost and output tokens, pairwise

Each row compares the same questions in both runs. "Fixed" is wrong in the baseline and
correct here; "broken" is the reverse.

| Group | Pairs | Baseline correct | Located correct | Fixed | Broken |
|---|---|---|---|---|---|
| Object counts (windows, doors, fixtures) | 23 (18 Finnish, 5 Swiss) | 11 | 11 | 1 | 1 |
| Room and layout counts | 19 (16 Finnish, 3 Swiss) | 14 | 18 | 4 | 0 |
| All count questions | 42 | 25 | 29 | 5 | 1 |
| Areas (baseline context in both runs) | 10 (Swiss) | 5 | 6 | 1 | 0 |

| Group | Cost per question, baseline | Cost per question, located | Ratio | Output tokens, baseline (mean / median) | Output tokens, located (mean / median) |
|---|---|---|---|---|---|
| Object counts | $0.0619 | $0.0807 | 1.30 | 5,539 / 4,068 | 7,351 / 6,546 |
| Room and layout counts | $0.0099 | $0.0174 | 1.76 | 623 / 203 | 1,300 / 557 |
| All count questions | $0.0384 | $0.0521 | 1.36 | 3,315 / 754 | 4,614 / 1,310 |
| Areas (baseline context) | $0.0724 | $0.0814 | 1.12 | 6,081 / 4,806 | 6,987 / 5,424 |

How to read this:

- **Object counts: no clear difference.** One fixed (4 sinks, where the baseline said 5), one
  broken (6 windows for 7). On Finnish plans both runs have 6 of 18 correct. The two large
  undercounts stay: 14 answered for 33 windows (baseline 15), 19 for 23 (baseline 17).
  Listing every item did not make the model find more of them.
- **Room and layout counts: 4 fixed, none broken.** With 19 pairs and one run of each this is
  an observation, not a finding. All four are kitchen, bathroom and bedroom counts on
  Finnish plans where the baseline answered one too many.
- **The area rows show the noise.** They have the same context in both runs and still differ
  by one answer and by 12% in cost.
- **Cost.** Reasoning tokens are output tokens and make up almost all of the cost. The most
  expensive answer used 31,580 output tokens ($0.32, the 33-window plan); the same question
  cost $0.20 in the baseline.

Not answered, because of the cost limit: 25 of the 67 count questions (9 object counts and
16 room and layout counts, all on Swiss plans) and 23 of the 33 area questions. The Swiss
side of the comparison is therefore thin: 8 count pairs, all correct in both runs.

Background, another model: Opus 5.5 with `baseline@1` on the same 52 questions
(`baseline-opus-5-5-100.jsonl`) has 12 of 23 object counts and 18 of 19 room counts correct,
for $0.024 and $0.012 per question, with a median of about 660 and 260 output tokens. It is
not part of the comparison.

## Readable replies and consistency

| Measure | Result |
|---|---|
| Count questions that asked for positions | 42 |
| Replies that could not be read as the JSON object | 0 |
| Replies that reached the output limit without an answer | 0 |
| Answers whose text contains a coordinate | 0 |
| Stated count differs from the number of marks | 0 |
| Marks outside the image | 0 |
| Marks very close to another mark | 0 |
| Marks in total | 125 |

The model kept its count and its list equal in every answer, so the consistency check never
had something to report. It says nothing about whether a mark is on the right object.

## Where the marks landed

Review set: every second question that asked for positions, up to 20, plus the plan with 23
windows; the plan with 33 windows was already in the set. That is 21 answers with 81 marks.
**Reviewed by the executing agent only**, from the rendered images and from enlarged cut-outs
around each mark. It is a first pass. The images are in
`var/evidence-review/located-kimi-k3-1-pro-modal-100/` of the worktree that made the run
(cut-outs in `crops/`), and the render command above makes them again from the result file.

"On the kind" counts marks on an item of the asked kind. An answer is `verifiable` when every
mark is on a different item of the asked kind, and `misleading` otherwise.

| Line | Plan | Subject | Expected | Answered | Marks | On the kind | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | colorful-13352 | kitchens | 1 | 1 | 1 | 1 | verifiable |
| 3 | high_quality-3650 | kitchens | 1 | 1 | 1 | 1 | verifiable |
| 5 | high_quality-5594 | storeys | 1 | 1 | 1 | 1 | verifiable |
| 7 | architectural-10063 | windows | 33 | 14 | 14 | 14 | verifiable; 19 windows have no mark |
| 9 | architectural-10267 | storeys | 1 | 1 | 1 | 1 | verifiable |
| 11 | architectural-10759 | doors | 6 | 5 | 5 | 4 | misleading: one mark is about 20 pixels beside its door, on a wall |
| 13 | architectural-11714 | sinks | 2 | 2 | 2 | 2 | verifiable |
| 15 | architectural-12283 | sinks | 4 | 6 | 6 | 4 | misleading: all six marks are on basin-like symbols, the reference has four; which two are wrong was not decided |
| 17 | architectural-13349 | kitchens | 1 | 1 | 1 | 1 | verifiable |
| 19 | architectural-13802 | storeys | 3 | 2 | 2 | 2 | verifiable; one storey has no mark |
| 21 | architectural-2001 | doors | 8 | 9 | 9 | 9 | verifiable; all nine are door swings, one more than the reference counts |
| 22 | architectural-3060 | windows | 23 | 19 | 19 | 19 | verifiable; 4 windows have no mark |
| 23 | architectural-3573 | storeys | 2 | 2 | 2 | 2 | verifiable |
| 25 | architectural-5027 | showers and bathtubs | 1 | 2 | 2 | 1 | misleading: one mark is on handwriting |
| 27 | architectural-6282 | bedrooms | 2 | 2 | 2 | 2 | verifiable |
| 29 | architectural-7071 | sinks | 1 | 2 | 2 | 1 | misleading: one mark is on another fixture |
| 31 | architectural-772 | windows | 4 | 4 | 4 | 4 | verifiable |
| 33 | architectural-900 | toilet bowls | 1 | 1 | 1 | 1 | verifiable |
| 35 | site1601_bldg2794_floor12912 | WCs | 2 | 2 | 2 | 0 | misleading: both marks are on washbasins (the plan's legend shows the two symbols); the count is right |
| 38 | site96_bldg108_floor47822 | apartments | 2 | 2 | 2 | 2 | verifiable |
| 42 | site1257_bldg2450_floor11435 | WCs | 2 | 2 | 2 | 2 | verifiable |

- **74 of 81 marks (91%) are on an item of the asked kind. 16 of 21 answers are verifiable.**
- The wrong marks share no direction, and the right ones sit on the symbol itself: on the
  window in the wall, in the swing of the door, on the basin. There is no sign of a scaling
  or offset fault.
- The marks did what they are for. In four of the five misleading answers the mark shows a
  person why the count is wrong or doubtful, and in line 35 it shows that a count scored as
  correct was made from the wrong objects. In lines 7, 19 and 22 the unmarked windows and
  storey are easy to see.
- A mark for a storey or a kitchen is a point somewhere in that storey or room. It confirms
  what the model meant, and little more.
- Limits of this review: one reviewer, an agent, working from images; on the dense plans
  (lines 7 and 22) each mark was judged from a cut-out of about 200 pixels. Whether two
  marks on one long window strip are two windows was not decided (line 22, marks 14 and 15).

## What was found about the endpoint

- **JSON output.** The endpoint accepted `response_format` of type `json_schema` with
  `strict`, of type `json_object`, and a request without either, and returned a JSON object
  in all three cases. The Kimi quickstart documents `json_schema` with `strict: true` for the
  final `message.content`
  ([quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart)); nothing was
  found in Modal's documentation. With the schema, the keys came back in alphabetical order
  (`answer` before `items`), not in schema order. Whether the endpoint enforces the schema
  was not tested with a reply that would break it, so the adapter also states the shape in
  the system prompt and reads the reply defensively.
- **Coordinates.** No documentation of a coordinate convention was found for the model or for
  the endpoint. The Kimi vision guide says only that image tokens are calculated dynamically
  and recommends images of at most 4K
  ([vision guide](https://platform.kimi.ai/docs/guide/use-kimi-vision-model)). The rule in
  `to_stored_pixels` is therefore empirical:
  - Asked for absolute pixel coordinates, the model answers in pixels of the file as it was
    sent. On a 2480x1753 plan the two toilets were marked at (885, 607) and (929, 607), on
    the symbols. Asked instead for values from 0 to 1000, it gave (356, 347) and (374, 347),
    which are the same places. The pixel wording was kept, because it needs no conversion.
  - The endpoint reports `image_tokens`. In the probes that number is exactly
    `ceil(width / 28) * ceil(height / 28)` (5,607 for 2480x1753). The result rows keep only
    the input tokens; in every answer of both runs, including a 3507x2480 plan, they are that
    number plus 158 to 547 tokens of text and schema (154 to 208 in the baseline). So the model gets the whole
    file in 28-pixel patches, and nothing makes it smaller first.
  - If an endpoint ever reports fewer image tokens than that, the adapter returns the
    answer without positions. Where the model's pixels point after a resize was never seen.

## Decision

By the thresholds that were set before the run (no clear loss of accuracy, at most 1.5 times
the cost, at most 1 unreadable reply, at least 85% of the reviewed marks in place and at least
15 verifiable answers), `located@1` passes on the questions that were answered. The gate is
not narrowed: room and layout counts did not drop.

It is not a recommendation to make it the default, and the default stays `baseline`:

- It does not answer more object counts correctly, which was the hope.
- It costs 36% more on count questions with a model whose cost per question is already high
  and uneven.
- Its value is the second judging criterion: a person can check a count on the drawing in
  seconds, and in this review the marks exposed a wrong basis in an answer that the rule
  scored as correct.

## Cost of the whole experiment

| Step | Cost |
|---|---|
| Discovery: 3 text-only probes of the JSON modes, 2 answered image probes, 1 streamed image probe | $0.18 |
| Discovery: 4 image probes that ended in a read timeout after 600 seconds | not known; at most $0.65 if they were billed up to their 16,000-token limit |
| Smoke run, 3 questions | $0.02 |
| Measured run, 52 questions | $3.00 |
| The request that ended in a connection reset during the measured run | not known; at most $0.33 |
| **Total** | **$3.20 recorded, at most $4.18** |

The approved budget was $5.00. The limit of the measured run was lowered from $3.60 to $3.30
after the reset, so that the total stays under the budget even if every failed request was
billed in full.

## Limits

- 52 of 100 questions; 42 count questions, of which 8 are on Swiss plans. The groups have 23
  and 19 pairs. One run of each composer. A difference of a few answers is not a finding.
- The baseline file belongs to another session and was still being written. The comparison
  uses its rows for the same 52 questions.
- Only the `modal` adapter asks for positions. The `claude` adapter ignores the request, and
  `located` was never run with it.
- The review of the marks was done by an agent and not repeated by a person.
- The four timed-out probes ran at the same time as each other and as another eval on the
  endpoint. One of the same questions, sent alone, was answered in 131 seconds.
