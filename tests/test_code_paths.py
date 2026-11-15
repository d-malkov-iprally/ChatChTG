import pytest

from chatchtg.evals import code_paths
from chatchtg.evals.code_paths import (
    Answer,
    Claim,
    DoubleClaimError,
    bootstrap,
    claims,
    llm_estimate,
    merge,
    summarise,
    tally_llm,
    weighted_score,
)

FI = "data/cubicasa5k/colorful-1.png"
FI2 = "data/cubicasa5k/colorful-2.png"
CH = "data/swiss_dwellings/site1_bldg1_floor1.png"
CH2 = "data/swiss_dwellings/site2_bldg2_floor2.png"

WINDOWS_FI = "How many windows are shown on this floor plan?"
BEDROOMS = "How many bedrooms (makuuhuone, usually marked MH) are on this floor plan?"
WINDOWS_CH = "How many windows are drawn on this floor plan?"
LABELLED = "How many rooms labelled 'Schlafzimmer' are there on this floor?"
APARTMENTS = "How many apartments (residential units) are shown on this floor plan?"
ROOM_AREA = "What is the net floor area of room 7.01 ('Balkon') in m²?"
FLOOR_AREA = (
    "What is the total net floor area in m² of all numbered indoor rooms on this floor (excluding "
    "balconies, loggias, terraces and other outdoor spaces, shafts and voids)?"
)


def q(path, question, answer, kind="count"):
    return {"path": path, "question": question, "type": kind, "answer": answer}


@pytest.mark.parametrize(
    ("path", "question", "kind", "expected"),
    [
        (FI, WINDOWS_FI, "count", Claim("objects-cubicasa", "finnish:window")),
        (FI, BEDROOMS, "count", Claim("rooms", "finnish:bedroom")),
        (CH, WINDOWS_CH, "count", Claim("objects-swiss", "swiss:window")),
        (CH, LABELLED, "count", Claim("rooms", "swiss:label")),
        (CH, APARTMENTS, "count", Claim("rooms", "swiss:apartments")),
        (CH, ROOM_AREA, "area", Claim("areas", "swiss:area-room")),
        (CH, FLOOR_AREA, "area", Claim("areas", "swiss:area-floor")),
    ],
)
def test_each_kind_of_question_is_claimed_by_one_path(path, question, kind, expected):
    assert claims({"path": path, "question": question, "type": kind}) == [expected]


def test_a_question_that_no_path_answers_is_not_claimed():
    assert claims({"path": FI, "question": "What is drawn here?", "type": "count"}) == []
    # A Finnish room question on a Swiss plan is not the room counter's.
    assert claims({"path": CH, "question": BEDROOMS, "type": "count"}) == []


def test_merge_scores_the_claiming_path_and_reports_the_rest():
    questions = [
        q(FI, WINDOWS_FI, 7),
        q(FI, BEDROOMS, 2),
        q(CH, ROOM_AREA, 10.0, "area"),
        q(CH, FLOOR_AREA, 100.0, "area"),
        q(FI, "What is drawn here?", 1),
    ]
    answers = {
        "objects-cubicasa": {(FI, WINDOWS_FI): Answer(7.0, "test")},
        "rooms": {(FI, BEDROOMS): Answer(1.0, "tuned", confident=False)},
        # 10.15 is within 2% of 10.0; the floor question has no stored answer.
        "areas": {(CH, ROOM_AREA): Answer(10.15, "held-out", confident=True)},
    }
    scored, unclaimed, not_run = merge(questions, answers)
    by_question = {row.question: row for row in scored}
    assert by_question[WINDOWS_FI].correct and by_question[WINDOWS_FI].held_out
    assert not by_question[BEDROOMS].correct and not by_question[BEDROOMS].held_out
    assert by_question[ROOM_AREA].correct and by_question[ROOM_AREA].held_out
    assert [row["question"] for row in unclaimed] == ["What is drawn here?"]
    assert [(row["question"], claim.path) for row, claim in not_run] == [(FLOOR_AREA, "areas")]


def test_an_answer_without_a_number_is_wrong():
    scored, _, _ = merge([q(CH, ROOM_AREA, 10.0, "area")], {"areas": {(CH, ROOM_AREA): Answer(None, "held-out")}})
    assert scored[0].correct is False


def test_merge_refuses_a_question_claimed_twice(monkeypatch):
    monkeypatch.setattr(code_paths, "claims", lambda row: [Claim("rooms", "a"), Claim("areas", "b")])
    with pytest.raises(DoubleClaimError):
        merge([q(FI, BEDROOMS, 1)], {})


def _scored(rows):
    """rows: (plan, kind, correct, held_out)."""
    return [
        code_paths.Scored(plan, f"q{i}", "rooms", kind, 1, 1.0, correct, "test" if held else "tuned", held, None)
        for i, (plan, kind, correct, held) in enumerate(rows)
    ]


def test_summary_counts_held_out_questions_and_plans_apart():
    scored = _scored([(FI, "finnish:bedroom", True, True), (FI2, "finnish:bedroom", False, False)])
    s = summarise(scored, {"finnish:bedroom": 10})["finnish:bedroom"]
    assert (s.benchmark, s.answered, s.correct) == (10, 2, 1)
    assert (s.held_out, s.held_out_correct, s.held_out_plans, s.plans) == (1, 1, 1, 2)


def test_bootstrap_point_is_the_question_weighted_mean_of_held_out_accuracies():
    scored = _scored([
        (FI, "finnish:bedroom", True, True),
        (FI2, "finnish:bedroom", False, True),
        (CH, "swiss:label", True, True),
        (CH2, "swiss:label", True, True),
        (CH2, "swiss:label", False, False),  # fitted plan: left out
    ])
    weights = {"finnish:bedroom": 3, "swiss:label": 1}
    point, low, high = bootstrap(scored, weights, draws=500, seed=1)
    assert point == pytest.approx(weighted_score({"finnish:bedroom": 0.5, "swiss:label": 1.0}, weights))
    assert point == pytest.approx(0.625)
    assert low <= point <= high
    assert bootstrap(scored, weights, draws=500, seed=1) == (point, low, high)


def test_bootstrap_resamples_plans_not_questions():
    # One plan with ten right answers, one with ten wrong ones: the only draws are 0, 0.5 and 1.
    rows = [(FI, "finnish:bedroom", True, True)] * 10 + [(FI2, "finnish:bedroom", False, True)] * 10
    point, low, high = bootstrap(_scored(rows), {"finnish:bedroom": 1}, draws=2000, seed=0)
    assert point == 0.5
    assert (low, high) == (0.0, 1.0)


def test_bootstrap_needs_a_held_out_question_of_every_kind():
    with pytest.raises(ValueError, match="swiss:label"):
        bootstrap(_scored([(FI, "finnish:bedroom", True, True)]), {"finnish:bedroom": 1, "swiss:label": 1})


def _run(path, question, kind, composer, correct, cost=0.01):
    return {"path": path, "question": question, "type": kind, "composer": composer,
            "model": "claude-opus-5-5", "correct": correct, "cost_usd": cost}


def test_llm_tally_keeps_plain_answers_and_the_latest_conventions():
    rows = [
        _run(FI, BEDROOMS, "count", "baseline@1", False),
        _run(FI, BEDROOMS, "count", "room-conventions@1", False),
        _run(FI, BEDROOMS, "count", "room-conventions@3", True),
        # The areas composer states a code result: it measures no LLM.
        _run(CH, ROOM_AREA, "area", "areas@2", True),
        # Its other questions get the baseline context.
        _run(CH, WINDOWS_CH, "count", "areas@2", True),
        {**_run(CH, WINDOWS_CH, "count", "baseline@1", True), "model": "kimi"},
    ]
    tallies = tally_llm(rows)
    assert tallies[("finnish:bedroom", "plain")].correct == 0
    conventions = tallies[("finnish:bedroom", "conventions")]
    assert (conventions.answers, conventions.correct) == (1, 1)
    assert ("swiss:area-room", "plain") not in tallies
    assert tallies[("swiss:window", "plain")].answers == 1


def test_llm_estimate_names_the_kinds_with_few_answers_and_widens_their_range():
    rows = [_run(FI, BEDROOMS, "count", "baseline@1", i % 2 == 0) for i in range(4)]
    estimate = llm_estimate(tally_llm(rows), {"finnish:bedroom": 1, "finnish:window": 1})
    assert set(estimate["uncertain"]) == {"finnish:bedroom", "finnish:window"}
    # Windows were not measured: the range runs from 0 to 1 for them.
    low, high = estimate["range"]
    assert low < estimate["estimate"] < high
    assert high - low > 0.5


def _analysis(missing_numbers):
    from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom

    def room(number, area):
        return PlanRoom(number=number, name="Zimmer", area=area, method="printed", printed_area=area,
                        measured_area=None, apartment="A", outdoor=False, seed=(0.0, 0.0))

    return PlanAnalysis(scale=100, scale_read=True, pixels_per_metre=59.0,
                        rooms=[room("1.01", 10.004), room("1.02", 20.0)], missing_numbers=missing_numbers)


def test_area_answers_are_read_from_the_cached_analyses_as_the_app_states_them(tmp_path):
    import json

    from chatchtg.evals.code_paths import load_area_answers

    (tmp_path / "site1_bldg1_floor1.json").write_text(json.dumps(_analysis(["1.03"]).to_json()))
    (tmp_path / "site2_bldg2_floor2.json").write_text(json.dumps({"error": "boom"}))
    one_room = "What is the net floor area of room 1.01 ('Zimmer') in m²?"
    questions = [
        q(CH, one_room, 10.0, "area"),
        q(CH, FLOOR_AREA, 30.0, "area"),
        q(CH2, one_room, 10.0, "area"),
        q("data/swiss_dwellings/site3_bldg3_floor3.png", one_room, 10.0, "area"),
        q(CH, LABELLED, 1),
    ]
    found = load_area_answers(questions, tmp_path, tuned={"site2_bldg2_floor2"})
    # The statement gives two decimals.
    assert found[(CH, one_room)] == Answer(10.0, "held-out", True)
    # A total on a plan that prints its areas and has a gap in its numbering goes to the LLM.
    assert found[(CH, FLOOR_AREA)] == Answer(30.0, "held-out", False)
    # A failed analysis answers nothing; a plan that was not analysed is not run.
    assert found[(CH2, one_room)] == Answer(None, "tuned", False)
    assert len(found) == 3
