"""The failure pattern of a measured room against its reference polygon."""

from chatchtg.evals.measure_score import edge_offsets, failure_pattern

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_a_room_within_two_percent_is_ok():
    assert failure_pattern(SQUARE, 10.0, SQUARE, 10.1) == "ok"


def test_a_missing_corner_is_a_lost_piece():
    measured = [(0, 0), (100, 0), (100, 60), (60, 60), (60, 100), (0, 100)]
    assert failure_pattern(SQUARE, 10.0, measured, 8.4) == "lost"


def test_an_added_part_is_a_leak():
    measured = [(0, 0), (140, 0), (140, 100), (0, 100)]
    assert failure_pattern(SQUARE, 10.0, measured, 14.0) == "leak"


def test_a_lost_piece_and_an_added_part_are_both():
    measured = [(0, 0), (140, 0), (140, 60), (60, 60), (60, 100), (0, 100)]
    assert failure_pattern(SQUARE, 10.0, measured, 10.8) == "both"


def test_a_room_that_is_only_shrunk_is_edges():
    measured = [(1.2, 1.2), (98.8, 1.2), (98.8, 98.8), (1.2, 98.8)]
    assert failure_pattern(SQUARE, 10.0, measured, 9.53) == "edges"


def test_a_room_without_an_area():
    assert failure_pattern(SQUARE, 10.0, [], None) == "no-area"


def test_edge_offsets_are_negative_inside():
    measured = [(0, 0), (98, 0), (98, 100), (0, 100)]
    offsets = edge_offsets(SQUARE, measured)
    assert sorted(round(o) for _, o in offsets) == [-2, 0, 0, 0]


def test_edge_offsets_skip_short_edges():
    truth = [(0, 0), (100, 0), (100, 100), (5, 100), (0, 95)]
    assert len(edge_offsets(truth, truth)) == 4
