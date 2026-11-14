"""The pure part of the label reader: text normalisation and table building.

No OCR model and no dataset is needed here.
"""

import pytest

from chatchtg.areas.labels import (
    TextLine,
    build_table,
    merge_fragments,
    merge_rereads,
    parse_area,
    parse_number,
    parse_scale,
    parse_tag,
    plan_language,
    repair_number,
    snap_name,
    to_plan_labels,
)


def line(text: str, cx: float, cy: float, height: float = 12.0, confidence: float = 0.9) -> TextLine:
    width = 7.0 * len(text)
    return TextLine(text, (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), confidence)


def label(cx: float, cy: float, *texts: str) -> list[TextLine]:
    """A centred label whose first line is at ``cy``; lines are 14 px apart."""
    return [line(text, cx, cy + 14 * k) for k, text in enumerate(texts)]


@pytest.mark.parametrize(
    ("text", "number"),
    [("7.16", "7.16"), ("U1.03", "U1.03"), ("12,07", "12.07"), ("1.104", "1.104"),
     ("3 . 38", "3.38"), ("Zimmer", None), ("15.61 m²", None), ("716", None)],
)
def test_parse_number(text, number):
    assert parse_number(text) == number


@pytest.mark.parametrize(
    ("text", "area"),
    [("15.61 m²", (15.61, True)), ("15,61m2", (15.61, True)), ("4.96", (4.96, False)),
     ("/4.50m²", (4.5, True)), ("2566/m2", (25.66, True)), ("868 m?", (8.68, True)),
     ("103.20 m²", (103.2, True)), ("Bad", None), ("2.1", None)],
)
def test_parse_area(text, area):
    assert parse_area(text) == area


@pytest.mark.parametrize(
    ("text", "letter"),
    [("Whg. A", "A"), ("App. B", "B"), ("Whg.C", "C"), ("Wha. D", "D"), ("Whg", None), ("Bad", None)],
)
def test_parse_tag(text, letter):
    assert parse_tag(text) == letter


def test_parse_scale():
    assert parse_scale("Massstab 1:200") == 200
    assert parse_scale("1:100 (A3)") == 100
    assert parse_scale("1:1000") is None
    assert parse_scale("Plan-Nr. 2082-9385") is None


@pytest.mark.parametrize(
    ("text", "prefix", "number"),
    [("4.Q2", "4", "4.02"), ("3:38", "3", "3.38"), ("338", "3", "3.38"), ("4.10.", "4", "4.10"),
     ("1,.10", "1", "1.10"), ("U1.O3", "U1", "U1.03"), ("01.03", "U1", "U1.03"),
     ("1016", "10", "10.16"), ("15.61", "1", None), ("5.38", "3", None), ("7.54 m²", "7", None),
     ("Bad", "1", None), ("1.03", "U1", None)],
)
def test_repair_number(text, prefix, number):
    assert repair_number(text, prefix) == number


def test_snap_name():
    assert snap_name("Kuche") == "Küche"
    assert snap_name("Sejour/repas") == "Séjour/repas"
    assert snap_name("Chambre a coucher") == "Chambre à coucher"
    assert snap_name("Treppenhous") == "Treppenhaus"
    assert snap_name("Reduit") == "Reduit"
    assert snap_name("Réduit") == "Réduit"
    assert snap_name("Kiiche") == "Küche"
    assert snap_name("Xylophon") is None
    assert snap_name("II") is None


def test_snap_name_within_the_plan_language_is_lenient():
    assert plan_language(["Zimmer", "Kuche", "Loggia", "2.01"]) == "de"
    assert plan_language(["Loggia", "Bagno", "Stanza", "Bad"]) == "it"
    assert plan_language(["Loggia", "2.01"]) is None
    assert snap_name("Bedl") is None
    assert snap_name("Bedl", "de") == "Bad"
    assert snap_name("Btil", "fr") == "Bain"
    assert snap_name("Bain", "it") == "Bagno"
    assert snap_name("Reduit", "fr") == "Réduit"
    assert snap_name("Xq", "de") is None


def test_merge_fragments_joins_a_split_number():
    merged = merge_fragments([line("4.", 100, 50), line("24", 116, 50), line("Bad", 300, 50)])
    assert sorted(l.text for l in merged) == ["4.24", "Bad"]


def test_build_table_groups_the_three_lines_of_a_label():
    lines = (
        label(100, 100, "Zimmer", "2.01", "15.61 m²")
        + label(300, 100, "Kuche", "2.02", "12.05 m2")
        + label(500, 100, "2.03")
        + [line("Whg. A", 100, 60)]
    )
    labels = to_plan_labels(build_table(lines), scale=100)
    rooms = {room.number: room for room in labels.rooms}
    assert set(rooms) == {"2.01", "2.02", "2.03"}
    assert (rooms["2.01"].name, rooms["2.01"].printed_area) == ("Zimmer", 15.61)
    assert (rooms["2.02"].name, rooms["2.02"].printed_area) == ("Küche", 12.05)
    assert (rooms["2.03"].name, rooms["2.03"].printed_area) == (None, None)
    assert labels.tags == {"A": (100, 60)}
    assert labels.missing_numbers == []
    assert len(labels.text_boxes) == 8
    assert labels.scale == 100


def test_seed_lies_between_name_and_number_or_just_above_the_number():
    lines = label(100, 100, "Bad", "1.01") + label(300, 100, "1.02") + label(500, 100, "Bad", "1.03")
    rooms = {room.number: room for room in to_plan_labels(build_table(lines), None).rooms}
    assert rooms["1.01"].number_xy == (100, 114)
    assert rooms["1.01"].seed == (100, 107)
    x, y = rooms["1.02"].seed
    assert x == 300 and 88 < y < 94


def test_unitless_area_below_a_number_is_an_area_even_on_a_matching_floor():
    lines = label(100, 100, "Bad", "4.01", "4.96") + label(300, 100, "Bad", "4.02", "4.03 m²")
    rooms = {room.number: room for room in to_plan_labels(build_table(lines), None).rooms}
    assert set(rooms) == {"4.01", "4.02"}
    assert rooms["4.01"].printed_area == 4.96


def test_sequence_gaps_are_reported():
    lines = label(100, 100, "Bad", "3.01") + label(300, 100, "Bad", "3.02") + label(500, 100, "Bad", "3.05")
    assert to_plan_labels(build_table(lines), None).missing_numbers == ["3.03", "3.04"]


def test_number_without_its_prefix_fills_the_gap():
    lines = (
        label(100, 100, "Bad", "3.01")
        + label(300, 100, "Bad", ".02")
        + label(500, 100, "Bad", "3.03")
    )
    labels = to_plan_labels(build_table(lines), None)
    assert [room.number for room in labels.rooms] == ["3.01", "3.02", "3.03"]
    assert labels.missing_numbers == []


def test_damaged_numbers_are_repaired_with_the_plan_prefix():
    lines = (
        label(100, 100, "Bad", "U1.01")
        + label(300, 100, "Bad", "U1.02")
        + label(500, 100, "Bad", "U1.O3")
        + label(700, 100, "Bad", "U104")
    )
    labels = to_plan_labels(build_table(lines), None)
    assert sorted(room.number for room in labels.rooms) == ["U1.01", "U1.02", "U1.03", "U1.04"]


def test_number_of_another_floor_is_not_a_room():
    lines = label(100, 100, "Bad", "3.01") + label(300, 100, "Bad", "3.02") + label(500, 100, "8.11")
    labels = to_plan_labels(build_table(lines), None)
    assert [room.number for room in labels.rooms] == ["3.01", "3.02"]


def test_duplicate_number_keeps_the_more_confident_read():
    lines = label(100, 100, "Bad", "3.01") + label(300, 100, "Bad", "3.02")
    lines.append(line("3.02", 500, 114, confidence=0.6))
    labels = to_plan_labels(build_table(lines), None)
    assert [room.number_xy for room in labels.rooms if room.number == "3.02"] == [(300, 114)]


def test_misread_duplicate_takes_the_missing_number_it_resembles():
    lines = (
        label(100, 100, "Bad", "7.03")
        + label(200, 100, "Bad", "7.64")  # really 7.04
        + label(300, 100, "Bad", "7.05")
        + [line("7.64", 900, 114, confidence=0.99), line("7.63", 800, 114), line("7.65", 1000, 114)]
        + [line(f"7.{k:02d}", 100 * k, 300) for k in list(range(1, 3)) + list(range(6, 63))]
    )
    labels = to_plan_labels(build_table(lines), None)
    rooms = {room.number: room for room in labels.rooms}
    assert rooms["7.04"].number_xy == (200, 114)
    assert not rooms["7.04"].number_certain
    assert rooms["7.64"].number_xy == (900, 114)
    assert labels.missing_numbers == []


def test_line_with_most_digits_lost_takes_the_gap_next_to_it():
    lines = [line(f"3.{k:02d}", 100 * k, 100) for k in (1, 2, 3, 5, 6)]
    lines += [line("3.", 350, 104), line("3", 400, 700)]
    labels = to_plan_labels(build_table(lines), None)
    assert {room.number: room.number_xy for room in labels.rooms}["3.04"] == (350, 104)


def test_number_far_beyond_the_sequence_is_dropped():
    lines = [line(f"2.{k:02d}", 100 * k, 100) for k in range(1, 6)] + [line("2.728", 300, 300)]
    labels = to_plan_labels(build_table(lines), None)
    assert [room.number for room in labels.rooms] == ["2.01", "2.02", "2.03", "2.04", "2.05"]
    assert labels.missing_numbers == []


def test_numbers_stacked_without_names_are_all_rooms():
    lines = [line("2.01", 100, 100), line("2.02", 100, 114), line("2.03", 100, 128)]
    labels = to_plan_labels(build_table(lines), None)
    assert [room.number for room in labels.rooms] == ["2.01", "2.02", "2.03"]


def test_fixture_mistaken_for_text_is_not_a_text_box():
    lines = label(100, 100, "Bad", "3.01") + label(300, 100, "Bad", "3.02")
    lines += [line("00", 500, 100), line("1", 600, 100)]
    assert len(to_plan_labels(build_table(lines), None).text_boxes) == 4


def test_orphan_lines_are_kept_as_text_boxes():
    lines = label(100, 100, "Bad", "3.01") + label(300, 100, "Bad", "3.03") + label(500, 100, "Zimmer")
    table = build_table(lines)
    assert [orphan.text for orphan in table.orphans] == ["Zimmer"]
    assert len(to_plan_labels(table, None).text_boxes) == 5


def test_reread_replaces_a_worse_read_and_adds_new_lines():
    first = label(100, 100, "Bad", "3.01") + label(300, 100, "Zimmer", "3,0")
    rereads = [line("3.02", 301, 114, confidence=0.8), line("Bod", 100, 100, confidence=0.99),
               line("9.50 m²", 300, 128)]
    merged = merge_rereads(first, rereads, "3")
    assert sorted(l.text for l in merged) == ["3.01", "3.02", "9.50 m²", "Bad", "Zimmer"]


def test_empty_plan():
    labels = to_plan_labels(build_table([]), None)
    assert labels.rooms == [] and labels.tags == {} and labels.missing_numbers == []
