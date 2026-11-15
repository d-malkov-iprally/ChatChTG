import pytest

from chatchtg.areas.questions import (
    AreaQuery,
    RoomRow,
    answer_query,
    label_matches,
    parse_question,
)


@pytest.mark.parametrize(
    "text, number, label",
    [
        ("What is the net floor area of room 7.01 ('Balkon') in m²?", "7.01", "Balkon"),
        ("What is the net floor area of room 3.20 ('Cage d'escalier') in m²?", "3.20",
         "Cage d'escalier"),
        ("Wie gross ist die Nettofläche von Raum U1.11 («Reduit») in m²?", "U1.11", "Reduit"),
        ("Quelle est la surface nette de la pièce 5.10 (« Séjour/repas ») en m² ?", "5.10",
         "Séjour/repas"),
        ("Qual è la superficie netta del locale 1.02 («Camera da letto») in m²?", "1.02",
         "Camera da letto"),
    ],
)  # fmt: skip
def test_parses_a_room_question(text, number, label):
    assert parse_question(text) == AreaQuery(kind="room", number=number, label=label)


@pytest.mark.parametrize(
    "text, label",
    [
        ("What is the total net area in m² of all rooms labelled 'Schlafzimmer' on this floor?",
         "Schlafzimmer"),
        ("What is the total net area in m² of all rooms labelled 'Jardin d'hiver' on this floor?",
         "Jardin d'hiver"),
        ("Wie gross ist die gesamte Nettofläche in m² aller Räume mit der Bezeichnung "
         "«Wohnen/Essen» in diesem Geschoss?", "Wohnen/Essen"),
        ("Quelle est la surface nette totale en m² de toutes les pièces désignées « Pièce » "
         "à cet étage ?", "Pièce"),
        ("Qual è la superficie netta totale in m² di tutti i locali denominati "
         "«Camera da letto» in questo piano?", "Camera da letto"),
    ],
)  # fmt: skip
def test_parses_a_label_question(text, label):
    assert parse_question(text) == AreaQuery(kind="label", label=label)


@pytest.mark.parametrize(
    "text, letter",
    [
        ("What is the net floor area of apartment B in m², summing all its rooms except "
         "balconies, loggias, terraces and other outdoor spaces?", "B"),
        ("Wie gross ist die Nettowohnfläche von Wohnung C in m² (Summe aller Räume ohne "
         "Balkone, Loggien, Terrassen und andere Aussenräume)?", "C"),
        ("Quelle est la surface nette de l'appartement A en m² (somme de toutes ses pièces, "
         "hors balcons, loggias, terrasses et autres espaces extérieurs) ?", "A"),
        ("Qual è la superficie netta dell'appartamento K in m² (somma di tutti i locali, "
         "esclusi balconi, logge, terrazze e altri spazi esterni)?", "K"),
    ],
)  # fmt: skip
def test_parses_an_apartment_question(text, letter):
    assert parse_question(text) == AreaQuery(kind="apartment", apartment=letter)


@pytest.mark.parametrize(
    "text",
    [
        "What is the total net floor area in m² of all numbered indoor rooms on this floor "
        "(excluding balconies, loggias, terraces and other outdoor spaces, shafts and voids)?",
        "Wie gross ist die gesamte Nettogeschossfläche in m² aller nummerierten Innenräume "
        "dieses Geschosses (ohne Balkone, Loggien, Terrassen und andere Aussenräume, Schächte "
        "und Lufträume)?",
        "Quelle est la surface nette totale en m² de toutes les pièces intérieures numérotées "
        "de cet étage (hors balcons, loggias, terrasses et autres espaces extérieurs, gaines "
        "et vides) ?",
        "Qual è la superficie netta totale in m² di tutti i locali interni numerati di questo "
        "piano (esclusi balconi, logge, terrazze e altri spazi esterni, cavedi e vuoti)?",
    ],
)
def test_parses_a_floor_question(text):
    assert parse_question(text) == AreaQuery(kind="floor")


def test_other_questions_are_not_parsed():
    assert parse_question("How many rooms does apartment B have?") is None
    assert parse_question("What is the scale of this drawing?") is None


ROOMS = [
    RoomRow("1.01", "Zimmer", 12.504, "A", False),
    RoomRow("1.02", "Zimmer", 10.004, "A", False),
    RoomRow("1.03", "Badezimmer", 4.0, "A", False),
    RoomRow("1.04", "Balkon", 6.0, "A", True),
    RoomRow("1.05", "Zimmer", 20.0, "B", False),
    RoomRow("1.06", None, 3.0, "B", True),
    RoomRow("1.07", "Treppenhaus", 15.0, None, False),
    RoomRow("1.08", "Lift", 2.5, None, False),
]


def test_room_is_found_by_its_number():
    answer = answer_query(AreaQuery("room", number="1.03", label="Bad"), ROOMS)
    assert answer.value == 4.0
    assert [room.number for room in answer.rooms] == ["1.03"]
    assert answer.note


def test_missing_room_gives_no_value():
    answer = answer_query(AreaQuery("room", number="9.99"), ROOMS)
    assert answer.value is None
    assert answer.rooms == []
    assert answer.missing == ["9.99"]


def test_room_read_twice_uses_the_reading_with_an_area():
    rooms = [RoomRow("2.01", "Bad", None, None, False), RoomRow("2.01", "Bad", 5.0, None, False)]
    assert answer_query(AreaQuery("room", number="2.01"), rooms).value == 5.0


def test_label_total_adds_unrounded_areas_of_the_exact_label():
    answer = answer_query(AreaQuery("label", label="Zimmer"), ROOMS)
    assert answer.value == pytest.approx(42.508)
    assert [room.number for room in answer.rooms] == ["1.01", "1.02", "1.05"]


def test_label_total_counts_outdoor_rooms():
    assert answer_query(AreaQuery("label", label="Balkon"), ROOMS).value == 6.0


def test_unknown_label_gives_no_value():
    assert answer_query(AreaQuery("label", label="Küche"), ROOMS).value is None


def test_apartment_total_leaves_out_outdoor_rooms():
    answer = answer_query(AreaQuery("apartment", apartment="A"), ROOMS)
    assert answer.value == pytest.approx(26.508)
    assert [room.number for room in answer.rooms] == ["1.01", "1.02", "1.03"]
    assert answer_query(AreaQuery("apartment", apartment="B"), ROOMS).value == 20.0
    assert answer_query(AreaQuery("apartment", apartment="C"), ROOMS).value is None


def test_floor_total_keeps_stairs_and_lifts_and_leaves_out_outdoor_rooms():
    answer = answer_query(AreaQuery("floor"), ROOMS)
    assert answer.value == pytest.approx(64.008)
    assert {"1.07", "1.08"} <= {room.number for room in answer.rooms}
    assert not any(room.outdoor for room in answer.rooms)


def test_room_without_area_gives_no_value_and_is_listed():
    rooms = [*ROOMS, RoomRow("1.09", "Zimmer", None, "B", False)]
    for query in (
        AreaQuery("label", label="Zimmer"),
        AreaQuery("apartment", apartment="B"),
        AreaQuery("floor"),
    ):
        answer = answer_query(query, rooms)
        assert answer.value is None
        assert answer.missing == ["1.09"]
    assert answer_query(AreaQuery("apartment", apartment="A"), rooms).value is not None


@pytest.mark.parametrize(
    "name, label",
    [
        ("Zimmer", "Zimmer"),
        ("ZIMMER", "Zimmer"),
        ("Zimrner", "Zimmer"),
        ("Zirnmer", "Zimmer"),
        ("Kuche", "Küche"),
        ("Sejour / repas", "Séjour/repas"),
        ("Reduit", "Réduit"),
        ("Cage d’escalier", "Cage d'escalier"),
        ("Wohnen/Esscn", "Wohnen/Essen"),
        ("8ad", "Bad"),
    ],
)
def test_label_matches_despite_accents_case_and_ocr_slips(name, label):
    assert label_matches(name, label)


@pytest.mark.parametrize(
    "name, label",
    [
        ("Badezimmer", "Zimmer"),
        ("Schlafzimmer", "Zimmer"),
        ("Zimmer", "Schlafzimmer"),
        ("Balcon", "Balkon"),
        ("Balcone", "Balcon"),
        ("Halle", "Hall"),
        ("Locale", "Local"),
        ("Wohnen/Essen", "Wohnen"),
        ("Küche/Essen", "Küche"),
        ("Bain", "Bad"),
        ("Lift", "Loggia"),
        (None, "Zimmer"),
        ("", "Zimmer"),
    ],
)
def test_label_does_not_match_another_room_name(name, label):
    assert not label_matches(name, label)


SHARED = [
    RoomRow("1.01", "Zimmer", 15.0, "A", False, region="1.01"),
    RoomRow("1.02", "Korridor", None, "A", False, region="1.01"),
    RoomRow("1.03", "Bad", 5.0, "A", False),
]


def test_rooms_measured_as_one_region_count_once_in_a_total():
    answer = answer_query(AreaQuery(kind="floor"), SHARED)

    assert answer.value == 20.0
    assert not answer.missing


@pytest.mark.parametrize(
    "query",
    [
        AreaQuery(kind="room", number="1.01"),
        AreaQuery(kind="room", number="1.02"),
        AreaQuery(kind="label", label="Zimmer"),
    ],
)
def test_a_part_of_a_shared_region_has_no_area_of_its_own(query):
    assert answer_query(query, SHARED).value is None


@pytest.mark.parametrize(
    "text",
    [
        "What is the net floor area of apartment A and B?",
        "What is the net floor area of room U1.03 and room U1.04?",
        "Wie gross ist die Nettofläche von Wohnung A und B?",
    ],
)
def test_a_question_about_two_rooms_or_apartments_is_not_read_as_one(text):
    assert parse_question(text) is None
