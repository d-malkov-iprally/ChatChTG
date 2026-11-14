import json

import pytest
from pydantic import ValidationError

from chatchtg.config import Settings
from chatchtg.extractions.models import Extraction, ImageSize, Item, ItemRef, Scale
from chatchtg.extractions.store import LocalExtractionStore
from chatchtg.wiring import build_extraction_store

DOC = "a" * 64
OTHER_DOC = "b" * 64

KITCHEN = Item(
    id="room-7.16",
    kind="room",
    label="Küche",
    box=(412, 300, 688, 540),
    polygon=[(412, 300), (688, 300), (688, 540), (412, 540)],
    measures={"area_m2": 11.4},
    attributes={"number": "7.16", "apartment": "B"},
)
DOOR = Item(id="door-1", kind="object", label="door", box=(10, 20, 30, 60), score=0.91)


def extraction(producer="areas", version=1, document_id=DOC, items=(KITCHEN,)):
    return Extraction(
        document_id=document_id,
        producer=producer,
        version=version,
        image=ImageSize(width=2480, height=1754),
        scale=Scale(ratio=100, pixels_per_metre=59.06),
        items=list(items),
    )


@pytest.fixture
def store(tmp_path):
    return LocalExtractionStore(tmp_path / "metadata")


def test_put_writes_one_file_per_document_and_producer_version(store, tmp_path):
    store.put(extraction(version=3))

    path = tmp_path / "metadata" / DOC / "areas@3.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["schema"] == 1
    assert stored["items"][0]["label"] == "Küche"
    assert [p.name for p in path.parent.iterdir()] == ["areas@3.json"]


def test_get_returns_what_was_put(store):
    original = extraction()
    store.put(original)

    assert store.get(DOC, "areas", 1) == original


def test_get_without_version_returns_the_highest(store):
    for version in (2, 10, 9):
        store.put(extraction(version=version))

    assert store.get(DOC, "areas").version == 10


def test_get_returns_none_when_the_producer_did_not_run(store):
    store.put(extraction())

    assert store.get(DOC, "rfdetr-large") is None
    assert store.get(DOC, "areas", 2) is None
    assert store.get(OTHER_DOC, "areas") is None


def test_no_items_means_ran_and_found_nothing(store):
    store.put(extraction(items=()))

    assert store.get(DOC, "areas").items == []


def test_put_replaces_the_same_version(store):
    store.put(extraction(items=(KITCHEN,)))
    store.put(extraction(items=(KITCHEN, DOOR)))

    assert len(store.get(DOC, "areas", 1).items) == 2
    assert len(store.list(DOC)) == 1


def test_list_returns_all_extractions_of_a_document(store):
    store.put(extraction("rfdetr-large", 2, items=(DOOR,)))
    store.put(extraction("areas", 1))
    store.put(extraction("areas", 1, document_id=OTHER_DOC))

    assert [(e.producer, e.version) for e in store.list(DOC)] == [("areas", 1), ("rfdetr-large", 2)]
    assert store.list("c" * 64) == []


def test_iter_returns_one_producer_version_over_all_documents(store):
    store.put(extraction("areas", 1))
    store.put(extraction("areas", 1, document_id=OTHER_DOC))
    store.put(extraction("areas", 2))
    store.put(extraction("rfdetr-large", 1, items=(DOOR,)))

    assert [e.document_id for e in store.iter("areas", 1)] == [DOC, OTHER_DOC]
    assert list(LocalExtractionStore(store._root / "missing").iter("areas", 1)) == []


def test_resolve_returns_the_item_of_a_reference(store):
    store.put(extraction(version=3, items=(KITCHEN, DOOR)))

    assert store.resolve(DOC, "areas@3#room-7.16") == KITCHEN
    assert store.resolve(DOC, extraction(version=3).ref("door-1")) == DOOR
    assert store.resolve(DOC, "areas@3#room-9.99") is None
    assert store.resolve(DOC, "areas@4#room-7.16") is None


def test_item_reference_has_a_text_form():
    ref = ItemRef.parse("rfdetr-large@12#door#3")

    assert ref == ItemRef(producer="rfdetr-large", version=12, item_id="door#3")
    assert str(ref) == "rfdetr-large@12#door#3"


@pytest.mark.parametrize("text", ["areas#room-1", "areas@0#room-1", "Areas@1#room-1", "areas@1#"])
def test_item_reference_refuses_other_text(text):
    with pytest.raises(ValueError):
        ItemRef.parse(text)


@pytest.mark.parametrize(
    "change",
    [
        {"document_id": "../../etc"},
        {"document_id": "A" * 64},
        {"producer": "Areas"},
        {"producer": "a/b"},
        {"version": 0},
        {"items": [KITCHEN, KITCHEN]},
    ],
)
def test_extraction_refuses_values_that_do_not_fit_the_format(change):
    fields = extraction().model_dump() | change

    with pytest.raises(ValidationError):
        Extraction(**fields)


def test_item_refuses_a_box_that_is_not_ordered():
    with pytest.raises(ValidationError):
        Item(id="door-1", kind="object", box=(30, 20, 10, 60))


def test_a_file_with_fields_of_a_later_format_still_loads(store, tmp_path):
    store.put(extraction())
    path = tmp_path / "metadata" / DOC / "areas@1.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["schema"] = 2
    stored["added_later"] = True
    stored["items"][0]["added_later"] = True
    path.write_text(json.dumps(stored), encoding="utf-8")

    loaded = store.get(DOC, "areas", 1)

    assert loaded.schema_version == 2
    assert loaded.items == [KITCHEN]


def test_files_that_are_not_extractions_are_ignored(store, tmp_path):
    store.put(extraction())
    (tmp_path / "metadata" / DOC / "areas@1.json.abc.tmp").write_text("{", encoding="utf-8")
    (tmp_path / "metadata" / "README.txt").write_text("notes", encoding="utf-8")

    assert len(store.list(DOC)) == 1
    assert len(list(store.iter("areas", 1))) == 1


def test_the_store_is_in_the_state_folder(tmp_path):
    settings = Settings(
        data_root=tmp_path / "data", state_dir=tmp_path / "var", composer="baseline",
        llm_provider="fake",
    )

    build_extraction_store(settings).put(extraction())

    assert (tmp_path / "var" / "metadata" / DOC / "areas@1.json").is_file()
