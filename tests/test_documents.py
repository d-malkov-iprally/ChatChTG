import pytest

from chatchtg.documents.library import DocumentNotFoundError, UnsupportedDocumentError
from conftest import JPEG, PNG


def test_upload_is_stored_and_read_back(library):
    record = library.add_upload(PNG, "plan.png")

    assert record.media_type == "image/png"
    assert record.size_bytes == len(PNG)
    assert record.location.startswith("cas:")
    assert library.get(record.id) == record
    assert library.read(record.id) == PNG


def test_same_content_gets_the_same_document(library):
    first = library.add_upload(PNG, "plan.png")
    second = library.add_upload(PNG, "copy.png")

    assert second == first
    assert library.count() == 1


def test_media_type_comes_from_content_not_from_the_name(library):
    assert library.add_upload(JPEG, "scan.png").media_type == "image/jpeg"
    with pytest.raises(UnsupportedDocumentError):
        library.add_upload(b"0\nSECTION\n", "drawing.png")


def test_registered_file_is_referenced_not_copied(library, data_root, tmp_path):
    path = data_root / "preview" / "plan.png"
    path.parent.mkdir()
    path.write_bytes(PNG)

    record = library.register_file(path, attributes={"source": "preview"})

    assert record.location == "ref:preview/plan.png"
    assert record.attributes == {"source": "preview"}
    assert library.read(record.id) == PNG
    assert not (tmp_path / "var" / "blobs").exists()

    assert library.register_file(path) == record
    assert library.count() == 1


def test_file_outside_the_data_root_can_be_registered(library, tmp_path):
    path = tmp_path / "elsewhere.jpg"
    path.write_bytes(JPEG)

    assert library.read(library.register_file(path).id) == JPEG


def test_patch_attributes_merges_and_removes(library):
    record = library.add_upload(PNG, "plan.png", attributes={"scale": "1:100", "language": "fi"})

    patched = library.patch_attributes(record.id, {"scale": "1:50", "language": None, "floor": 2})

    assert patched.attributes == {"scale": "1:50", "floor": 2}
    assert library.get(record.id).attributes == {"scale": "1:50", "floor": 2}


def test_unknown_document(library):
    with pytest.raises(DocumentNotFoundError):
        library.get("missing")
    with pytest.raises(DocumentNotFoundError):
        library.patch_attributes("missing", {"a": 1})


def test_list_is_paged(library):
    ids = {library.add_upload(PNG + bytes([i]), f"{i}.png").id for i in range(3)}

    assert {r.id for r in library.list(limit=2)} | {r.id for r in library.list(limit=2, offset=2)} == ids
