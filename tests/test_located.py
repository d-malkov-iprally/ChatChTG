import io

from conftest import PNG
from PIL import Image

from chatchtg.context.baseline import BaselineComposer
from chatchtg.context.composer import ComposerInput
from chatchtg.context.types import Context, ImagePart, LocationRequest, TextPart
from chatchtg.llm.client import Completion, Location
from chatchtg.llm.fake import FakeLLM


def real_png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("L", (width, height), 255).save(out, format="PNG")
    return out.getvalue()


def test_image_part_reports_its_size_or_none():
    assert ImagePart(media_type="image/png", data=real_png(40, 30)).size() == (40, 30)
    assert ImagePart(media_type="image/png", data=PNG).size() is None


def test_existing_contexts_ask_for_no_locations(library):
    document = library.add_upload(PNG, "plan.png")
    context = BaselineComposer().compose(ComposerInput(document=document, content=PNG, message="Hi"))
    assert context.locations is None
    assert "locations" not in context.summary()


def test_summary_records_a_location_request():
    context = Context(system="s", parts=[TextPart(text="q")], locations=LocationRequest())
    assert context.summary()["locations"] == {"shape": "point"}


def test_fake_llm_returns_locations_only_when_asked():
    marks = [Location(label="door", point=(10, 20))]
    llm = FakeLLM(reply="There is 1 door.", locations=marks)
    plain = Context(system="s", parts=[TextPart(text="q")])
    asked = plain.model_copy(update={"locations": LocationRequest()})
    assert llm.complete(plain).locations is None
    assert llm.complete(asked).locations == marks
    assert FakeLLM().complete(asked).locations == []
    assert Completion(text="t", model="m").locations is None
