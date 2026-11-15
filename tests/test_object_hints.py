import io

from PIL import Image

from chatchtg.answering import AnswerService
from chatchtg.config import Settings
from chatchtg.context.baseline import BaselineComposer
from chatchtg.context.composer import ComposerInput, composer_reads
from chatchtg.context.object_hints import (
    HINT_SYSTEM_PROMPT,
    MAX_LONG_EDGE,
    MIN_LONG_EDGE,
    ObjectHintsComposer,
    candidates,
    marked_scale,
)
from chatchtg.context.registry import get_composer
from chatchtg.context.types import ImagePart, TextPart
from chatchtg.evals.scoring import extract_number
from chatchtg.extractions.models import Extraction, ImageSize, Item
from chatchtg.extractions.store import LocalExtractionStore
from chatchtg.llm.fake import FakeLLM
from chatchtg.objects.counter import PRODUCER, PRODUCER_VERSION, THRESHOLDS
from chatchtg.objects.questions import ObjectQuestion
from chatchtg.schemas import AnswerRequest
from chatchtg.wiring import build_answer_service

WINDOWS = "How many windows are shown on this floor plan?"
SINKS = (
    "How many sinks and wash basins (kitchen and bathroom combined; a double-bowl sink counts "
    "as one) are drawn on this floor plan?"
)


def _png(width: int = 200, height: int = 100) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(out, format="PNG")
    return out.getvalue()


def _item(item_id: str, label: str, score: float, box=(10, 10, 40, 20)) -> Item:
    return Item(id=item_id, kind="object", label=label, box=box, score=score)


def _extraction(document_id: str, items) -> Extraction:
    return Extraction(
        document_id=document_id,
        producer=PRODUCER,
        version=PRODUCER_VERSION,
        image=ImageSize(width=200, height=100),
        items=list(items),
    )


WINDOW_ITEMS = [
    _item("window-1", "window", 0.95, (10, 10, 40, 20)),
    _item("window-2", "window", 0.30, (100, 60, 130, 70)),  # doubtful: below 0.46, above 0.23
    _item("window-3", "window", 0.10, (150, 60, 180, 70)),  # too low to show
    _item("door-1", "door", 0.99, (60, 10, 80, 40)),  # another class
]


def _input(library, message, items=WINDOW_ITEMS, content=None):
    content = content or _png()
    document = library.add_upload(content, "plan.png")
    extractions = [_extraction(document.id, items)] if items is not None else []
    return ComposerInput(document=document, content=content, message=message, extractions=extractions)


def test_candidates_are_the_counted_and_the_doubtful_detections_of_the_asked_class():
    marks = candidates(_extraction("a" * 64, WINDOW_ITEMS), ObjectQuestion("window"))

    assert [(c.number, c.item.id, c.doubtful) for c in marks] == [(1, "window-1", False), (2, "window-2", True)]
    assert 0.30 < THRESHOLDS["window"]


def test_showers_and_bathtubs_are_candidates_together():
    items = [_item("shower-1", "shower", 0.9), _item("bathtub-1", "bathtub", 0.6, (50, 50, 90, 70))]
    marks = candidates(_extraction("a" * 64, items), ObjectQuestion("bath_shower"))

    # A bathtub must score 0.9 to be counted; at 0.6 it is shown as doubtful.
    assert [(c.item.label, c.doubtful) for c in marks] == [("shower", False), ("bathtub", True)]


def test_a_hinted_context_has_the_plain_and_the_marked_drawing_and_lists_the_candidates(library):
    request = _input(library, WINDOWS)

    context = ObjectHintsComposer().compose(request)

    assert context.system == HINT_SYSTEM_PROMPT
    plain, marked, hint, question = context.parts
    assert isinstance(plain, ImagePart) and plain.data == request.content
    assert isinstance(marked, ImagePart) and marked.media_type == "image/png"
    assert max(marked.size()) == MIN_LONG_EDGE  # a small drawing is enlarged for the numbers
    assert isinstance(hint, TextPart)
    assert "1: window, score 0.95, counted" in hint.text
    assert "2?: window, score 0.30, doubtful" in hint.text
    assert "0.10" not in hint.text and "door" not in hint.text
    assert "would answer 1." in hint.text
    assert question.text == WINDOWS
    assert context.locations is None and context.computed is None and context.method is None


def test_the_marks_change_the_drawing_and_leave_the_unmarked_parts_as_they_were(library):
    request = _input(library, WINDOWS, content=_png(MIN_LONG_EDGE, 700))

    marked = ObjectHintsComposer().compose(request).parts[1]

    with Image.open(io.BytesIO(marked.data)) as image:
        image = image.convert("RGB")
        assert image.size == (MIN_LONG_EDGE, 700)
        assert image.getpixel((700, 400)) == (255, 255, 255)
        # The outline of window-1 runs a few pixels outside its box, not over the symbol.
        assert image.getpixel((25, 15)) == (255, 255, 255)
        assert image.getpixel((5, 15)) != (255, 255, 255)


def test_a_large_drawing_is_reduced_to_what_the_api_takes():
    assert marked_scale(5152, 1000) == MAX_LONG_EDGE / 5152
    assert marked_scale(3000, 3000) < MAX_LONG_EDGE / 3000
    assert marked_scale(2000, 1000) == 1.0


def test_no_candidates_still_sends_the_drawing_and_says_so(library):
    context = ObjectHintsComposer().compose(_input(library, SINKS))

    assert [type(part) for part in context.parts] == [ImagePart, TextPart, TextPart]
    assert "no candidates" in context.parts[1].text


def test_other_questions_and_drawings_without_detections_get_the_baseline_context(library):
    composer = ObjectHintsComposer()
    for request in (
        _input(library, "How many bedrooms are there?"),
        _input(library, WINDOWS, items=None),
        _input(library, "Kuinka monta makuuhuonetta pohjapiirustuksessa on?"),
    ):
        assert composer.compose(request) == BaselineComposer().compose(request)


def test_the_answer_service_hands_the_stored_detections_to_the_composer(tmp_path, library):
    content = _png()
    document = library.add_upload(content, "plan.png")
    store = LocalExtractionStore(tmp_path / "var" / "metadata")
    store.put(_extraction(document.id, WINDOW_ITEMS))
    service = AnswerService(library, ObjectHintsComposer(), FakeLLM(reply="Answer: 2"), extractions=store)

    trace = service.answer_traced(AnswerRequest(document_id=document.id, message=WINDOWS))

    assert trace.context.system == HINT_SYSTEM_PROMPT
    assert sum(isinstance(part, ImagePart) for part in trace.context.parts) == 2
    assert extract_number(trace.response.text) == 2
    assert trace.response.composer == "object-hints@1"
    assert trace.response.method is None  # the LLM decided the count, not code


def test_without_a_store_or_for_a_composer_that_reads_nothing_no_extraction_is_loaded(tmp_path, library):
    class Recording(BaselineComposer):
        def compose(self, request):
            self.seen = request.extractions
            return super().compose(request)

    document = library.add_upload(_png(), "plan.png")
    store = LocalExtractionStore(tmp_path / "var" / "metadata")
    store.put(_extraction(document.id, WINDOW_ITEMS))
    composer = Recording()
    AnswerService(library, composer, FakeLLM(), extractions=store).answer(
        AnswerRequest(document_id=document.id, message=WINDOWS)
    )
    assert composer.seen == []
    hints = AnswerService(library, ObjectHintsComposer(), FakeLLM()).answer_traced(
        AnswerRequest(document_id=document.id, message=WINDOWS)
    )
    assert hints.context.system != HINT_SYSTEM_PROMPT


def test_a_delegating_composer_reads_what_its_parts_read():
    class Delegating:
        name, version = "both", 1
        parts = (ObjectHintsComposer(), BaselineComposer(), ObjectHintsComposer())

    assert composer_reads(Delegating()) == ((PRODUCER, PRODUCER_VERSION),)
    assert composer_reads(BaselineComposer()) == ()


def test_the_app_wiring_gives_the_composer_the_extraction_store(tmp_path, library):
    content = _png()
    document = library.add_upload(content, "plan.png")
    LocalExtractionStore(tmp_path / "var" / "metadata").put(_extraction(document.id, WINDOW_ITEMS))
    settings = Settings.from_env(
        {"CHATCHTG_STATE_DIR": str(tmp_path / "var"), "CHATCHTG_COMPOSER": "object-hints"}
    )

    trace = build_answer_service(settings, library).answer_traced(
        AnswerRequest(document_id=document.id, message=WINDOWS)
    )

    assert get_composer("object-hints").name == "object-hints"
    assert trace.context.system == HINT_SYSTEM_PROMPT
    assert trace.response.model == "fake"
