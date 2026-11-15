"""Located composer: for a count question, also ask for the position of each counted item.

Every other question gets exactly the baseline context. The request for positions is the
provider-neutral ``Context.locations``; the LLM adapter decides how the model returns them.
"""

import re

from chatchtg.context.baseline import SYSTEM_PROMPT, BaselineComposer
from chatchtg.context.composer import ComposerInput
from chatchtg.context.types import Context, ImagePart, LocationRequest, TextPart

# The counting phrases of the five languages of the benchmark. Every count question of the
# floor-plan benchmark starts with one of them, and no area question does.
_COUNT = re.compile(r"^\s*(how many|kuinka monta|wie viele|combien|quant[ie])\b", re.IGNORECASE)

# The text is part of the version: any change needs a new version number.
LOCATE_PROMPT = (
    "The question asks for a count. List every item that you count: a short label and the "
    "position of the centre of the item, in absolute pixel coordinates of the image, with x "
    "to the right and y down from the top-left corner. Count each item once. Then give the "
    "answer in one or two sentences, in the language of the question. The number in the "
    "answer must equal the number of items that you listed."
)

_IMAGE_TYPES = frozenset({"image/png", "image/jpeg"})


def asks_for_count(message: str) -> bool:
    """True for a question that starts with a counting phrase."""
    return _COUNT.match(message) is not None


class LocatedComposer:
    name = "located"
    version = 1

    def compose(self, request: ComposerInput) -> Context:
        if not asks_for_count(request.message) or request.document.media_type not in _IMAGE_TYPES:
            return BaselineComposer().compose(request)
        return Context(
            system=f"{SYSTEM_PROMPT} {LOCATE_PROMPT}",
            parts=[
                ImagePart(media_type=request.document.media_type, data=request.content),
                TextPart(text=request.message),
            ],
            locations=LocationRequest(),
        )
