"""Baseline composer: system prompt, then the drawing, then the user's message."""

from chatchtg.context.composer import ComposerInput
from chatchtg.context.types import Context, ImagePart, TextPart

SYSTEM_PROMPT = (
    "You answer questions about one construction drawing. "
    "Use only what the drawing shows. "
    "If the question asks for a quantity, end your answer with the number and its unit."
)


class BaselineComposer:
    name = "baseline"
    version = 1

    def compose(self, request: ComposerInput) -> Context:
        return Context(
            system=SYSTEM_PROMPT,
            parts=[
                ImagePart(media_type=request.document.media_type, data=request.content),
                TextPart(text=request.message),
            ],
        )
