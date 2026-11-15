"""Combined composer: the area composer for area questions, the room conventions for the rest.

The two composers change disjoint sets of questions (area questions on Swiss plans, room
counts on Finnish plans), and each sends every other question as the baseline does. This
composer picks one of them for each question and returns that composer's context
unchanged, so it gets both gains. It holds no logic of its own beyond the choice: when a
delegate changes, this composer changes with it, and its id says which versions it uses.
"""

from __future__ import annotations

from chatchtg.areas.questions import parse_question
from chatchtg.context.composer import Composer, ComposerInput
from chatchtg.context.types import Context


class CombinedComposer:
    name = "combined"
    version = 1
    """The version of the choice only. The versions of the delegates are in ``parts``."""

    def __init__(self, areas: Composer, room_conventions: Composer) -> None:
        self._areas = areas
        self._room_conventions = room_conventions

    @property
    def parts(self) -> tuple[Composer, ...]:
        return (self._areas, self._room_conventions)

    def delegate(self, request: ComposerInput) -> Composer:
        """The composer that builds the context for ``request``."""
        if parse_question(request.message) is not None:
            return self._areas
        return self._room_conventions

    def compose(self, request: ComposerInput) -> Context:
        return self.delegate(request).compose(request)
