"""The named composers. To add a strategy, write a composer and list it here."""

from chatchtg.context.baseline import BaselineComposer
from chatchtg.context.composer import Composer
from chatchtg.context.room_conventions import RoomConventionsComposer

COMPOSERS: dict[str, Composer] = {c.name: c for c in (BaselineComposer(), RoomConventionsComposer())}


def get_composer(name: str) -> Composer:
    try:
        return COMPOSERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown composer {name!r}. Available: {', '.join(sorted(COMPOSERS))}"
        ) from None
