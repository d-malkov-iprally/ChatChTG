"""The named composers. To add a strategy, write a composer and list it here."""

from chatchtg.context.areas import AreaComposer
from chatchtg.context.baseline import BaselineComposer
from chatchtg.context.combined import CombinedComposer
from chatchtg.context.composer import Composer
from chatchtg.context.located import LocatedComposer
from chatchtg.context.object_hints import ObjectHintsComposer
from chatchtg.context.room_conventions import RoomConventionsComposer

_AREAS = AreaComposer()
_ROOM_CONVENTIONS = RoomConventionsComposer()

COMPOSERS: dict[str, Composer] = {
    c.name: c
    for c in (
        BaselineComposer(),
        _AREAS,
        _ROOM_CONVENTIONS,
        LocatedComposer(),
        ObjectHintsComposer(),
        CombinedComposer(areas=_AREAS, room_conventions=_ROOM_CONVENTIONS),
    )
}


def get_composer(name: str) -> Composer:
    try:
        return COMPOSERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown composer {name!r}. Available: {', '.join(sorted(COMPOSERS))}"
        ) from None
