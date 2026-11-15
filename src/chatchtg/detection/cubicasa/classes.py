"""The fixed classes of the CubiCasa5K model. This module needs no torch."""

from __future__ import annotations

# The index of a name is the class number of the model. None is a class that is not shown.
ICON_CLASSES = [
    None,  # no icon
    "window",
    "door",
    "closet",
    "electrical appliance",
    "toilet",
    "sink",
    "sauna bench",
    "fire place",
    "bathtub",
    "chimney",
]
ROOM_CLASSES = [
    None,  # background
    "outdoor",
    None,  # wall
    "kitchen",
    "living room",
    "bed room",
    "bath",
    "entry",
    None,  # railing
    "storage",
    "garage",
    None,  # undefined
]
CLASSES = tuple(name for name in ICON_CLASSES + ROOM_CLASSES if name)

# Other words for a class, without spaces.
ALIASES = {
    "appliance": "electrical appliance",
    "balcony": "outdoor",
    "bathroom": "bath",
    "fireplace": "fire place",
    "sauna": "sauna bench",
    "washbasin": "sink",
    "wardrobe": "closet",
    "wc": "toilet",
}


def canonical_label(label: str) -> str | None:
    """The class that ``label`` names, or None. "Bedroom" and "bed room" name the same class."""
    compact = label.lower().replace(" ", "").replace("-", "")
    if compact in ALIASES:
        return ALIASES[compact]
    for name in CLASSES:
        if compact in (name.replace(" ", ""), name.replace(" ", "") + "s"):
            return name
    return None
