"""Facts that a producer derived from one drawing: rooms, objects, their geometry and measures.

The format is described in ``docs/metadata-format.md``. New fields must be optional
or have defaults, so that files written before the change still load.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

_DOCUMENT_ID = re.compile(r"[0-9a-f]{64}")
_PRODUCER = re.compile(r"[a-z0-9][a-z0-9-]*")
_REF = re.compile(r"(?P<producer>[a-z0-9][a-z0-9-]*)@(?P<version>[1-9][0-9]*)#(?P<item_id>.+)")

Point = tuple[float, float]
Box = tuple[float, float, float, float]


class ImageSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Scale(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    ratio: float | None = Field(default=None, gt=0)
    """The printed scale: 100 for 1:100."""
    pixels_per_metre: float | None = Field(default=None, gt=0)


class Item(BaseModel):
    """One thing found on the drawing. Coordinates are image pixels, origin top-left, y down."""

    # JSON has no NaN or infinity. They would be written as null, and the file would not load.
    model_config = ConfigDict(allow_inf_nan=False)

    id: str = Field(min_length=1)
    """Unique in its extraction, and the same for each run of the same producer version."""
    kind: str = Field(min_length=1)
    """``room``, ``object`` or ``text``. The set is open: skip the kinds you do not know."""
    label: str = ""
    """The object class (``door``) or the printed room name (``Küche``)."""
    box: Box
    """``(x0, y0, x1, y1)``."""
    polygon: list[Point] | None = None
    """The outline, when the producer has one."""
    holes: list[list[Point]] = Field(default_factory=list)
    """Outlines of the areas inside ``polygon`` that are not part of the item."""
    measures: dict[str, float] = Field(default_factory=dict)
    """Quantities, with the unit in the key: ``area_m2``, ``length_m``."""
    score: float | None = Field(default=None, ge=0, le=1)
    """The producer's confidence. Scores of different producers are not comparable."""
    flags: list[str] = Field(default_factory=list)
    """Warnings of the producer, for example ``open-region``."""
    attributes: dict[str, Any] = Field(default_factory=dict)
    """More facts about the item. Rooms: ``number``, ``apartment``, ``outdoor``,
    ``printed_area_m2``."""

    @field_validator("box")
    @classmethod
    def _box_is_ordered(cls, box: Box) -> Box:
        x0, y0, x1, y1 = box
        if x0 > x1 or y0 > y1:
            raise ValueError(f"box must be (x0, y0, x1, y1) with x0 <= x1 and y0 <= y1: {box}")
        return box


class ItemRef(BaseModel):
    """Points at one item of one document. The text form is ``producer@version#item_id``."""

    model_config = ConfigDict(frozen=True)

    producer: str
    version: int
    item_id: str

    @classmethod
    def parse(cls, text: str) -> ItemRef:
        match = _REF.fullmatch(text)
        if match is None:
            raise ValueError(f"Not an item reference (producer@version#item_id): {text!r}")
        return cls(
            producer=match["producer"], version=int(match["version"]), item_id=match["item_id"]
        )

    def __str__(self) -> str:
        return f"{self.producer}@{self.version}#{self.item_id}"


class Extraction(BaseModel):
    """The result of one producer version for one drawing.

    An extraction without items means that the producer ran and found nothing.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: int = Field(default=SCHEMA_VERSION, alias="schema")
    """The version of this format. It is ``schema`` in the file."""
    document_id: str
    """The SHA-256 of the drawing's bytes, the same value as ``DocumentRecord.id``."""
    producer: str
    version: int = Field(ge=1)
    """Increase when the producer can give a different result for the same drawing."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    image: ImageSize
    """The size of the image that the coordinates refer to."""
    scale: Scale | None = None
    """The scale of the drawing, when the producer found it."""
    params: dict[str, Any] = Field(default_factory=dict)
    """How the producer ran: model name, thresholds."""
    source: str | None = None
    """The dataset path that the producer read. It joins the extraction with question rows."""
    attributes: dict[str, Any] = Field(default_factory=dict)
    """Facts about the whole drawing."""
    items: list[Item] = Field(default_factory=list)

    @field_validator("document_id")
    @classmethod
    def _document_id_is_sha256(cls, value: str) -> str:
        if _DOCUMENT_ID.fullmatch(value) is None:
            raise ValueError("document_id must be a SHA-256 in lower-case hexadecimal")
        return value

    @field_validator("producer")
    @classmethod
    def _producer_is_a_name(cls, value: str) -> str:
        if _PRODUCER.fullmatch(value) is None:
            raise ValueError("producer must contain only a-z, 0-9 and '-', and not start with '-'")
        return value

    @model_validator(mode="after")
    def _item_ids_are_unique(self) -> Extraction:
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise ValueError(f"Two items have the id {item.id!r}")
            seen.add(item.id)
        return self

    def item(self, item_id: str) -> Item | None:
        return next((item for item in self.items if item.id == item_id), None)

    def ref(self, item_id: str) -> ItemRef:
        return ItemRef(producer=self.producer, version=self.version, item_id=item_id)
