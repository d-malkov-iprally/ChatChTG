# Metadata format

Facts that code derives from a drawing (rooms and their areas, detected objects) are stored in one format, so that every producer writes the same shape and every reader finds it in the same place. The reasons for the storage choice are in [ADR 0004](ADR/0004-extractions-as-json-files.md).

- A **producer** is the code that derives the facts: the area pipeline, an object detector.
- An **extraction** is the result of one producer version for one drawing. It is one file.
- An **item** is one thing found on the drawing: a room, a door.

The models and the store are in `src/chatchtg/extractions/`.

## Layout

```
var/metadata/<document_id>/<producer>@<version>.json
```

- `document_id` is the SHA-256 of the drawing's bytes in lower-case hexadecimal. It is the same value as `DocumentRecord.id`. If you have only a file, hash its bytes: `hashlib.sha256(path.read_bytes()).hexdigest()`.
- `producer` contains only `a-z`, `0-9` and `-`.
- `version` is an integer from 1. Increase it when the producer can give a different result for the same drawing. Versions are stored side by side, so an eval can compare them.
- A file with `"items": []` means that the producer ran and found nothing. No file means that the producer did not run.
- To write the same document, producer and version again replaces the file.

## Example

`var/metadata/3fa9…c1/areas@3.json`:

```json
{
  "schema": 1,
  "document_id": "3fa9…c1",
  "producer": "areas",
  "version": 3,
  "created_at": "2026-11-14T10:02:11Z",
  "image": {"width": 2480, "height": 1754},
  "scale": {"ratio": 100, "pixels_per_metre": 59.06},
  "params": {"ocr": "rapidocr"},
  "source": "data/swiss_dwellings/site1000_bldg2082_floor9385.png",
  "attributes": {"missing_numbers": ["7.18"]},
  "items": [
    {
      "id": "room-7.16",
      "kind": "room",
      "label": "Küche",
      "box": [412, 300, 688, 540],
      "polygon": [[412, 300], [688, 300], [688, 540], [412, 540]],
      "holes": [],
      "measures": {"area_m2": 11.4},
      "score": null,
      "flags": [],
      "attributes": {"number": "7.16", "apartment": "B", "outdoor": false, "printed_area_m2": 11.5}
    }
  ]
}
```

## Extraction

| Field | Type | Meaning |
|---|---|---|
| `schema` | integer | Version of this format. Now `1`. |
| `document_id` | string | SHA-256 of the drawing's bytes. |
| `producer` | string | Name of the producer. |
| `version` | integer | Version of the producer. |
| `created_at` | date and time, UTC | When the extraction was made. Default: now. |
| `image` | `{width, height}` | Size in pixels of the image that the coordinates refer to. |
| `scale` | `{ratio, pixels_per_metre}` or null | Scale of the drawing, when the producer found it. `ratio` is 100 for 1:100. Each part can be null. |
| `params` | object | How the producer ran: model name, thresholds. Free form. |
| `source` | string or null | Dataset path that the producer read. It joins the extraction with question rows. |
| `attributes` | object | Facts about the whole drawing. Free form. |
| `items` | list of items | |

Required: `document_id`, `producer`, `version`, `image`.

## Item

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique in the file. The same for each run of the same producer version, so that a stored reference stays valid. |
| `kind` | string | `room`, `object` or `text`. The set is open: a reader skips the kinds that it does not know. |
| `label` | string | The object class (`door`) or the printed room name (`Küche`). |
| `box` | `[x0, y0, x1, y1]` | Bounding box. |
| `polygon` | list of `[x, y]`, or null | Outline, when the producer has one. |
| `holes` | list of polygons | Areas inside `polygon` that are not part of the item. |
| `measures` | object of numbers | Quantities, with the unit in the key: `area_m2`, `length_m`. |
| `score` | number from 0 to 1, or null | Confidence of the producer. Scores of different producers are not comparable. |
| `flags` | list of strings | Warnings of the producer: `open-region`, `shared-region`, `no-region`. |
| `attributes` | object | More facts about the item. Rooms use `number`, `apartment`, `outdoor`, `printed_area_m2`. |

Required: `id`, `kind`, `box`.

All coordinates are pixels of the image: the origin is the top-left corner, x goes right and y goes down. Areas are in square metres.

## Reference to an item

```
<producer>@<version>#<item_id>        areas@3#room-7.16
```

A reference is valid in one document. An answer stores the references of the items that it used, and the interface finds their geometry with `ExtractionStore.resolve(document_id, reference)`.

## How to write and read

```python
from chatchtg.config import Settings
from chatchtg.extractions.models import Extraction, ImageSize, Item
from chatchtg.wiring import build_extraction_store

store = build_extraction_store(Settings.from_env())

store.put(Extraction(
    document_id=document_id,
    producer="rfdetr-large",
    version=1,
    image=ImageSize(width=width, height=height),
    items=[Item(id="door-1", kind="object", label="door", box=(10, 20, 30, 60), score=0.91)],
))

store.get(document_id, "rfdetr-large")       # the highest version, or None
store.list(document_id)                      # all extractions of the drawing
store.iter("rfdetr-large", 1)                # this producer version, for each drawing
store.resolve(document_id, "rfdetr-large@1#door-1")
```

Code that cannot import `chatchtg` (a training job) writes the JSON directly. `put` writes to a temporary file and renames it, so a reader never sees a file that is half written. Do the same.

## How the current outputs fit

| Output | In this format |
|---|---|
| Area pipeline: `RoomRow`, `RoomMeasure`, `RoomLabel` | `kind` is `room`. `label` is the room name. `measures.area_m2` is the measured area. `polygon`, `holes`, `box` and `flags` are stored as they are. `attributes` holds `number`, `apartment`, `outdoor` and `printed_area_m2`. `PlanLabels.scale` goes to `scale.ratio`. |
| `detection.Detection(label, score, box)` | `kind` is `object`. `label`, `score` and `box` are stored as they are. |
| Object detector: COCO predictions | `kind` is `object`. The COCO `bbox` is `[x, y, width, height]`: store `[x, y, x + width, y + height]`. `document_id` is the hash of the image file. |
| Room counting: OCR tokens | These stay in `var/ocr-cache`. The kind `text` is reserved for them. |

Ground truth made from the DXF files (`var/room-truth`) is not stored here. An answer must not be able to read it.

## How to change the format

- Add a field only if it is optional or has a default. Then old files still load, and old code ignores the new field.
- A change that old code cannot read increases `schema`.
