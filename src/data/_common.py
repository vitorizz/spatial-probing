"""Standard-library helpers for the COCO-backed evaluation datasets."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

COCO_NAME = re.compile(r"(?:COCO_(train2014|val2014|train2017|val2017)_)?(\d{12})\.jpg")
COCO_SPLITS = {"train2014", "val2014", "train2017", "val2017"}


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read JSON objects, preserving source line numbers for error messages."""
    found = False
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            found = True
            yield line_number, row
    if not found:
        raise ValueError(f"{path}: no annotation records")


def required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} must be a nonempty string")
    return value


def image_fields(
    filename: str,
    image_root: str | Path,
    *,
    image_link: str | None = None,
    check_images: bool = True,
) -> dict[str, Any]:
    """Resolve exact COCO filenames in a flat folder or a root of split folders.

    VSR supplies the COCO split in image_link; POPE supplies it in the filename.
    Never substitute a different COCO year's filename when an image is missing.
    """
    match = COCO_NAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"unrecognized COCO filename: {filename!r}")
    split, digits = match.groups()
    if image_link is not None:
        parts = urlsplit(image_link)
        link_path = Path(parts.path)
        if (
            parts.scheme not in {"http", "https"}
            or parts.hostname != "images.cocodataset.org"
            or link_path.name != filename
            or link_path.parent.name not in COCO_SPLITS
        ):
            raise ValueError(f"image_link does not identify the COCO image: {image_link!r}")
        linked_split = link_path.parent.name
        if split is not None and split != linked_split:
            raise ValueError("image filename and image_link have different COCO splits")
        split = linked_split

    root = Path(image_root)
    flat = root / filename
    nested = root / split / filename if split and root.name != split else flat
    path = flat if flat.is_file() else nested
    if check_images and not path.is_file():
        raise FileNotFoundError(f"COCO image {filename!r} not found under {root}")
    image_id = int(digits)
    return {"image_path": str(path), "coco_image_id": image_id, "group": str(image_id)}


def summary(items: list[dict[str, Any]], check_images: bool) -> dict[str, Any]:
    return {
        "n": len(items),
        "n_images": len({item["coco_image_id"] for item in items}),
        "labels": dict(Counter(item["label"] for item in items)),
        "checked_image_paths": check_images,
    }
