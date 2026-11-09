"""Load the original COCO POPE JSONL annotations (not POPEv2).

Source: https://github.com/RUCAIBox/POPE/tree/main/output/coco
The coco_pope_{random,popular,adversarial}.json files are JSON Lines, despite
their .json extension. Their image filenames refer to COCO val2014.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data._common import image_fields, read_jsonl, required_text, summary

VARIANTS = ("random", "popular", "adversarial")


def load_pope(
    annotations: str | Path,
    image_root: str | Path,
    *,
    variant: str = "random",
    check_images: bool = True,
) -> list[dict[str, Any]]:
    """Retain source order and qualify question IDs by variant to avoid collisions.

    Questions remain unformatted; eval will add the configured yes/no suffix.
    Repeated images are expected and share a group for later paired statistics.
    """
    if variant not in VARIANTS:
        raise ValueError(f"expected variant in {VARIANTS}")
    items = []
    seen = set()
    for line_number, row in read_jsonl(annotations):
        try:
            question_id = row.get("question_id")
            if type(question_id) is not int or question_id < 1:
                raise ValueError("question_id must be a positive integer")
            if question_id in seen:
                raise ValueError(f"duplicate question_id: {question_id}")
            label = required_text(row, "label").strip().lower()
            if label not in ("yes", "no"):
                raise ValueError("POPE label must be yes or no")
            question = required_text(row, "text")
            fields = image_fields(
                required_text(row, "image"), image_root, check_images=check_images,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise type(exc)(f"{annotations}:{line_number}: {exc}") from exc
        seen.add(question_id)
        items.append({
            "id": f"pope:{variant}:{question_id}",
            "dataset": "pope", "variant": variant,
            **fields,
            "question": question, "label": label,
        })
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, required=True, help="official JSONL file")
    parser.add_argument("--image-root", type=Path, required=True, help="COCO folder or val2014 folder")
    parser.add_argument("--variant", choices=VARIANTS, default="random")
    parser.add_argument("--no-check-images", action="store_true", help="validate annotations only")
    args = parser.parse_args()
    try:
        items = load_pope(
            args.annotations, args.image_root, variant=args.variant,
            check_images=not args.no_check_images,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({"variant": args.variant,
                      **summary(items, not args.no_check_images)}, indent=2))


if __name__ == "__main__":
    main()
