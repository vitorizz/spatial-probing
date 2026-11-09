"""Load official VSR JSONL annotations without downloading data or loading a model.

Source: https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot
The random variant is at cambridgeltl/vsr_random. Each has train/dev/test.jsonl.
Images are separate COCO files; image_link identifies their original split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data._common import image_fields, read_jsonl, required_text, summary

VARIANTS = ("random", "zeroshot")
SPLITS = ("train", "dev", "test")


def load_vsr(
    annotations: str | Path,
    image_root: str | Path,
    *,
    variant: str = "zeroshot",
    split: str = "test",
    check_images: bool = True,
) -> list[dict[str, Any]]:
    """Normalize labels to yes/no, retaining every statement in source order.

    IDs include variant, split and the one-based record index (ignoring blank
    lines). Use the same annotation file for baseline and adapter evaluation.
    Captions remain unformatted so eval can apply configs/base.yaml's prompt.
    """
    if variant not in VARIANTS or split not in SPLITS:
        raise ValueError(f"expected variant in {VARIANTS} and split in {SPLITS}")
    items = []
    for index, (line_number, row) in enumerate(read_jsonl(annotations), start=1):
        try:
            label = row.get("label")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("VSR label must be integer 0 or 1")
            caption = required_text(row, "caption")
            relation = required_text(row, "relation")
            fields = image_fields(
                required_text(row, "image"), image_root,
                image_link=required_text(row, "image_link"), check_images=check_images,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise type(exc)(f"{annotations}:{line_number}: {exc}") from exc
        items.append({
            "id": f"vsr:{variant}:{split}:{index}",
            "dataset": "vsr", "variant": variant, "split": split,
            **fields,
            "caption": caption, "relation": relation,
            "label": "yes" if label == 1 else "no",
        })
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, required=True, help="official JSONL file")
    parser.add_argument("--image-root", type=Path, required=True, help="COCO folder or split folder")
    parser.add_argument("--variant", choices=VARIANTS, default="zeroshot")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--no-check-images", action="store_true", help="validate annotations only")
    args = parser.parse_args()
    try:
        items = load_vsr(
            args.annotations, args.image_root, variant=args.variant, split=args.split,
            check_images=not args.no_check_images,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({"variant": args.variant, "split": args.split,
                      **summary(items, not args.no_check_images)}, indent=2))


if __name__ == "__main__":
    main()
