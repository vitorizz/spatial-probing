"""Build COCO spatial training questions.

Usage:
    python -m src.data.coco_spatial exclude
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

from src.common import data_dir, write_json

VSR_DATASETS = ["cambridgeltl/vsr_random", "cambridgeltl/vsr_zeroshot"]
POPE_URL = "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_{split}.json"
POPE_SPLITS = ["random", "popular", "adversarial"]

COCO_ID = re.compile(r"(\d{12})")


def coco_id(filename: str) -> int:
    """'000000296471.jpg' or 'COCO_val2014_000000310196.jpg' -> COCO image id."""
    m = COCO_ID.search(filename)
    if m is None:
        raise ValueError(f"no COCO id in {filename!r}")
    return int(m.group(1))


def vsr_ids() -> set[int]:
    from datasets import load_dataset

    ids = set()
    for name in VSR_DATASETS:
        for split in load_dataset(name).values():
            ids.update(coco_id(f) for f in split["image"])
    return ids


def pope_ids() -> set[int]:
    ids = set()
    for split in POPE_SPLITS:
        with urllib.request.urlopen(POPE_URL.format(split=split)) as r:
            for line in r.read().decode("utf-8").splitlines():
                if line.strip():
                    ids.add(coco_id(json.loads(line)["image"]))
    return ids


def build_exclusion_list() -> None:
    vsr, pope = vsr_ids(), pope_ids()
    ids = sorted(vsr | pope)
    out = write_json(
        {"counts": {"vsr": len(vsr), "pope": len(pope), "total": len(ids)}, "ids": ids},
        data_dir() / "exclude_ids.json",
    )
    print(f"VSR: {len(vsr)} images, POPE: {len(pope)} images, total: {len(ids)} -> {out}")


def load_exclusion_list() -> set[int]:
    with open(data_dir() / "exclude_ids.json", encoding="utf-8") as f:
        return set(json.load(f)["ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("exclude", help="save COCO ids used by VSR and POPE")
    args = parser.parse_args()

    if args.cmd == "exclude":
        build_exclusion_list()


if __name__ == "__main__":
    main()
