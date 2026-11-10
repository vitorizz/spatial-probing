"""Download and parse What'sUp Controlled Images B (the `controlled_clevr` folder).

    python -m src.data.whatsup            # download if missing, then run the checks

Items are dicts with id, image_path, o1, o2, label (left/right/front/behind) and
group (the sorted object pair, used by GroupKFold and the bootstrap).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

from src.common import data_dir

ANNOT_ID = "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z"
IMAGES_ID = "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG"
ANNOT_FILE = "controlled_clevr_dataset.json"
IMAGES_TAR = "controlled_clevr.tar.gz"
IMAGES_DIR = "controlled_clevr"

RELATION_TOKENS = {"left_of": "left", "right_of": "right", "in-front_of": "front", "behind": "behind"}
LABELS = ["left", "right", "front", "behind"]
CAPTION_PHRASES = {
    "left": "to the left of",
    "right": "to the right of",
    "front": "in front of",
    "behind": "behind",
}


def download(root: str | Path | None = None) -> Path:
    """Fetch annotations and images into `root` (default data_dir()) if missing."""
    root = Path(root) if root else data_dir()
    root.mkdir(parents=True, exist_ok=True)
    annot = root / ANNOT_FILE
    if not annot.exists():
        _gdown(ANNOT_ID, annot)
    img_dir = root / IMAGES_DIR
    if not img_dir.is_dir() or not any(img_dir.glob("*.jpeg")):
        tar = root / IMAGES_TAR
        if not tar.exists():
            _gdown(IMAGES_ID, tar)
        with tarfile.open(tar) as tf:
            members = [m for m in tf.getmembers() if m.name.startswith(IMAGES_DIR + "/")]
            for m in members:  # refuse absolute paths and ../ escapes
                assert not m.name.startswith("/") and ".." not in Path(m.name).parts, m.name
            tf.extractall(root, members=members)
    return root


def _gdown(file_id: str, out: Path) -> None:
    # gdown >= 5 takes the id positionally (the spec's `--id` flag was removed).
    subprocess.run(["gdown", file_id, "-O", str(out)], check=True)


def parse_filename(name: str) -> tuple[str, str, str]:
    """`{o1}_{relation}_{o2}.jpeg` -> (o1, label, o2)."""
    stem = Path(name).stem
    o1, rest = stem.split("_", 1)
    rel, o2 = rest.rsplit("_", 1)
    if rel not in RELATION_TOKENS:
        raise ValueError(f"unknown relation {rel!r} in {name}")
    return o1, RELATION_TOKENS[rel], o2


def group_id(o1: str, o2: str) -> str:
    return "|".join(sorted((o1, o2)))


def load_whatsup(root: str | Path | None = None) -> list[dict]:
    """All items, sorted by id so every run sees the same order."""
    root = Path(root) if root else data_dir()
    items = []
    for path in sorted((root / IMAGES_DIR).glob("*.jpeg")):
        o1, label, o2 = parse_filename(path.name)
        items.append(
            {
                "id": path.stem,
                "image_path": str(path),
                "o1": o1,
                "o2": o2,
                "label": label,
                "group": group_id(o1, o2),
            }
        )
    return items


def check(items: list[dict], root: str | Path | None = None) -> dict:
    """Spec checks: 408 items, 102 per class, every image opens. Also checks the
    filename relation against the first caption in the annotation JSON."""
    from PIL import Image

    root = Path(root) if root else data_dir()
    counts = Counter(it["label"] for it in items)
    assert len(items) == 408, f"expected 408 items, got {len(items)}"
    assert all(counts[c] == 102 for c in LABELS), f"class counts {dict(counts)}"
    for it in items:
        with Image.open(it["image_path"]) as im:
            im.verify()

    with open(root / ANNOT_FILE, encoding="utf-8") as f:
        annot = {Path(a["image_path"]).stem: a["caption_options"][0] for a in json.load(f)}
    assert set(annot) == {it["id"] for it in items}, "annotation and image ids differ"
    for it in items:
        cap = annot[it["id"]]
        want = f"{it['o1']} {CAPTION_PHRASES[it['label']]} a"
        assert want in cap.lower(), f"{it['id']}: caption {cap!r} disagrees with filename"

    groups = Counter(it["group"] for it in items)
    return {
        "n": len(items),
        "per_class": dict(counts),
        "n_groups": len(groups),
        "items_per_group": dict(Counter(groups.values())),
        "objects": sorted({it["o1"] for it in items} | {it["o2"] for it in items}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="data folder (default: data_dir())")
    args = ap.parse_args()
    root = download(args.root)
    stats = check(load_whatsup(root), root)
    print(json.dumps(stats, indent=2))
    print("What'sUp Controlled B: all checks passed")


if __name__ == "__main__":
    main()
