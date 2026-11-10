"""Build COCO spatial training questions.

Usage:
    python -m src.data.coco_spatial exclude
    python -m src.data.coco_spatial pairs --split val2017
    python -m src.data.coco_spatial download --split val2017
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import combinations
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.common import CONFIG_DIR, data_dir, load_config, save_config, write_json

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


def annotations_file(split: str, url: str) -> Path:
    """Path to instances_{split}.json, downloading and extracting the zip if needed."""
    coco_dir = data_dir() / "coco"
    path = coco_dir / "annotations" / f"instances_{split}.json"
    if path.exists():
        return path
    coco_dir.mkdir(parents=True, exist_ok=True)
    zip_path = coco_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"downloading {url}")
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if Path(name).name.startswith("instances_"):
                z.extract(name, coco_dir)
    return path


def lateral_relation(a: dict, b: dict, width: int, cfg: dict) -> tuple[str | None, float]:
    """Relation of box a to box b ("left"/"right"), or None if unclear; plus the centre gap."""
    ax, _, aw, _ = a["bbox"]
    bx, _, bw, _ = b["bbox"]
    acx, bcx = (ax + aw / 2) / width, (bx + bw / 2) / width
    gap = abs(acx - bcx)
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    clear = gap >= cfg["lateral_min_gap"] and overlap <= cfg["lateral_max_overlap"] * min(aw, bw)
    if not clear:
        return None, gap
    return ("left" if acx < bcx else "right"), gap


def build_pairs(split: str, cfg: dict) -> None:
    from pycocotools.coco import COCO

    pcfg = cfg["pairs"]
    coco = COCO(str(annotations_file(split, cfg["annotations_url"])))
    names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    excluded = load_exclusion_list()
    rng = random.Random(cfg["seed"])

    out_dir = data_dir() / "coco_spatial"
    out_path = out_dir / f"pairs_{split}.jsonl"
    n_images = n_pairs = n_lateral = n_excluded = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for img_id in tqdm(sorted(coco.getImgIds()), desc=f"pairs {split}"):
            if img_id in excluded:
                n_excluded += 1
                continue
            img = coco.loadImgs(img_id)[0]
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)) if not a["iscrowd"]]
            counts = Counter(a["category_id"] for a in anns)
            min_area = pcfg["min_area_frac"] * img["width"] * img["height"]
            kept = [a for a in anns if counts[a["category_id"]] == 1 and a["area"] >= min_area]

            pairs = list(combinations(kept, 2))
            if not pairs:
                continue
            pairs = rng.sample(pairs, min(len(pairs), pcfg["max_pairs_per_image"]))

            n_images += 1
            for a, b in pairs:
                relation, gap = lateral_relation(a, b, img["width"], pcfg)
                n_pairs += 1
                n_lateral += relation is not None
                f.write(json.dumps({
                    "coco_image_id": img_id,
                    "file_name": img["file_name"],
                    "coco_url": img["coco_url"],
                    "width": img["width"],
                    "height": img["height"],
                    "a": {"ann_id": a["id"], "name": names[a["category_id"]], "bbox": a["bbox"]},
                    "b": {"ann_id": b["id"], "name": names[b["category_id"]], "bbox": b["bbox"]},
                    "lateral": relation,
                    "cx_gap": round(gap, 4),
                }) + "\n")

    save_config(cfg, out_dir, f"config_pairs_{split}.yaml")
    print(f"{split}: {n_images} images with pairs, {n_pairs} pairs, "
          f"{n_lateral} with a clear left/right, {n_excluded} images excluded -> {out_path}")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def image_dir(split: str) -> Path:
    return data_dir() / "coco_images" / split


def download_one(url: str, target: Path, max_side: int, quality: int, retries: int = 3) -> None:
    """Download, shrink so the long side is at most max_side, save as JPEG.
    Writes to a temp file first so an interrupted save never looks finished."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                img = Image.open(BytesIO(r.read())).convert("RGB")
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            tmp = target.with_suffix(".part")
            img.save(tmp, "JPEG", quality=quality)
            tmp.replace(target)
            return
        except Exception:
            if attempt == retries - 1:
                raise


def download_images(split: str, cfg: dict) -> None:
    icfg = cfg["images"]
    pairs = read_jsonl(data_dir() / "coco_spatial" / f"pairs_{split}.jsonl")
    urls = {p["file_name"]: p["coco_url"] for p in pairs}
    out_dir = image_dir(split)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = {name: url for name, url in urls.items() if not (out_dir / name).exists()}
    print(f"{split}: {len(urls)} images needed, {len(urls) - len(todo)} already on disk")

    failed = []
    with ThreadPoolExecutor(icfg["workers"]) as pool:
        futures = {
            pool.submit(download_one, url, out_dir / name, icfg["max_side"], icfg["jpeg_quality"]): name
            for name, url in todo.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"download {split}"):
            if fut.exception() is not None:
                failed.append(futures[fut])

    print(f"done: {len(urls) - len(failed)} images in {out_dir}, {len(failed)} failed")
    if failed:
        print("rerun the command to retry:", failed[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DIR / "coco_spatial.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("exclude", help="save COCO ids used by VSR and POPE")
    p = sub.add_parser("pairs", help="pick object pairs and left/right labels from annotations")
    p.add_argument("--split", default="val2017", choices=["val2017", "train2017"])
    d = sub.add_parser("download", help="download and shrink the images that have pairs")
    d.add_argument("--split", default="val2017", choices=["val2017", "train2017"])
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "exclude":
        build_exclusion_list()
    elif args.cmd == "pairs":
        build_pairs(args.split, cfg)
    elif args.cmd == "download":
        download_images(args.split, cfg)


if __name__ == "__main__":
    main()
