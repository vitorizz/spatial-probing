"""Download and validate selected VSR/POPE annotations and referenced COCO images."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from http.client import HTTPException
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.data._common import COCO_NAME, read_jsonl, summary
from src.data.pope import VARIANTS as POPE_VARIANTS, load_pope
from src.data.vsr import SPLITS, VARIANTS as VSR_VARIANTS, load_vsr

VSR_URL = "https://huggingface.co/datasets/cambridgeltl/vsr_{variant}/resolve/main/{split}.jsonl"
POPE_URL = "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_{variant}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError("expected a COCO JPEG image")
        image.load()  # Decode pixels too: merely opening a JPEG can miss truncation.
        width, height = image.size
    return {"sha256": _sha256(path), "width": width, "height": height}


def _fetch(url: str, destination: Path, validate: Callable[[Path], Any]) -> tuple[Any, bool]:
    """Reuse validated cache files; publish a new file only after full validation.

    Temporary files live beside the destination so replace is atomic. Failures
    remove only this attempt's temporary file. No existing cache file is replaced.
    """
    if destination.exists():
        try:
            return validate(destination), False
        except (OSError, ValueError) as exc:
            raise ValueError(f"cached file failed validation: {destination}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=30) as response:
                shutil.copyfileobj(response, stream)
                length = response.headers.get("Content-Length")
            stream.flush()
        if length is not None and temporary.stat().st_size != int(length):
            raise ValueError(f"incomplete download: {url}")
        result = validate(temporary)
        temporary.replace(destination)
        return result, True
    except (OSError, ValueError, HTTPException) as exc:
        raise ValueError(f"could not prepare {destination} from {url}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_dataset(
    dataset: str, root: str | Path, *, variant: str | None = None,
    split: str | None = None, annotations_only: bool = False, limit: int | None = None,
) -> dict[str, Any]:
    """Prepare one benchmark, preserving source bytes and exact COCO filenames.

    Reruns reuse validated files, so interrupted preparation resumes per file.
    A saved manifest pins annotation/image hashes for subsequent runs of the
    same selection. Limited and annotation-only manifests have separate names.
    """
    if dataset not in {"vsr", "pope"}:
        raise ValueError("dataset must be vsr or pope")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    root = Path(root).resolve()
    if dataset == "vsr":
        variant, split = variant or "zeroshot", split or "test"
        if variant not in VSR_VARIANTS or split not in SPLITS:
            raise ValueError(f"VSR requires variant in {VSR_VARIANTS} and split in {SPLITS}")
        annotations = root / "vsr" / variant / f"{split}.jsonl"
        url = VSR_URL.format(variant=variant, split=split)
        loader = lambda path: load_vsr(path, root / "coco", variant=variant, split=split, check_images=False)
    else:
        variant = variant or "random"
        if variant not in POPE_VARIANTS or split is not None:
            raise ValueError(f"POPE requires variant in {POPE_VARIANTS} and no split")
        annotations = root / "pope" / f"coco_pope_{variant}.json"
        url = POPE_URL.format(variant=variant)
        loader = lambda path: load_pope(path, root / "coco", variant=variant, check_images=False)
    stem = "_".join(part for part in (dataset, variant, split) if part)
    stem += f"_limit{limit}" if limit is not None else ""
    stem += "_annotations" if annotations_only else ""
    manifest_path = root / "prepared" / f"{stem}.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous is not None and (
        not isinstance(previous, dict) or previous.get("schema_version") != 1
        or not isinstance(previous.get("annotations"), dict)
        or not isinstance(previous["annotations"].get("sha256"), str)
        or not isinstance(previous.get("images"), list)
        or any(not isinstance(image, dict) or not isinstance(image.get("file"), str) for image in previous["images"])
    ):
        raise ValueError(f"invalid previous preparation manifest: {manifest_path}")

    def validate_annotations(path: Path) -> list[dict[str, Any]]:
        items = loader(path)
        if dataset == "pope" and any(COCO_NAME.fullmatch(Path(item["image_path"]).name).group(1) != "val2014" for item in items):
            raise ValueError("original COCO POPE images must use val2014 filenames")
        if previous is not None and _sha256(path) != previous["annotations"]["sha256"]:
            raise ValueError("annotations changed since the previous preparation; preserve the frozen benchmark")
        return items

    items, downloaded_annotations = _fetch(url, annotations, validate_annotations)
    selected = items[:limit] if limit is not None else items
    raw = [row for _, row in read_jsonl(annotations)][:len(selected)]
    jobs = {}
    for item, record in zip(selected, raw):
        path = Path(item["image_path"])
        if dataset == "vsr":
            image_url = record["image_link"]
        else:
            coco_split = COCO_NAME.fullmatch(record["image"]).group(1)
            if coco_split != "val2014":
                raise ValueError("original COCO POPE images must use val2014 filenames")
            image_url = f"http://images.cocodataset.org/{coco_split}/{record['image']}"
        if path in jobs and jobs[path] != image_url:
            raise ValueError(f"conflicting source URLs for {path}")
        jobs[path] = image_url
    old_hashes = {image["file"]: image["sha256"] for image in previous.get("images", []) if "sha256" in image} if previous else {}
    images, n_downloaded = [], 0
    for index, (path, image_url) in enumerate(jobs.items(), start=1):
        relative = path.relative_to(root).as_posix()
        entry = {"file": relative, "url": image_url}
        if not annotations_only:
            def validate_image(candidate: Path) -> dict[str, Any]:
                info = _image_info(candidate)
                if relative in old_hashes and info["sha256"] != old_hashes[relative]:
                    raise ValueError("image bytes changed since the previous preparation")
                return info

            info, downloaded = _fetch(image_url, path, validate_image)
            entry.update(info)
            n_downloaded += int(downloaded)
            if index % 25 == 0 or index == len(jobs):
                print(f"{stem}: {index}/{len(jobs)} images validated", file=sys.stderr, flush=True)
        images.append(entry)
    manifest = {
        "schema_version": 1, "dataset": dataset, "variant": variant, "split": split,
        **summary(selected, not annotations_only), "n_available": len(items), "limit": limit,
        "images_verified": not annotations_only,
        "annotations": {"file": annotations.relative_to(root).as_posix(), "url": url, "sha256": _sha256(annotations)},
        "question_ids": [item["id"] for item in selected],
        "coco_image_ids": sorted({item["coco_image_id"] for item in selected}), "images": images,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=manifest_path.parent, delete=False) as stream:
            temp_manifest = Path(stream.name)
            json.dump(manifest, stream, indent=2, allow_nan=False)
        temp_manifest.replace(manifest_path)
    finally:
        if temp_manifest is not None:
            temp_manifest.unlink(missing_ok=True)
    return {"manifest": str(manifest_path), "annotations_downloaded": downloaded_annotations,
            "images_downloaded": n_downloaded,
            "images_cached": len(images) - n_downloaded if not annotations_only else 0,
            "n": len(selected), "n_images": len(images), "images_verified": not annotations_only}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vsr", "pope"), required=True)
    parser.add_argument("--root", type=Path, help="data folder (default: shared data_dir())")
    parser.add_argument("--variant", help="VSR: random/zeroshot; POPE: random/popular/adversarial")
    parser.add_argument("--split", help="VSR: train/dev/test (default: test)")
    parser.add_argument("--annotations-only", action="store_true", help="skip image downloads")
    parser.add_argument("--limit", type=int, help="prepare images for the first N questions (smoke run)")
    args = parser.parse_args(argv)
    try:
        if args.root is None:
            from src.common import data_dir

            args.root = data_dir()
        result = prepare_dataset(args.dataset, args.root, variant=args.variant, split=args.split,
                                 annotations_only=args.annotations_only, limit=args.limit)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
