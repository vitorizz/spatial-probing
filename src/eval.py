"""Evaluate What'sUp, VSR or POPE using shared option scoring, with an optional adapter."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import REPO_ROOT, load_config, results_dir, set_seed, write_json
from src.data.pope import load_pope
from src.data.vsr import load_vsr
from src.stats import question_manifest_hash, summarize_predictions


def prepare_items(
    dataset: str, items: list[dict[str, Any]], cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the selected questions before loading the model, and build prompts."""
    if dataset not in {"whatsup", "vsr", "pope"}:
        raise ValueError(f"unknown dataset: {dataset}")
    options = cfg["options"]["fourway" if dataset == "whatsup" else "yesno"]
    if (
        not isinstance(options, list) or not options
        or any(not isinstance(option, str) for option in options)
        or len(set(options)) != len(options)
    ):
        raise ValueError("answer options must be unique strings; quote yes/no in YAML")
    if dataset != "whatsup" and set(options) != {"yes", "no"}:
        raise ValueError("VSR/POPE require yes/no options")
    if not items:
        raise ValueError("no evaluation questions found")
    prepared, seen = [], set()
    for item in items:
        if not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("every question must have a nonempty string id")
        if item["id"] in seen:
            raise ValueError(f"duplicate question id: {item['id']}")
        seen.add(item["id"])
        if item.get("label") not in options:
            raise ValueError(f"invalid label for {item['id']}: {item.get('label')!r}")
        if not item.get("group"):
            raise ValueError(f"missing group for {item['id']}")
        if not Path(item["image_path"]).is_file():
            raise FileNotFoundError(f"{item['id']}: missing image {item['image_path']}")
        if dataset == "whatsup":
            from src.model_utils import fourway_question

            question = fourway_question(item["o1"], item["o2"], cfg)
        elif dataset == "vsr":
            question = cfg["prompts"]["vsr"].format(caption=item["caption"])
        else:
            question = item["question"] + cfg["prompts"]["pope_suffix"]
        prepared.append({**item, "dataset": dataset, "question": question})
    return prepared, options


def _code_version() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return {"revision": revision, "tracked_changes": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "tracked_changes": None}


def _environment() -> dict[str, Any]:
    env = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "Pillow", "numpy", "PyYAML"):
        try:
            env[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            env[package] = None
    return env


def evaluate(
    dataset: str,
    items: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    out: str | Path,
    model_name: str = "base",
    adapter: str | None = None,
    variant: str | None = None,
    split: str | None = None,
    annotations: str | Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score one dataset and write a CSV plus a JSON summary after a successful run.

    Uses Person A's model loading, input building, token selection and scoring.
    Applies vision FP32 after loading, keeping the language backbone in FP16.
    No hidden states are requested. Limited runs get a separate output filename.
    """
    cfg = copy.deepcopy(cfg)
    if cfg["model"]["dtype"] != "float16":
        raise ValueError("evaluation requires a float16 language backbone with float32 vision")
    cfg["model"]["vision_dtype"] = "float32"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_name):
        raise ValueError("model_name must be a simple filename prefix")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if dataset == "whatsup" and (variant is not None or split is not None):
        raise ValueError("What'sUp does not use variant/split arguments")
    if dataset == "pope" and split is not None:
        raise ValueError("POPE does not use a split argument")
    if dataset == "vsr" and (variant not in {"random", "zeroshot"} or split not in {"train", "dev", "test"}):
        raise ValueError("VSR requires a variant (random/zeroshot) and split (train/dev/test)")
    if dataset == "pope" and variant not in {"random", "popular", "adversarial"}:
        raise ValueError("POPE requires a variant (random/popular/adversarial)")
    key = "_".join(part for part in (dataset, variant, split) if part)
    stem = f"{model_name}_{key}" + (f"_limit{limit}" if limit is not None else "")
    out = Path(out)
    predictions_path = out / "preds" / f"{stem}.csv"
    metrics_path = out / "metrics" / f"{stem}.json"
    if not overwrite and (predictions_path.exists() or metrics_path.exists()):
        raise FileExistsError(f"results already exist for {stem}; choose another model name or --overwrite")

    n_available = len(items)
    selected = items[:limit] if limit is not None else items
    for item in selected:
        for field, expected in (("dataset", dataset), ("variant", variant), ("split", split)):
            if field in item and item[field] != expected:
                raise ValueError(f"{item.get('id')}: {field} does not match the requested evaluation")
    prepared, options = prepare_items(dataset, selected, cfg)
    manifest_hash = question_manifest_hash(prepared)
    annotations_hash = hashlib.sha256(Path(annotations).read_bytes()).hexdigest() if annotations else None

    from PIL import Image
    from src import model_utils

    set_seed(cfg["seed"])
    start = time.monotonic()
    model, processor = model_utils.load_model(cfg, adapter=adapter)
    # Apply after PEFT loading too, in this evaluation process. Vision-only FP32
    # avoids the observed T4 NaNs without changing the language backbone/kernels.
    paths = model_utils.find_module_paths(model)
    vision = model.get_submodule(paths["visual"])
    vision.float()
    language = model.get_submodule(paths["lm_layers"])
    vision_dtype = str(next(vision.parameters()).dtype)
    language_dtype = str(next(language.parameters()).dtype)
    if vision_dtype.removeprefix("torch.") != "float32" or language_dtype.removeprefix("torch.") != "float16":
        raise ValueError(f"unexpected evaluation precision: vision={vision_dtype}, language={language_dtype}")
    loaded = time.monotonic()
    option_ids = model_utils.option_token_ids(processor.tokenizer, options)
    rows = []
    for index, item in enumerate(prepared, start=1):
        with Image.open(item["image_path"]) as image:
            inputs = model_utils.make_inputs(processor, image.convert("RGB"), item["question"], device=model.device)
        result = model_utils.score_options(model, inputs, option_ids)
        if result.pred not in options or set(result.scores) != set(options):
            raise ValueError(f"{item['id']}: model returned invalid answer options")
        scores = {f"score_{option}": float(result.scores[option]) for option in options}
        if not all(math.isfinite(score) for score in scores.values()):
            raise ValueError(f"{item['id']}: model returned non-finite scores")
        rows.append({**item, "pred": result.pred, "correct": int(result.pred == item["label"]), **scores})
        if index % 25 == 0 or index == len(prepared):
            print(f"{stem}: {index}/{len(prepared)}", file=sys.stderr, flush=True)
    finished = time.monotonic()

    metrics = {
        "schema_version": 1,
        "model": model_name, "dataset": dataset, "variant": variant, "split": split,
        "adapter": adapter,
        **summarize_predictions(dataset, rows, options),
        "n_available": n_available, "limit": limit,
        "complete_dataset": len(prepared) == n_available,
        "manifest_sha256": manifest_hash,
        "annotations": str(annotations) if annotations else None,
        "annotations_sha256": annotations_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"load_s": round(loaded - start, 3), "main_pass_s": round(finished - loaded, 3)},
        "env": _environment() | {"device": str(model.device), "dtype": language_dtype,
                                 "vision_dtype": vision_dtype, "language_dtype": language_dtype},
        "code": _code_version(), "config": cfg,
        "predictions_file": str(predictions_path.relative_to(out)),
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metrics["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    write_json(metrics, metrics_path)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("whatsup", "vsr", "pope"), required=True)
    parser.add_argument("--annotations", type=Path, help="VSR/POPE JSONL file")
    parser.add_argument("--image-root", type=Path, help="VSR/POPE COCO image folder")
    parser.add_argument("--data-root", type=Path, help="What'sUp folder, as used by Person A's loader")
    parser.add_argument("--variant", help="VSR: random/zeroshot; POPE: random/popular/adversarial")
    parser.add_argument("--split", choices=("train", "dev", "test"), help="VSR split (default: test)")
    parser.add_argument("--config", type=Path, help="YAML overrides for shared base config")
    parser.add_argument("--adapter", help="optional PEFT adapter path or model ID")
    parser.add_argument("--model-name", help="output prefix (default: base, or adapted with --adapter)")
    parser.add_argument("--out", type=Path, help="results root (default: results_dir())")
    parser.add_argument("--limit", type=int, help="first N questions, for a smoke run")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run's outputs")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.dataset != "whatsup" and (args.annotations is None or args.image_root is None):
        parser.error("VSR/POPE require --annotations and --image-root")
    if args.dataset == "whatsup" and any((args.annotations, args.image_root, args.variant, args.split)):
        parser.error("What'sUp uses --data-root, without annotations/variant/split flags")
    if args.dataset != "whatsup" and args.data_root is not None:
        parser.error("--data-root is only used for What'sUp")
    if args.dataset == "pope" and args.split is not None:
        parser.error("--split is only used for VSR")
    variant, split = None, None
    try:
        if args.dataset == "vsr":
            variant, split = args.variant or "zeroshot", args.split or "test"
            items = load_vsr(args.annotations, args.image_root, variant=variant, split=split, check_images=False)
        elif args.dataset == "pope":
            variant = args.variant or "random"
            items = load_pope(args.annotations, args.image_root, variant=variant, check_images=False)
        else:
            try:
                from src.data.whatsup import check as check_whatsup, load_whatsup
            except ModuleNotFoundError as exc:
                if exc.name != "src.data.whatsup":
                    raise
                parser.exit(1, "What'sUp evaluation requires Person A's src/data/whatsup.py to be integrated.\n")
            items = load_whatsup(args.data_root)
            if args.limit is None:
                try:
                    check_whatsup(items, args.data_root)
                except AssertionError as exc:
                    raise ValueError(f"What'sUp dataset check failed: {exc}") from exc
        metrics = evaluate(
            args.dataset, items, load_config(args.config), out=args.out or results_dir(),
            model_name=args.model_name or ("adapted" if args.adapter else "base"), adapter=args.adapter,
            variant=variant, split=split, annotations=args.annotations,
            limit=args.limit, overwrite=args.overwrite,
        )
    except NotImplementedError:
        parser.exit(1, "Evaluation requires Person A's shared model_utils implementation to be integrated.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
