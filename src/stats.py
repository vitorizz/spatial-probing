"""Metrics computed from saved per-question predictions, without model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def question_manifest_hash(rows: list[dict[str, Any]]) -> str:
    """Hash the question manifest in file order, excluding model and machine paths.

    This preserves src.eval's schema-1 hash for existing standard dataset records.
    Both Windows and POSIX image paths are accepted when reading portable CSVs.
    """
    manifest = [
        {key: row[key] for key in ("id", "group", "label", "question")}
        | {"image": str(row["image_path"]).replace("\\", "/").rsplit("/", 1)[-1]}
        for row in rows
    ]
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def summarize_predictions(
    dataset: str, rows: list[dict[str, Any]], options: list[str],
) -> dict[str, Any]:
    """Accuracy and confusion for all datasets; POPE treats yes as positive.

    Missing classes/subsets have null accuracy. Undefined POPE precision/recall
    and F1 are reported as zero. Confusion rows are true labels, columns predicted.
    Correctness is recomputed from label/pred, never trusted from a CSV field.
    """
    if dataset not in {"whatsup", "vsr", "pope"}:
        raise ValueError(f"unknown dataset: {dataset}")
    if not rows:
        raise ValueError("cannot summarize an empty prediction set")
    if not options or len(set(options)) != len(options):
        raise ValueError("options must be nonempty and unique")
    if dataset in {"vsr", "pope"} and set(options) != {"yes", "no"}:
        raise ValueError("VSR/POPE require yes/no options")
    for row in rows:
        if row["label"] not in options or row["pred"] not in options:
            raise ValueError(f"label/pred outside answer options: {row.get('id')}")

    def accuracy(subset: list[dict[str, Any]]) -> float | None:
        return sum(r["label"] == r["pred"] for r in subset) / len(subset) if subset else None

    result = {
        "n": len(rows),
        "accuracy": accuracy(rows),
        "per_class_accuracy": {
            label: accuracy([r for r in rows if r["label"] == label]) for label in options
        },
        "pred_counts": {label: sum(r["pred"] == label for r in rows) for label in options},
        "confusion": {
            "labels": options,
            "rows_true_cols_pred": [
                [sum(r["label"] == true and r["pred"] == pred for r in rows) for pred in options]
                for true in options
            ],
        },
    }
    if dataset == "whatsup":
        result["lateral_accuracy"] = accuracy([r for r in rows if r["label"] in {"left", "right"}])
        result["depth_accuracy"] = accuracy([r for r in rows if r["label"] in {"front", "behind"}])
    if dataset == "vsr":
        relations = sorted({r["relation"] for r in rows if "relation" in r})
        result["per_relation_accuracy"] = {
            relation: accuracy([r for r in rows if r.get("relation") == relation]) for relation in relations
        }
    if dataset == "pope":
        tp = sum(r["label"] == "yes" and r["pred"] == "yes" for r in rows)
        fp = sum(r["label"] == "no" and r["pred"] == "yes" for r in rows)
        fn = sum(r["label"] == "yes" and r["pred"] == "no" for r in rows)
        result.update({
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "yes_ratio": sum(r["pred"] == "yes" for r in rows) / len(rows),
        })
    return result


@dataclass
class EvaluationRun:
    path: Path
    metadata: dict[str, Any]
    rows: list[dict[str, str]]
    options: list[str]


def load_run(path: str | Path) -> EvaluationRun:
    """Read src.eval schema-1 artifacts and verify CSV and question-manifest hashes.

    The JSON stays under <results>/metrics/ and predictions_file is relative to
    <results>. No source images, annotations or model weights are needed here.
    """
    path = Path(path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("expected src.eval schema_version 1")
        dataset = metadata["dataset"]
        if dataset not in {"whatsup", "vsr", "pope"}:
            raise ValueError("unknown evaluation dataset")
        if dataset == "vsr" and (metadata["variant"] not in {"random", "zeroshot"} or metadata["split"] not in {"train", "dev", "test"}):
            raise ValueError("invalid VSR variant/split")
        if dataset == "pope" and (metadata["variant"] not in {"random", "popular", "adversarial"} or metadata["split"] is not None):
            raise ValueError("invalid POPE variant/split")
        if dataset == "whatsup" and (metadata["variant"] is not None or metadata["split"] is not None):
            raise ValueError("What'sUp does not have variant/split metadata")
        if not isinstance(metadata["model"], str) or not metadata["model"]:
            raise ValueError("missing model name")
        cfg = metadata["config"]
        options = cfg["options"]["fourway" if dataset == "whatsup" else "yesno"]
        if (not isinstance(options, list) or not options
                or any(not isinstance(option, str) for option in options) or len(set(options)) != len(options)):
            raise ValueError("answer options must be unique strings")
        if not isinstance(cfg["model"], dict) or not cfg["model"].get("id"):
            raise ValueError("missing base-model configuration")
        if type(cfg["seed"]) is not int:
            raise ValueError("evaluation seed must be an integer")
        if metadata["adapter"] is not None and (not isinstance(metadata["adapter"], str) or not metadata["adapter"]):
            raise ValueError("adapter must be null or a nonempty string")
        if not isinstance(metadata.get("env", {}), dict):
            raise ValueError("environment metadata must be a dictionary")
        relative = Path(metadata["predictions_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("predictions_file must stay relative to the results root")
        predictions = path.parent.parent / relative
        contents = predictions.read_bytes()
        if hashlib.sha256(contents).hexdigest() != metadata["predictions_sha256"]:
            raise ValueError("prediction CSV checksum does not match metadata")
        with predictions.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"id", "group", "label", "pred", "question", "image_path", "dataset"}
            if (not reader.fieldnames or not required.issubset(reader.fieldnames)
                    or len(set(reader.fieldnames)) != len(reader.fieldnames)):
                raise ValueError("prediction CSV has missing or duplicate columns")
            rows = list(reader)
        seen = set()
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("prediction CSV has a malformed row")
            if not row["id"] or row["id"] in seen:
                raise ValueError("prediction CSV has empty or duplicate question IDs")
            seen.add(row["id"])
            if not row["group"] or not row["question"] or not row["image_path"]:
                raise ValueError(f"missing group, question or image path for {row['id']}")
            for field in ("dataset", "variant", "split"):
                if field in row and row[field] != (metadata[field] or ""):
                    raise ValueError(f"CSV {field} differs from run metadata")
        if type(metadata["n"]) is not int or metadata["n"] != len(rows):
            raise ValueError("prediction count differs from metadata")
        if type(metadata["n_available"]) is not int or metadata["n_available"] < len(rows):
            raise ValueError("invalid total-question count")
        if metadata["limit"] is not None and (type(metadata["limit"]) is not int or metadata["limit"] < 1):
            raise ValueError("invalid evaluation limit")
        if type(metadata["complete_dataset"]) is not bool or metadata["complete_dataset"] != (len(rows) == metadata["n_available"]):
            raise ValueError("inconsistent dataset-completeness metadata")
        summarize_predictions(dataset, rows, options)
        if question_manifest_hash(rows) != metadata["manifest_sha256"]:
            raise ValueError("question-manifest checksum does not match predictions")
        return EvaluationRun(path, metadata, rows, options)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _metric_values(confusion: np.ndarray, dataset: str, options: list[str]) -> dict[str, np.ndarray]:
    """Evaluate rates on (..., model, true, predicted) confusion arrays."""
    def divide(numerator: np.ndarray, denominator: np.ndarray, empty: float = np.nan) -> np.ndarray:
        return np.divide(numerator, denominator, out=np.full(numerator.shape, empty, dtype=float), where=denominator != 0)

    correct = np.diagonal(confusion, axis1=-2, axis2=-1)
    total = confusion.sum(axis=(-2, -1))
    result = {"accuracy": divide(correct.sum(axis=-1), total)}
    if dataset == "whatsup":
        for name, labels in (("lateral_accuracy", {"left", "right"}), ("depth_accuracy", {"front", "behind"})):
            indices = [index for index, option in enumerate(options) if option in labels]
            result[name] = divide(correct[..., indices].sum(axis=-1), confusion[..., indices, :].sum(axis=(-2, -1)))
    if dataset == "pope":
        yes, no = options.index("yes"), options.index("no")
        tp, fp, fn = confusion[..., yes, yes], confusion[..., no, yes], confusion[..., yes, no]
        result.update({
            "precision": divide(tp, tp + fp, 0.0),
            "recall": divide(tp, tp + fn, 0.0),
            "f1": divide(2 * tp, 2 * tp + fp + fn, 0.0),
            "yes_ratio": divide(confusion[..., :, yes].sum(axis=-1), total),
        })
    return result


def _compare(
    baseline: EvaluationRun, adapter: EvaluationRun, *, n_bootstrap: int,
    confidence: float, seed: int | None, allow_limited: bool,
) -> dict[str, Any]:
    before, after = baseline.metadata, adapter.metadata
    for field in ("dataset", "variant", "split", "annotations_sha256"):
        if before.get(field) != after.get(field):
            raise ValueError(f"cannot compare different {field} values")
    if before["adapter"] is not None or after["adapter"] is None:
        raise ValueError("comparison requires a base run followed by an adapter run")
    if baseline.options != adapter.options:
        raise ValueError("answer options or their order differ between runs")
    for field in ("model", "seed"):
        if before["config"][field] != after["config"][field]:
            raise ValueError(f"evaluation {field} configuration differs between runs")
    limited = any(meta["limit"] is not None or not meta["complete_dataset"] for meta in (before, after))
    if limited and not allow_limited:
        raise ValueError("limited evaluations require --allow-limited and are labeled in the report")
    left = {row["id"]: row for row in baseline.rows}
    right = {row["id"]: row for row in adapter.rows}
    if left.keys() != right.keys():
        raise ValueError("question ID sets differ; comparisons never use a silent intersection")
    for identifier, row in left.items():
        for field in ("label", "group", "question", "relation", "coco_image_id"):
            if row.get(field) != right[identifier].get(field):
                raise ValueError(f"{identifier}: {field} differs between runs")
        image = lambda value: value.replace("\\", "/").rsplit("/", 1)[-1]
        if image(row["image_path"]) != image(right[identifier]["image_path"]):
            raise ValueError(f"{identifier}: image differs between runs")
    if type(n_bootstrap) is not int or n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("bootstrap requires at least 2 resamples and confidence between 0 and 1")
    seed = before["config"]["seed"] if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")

    dataset, options = before["dataset"], baseline.options
    groups = sorted({row["group"] for row in baseline.rows})
    group_index, option_index = {group: i for i, group in enumerate(groups)}, {option: i for i, option in enumerate(options)}
    counts = np.zeros((len(groups), 2, len(options), len(options)), dtype=np.int64)
    for identifier in sorted(left):
        for model, row in enumerate((left[identifier], right[identifier])):
            counts[group_index[row["group"]], model, option_index[row["label"]], option_index[row["pred"]]] += 1
    points = _metric_values(counts.sum(axis=0), dataset, options)
    draws = {metric: [] for metric in points}
    rng = np.random.default_rng(seed)
    # Draw whole groups and use the same draw for both models and every metric.
    # Batch to bound memory on the larger VSR sets. Rates use summed counts, not
    # an unweighted mean of per-image rates; groups may contain different N.
    for start in range(0, n_bootstrap, 128):
        indices = rng.integers(len(groups), size=(min(128, n_bootstrap - start), len(groups)))
        values = _metric_values(counts[indices].sum(axis=1), dataset, options)
        for metric, value in values.items():
            draws[metric].append(value[:, 1] - value[:, 0])
    metrics = {}
    for metric, point in points.items():
        indices = list(range(len(options)))
        if metric in {"lateral_accuracy", "depth_accuracy"}:
            labels = {"left", "right"} if metric == "lateral_accuracy" else {"front", "behind"}
            indices = [i for i, option in enumerate(options) if option in labels]
        subset = counts[:, 0, indices, :].sum(axis=(-2, -1))
        n, n_groups = int(subset.sum()), int(np.count_nonzero(subset))
        differences = np.concatenate(draws[metric])
        differences = differences[np.isfinite(differences)]
        interval = (None, None)
        if n_groups >= 2 and len(differences) >= 2:
            tail = (1 - confidence) / 2
            interval = tuple(float(value) for value in np.quantile(differences, [tail, 1 - tail]))
        valid_points = bool(np.isfinite(point).all())
        metrics[metric] = {
            "n": n, "n_groups": n_groups,
            "baseline": float(point[0]) if valid_points else None,
            "adapter": float(point[1]) if valid_points else None,
            "delta": float(point[1] - point[0]) if valid_points else None,
            "ci_low": interval[0], "ci_high": interval[1],
            "n_bootstrap_valid": len(differences),
        }
    env_keys = set(before.get("env", {})) | set(after.get("env", {}))
    return {
        "dataset": dataset, "variant": before["variant"], "split": before["split"],
        "baseline": {"metrics_file": str(baseline.path), "model": before["model"],
                     "predictions_sha256": before["predictions_sha256"]},
        "adapter": {"metrics_file": str(adapter.path), "model": after["model"], "adapter": after["adapter"],
                    "predictions_sha256": after["predictions_sha256"]},
        "limited": limited, "n": len(left), "n_groups": len(groups),
        "method": "paired_cluster_percentile_bootstrap", "n_bootstrap": n_bootstrap,
        "confidence": confidence, "seed": seed, "metrics": metrics,
        "environment_differences": {
            key: {"baseline": before.get("env", {}).get(key), "adapter": after.get("env", {}).get(key)}
            for key in sorted(env_keys) if before.get("env", {}).get(key) != after.get("env", {}).get(key)
        },
    }


def compare_runs(
    baseline: str | Path, adapter: str | Path, *, n_bootstrap: int = 2000,
    confidence: float = 0.95, seed: int | None = None, allow_limited: bool = False,
) -> dict[str, Any]:
    """Match question IDs and estimate adapter-minus-baseline metric differences.

    Images are the resampling units for VSR/POPE; object-pair groups for What'sUp.
    Intervals are null with fewer than two groups contributing to a metric.
    """
    return _compare(load_run(baseline), load_run(adapter), n_bootstrap=n_bootstrap,
                    confidence=confidence, seed=seed, allow_limited=allow_limited)


def write_report(comparisons: list[dict[str, Any]], out: str | Path, *, overwrite: bool = False) -> None:
    """Write machine-readable rates plus a Markdown table in percent/percentage points."""
    if not comparisons:
        raise ValueError("no comparisons to report")
    out = Path(out)
    paths = [out / f"comparison.{suffix}" for suffix in ("json", "csv", "md")]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError(f"comparison outputs already exist in {out}; choose another --out or --overwrite")
    rows, lines = [], [
        "# Baseline / adapter comparison", "",
        "Rates are percentages; changes and confidence intervals are percentage points (adapter minus baseline).", "",
        "| Dataset | Metric | N | Groups | Baseline (%) | Adapter (%) | Change (pp) | CI for change (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    def percent(value: float | None, signed: bool = False) -> str:
        return "—" if value is None else format(value * 100, "+.2f" if signed else ".2f")

    for comparison in comparisons:
        label = "/".join(str(comparison[key]) for key in ("dataset", "variant", "split") if comparison[key] is not None)
        if comparison["limited"]:
            label += " (limited)"
        for metric, values in comparison["metrics"].items():
            rows.append({"dataset": comparison["dataset"], "variant": comparison["variant"], "split": comparison["split"],
                         "baseline_model": comparison["baseline"]["model"], "adapter_model": comparison["adapter"]["model"],
                         "limited": comparison["limited"], "metric": metric, **values,
                         "confidence": comparison["confidence"], "n_bootstrap": comparison["n_bootstrap"], "seed": comparison["seed"]})
            interval = "—" if values["ci_low"] is None else (
                f"{comparison['confidence'] * 100:g}% [{percent(values['ci_low'], True)}, {percent(values['ci_high'], True)}]"
            )
            lines.append(f"| {label} | {metric} | {values['n']} | {values['n_groups']} | {percent(values['baseline'])} | "
                         f"{percent(values['adapter'])} | {percent(values['delta'], True)} | {interval} |")
    lines += ["", "Paired percentile bootstrap resamples image groups (VSR/POPE) or object-pair groups (What'sUp).",
              "Each draw recomputes metrics from all questions in the sampled groups. JSON records resample counts and seeds.",
              "Intervals describe evaluation-group sampling, not variation across training runs.",
              "An absent interval means fewer than two contributing groups or fewer than two valid resamples.",
              "See JSON for any recorded environment differences between the runs.", ""]
    out.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps({"schema_version": 1, "comparisons": comparisons}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with paths[1].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths[2].write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    from src.common import results_dir

    parser = argparse.ArgumentParser(description="Compare saved baseline/adapter evaluations without loading a model.")
    parser.add_argument("--baseline", nargs="+", type=Path, required=True, help="baseline metric JSON files")
    parser.add_argument("--adapter", nargs="+", type=Path, required=True, help="adapter metric JSON files; matched by dataset/variant/split")
    parser.add_argument("--out", type=Path, help="report folder (default: results/comparisons)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="number of group resamples (default: 2000)")
    parser.add_argument("--confidence", type=float, default=0.95, help="interval coverage (default: 0.95)")
    parser.add_argument("--seed", type=int, help="bootstrap seed (default: each baseline's evaluation seed)")
    parser.add_argument("--allow-limited", action="store_true", help="explicitly allow and label smoke-run comparisons")
    parser.add_argument("--overwrite", action="store_true", help="replace existing report files")
    args = parser.parse_args(argv)
    try:
        def by_dataset(paths: list[Path]) -> dict[tuple, EvaluationRun]:
            runs = {}
            for path in paths:
                run = load_run(path)
                key = tuple(run.metadata[field] for field in ("dataset", "variant", "split"))
                if key in runs:
                    raise ValueError(f"duplicate dataset/variant/split: {key}")
                runs[key] = run
            return runs

        baseline, adapter = by_dataset(args.baseline), by_dataset(args.adapter)
        if baseline.keys() != adapter.keys():
            raise ValueError("baseline and adapter files must cover the same dataset/variant/split sets")
        comparisons = [
            _compare(run, adapter[key], n_bootstrap=args.bootstrap, confidence=args.confidence,
                     seed=args.seed, allow_limited=args.allow_limited)
            for key, run in baseline.items()
        ]
        out = args.out or results_dir() / "comparisons"
        write_report(comparisons, out, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Wrote {len(comparisons)} dataset comparisons to {out}")


if __name__ == "__main__":
    main()
