"""Metrics computed from saved per-question predictions, without model inference."""
from __future__ import annotations

from typing import Any


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
