"""Stage 2b: linear probes on the saved What'sUp hidden states, one per layer and task.

    python -m src.probe                       # base model, all layers, all checks
    python -m src.probe --model-name lora_targeted   # re-probe a fine-tuned model (Stage 5, optional)

Reads  hidden/{model}_whatsup_last.npy + hidden/{model}_whatsup_index.csv  (from extract_hidden.py).
Writes probes/{model}_probe_curves.csv   layer, task, C, mean_acc, std_acc, shuffled_mean, shuffled_max
       probes/{model}_probe_summary.json  acceptance checks + final-layer probe vs model accuracy
       figures/{model}_probe_curves.png
For the base model the files are also copied to the spec's names
(probes/probe_curves.csv, figures/probe_curves.png).

Probe: StandardScaler + LogisticRegression(C=0.05), fit inside each fold.
Evaluation: GroupKFold(5) with groups = object pairs, so no pair is in train and test.
Controls: 5 label permutations (seeds 0-4) per layer and task; depth repeated with C = 0.005 and 0.5.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import results_dir, write_json

TASKS = {
    "4way": ("left", "right", "front", "behind"),
    "lateral": ("left", "right"),
    "depth": ("front", "behind"),
}
CHANCE = {"4way": 0.25, "lateral": 0.5, "depth": 0.5}
MAIN_C = 0.05
EXTRA_C = (0.005, 0.5)  # regularization check, depth task only
N_FOLDS = 5
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)


def load_hidden(out: Path, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    H = np.load(out / "hidden" / f"{name}_whatsup_last.npy")
    index = pd.read_csv(out / "hidden" / f"{name}_whatsup_index.csv")
    assert len(H) == len(index), f"{len(H)} hidden rows vs {len(index)} index rows"
    assert H.ndim == 3, H.shape
    return H, index


def cv_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float) -> np.ndarray:
    """Held-out accuracy per fold. The scaler is inside the pipeline, so it is fit on
    the training folds only."""
    accs = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return np.asarray(accs)


def probe_one(X: np.ndarray, y: np.ndarray, groups: np.ndarray, layer: int, task: str,
              C: float, shuffled: bool) -> dict:
    X = X.astype(np.float32)
    accs = cv_accuracy(X, y, groups, C)
    row = {"layer": layer, "task": task, "C": C,
           "mean_acc": float(accs.mean()), "std_acc": float(accs.std()),
           "shuffled_mean": np.nan, "shuffled_max": np.nan}
    if shuffled:
        sh = [cv_accuracy(X, np.random.default_rng(s).permutation(y), groups, C).mean()
              for s in SHUFFLE_SEEDS]
        row["shuffled_mean"] = float(np.mean(sh))
        row["shuffled_max"] = float(np.max(sh))
    return row


def run_probes(H: np.ndarray, index: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    labels = index["label"].to_numpy()
    groups = index["group"].to_numpy()
    jobs = []
    for task, classes in TASKS.items():
        m = np.isin(labels, classes)
        Ht, yt, gt = H[m], labels[m], groups[m]
        for layer in range(H.shape[1]):
            jobs.append((Ht[:, layer], yt, gt, layer, task, MAIN_C, True))
            if task == "depth":
                for C in EXTRA_C:
                    jobs.append((Ht[:, layer], yt, gt, layer, task, C, False))
    rows = Parallel(n_jobs=n_jobs, verbose=5)(delayed(probe_one)(*j) for j in jobs)
    return pd.DataFrame(rows).sort_values(["task", "C", "layer"]).reset_index(drop=True)


def plot_curves(curves: pd.DataFrame, path: Path, title: str,
                selected: list[int] | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"4way": "#4C72B0", "lateral": "#55A868", "depth": "#C44E52"}
    names = {"4way": "4-way", "lateral": "lateral (left/right)", "depth": "depth (front/behind)"}
    fig, ax = plt.subplots(figsize=(9, 5))
    if selected:
        lo, hi = min(selected), max(selected)
        ax.axvspan(lo - 0.5, hi + 0.5, color="#888888", alpha=0.15, lw=0,
                   label=f"targeted layers S = {lo}-{hi}")
    for task in TASKS:
        d = curves[(curves.task == task) & (curves.C == MAIN_C)].sort_values("layer")
        c = colors[task]
        ax.plot(d.layer, d.mean_acc, color=c, lw=2, label=names[task])
        ax.fill_between(d.layer, d.mean_acc - d.std_acc, d.mean_acc + d.std_acc, color=c, alpha=0.2, lw=0)
        ax.plot(d.layer, d.shuffled_mean, color=c, lw=1.2, ls="--", label=f"{names[task]}, shuffled labels")
    for ch in sorted(set(CHANCE.values())):
        ax.axhline(ch, color="black", lw=0.8, ls=":")
        ax.text(curves.layer.max() + 0.3, ch, f"chance {ch:.2f}", va="center", fontsize=8)
    ax.set_xlabel("layer (0 = input embedding)")
    ax.set_ylabel(f"probe accuracy ({N_FOLDS}-fold group CV)")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, curves.layer.max() + 0.5)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summarize(curves: pd.DataFrame, out: Path, name: str) -> dict:
    main = curves[curves.C == MAIN_C]
    gaps = {t: float((main[main.task == t].shuffled_mean - CHANCE[t]).abs().max()) for t in TASKS}
    worst = {t: float((main[main.task == t].shuffled_max - CHANCE[t]).abs().max()) for t in TASKS}
    L = int(curves.layer.max())

    reg = {}
    base = curves[(curves.task == "depth") & (curves.C == MAIN_C)].set_index("layer").mean_acc
    for C in EXTRA_C:
        other = curves[(curves.task == "depth") & (curves.C == C)].set_index("layer").mean_acc
        reg[str(C)] = {
            "pearson_r_vs_C0.05": float(np.corrcoef(base, other)[0, 1]),
            "best_layer": int(other.idxmax()),
            "max_abs_diff": float((base - other).abs().max()),
        }

    summary = {
        "model": name,
        "n_layers_probed": L + 1,
        "best_layer": {t: int(main[main.task == t].set_index("layer").mean_acc.idxmax()) for t in TASKS},
        "best_acc": {t: float(main[main.task == t].mean_acc.max()) for t in TASKS},
        "final_layer_acc": {t: float(main[(main.task == t) & (main.layer == L)].mean_acc.iloc[0]) for t in TASKS},
        "max_fold_std": {t: float(main[main.task == t].std_acc.max()) for t in TASKS},
        "shuffled_control": {
            "max_abs_gap_from_chance_mean_of_5": gaps,
            "max_abs_gap_from_chance_single_perm": worst,
            "pass_within_5_points": bool(all(g <= 0.05 for g in gaps.values())),
        },
        "regularization_check_depth": reg,
    }
    metrics_path = out / "metrics" / f"{name}_whatsup.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        summary["probe_vs_model"] = {
            "final_layer_depth_probe_acc": summary["final_layer_acc"]["depth"],
            "model_depth_acc": m.get("depth_accuracy"),
            "final_layer_lateral_probe_acc": summary["final_layer_acc"]["lateral"],
            "model_lateral_acc": m.get("lateral_accuracy"),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2b: probes on What'sUp hidden states")
    ap.add_argument("--model-name", default="base")
    ap.add_argument("--out", default=None, help="results root (default: results_dir())")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    out = Path(args.out) if args.out else results_dir()
    name = args.model_name
    H, index = load_hidden(out, name)
    print(f"hidden states {H.shape} {H.dtype}; labels {index.label.value_counts().to_dict()}; "
          f"{index.group.nunique()} groups")

    t0 = time.time()
    curves = run_probes(H, index, n_jobs=args.n_jobs)
    runtime = time.time() - t0

    probes = out / "probes"
    probes.mkdir(parents=True, exist_ok=True)
    curves_path = probes / f"{name}_probe_curves.csv"
    curves.to_csv(curves_path, index=False)
    fig_path = plot_curves(curves, out / "figures" / f"{name}_probe_curves.png",
                           f"Probe accuracy by layer: {name} model, What'sUp Controlled B")
    summary = summarize(curves, out, name)
    summary["runtime_s"] = round(runtime, 1)
    write_json(summary, probes / f"{name}_probe_summary.json")
    if name == "base":  # the spec's file names
        shutil.copyfile(curves_path, probes / "probe_curves.csv")
        shutil.copyfile(fig_path, out / "figures" / "probe_curves.png")

    print(json.dumps({k: summary[k] for k in
                      ("best_layer", "best_acc", "final_layer_acc", "shuffled_control", "probe_vs_model")
                      if k in summary}, indent=2))
    if not summary["shuffled_control"]["pass_within_5_points"]:
        print("WARNING: a shuffled control is more than 5 points from chance; check for leakage.")
    if max(summary["max_fold_std"].values()) > 0.10:
        print("NOTE: some fold std > 10 points (spec risk 'probe curves too noisy').")
    print(f"wrote {curves_path} and {fig_path} ({runtime:.0f} s)")


if __name__ == "__main__":
    main()
