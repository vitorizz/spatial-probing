"""Builds the final report's figures from the files under results/.

    python report/make_figures.py        # run from the repo root; writes report/figures/*.pdf and *.png
"""
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "figures"
MILESTONE_EVAL = RES / "evaluation_milestone_vision_fp32" / "preds"
FINAL_EVAL = RES / "evaluation_final" / "preds"

COL, FULL = 3.33, 7.0  # acmart sigconf column and text width, inches
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
# Colour follows the model in every figure; line style and marker repeat the identity for print.
STYLE = {
    "base": dict(color=MUTED, ls=":", marker="x", label="base"),
    "lora_targeted": dict(color="#2a78d6", ls="-", marker="o", label="targeted (blocks 28-35)"),
    "lora_all": dict(color="#eb6834", ls="--", marker="s", label="all 36 blocks"),
    "lora_random": dict(color="#1baf7a", ls="-.", marker="^", label="random 8 blocks"),
}
RUNS = ["lora_targeted", "lora_all", "lora_random"]
RANDOM_BLOCKS = [2, 12, 16, 24, 25, 26, 29, 31]  # runs/lora_random/lora_modules.txt
TARGETED_BLOCKS = list(range(28, 36))

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_overview():
    """Pipeline sketch plus the blocks each adapter trains."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FULL, 2.55), gridspec_kw={"height_ratios": [1.25, 1]})
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f2ee", ec=MUTED, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", **kw)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.8))

    box(0.5, 8, 14.5, 14, "image +\n4-way\nquestion")
    box(19.5, 8, 12.5, 14, "vision\nencoder\n(frozen)")
    arrow(15.7, 15, 18.8, 15); arrow(32.7, 15, 35.8, 15)
    x0, w = 36.5, 1.25
    for b in range(36):  # decoder blocks
        sel = b in TARGETED_BLOCKS
        ax.add_patch(Rectangle((x0 + b * w, 9), w * 0.82, 12, fc="#2a78d6" if sel else "#f3f2ee", ec=MUTED, lw=0.4))
    ax.text(x0 + 18 * w, 23.2, "36 language-model blocks (hidden states $h_1 \\dots h_{36}$ at the last prompt token)",
            ha="center", va="bottom")
    ax.text(x0 + 32 * w, 6.6, "LoRA on blocks 28-35\n(case B: last 8)", ha="center", va="top", color=INK)
    ax.text(x0 + 11 * w, 6.6, "one linear probe per layer:\n4-way, lateral, depth", ha="center", va="top")
    for b in range(0, 36, 3):
        ax.plot([x0 + b * w + w * 0.41] * 2, [8.6, 7.4], color=MUTED, lw=0.6)
    arrow(x0 + 36 * w + 0.4, 15, x0 + 36 * w + 3.6, 15)
    box(86, 8, 13.5, 14, "option logits:\nleft, right,\nfront, behind")

    ax2.set_xlim(-9.5, 36.3); ax2.set_ylim(-0.4, 3.2); ax2.axis("off")
    rows = [("lora_targeted", TARGETED_BLOCKS, "1.64M"), ("lora_all", list(range(36)), "7.37M"),
            ("lora_random", RANDOM_BLOCKS, "1.64M")]
    for r, (run, blocks, params) in enumerate(rows):
        y = 2 - r
        ax2.text(-0.4, y + 0.4, STYLE[run]["label"].split(" (")[0], ha="right", va="center")
        for b in range(36):
            on = b in blocks
            ax2.add_patch(Rectangle((b, y), 0.86, 0.8, fc=STYLE[run]["color"] if on else "#f3f2ee", ec=MUTED, lw=0.4))
        ax2.text(36.3, y + 0.4, params, ha="left", va="center", color=MUTED)
    for b in (0, 5, 10, 15, 20, 25, 30, 35):
        ax2.text(b + 0.43, -0.12, str(b), ha="center", va="top", color=MUTED, fontsize=6.5)
    ax2.text(18, 3.0, "decoder blocks adapted by each run (0-based index); right: trainable parameters",
             ha="center", va="bottom")
    fig.subplots_adjust(hspace=0.12)
    save(fig, "overview")


def fig_training():
    """Training loss and validation accuracy of the three full runs."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    for run in RUNS:
        log = pd.read_csv(RES / "training_final" / run / "train_log.csv")
        s = STYLE[run]
        ax.plot(log.step, log.loss, color=s["color"], lw=0.5, alpha=0.3)
        ax.plot(log.step, log.loss.rolling(7, center=True, min_periods=3).mean(), color=s["color"], ls=s["ls"],
                lw=1.4, label=s["label"])
    ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss (answer tokens)")
    ax.set_ylim(0, 0.8); ax.set_xlim(0, 750); ax.grid(axis="y")
    ax.legend(loc="upper right")
    ax.set_title("(a) loss, mean of 7 logged points", loc="left")
    for ax, col, title in ((axes[1], "acc_depth", "(b) validation, depth"), (axes[2], "acc_lateral", "(c) validation, lateral")):
        for run in RUNS:
            val = pd.read_csv(RES / "training_final" / run / "val_curve.csv")
            s = STYLE[run]
            ax.plot(val.step, 100 * val[col], color=s["color"], ls=s["ls"], marker=s["marker"], ms=3.5, lw=1.4)
        ax.set_xlabel("optimizer step"); ax.set_xlim(-20, 770); ax.grid(axis="y")
        ax.set_title(title, loc="left")
    axes[1].set_ylabel("accuracy (%)"); axes[1].set_ylim(50, 100); axes[2].set_ylim(50, 100)
    fig.tight_layout(w_pad=1.2)
    save(fig, "training")


def whatsup_preds():
    return {
        "base": pd.read_csv(MILESTONE_EVAL / "base_whatsup.csv"),
        "milestone": pd.read_csv(MILESTONE_EVAL / "adapted_whatsup.csv"),
        **{run: pd.read_csv(FINAL_EVAL / f"{run}_whatsup.csv") for run in RUNS},
    }


def fig_margin():
    """On What'sUp depth items: best depth-option logit minus best lateral-option logit."""
    preds = whatsup_preds()
    order = ["base", "milestone", "lora_targeted", "lora_all", "lora_random"]
    names = {"base": "base", "milestone": "targeted, 2.9k\n(milestone)", "lora_targeted": "targeted, 12k",
             "lora_all": "all blocks, 12k", "lora_random": "random 8, 12k"}
    colors = {"milestone": STYLE["lora_targeted"]["color"], **{k: v["color"] for k, v in STYLE.items()}}
    fig, ax = plt.subplots(figsize=(COL, 2.25))
    rng = np.random.default_rng(0)
    for i, key in enumerate(order):
        d = preds[key]
        d = d[d.label.isin(["front", "behind"])]
        margin = d[["score_front", "score_behind"]].max(axis=1) - d[["score_left", "score_right"]].max(axis=1)
        y = len(order) - 1 - i
        ax.scatter(margin, y + rng.uniform(-0.28, 0.28, len(margin)), s=3, color=colors[key],
                   alpha=0.35 if key != "milestone" else 0.2, linewidths=0)
        ax.plot([margin.median()] * 2, [y - 0.36, y + 0.36], color=INK, lw=1.2)
        ax.text(12.9, y, f"{100 * d.correct.mean():.1f}%", va="center", ha="right", color=INK)
    ax.axvline(0, color=MUTED, lw=0.7, ls="--")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in reversed(order)])
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(-4, 13); ax.set_ylim(-0.6, len(order) - 0.25)
    ax.set_xticks(range(-4, 11, 2))
    ax.set_xlabel("depth-option logit minus lateral-option logit")
    ax.text(12.9, len(order) - 0.42, "depth acc.", ha="right", va="bottom", color=MUTED)
    ax.text(-0.15, len(order) - 0.42, "says left/right", ha="right", va="bottom", color=MUTED)
    ax.text(0.15, len(order) - 0.42, "says front/behind", ha="left", va="bottom", color=MUTED)
    ax.grid(axis="x")
    save(fig, "margin")


def fig_probes():
    """Depth probe curve of every model, and its difference from the base model."""
    curves = {"base": pd.read_csv(RES / "milestone" / "probing" / "base_probe_curves.csv"),
              **{run: pd.read_csv(RES / "probing_final" / f"{run}_probe_curves.csv") for run in RUNS}}
    depth = {k: c[(c.task == "depth") & (c.C == 0.05)].sort_values("layer").set_index("layer").mean_acc
             for k, c in curves.items()}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(COL, 3.1), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for a in (ax, ax2):
        a.axvspan(28.5, 36.5, color="#2a78d6", alpha=0.08, lw=0)
        a.grid(axis="y")
    for key in [*RUNS, "base"]:  # base last: it coincides with the targeted curve
        s = STYLE[key]
        ax.plot(depth[key].index, 100 * depth[key], color=INK if key == "base" else s["color"], ls=s["ls"],
                lw=1.0 if key == "base" else 1.3, label=s["label"])
    ax.axhline(50, color=MUTED, lw=0.6, ls=":")
    ax.text(36.3, 51, "chance", ha="right", va="bottom", color=MUTED)
    ax.set_ylabel("depth probe accuracy (%)"); ax.set_ylim(45, 102)
    ax.legend(loc="lower left", bbox_to_anchor=(0.1, 0.06), handlelength=2.6)
    ax.text(32.5, 100.6, "$S$", ha="center", va="bottom", color="#2a78d6")
    for run in RUNS:
        s = STYLE[run]
        ax2.plot(depth[run].index, 100 * (depth[run] - depth["base"]), color=s["color"], ls=s["ls"], lw=1.3)
    ax2.axhline(0, color=MUTED, lw=0.6)
    ax2.set_ylabel("change vs. base (pp)"); ax2.set_ylim(-6, 6)
    ax2.set_xlabel("layer (0 = input embedding)"); ax2.set_xlim(-0.5, 36.5)
    fig.tight_layout(h_pad=0.6)
    save(fig, "probes_adapters")


if __name__ == "__main__":
    fig_overview()
    fig_training()
    fig_margin()
    fig_probes()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RES / "milestone" / "probing" / "probe_curves_selected.png", OUT / "probe_curves_base.png")
