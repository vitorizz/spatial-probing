"""Stage 1 (What'sUp) + Stage 2a: one pass gives predictions, metrics and hidden states.

    python -m src.extract_hidden                      # base model, all 408 items
    python -m src.extract_hidden --adapter runs/lora_targeted/adapter --model-name lora_targeted
    python -m src.extract_hidden --limit 8 --out /tmp/smoke   # quick smoke test

Writes, under --out (default results_dir()):
    preds/{model}_whatsup.csv           id, group, label, pred, correct, score_<option>...
    preds/{model}_whatsup_gen.csv       free-generation sanity check (Model details)
    metrics/{model}_whatsup.json        accuracy (all/lateral/depth), per class, confusion,
                                        runtime, determinism + generation-agreement checks
    hidden/{model}_whatsup_last.npy     float16 (N, L+1, d): hidden_states[l][0, -1, :]
    hidden/{model}_whatsup_index.csv    id, group, label per row of the .npy

Progress is checkpointed to hidden/{model}_whatsup_partial.* every --save-every items
and resumed automatically, so a Colab disconnect only loses the last few items.
"""
from __future__ import annotations

import argparse
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.common import load_config, read_json, results_dir, set_seed, write_json
from src.data.whatsup import LABELS, download, load_whatsup
from src.model_utils import (
    find_module_paths,
    fourway_question,
    generate_text,
    get_arch_facts,
    load_model,
    make_inputs,
    option_token_ids,
    parse_option,
    score_options,
)

LATERAL = ("left", "right")
DEPTH = ("front", "behind")


def whatsup_metrics(labels: list[str], preds: list[str]) -> dict:
    """Accuracy overall, on lateral and depth subsets, per class, plus the 4x4 confusion
    matrix (rows = true label, columns = prediction, both in LABELS order)."""
    y = np.asarray(labels)
    p = np.asarray(preds)
    ok = y == p

    def acc(mask: np.ndarray) -> float | None:
        return float(ok[mask].mean()) if mask.any() else None

    conf = [[int(((y == t) & (p == q)).sum()) for q in LABELS] for t in LABELS]
    return {
        "n": int(len(y)),
        "accuracy": acc(np.ones_like(ok)),
        "lateral_accuracy": acc(np.isin(y, LATERAL)),
        "depth_accuracy": acc(np.isin(y, DEPTH)),
        "per_class_accuracy": {c: acc(y == c) for c in LABELS},
        "pred_counts": {c: int((p == c).sum()) for c in LABELS},
        "confusion": {"labels": LABELS, "rows_true_cols_pred": conf},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="What'sUp baseline + hidden-state extraction")
    ap.add_argument("--config", default=None, help="YAML merged over configs/base.yaml")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (PeftModel)")
    ap.add_argument("--model-name", default=None, help="file prefix (default: base, or the adapter's run name)")
    ap.add_argument("--out", default=None, help="results root (default: results_dir())")
    ap.add_argument("--data-root", default=None, help="What'sUp folder (default: data_dir())")
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (default: cuda if available, else cpu)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke tests)")
    ap.add_argument("--n-gen", type=int, default=20, help="items for the free-generation check")
    ap.add_argument("--n-repeat", type=int, default=10, help="items re-scored for the determinism check")
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint interval (items)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    name = args.model_name or ("base" if args.adapter is None else Path(args.adapter).parent.name)
    out = Path(args.out) if args.out else results_dir()
    for sub in ("preds", "metrics", "hidden"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    items = load_whatsup(download(args.data_root))
    if args.limit:
        items = items[: args.limit]

    t_load = time.time()
    model, processor = load_model(cfg, adapter=args.adapter, device=args.device)
    t_load = time.time() - t_load
    device = model.device
    arch = get_arch_facts(model)
    options = cfg["options"]["fourway"]
    option_ids = option_token_ids(processor.tokenizer, options)

    def inputs_for(it: dict):
        with Image.open(it["image_path"]) as im:
            image = im.convert("RGB")
        return make_inputs(processor, image, fourway_question(it["o1"], it["o2"], cfg), device)

    # --- main pass, with resume -------------------------------------------------------
    part_rows = out / "hidden" / f"{name}_whatsup_partial.csv"
    part_hid = out / "hidden" / f"{name}_whatsup_partial.npy"
    rows: list[dict] = []
    hidden: list[np.ndarray] = []
    if part_rows.exists() and part_hid.exists():
        prev = pd.read_csv(part_rows)
        prev_h = np.load(part_hid)
        if len(prev) == len(prev_h) and list(prev["id"]) == [it["id"] for it in items[: len(prev)]]:
            rows = prev.to_dict("records")
            hidden = list(prev_h)
            print(f"resuming after {len(rows)} items")

    def checkpoint() -> None:
        pd.DataFrame(rows).to_csv(part_rows, index=False)
        np.save(part_hid, np.stack(hidden))

    t0 = time.time()
    n_new = 0
    for i in tqdm(range(len(rows), len(items)), desc=f"{name} whatsup", initial=len(rows), total=len(items)):
        it = items[i]
        res = score_options(model, inputs_for(it), option_ids, output_hidden_states=True)
        rows.append(
            {
                "id": it["id"], "group": it["group"], "o1": it["o1"], "o2": it["o2"],
                "label": it["label"], "pred": res.pred, "correct": int(res.pred == it["label"]),
                **{f"score_{o}": res.scores[o] for o in options},
            }
        )
        hidden.append(res.hidden)
        n_new += 1
        if n_new % args.save_every == 0:
            checkpoint()
    t_main = time.time() - t0

    H = np.stack(hidden).astype(np.float16)
    assert H.shape == (len(items), arch.num_layers + 1, arch.hidden_size), H.shape
    assert np.isfinite(H).all(), "non-finite hidden states"
    preds = pd.DataFrame(rows)

    # --- determinism: re-score the first n items ----------------------------------------
    rep = items[: min(args.n_repeat, len(items))]
    same_pred, max_diff = 0, 0.0
    for it, row in zip(rep, rows):
        r = score_options(model, inputs_for(it), option_ids)
        same_pred += int(r.pred == row["pred"])
        max_diff = max(max_diff, max(abs(r.scores[o] - row[f"score_{o}"]) for o in options))
    determinism = {
        "n": len(rep),
        "identical_preds": same_pred,
        "all_identical": same_pred == len(rep),
        "max_abs_score_diff": max_diff,
    }

    # --- free-generation sanity check -------------------------------------------------
    gen_idx = sorted(random.Random(cfg["seed"]).sample(range(len(items)), min(args.n_gen, len(items))))
    gen_rows = []
    for i in tqdm(gen_idx, desc="free generation"):
        text = generate_text(model, processor, inputs_for(items[i]), max_new_tokens=5)
        parsed = parse_option(text, options)
        gen_rows.append(
            {"id": items[i]["id"], "label": items[i]["label"], "scored_pred": rows[i]["pred"],
             "generated": text, "parsed": parsed, "agree": int(parsed == rows[i]["pred"])}
        )
    gen = pd.DataFrame(gen_rows)

    # --- write outputs ----------------------------------------------------------------
    preds.to_csv(out / "preds" / f"{name}_whatsup.csv", index=False)
    gen.to_csv(out / "preds" / f"{name}_whatsup_gen.csv", index=False)
    np.save(out / "hidden" / f"{name}_whatsup_last.npy", H)
    preds[["id", "group", "label"]].to_csv(out / "hidden" / f"{name}_whatsup_index.csv", index=False)

    import torch
    import transformers

    metrics = {
        "model": name,
        "dataset": "whatsup_controlled_b",
        "adapter": args.adapter,
        **whatsup_metrics(preds["label"].tolist(), preds["pred"].tolist()),
        "determinism": determinism,
        "free_generation": {
            "n": len(gen),
            "agreement": float(gen["agree"].mean()) if len(gen) else None,
            "unparsed": int(gen["parsed"].isna().sum()) if len(gen) else 0,
        },
        "runtime": {
            "load_s": round(t_load, 1),
            "main_pass_s": round(t_main, 1),
            "items_this_session": n_new,
            "s_per_item": round(t_main / n_new, 3) if n_new else None,
        },
        "hidden": {
            "file": f"hidden/{name}_whatsup_last.npy",
            "shape": list(H.shape),
            "dtype": "float16",
            "position": "last prompt token; layer 0 = input embedding",
        },
        "arch": vars(arch),
        "module_paths": find_module_paths(model),
        "option_token_ids": option_ids,
        "env": {
            "device": str(device),
            "dtype": str(model.dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": cfg,
    }
    write_json(metrics, out / "metrics" / f"{name}_whatsup.json")
    part_rows.unlink(missing_ok=True)
    part_hid.unlink(missing_ok=True)

    m = read_json(out / "metrics" / f"{name}_whatsup.json")
    print(
        f"\n{name} What'sUp: acc {m['accuracy']:.3f} | lateral {m['lateral_accuracy']}"
        f" | depth {m['depth_accuracy']} | n {m['n']}"
    )
    print(f"determinism: {determinism} | generation agreement: {m['free_generation']['agreement']}")
    if m["depth_accuracy"] is not None and m["depth_accuracy"] >= 0.85 and len(items) == 408:
        print("WARNING: depth accuracy >= 85%: the spec's decision point says stop and tell the team.")
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
