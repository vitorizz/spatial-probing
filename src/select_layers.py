"""Stage 2c: choose the targeted layer set S from the depth probe curve. Frozen before Stage 4.

    python -m src.select_layers            # writes probes/selected_layers.json (refuses to overwrite)
    python -m src.select_layers --force    # only if the probes themselves were wrong

Rule (spec, Stage 2c). a(l) = depth probe accuracy for l = 1..L (C = 0.05), smoothed with a
centered 3-layer moving average (at l = 1 and l = L, the mean of the two available values).
p = argmax a(l), k = 8.
  A. drop:        a(L) <= a(p) - 0.05.  t = first layer after p with a(l) < a(p) - 0.02;
                  S = t-2 .. t+k-3, clipped to 1..L.
  B. stays high:  a(L) >  a(p) - 0.05 and a(p) >= 0.75.  S = L-k+1 .. L.
  C. never high:  a(p) < 0.65.  S = 1 .. k  (bottleneck may be the vision encoder).
  otherwise:      case A using the largest drop between consecutive smoothed values.
Cases are tested in that order (A, B, C, otherwise); every condition's value is saved.

Layer numbering: l is the hidden_states index, so l = 0 is the input embedding and l is the
output of decoder block l (1-based). PEFT's `layers_to_transform` counts blocks from 0, so
the JSON also gives `peft_layers_to_transform` = [l - 1 for l in S]. Use that one in train_lora.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import read_json, results_dir, write_json
from src.probe import MAIN_C, plot_curves

K = 8
DROP = 0.05
FALL = 0.02
HIGH = 0.75
LOW = 0.65


def smooth3(a: np.ndarray) -> np.ndarray:
    """Centered 3-point moving average; the ends average the two values available."""
    out = np.empty_like(a, dtype=float)
    for i in range(len(a)):
        out[i] = a[max(0, i - 1): i + 2].mean()
    return out


def window(t: int, L: int, k: int = K) -> list[int]:
    """t-2 .. t+k-3, clipped to 1..L."""
    return [l for l in range(t - 2, t + k - 2) if 1 <= l <= L]


def select(depth_acc: dict[int, float], k: int = K) -> dict:
    """depth_acc: layer -> probe accuracy for layers 0..L (layer 0 is ignored)."""
    L = max(depth_acc)
    layers = np.arange(1, L + 1)
    raw = np.array([depth_acc[l] for l in layers])
    a = smooth3(raw)
    sa = dict(zip(layers.tolist(), a.tolist()))

    p = int(layers[np.argmax(a)])  # first maximum
    ap, aL = sa[p], sa[L]
    cond = {
        "A_drop": aL <= ap - DROP,
        "B_stays_high": aL > ap - DROP and ap >= HIGH,
        "C_never_high": ap < LOW,
    }
    t = None
    if cond["A_drop"]:
        case = "A"
        t = next(l for l in range(p + 1, L + 1) if sa[l] < ap - FALL)
        S = window(t, L, k)
    elif cond["B_stays_high"]:
        case = "B"
        S = list(range(L - k + 1, L + 1))
    elif cond["C_never_high"]:
        case = "C"
        S = list(range(1, k + 1))
    else:
        case = "otherwise"
        drops = {l: sa[l - 1] - sa[l] for l in range(2, L + 1)}
        t = max(drops, key=drops.get)  # layer right after the largest drop
        S = window(t, L, k)

    meaning = {
        "A": "Depth information is present, then lost; adapt where it is lost.",
        "B": "Depth information is present but not used; adapt the layers that form the answer.",
        "C": "Depth information never becomes decodable in the language model; "
             "the bottleneck may be the vision encoder.",
        "otherwise": "No clear pattern; case A applied at the largest drop between consecutive layers.",
    }[case]
    return {
        "case": case,
        "interpretation": meaning,
        "conditions": cond,
        "L": L,
        "k": k,
        "p": p,
        "a_p": ap,
        "a_L": aL,
        "t": t,
        "S": S,
        "peft_layers_to_transform": [l - 1 for l in S],
        "a_smoothed": {str(l): v for l, v in sa.items()},
        "a_raw": {str(l): float(v) for l, v in zip(layers, raw)},
        "thresholds": {"drop": DROP, "fall": FALL, "high": HIGH, "low": LOW},
        "layer_numbering": "S uses hidden_states indices (1..L = output of decoder block l, 1-based). "
                           "peft_layers_to_transform = S - 1 (0-based block index for PEFT).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2c: choose targeted layers S")
    ap.add_argument("--model-name", default="base")
    ap.add_argument("--out", default=None, help="results root (default: results_dir())")
    ap.add_argument("--force", action="store_true", help="overwrite an existing selection")
    args = ap.parse_args()

    out = Path(args.out) if args.out else results_dir()
    target = out / "probes" / "selected_layers.json"
    if target.exists() and not args.force:
        prev = read_json(target)
        raise SystemExit(
            f"{target} already exists (case {prev['case']}, S = {prev['S']}). S is frozen "
            "before Stage 4; pass --force only if the probe curves themselves were wrong."
        )

    curves_path = out / "probes" / f"{args.model_name}_probe_curves.csv"
    curves = pd.read_csv(curves_path)
    d = curves[(curves.task == "depth") & (curves.C == MAIN_C)]
    result = select(dict(zip(d.layer.astype(int), d.mean_acc)))
    result.update({
        "model": args.model_name,
        "source": str(curves_path.name),
        "source_sha256": hashlib.sha256(curves_path.read_bytes()).hexdigest(),
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    })
    write_json(result, target)

    fig = plot_curves(curves, out / "figures" / "probe_curves_selected.png",
                      f"Probe accuracy by layer ({args.model_name}), targeted layers shaded",
                      selected=result["S"])
    print(f"case {result['case']}: {result['interpretation']}")
    print(f"p = {result['p']} (a = {result['a_p']:.3f}), a(L) = {result['a_L']:.3f}, t = {result['t']}")
    print(f"S (hidden_states layers)          = {result['S']}")
    print(f"peft layers_to_transform (0-based) = {result['peft_layers_to_transform']}")
    if result["case"] != "C" and result["conditions"]["C_never_high"]:
        print("NOTE: a(p) < 0.65 as well, so case C's condition also holds; the spec's order picked "
              f"case {result['case']}. Flag this to the team.")
    print(f"wrote {target} and {fig}")


if __name__ == "__main__":
    main()
