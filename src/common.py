"""Paths, seeding and config helpers shared by all stages."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/base.yaml, with the config at `path` merged on top."""
    with open(CONFIG_DIR / "base.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def save_config(cfg: dict[str, Any], out_dir: str | Path, name: str = "config.yaml") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def output_root() -> Path:
    """The Drive folder if Drive is mounted (Colab), otherwise the repo root."""
    drive = Path(load_config()["paths"]["drive_root"])
    if drive.parent.exists():
        return drive
    return REPO_ROOT


def data_dir() -> Path:
    return _mk(output_root() / "data")


def results_dir() -> Path:
    return _mk(output_root() / "results")


def runs_dir() -> Path:
    return _mk(output_root() / "runs")


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
