"""LoRA fine-tuning of the language model on the COCO spatial questions.

Usage:
    python -m src.train_lora --config configs/lora_targeted.yaml
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

import torch

from src.common import load_config, read_json, results_dir, runs_dir, save_config, set_seed
from src.model_utils import find_module_paths, get_arch_facts, load_model


def choose_layers(cfg: dict[str, Any], num_layers: int) -> list[int]:
    """0-based decoder block indices to adapt."""
    mode = cfg["layers"]
    if mode == "all":
        return list(range(num_layers))
    if mode == "selected":
        return sorted(read_json(results_dir() / "probes" / "selected_layers.json")["peft_layers_to_transform"])
    if mode == "random":
        return sorted(random.Random(cfg["seed"]).sample(range(num_layers), cfg["random_k"]))
    raise ValueError(f"unknown layers mode: {mode}")


def add_lora(model: Any, layers: list[int], lcfg: dict[str, Any]) -> Any:
    """Wrap the model with LoRA on the attention projections of `layers` only.

    The regex names the language-model layer path explicitly, so nothing in the
    vision tower or the merger can match.
    """
    from peft import LoraConfig, get_peft_model

    lm_layers = find_module_paths(model)["lm_layers"]
    pattern = (rf"{re.escape(lm_layers)}\.({'|'.join(map(str, layers))})"
               rf"\.self_attn\.({'|'.join(lcfg['target_modules'])})")
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=pattern, bias="none",
    ))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def check_lora(model: Any, layers: list[int], lcfg: dict[str, Any], out_dir: Path) -> dict[str, int]:
    """The spec's mandatory checks; writes lora_modules.txt and returns parameter counts."""
    from peft.tuners.lora import LoraLayer

    names = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    assert names, "no LoRA modules were added"
    assert not any("visual" in n for n in names), "LoRA module inside the vision tower"
    adapted = {int(re.search(r"\.layers\.(\d+)\.", n).group(1)) for n in names}
    assert adapted == set(layers), f"adapted layers {sorted(adapted)} != requested {layers}"
    assert len(names) == len(layers) * len(lcfg["target_modules"]), f"{len(names)} LoRA modules"
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lora_modules.txt", "w", encoding="utf-8") as f:
        f.write(f"layers (0-based): {layers}\n")
        f.write(f"trainable params: {trainable}\ntotal params: {total}\n\n")
        f.write("\n".join(names) + "\n")
    return {"trainable_params": trainable, "total_params": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = runs_dir() / cfg["run_name"]

    model, processor = load_model(cfg)
    layers = choose_layers(cfg, get_arch_facts(model).num_layers)
    model = add_lora(model, layers, cfg["lora"])
    counts = check_lora(model, layers, cfg["lora"], out_dir)
    save_config({**cfg, "resolved_layers": layers, **counts}, out_dir)
    print(f"{cfg['run_name']}: layers {layers}, {counts['trainable_params']:,} trainable "
          f"of {counts['total_params']:,} params -> {out_dir}")


if __name__ == "__main__":
    main()
