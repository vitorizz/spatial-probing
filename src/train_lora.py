"""LoRA fine-tuning of the language model on the COCO spatial questions.

Usage:
    python -m src.train_lora --config configs/lora_targeted.yaml
    python -m src.train_lora --config configs/lora_targeted.yaml --max-examples 200   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.common import data_dir, load_config, read_json, results_dir, runs_dir, save_config, set_seed
from src.data.coco_spatial import read_jsonl
from src.model_utils import (chat_text, find_module_paths, get_arch_facts, load_model, make_inputs,
                             option_token_ids, score_options)


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


def end_of_turn(processor: Any) -> str:
    """Text the chat template puts after an assistant answer, minus the trailing newline."""
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]}
    reply = {"role": "assistant", "content": [{"type": "text", "text": "ANSWER"}]}
    prompt = chat_text(processor, "Q")
    full = processor.apply_chat_template([user, reply], tokenize=False)
    assert full.startswith(prompt), "chat template does not extend the eval prompt"
    return full[len(prompt):].split("ANSWER", 1)[1].rstrip()


class Collator:
    """Builds a batch from question rows: eval prompt + answer + end-of-turn, with labels
    set to -100 everywhere except the answer and end-of-turn tokens."""

    def __init__(self, processor: Any, image_root: Path):
        self.processor = processor
        self.image_root = image_root
        self.end = end_of_turn(processor)

    def __call__(self, rows: list[dict]) -> dict:
        texts = [chat_text(self.processor, r["question"]) + r["answer"] + self.end for r in rows]
        images = [Image.open(self.image_root / r["image_path"]).convert("RGB") for r in rows]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = torch.full_like(batch["input_ids"], -100)
        for i, r in enumerate(rows):
            target = self.processor.tokenizer(r["answer"] + self.end, add_special_tokens=False)["input_ids"]
            pos = batch["attention_mask"][i].nonzero().squeeze(-1)[-len(target):]
            assert batch["input_ids"][i, pos].tolist() == target, f"answer tokens misaligned for {r['id']}"
            labels[i, pos] = batch["input_ids"][i, pos]
        batch["labels"] = labels
        return batch


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def evaluate(model: Any, processor: Any, rows: list[dict], cfg: dict) -> dict[str, float]:
    """Option-scoring accuracy on generated val questions, overall and per axis."""
    option_ids = {
        "4way": option_token_ids(processor.tokenizer, cfg["options"]["fourway"]),
        "yesno": option_token_ids(processor.tokenizer, cfg["options"]["yesno"]),
    }
    model.eval()
    correct = defaultdict(list)
    for r in rows:
        image = Image.open(data_dir() / r["image_path"]).convert("RGB")
        inputs = make_inputs(processor, image, r["question"], model.device)
        ok = score_options(model, inputs, option_ids[r["qtype"]]).pred == r["answer"]
        correct["all"].append(ok)
        correct[r["axis"]].append(ok)
    model.train()
    return {f"acc_{k}": round(sum(v) / len(v), 4) for k, v in correct.items()}


def train(model: Any, processor: Any, cfg: dict, out_dir: Path, max_examples: int | None) -> None:
    tcfg = cfg["train"]
    bs, accum = tcfg["batch_size"], tcfg["grad_accum"]

    rows = read_jsonl(data_dir() / cfg["data"]["train"])
    random.Random(cfg["seed"]).shuffle(rows)
    if max_examples is not None:
        rows = rows[:max_examples]
    rows = rows * tcfg["epochs"]
    total_steps = len(rows) // (bs * accum)
    rows = rows[: total_steps * bs * accum]  # drop the last incomplete optimizer step
    val_rows = read_jsonl(data_dir() / cfg["data"]["val"])[: tcfg["val_questions"]]

    params =[p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, math.ceil(tcfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    print(f"{len(rows)} examples, {total_steps} optimizer steps of {bs} x {accum}")

    eval_seconds = 0.0

    def log_val(step: int) -> None:
        nonlocal eval_seconds
        t = time.time()
        metrics = evaluate(model, processor, val_rows, cfg)
        eval_seconds += time.time() - t
        append_csv(out_dir / "val_curve.csv", {"step": step, **metrics})
        print(f"step {step}: {metrics}")

    ckpt_path = out_dir / "checkpoint.pt"
    start_step = 0
    if ckpt_path.exists():
        start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)
        for name in ("train_log.csv", "val_curve.csv"):
            drop_rows_after(out_dir / name, start_step)
        print(f"resumed from step {start_step}")
    else:
        for name in ("train_log.csv", "val_curve.csv"):
            (out_dir / name).unlink(missing_ok=True)
        log_val(0)

    # Skip the batches already trained on; the data order is fixed, so this is the data position.
    skip = start_step * accum * bs
    loader = torch.utils.data.DataLoader(
        rows[skip:], batch_size=bs, shuffle=False, collate_fn=Collator(processor, data_dir()),
        num_workers=tcfg["num_workers"],
    )
    model.train()
    step, window, t0 = start_step, [], time.time()
    eval_seconds = 0.0
    for i, batch in enumerate(loader):
        batch = batch.to(model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss
        scaler.scale(loss / accum).backward()
        window.append(loss.item())
        if (i + 1) % accum:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1

        if step % tcfg["log_every"] == 0 or step == total_steps:
            append_csv(out_dir / "train_log.csv", {
                "step": step, "loss": round(sum(window) / len(window), 5),
                "lr": scheduler.get_last_lr()[0],
                "sec_per_step": round((time.time() - t0 - eval_seconds) / (step - start_step), 2),
                "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            })
        window = []
        if step % tcfg["eval_every"] == 0 and step != total_steps:
            log_val(step)
        if step % tcfg["save_every"] == 0 and step != total_steps:
            save_checkpoint(ckpt_path, step, model, optimizer, scheduler, scaler)

    log_val(step)
    model.save_pretrained(out_dir / "adapter")
    ckpt_path.unlink(missing_ok=True)
    print(f"saved adapter -> {out_dir / 'adapter'}")


def save_checkpoint(path: Path, step: int, model: Any, optimizer: Any, scheduler: Any, scaler: Any) -> None:
    """LoRA weights + optimizer, scheduler, scaler and RNG states. Written to a temp file
    and renamed, so a disconnect mid-save leaves the previous checkpoint intact."""
    from peft import get_peft_model_state_dict

    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "lora": get_peft_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
    }, tmp)
    tmp.replace(path)
    print(f"checkpoint at step {step}")


def load_checkpoint(path: Path, model: Any, optimizer: Any, scheduler: Any, scaler: Any) -> int:
    from peft import set_peft_model_state_dict

    state = torch.load(path, map_location="cpu", weights_only=False)
    result = set_peft_model_state_dict(model, state["lora"])
    assert not getattr(result, "unexpected_keys", []), "checkpoint does not match this LoRA setup"
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["rng_cpu"])
    torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state["step"]


def drop_rows_after(path: Path, step: int) -> None:
    """Remove log rows written after the checkpoint, so a resumed run doesn't log them twice."""
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if int(r["step"]) <= step]
    with open(path, "w", newline="", encoding="utf-8") as f:
        if keep:
            w = csv.DictWriter(f, fieldnames=list(keep[0]))
            w.writeheader()
            w.writerows(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="train on the first N examples only (smoke test); writes to runs/<run>_smoke<N>")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_name = cfg["run_name"] + (f"_smoke{args.max_examples}" if args.max_examples else "")
    out_dir = runs_dir() / run_name
    if (out_dir / "adapter" / "adapter_config.json").exists():
        print(f"{out_dir / 'adapter'} already exists; this run is finished. Delete it to retrain.")
        return

    model, processor = load_model(cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    layers = choose_layers(cfg, get_arch_facts(model).num_layers)
    model = add_lora(model, layers, cfg["lora"])
    counts = check_lora(model, layers, cfg["lora"], out_dir)
    save_config({**cfg, "run_name": run_name, "max_examples": args.max_examples,
                 "resolved_layers": layers, **counts}, out_dir)
    print(f"{run_name}: layers {layers}, {counts['trainable_params']:,} trainable "
          f"of {counts['total_params']:,} params -> {out_dir}")

    train(model, processor, cfg, out_dir, args.max_examples)


if __name__ == "__main__":
    main()
