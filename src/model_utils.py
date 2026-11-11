"""Load Qwen2.5-VL, build prompts and score answer options.

Signatures are shared by eval.py, extract_hidden.py and train_lora.py; tell the
team before changing one. Constants come from configs/base.yaml.

Typical use:

    cfg = load_config()
    model, processor = load_model(cfg)
    option_ids = option_token_ids(processor.tokenizer, cfg["options"]["fourway"])
    inputs = make_inputs(processor, image, fourway_question("mug", "knife", cfg), model.device)
    res = score_options(model, inputs, option_ids, output_hidden_states=True)
    res.pred, res.scores, res.hidden.shape   # "left", {...}, (L+1, d)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def default_device() -> str:
    """cuda (Colab) if available, else cpu.

    MPS is never picked automatically: on macOS 14 with torch 2.15, conv3d (the
    vision patch embedding) crashes on MPS. Pass device="mps" to try it anyway.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_processor(cfg: dict[str, Any]) -> Any:
    from transformers import AutoProcessor

    m = cfg["model"]
    # Exactly the spec's call. Do not switch to size={...}: transformers 5.22
    # silently ignores it and images use ~3x more tokens.
    proc = AutoProcessor.from_pretrained(
        m["id"], min_pixels=m["min_pixels"], max_pixels=m["max_pixels"]
    )
    ip = proc.image_processor
    got = getattr(ip, "max_pixels", None) or ip.size.get("longest_edge")
    assert got == m["max_pixels"], f"pixel limit not applied: {got} != {m['max_pixels']}"
    return proc


def load_model(
    cfg: dict[str, Any], adapter: str | None = None, device: str | None = None
) -> tuple[Any, Any]:
    """Return (model, processor) in eval mode, with the LoRA adapter applied if given."""
    import transformers
    from packaging.version import Version
    from transformers import Qwen2_5_VLForConditionalGeneration

    m = cfg["model"]
    device = device or default_device()
    dtype = _DTYPES[m["dtype"]]  # kept on CPU too: fp32 needs ~15 GB of RAM

    # `torch_dtype` was renamed to `dtype` in transformers 4.56.
    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        m["id"],
        attn_implementation=m["attn_implementation"],
        # Loading straight onto MPS runs out of memory during weight init on a
        # 16 GB Mac; load on CPU and move instead.
        device_map="cpu" if device == "mps" else device,
        **{dtype_kw: dtype},
    )
    if device == "mps":
        model = model.to("mps")
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, load_processor(cfg)


@dataclass
class ArchFacts:
    num_layers: int
    hidden_size: int
    vision_depth: int


def get_arch_facts(model_or_config: Any) -> ArchFacts:
    """Handles both config.text_config/vision_config and the older flat config."""
    cfg = getattr(model_or_config, "config", model_or_config)
    text = getattr(cfg, "text_config", None) or cfg
    vision = getattr(cfg, "vision_config", None)
    num_layers = getattr(text, "num_hidden_layers", None) or cfg.num_hidden_layers
    hidden_size = getattr(text, "hidden_size", None) or cfg.hidden_size
    vision_depth = getattr(vision, "depth", None) if vision is not None else None
    if vision_depth is None:
        vision_depth = getattr(vision, "num_hidden_layers", -1) if vision is not None else -1
    return ArchFacts(int(num_layers), int(hidden_size), int(vision_depth))


def find_module_paths(model: Any) -> dict[str, str]:
    """Return {"lm_layers": ..., "visual": ...} found via named_modules()."""
    n_layers = get_arch_facts(model).num_layers
    lm_layers = visual = None
    for name, mod in model.named_modules():
        parts = name.split(".")
        if visual is None and parts[-1] == "visual":
            visual = name
        if (
            lm_layers is None
            and parts[-1] == "layers"
            and "visual" not in parts
            and isinstance(mod, torch.nn.ModuleList)
            and len(mod) == n_layers
        ):
            lm_layers = name
    if lm_layers is None or visual is None:
        raise RuntimeError(f"module paths not found: lm_layers={lm_layers}, visual={visual}")
    return {"lm_layers": lm_layers, "visual": visual}


def fourway_question(o1: str, o2: str, cfg: dict[str, Any]) -> str:
    return cfg["prompts"]["fourway"].format(o1=o1, o2=o2)


def chat_text(processor: Any, question: str) -> str:
    """Chat-template string for one image + question, with the generation prompt."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_inputs(processor: Any, image: Any, question: str, device: Any = None) -> Any:
    """Processor outputs for one (image, question), moved to `device` (default: auto)."""
    if isinstance(image, str):
        from PIL import Image

        image = Image.open(image)
    if getattr(image, "mode", "RGB") != "RGB":
        image = image.convert("RGB")
    text = chat_text(processor, question)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs.to(device or default_device())


def option_token_ids(tokenizer: Any, options: list[str]) -> dict[str, list[int]]:
    """First token ids of "x", "X", " x", " X" per option; asserts no overlap."""
    out: dict[str, list[int]] = {}
    for opt in options:
        ids: list[int] = []
        for v in (opt, opt.capitalize(), " " + opt, " " + opt.capitalize()):
            tid = tokenizer.encode(v, add_special_tokens=False)[0]
            if tid not in ids:
                ids.append(tid)
        out[opt] = ids
    seen: dict[int, str] = {}
    for opt, ids in out.items():
        for tid in ids:
            assert tid not in seen, f"token {tid} shared by {seen[tid]!r} and {opt!r}"
            seen[tid] = opt
    return out


@dataclass
class ScoreResult:
    scores: dict[str, float]
    pred: str
    hidden: Any | None  # (L+1, d) float16 at the last prompt position


@torch.inference_mode()
def score_options(
    model: Any,
    inputs: Any,
    option_ids: dict[str, list[int]],
    output_hidden_states: bool = False,
) -> ScoreResult:
    """One forward pass; score = max logit over each option's first-token ids.

    Ties go to the option listed first. `hidden` is a numpy float16 array of
    hidden_states[l][0, -1, :] for l = 0..L (l = 0 is the input embedding).
    """
    out = model(**inputs, output_hidden_states=output_hidden_states, use_cache=False)
    last = out.logits[0, -1].float()
    scores = {opt: float(last[ids].max()) for opt, ids in option_ids.items()}
    pred = max(scores, key=scores.get)  # first max wins on ties: deterministic
    hidden = None
    if output_hidden_states:
        hidden = (
            torch.stack([h[0, -1] for h in out.hidden_states])
            .to(torch.float16)
            .cpu()
            .numpy()
        )
    return ScoreResult(scores=scores, pred=pred, hidden=hidden)


@torch.inference_mode()
def generate_text(model: Any, processor: Any, inputs: Any, max_new_tokens: int = 5) -> str:
    """Greedy decode; returns only the newly generated text."""
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, inputs["input_ids"].shape[1]:]
    return processor.batch_decode([new], skip_special_tokens=True)[0].strip()


def parse_option(text: str, options: list[str]) -> str | None:
    """First option word found in free-generated text, or None."""
    words = [w.strip(".,!?:;\"'()").lower() for w in text.split()]
    for w in words:
        if w in options:
            return w
    return None

