"""Load Qwen2.5-VL, build prompts and score answer options.

Signatures are shared by eval.py, extract_hidden.py and train_lora.py; tell the
team before changing one. Constants come from configs/base.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def load_model(cfg: dict[str, Any], adapter: str | None = None) -> tuple[Any, Any]:
    """Return (model, processor) in eval mode, with the LoRA adapter applied if given."""
    raise NotImplementedError


@dataclass
class ArchFacts:
    num_layers: int
    hidden_size: int
    vision_depth: int


def get_arch_facts(model_or_config: Any) -> ArchFacts:
    """Handles both config.text_config/vision_config and the older flat config."""
    raise NotImplementedError


def find_module_paths(model: Any) -> dict[str, str]:
    """Return {"lm_layers": ..., "visual": ...} found via named_modules()."""
    raise NotImplementedError


def fourway_question(o1: str, o2: str, cfg: dict[str, Any]) -> str:
    raise NotImplementedError


def chat_text(processor: Any, question: str) -> str:
    """Chat-template string for one image + question, with the generation prompt."""
    raise NotImplementedError


def make_inputs(processor: Any, image: Any, question: str, device: str = "cuda") -> Any:
    raise NotImplementedError


def option_token_ids(tokenizer: Any, options: list[str]) -> dict[str, list[int]]:
    """First token ids of "x", "X", " x", " X" per option; asserts no overlap."""
    raise NotImplementedError


@dataclass
class ScoreResult:
    scores: dict[str, float]
    pred: str
    hidden: Any | None  # (L+1, d) float16 at the last prompt position


def score_options(
    model: Any,
    inputs: Any,
    option_ids: dict[str, list[int]],
    output_hidden_states: bool = False,
) -> ScoreResult:
    raise NotImplementedError


def generate_text(model: Any, processor: Any, inputs: Any, max_new_tokens: int = 5) -> str:
    raise NotImplementedError
