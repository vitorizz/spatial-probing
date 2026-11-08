# NOTES

Decisions, deviations from the spec, and observed facts (model paths, dataset fields, version pins).

- 2026-11-07: `requirements.txt` holds lower bounds only. Replace them with exact pins once the Colab environment is confirmed.
- 2026-11-08 (Person A): observed facts and deviations for model_utils / What'sUp.
  - Model config (transformers 5.22.2): 36 LM layers, hidden size 2048, vision depth 32, as the spec expects. Values live under `config.text_config` / `config.vision_config`; `get_arch_facts` also handles the older flat config.
  - Module paths found via `named_modules()`: LM layers `model.language_model.layers`, vision tower `model.visual`.
  - `lm_head` is tied to `embed_tokens` (`tie_word_embeddings: true`). The transformers 5.x load report lists `lm_head.weight` as MISSING; this is cosmetic (checked: same tensor after load).
  - Pixel limits: pass `min_pixels`/`max_pixels` to `AutoProcessor.from_pretrained` exactly as the spec says. Passing `size={"shortest_edge", "longest_edge"}` instead is silently ignored in 5.22 (1,587 tokens instead of 540). `load_processor` asserts the limit took effect. 5.22 prints a FutureWarning about this kwarg; ignore it unless we upgrade.
  - With these limits a What'sUp photo (1280x960) becomes a 38x52 patch grid, and the full prompt is 540 tokens.
  - Option token ids (first token of "x", "X", " x", " X"): left [2359, 5415, 2115, 13727], right [1291, 5979, 1290, 10083], front [6951, 23395, 4065, 14902], behind [29898, 73642, 4815, 42374]; yes [9693, 9454, 9834, 7414], no [2152, 2753, 902, 2308]. Bare "behind" is two tokens (29898 + 484); its first token is used. No overlaps.
  - `configs/base.yaml`: `options.yesno` was `[yes, no]`, which YAML parses as `[True, False]`. Quoted it.
  - transformers 5.x needs `torchvision` to load the Qwen processor (added to requirements.txt; Colab has it preinstalled).
  - gdown 5.x removed `--id`: use `gdown <id> -O <file>` (the spec's annotation command fails as written).
  - What'sUp Controlled B: 408 images, 102 per class, 102 object pairs (groups) with exactly 4 photos each, 18 object names. Filenames agree with the first caption in the annotation JSON for every item.
  - Local Mac (M1 Pro, macOS 14.7, torch 2.15): MPS crashes in conv3d (vision patch embedding), so `default_device()` returns cuda or cpu, never mps. CPU fp16 takes about 34 s per What'sUp item (about 4 h for all 408). Use Colab for full runs.
  - Hidden states: `score_options` keeps `hidden_states[l][0, -1, :]` for l = 0..36. In HF Qwen2 the last entry (l = 36) is after the final RMSNorm.
