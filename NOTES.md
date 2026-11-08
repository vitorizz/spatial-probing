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
- 2026-11-08 (Person A): What'sUp baseline on Colab (T4, base model, `python -m src.extract_hidden`).
  - Accuracy 0.500 overall, lateral 0.995 (203/204), depth 0.005 (1/204). Far below the 85% depth stop rule, so the project premise holds.
  - Determinism: 10/10 identical predictions on a rerun, max score difference 0.0. Free-generation agreement with option scoring: 20/20 (1.0).
  - Speed: 1.88 items/s on the T4 (408 items in 3 min 36 s); free generation 1.68 items/s. Model load about 30 s after a ~50 s download.
  - Outputs on Drive under `spatial_probe/results/`: `preds/base_whatsup.csv`, `preds/base_whatsup_gen.csv`, `metrics/base_whatsup.json`, `hidden/base_whatsup_last.npy` (408 x 37 x 2048 float16) + `hidden/base_whatsup_index.csv`.
  - Colab setup: install requirements without torch/torchvision (`grep -v -E '^torch' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt`). A plain `pip install -r requirements.txt` replaced Colab's CUDA torch with a CPU-only build ("Torch not compiled with CUDA enabled"); only "Disconnect and delete runtime" undoes it.
  - The first Colab attempt stopped silently at 91/408 with no traceback (cause unknown, RAM was fine). Rerunning with output to a log finished cleanly with exit code 0.
