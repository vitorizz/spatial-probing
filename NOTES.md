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
- 2026-11-09 (Person A): Stage 2b probes and Stage 2c layer selection on the base model (Colab, `python -m src.probe`, `python -m src.select_layers`).
  - All three probes (4-way, lateral, depth) reach about 1.0 from about layer 19 and stay there to the final layer 36. Best layers (first maximum): 4-way 23, lateral 19, depth 21. Depth is decodable at least as early and as well as lateral, so H1 (lateral stays decodable longer than depth) is not supported on What'sUp.
  - Probe vs model: the final-layer depth probe gets 1.0, but the model answers depth questions correctly 0.5% of the time (lateral: probe 1.0, model 0.995). The depth information is present at the last layer but not used for the answer.
  - Layer selection: case B (stays high): p = 24, a(p) = 1.000, a(L) = 1.000. S = layers 29-36 (hidden_states numbering). For PEFT `layers_to_transform` use the 0-based blocks 28-35. Frozen in `results/probes/selected_layers.json`; do not change it after seeing fine-tuning results.
  - Layer 0 is the input embedding of the last prompt token, the same token for every photo, so its vector is identical for all 408 items. Probes there sit at exactly chance (0.25 / 0.50) and the shuffled controls fall below chance by construction (0.198 / 0.441): with identical inputs a probe predicts the training folds' majority class, which is the test fold's minority. Reproduced exactly on synthetic data with a constant layer 0. Constant layers are therefore excluded from the shuffled-control check; at every other layer the controls are within 5 points of chance.
  - Fold standard deviation is above 10 points at some layers between 1 and 18 (spec risk "probe curves too noisy"). The selection depends on layers 19-36, where the spread is near 0. The spec's mitigation (also probe on 2,000 COCO 4-way questions) is optional and not done.
  - Runtime: 7 minutes on a 2-core Colab CPU; about 15 s on an 8-core Mac.
- 2026-11-10 (Person B): reran Person A's pipeline (whatsup, extract_hidden, probe, select_layers) on a second Colab account/Drive. Results reproduced exactly: What'sUp acc 0.500 / lateral 0.995 / depth 0.005; same best layers (4-way 23, lateral 19, depth 21); case B, p = 24, S = 29-36, PEFT layers 28-35. Fold std > 10 points at lateral layers 1, 2, 3, 4, 6 in this run.
- 2026-11-10 (Person B): on Colab, recent `peft` refuses to create LoRA layers when the preinstalled `torchao` 0.10.0 is present ("only versions above 0.16.0 are supported"). We don't use torchao; the training notebook runs `pip uninstall -y torchao` after installing requirements.
- 2026-11-10 (Person B): LoRA smoke test, `lora_targeted` (PEFT layers 28-35), 200 examples = 12 optimizer steps of 2 x 8, Colab T4.
  - Loss falls: 0.884 (mean of steps 1-10) -> 0.486 (steps 11-12).
  - Memory: peak 9.4 GB allocated of 15 GB, so batch 2 x accumulation 8 fits; no fallback needed.
  - Speed: 20.1 s per optimizer step (16 examples), 1.26 s per example.
  - Extrapolated: milestone run (2,864 questions, 179 steps) about 60 min of training; full run (12,000 questions, 750 steps) about 4.2 h of training per run, plus about 3 min per 300-question validation pass.
  - Generated val set (148 questions), base model at step 0: acc 0.703, lateral 0.892, depth 0.514. After 12 steps: 0.710 / 0.905 / 0.514.
