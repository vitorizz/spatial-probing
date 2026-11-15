# Where Do VLMs Lose Spatial Information?

Qwen2.5-VL-3B-Instruct answers left/right questions almost perfectly and front/behind
questions almost never. We probe every language-model layer to see whether the depth
relation is present in the hidden states, pick a layer set from the probe curve with a
rule fixed in advance, and fine-tune only those layers with LoRA on automatically
labelled COCO questions.

Milestone result: the depth relation is linearly decodable at every layer from 19 to 36
(probe accuracy 1.0) while the model gets 0.5% of What'sUp depth questions right. Adapting
only the last 8 blocks on 2,864 COCO questions takes What'sUp depth from 0.5% to 60.8%
without hurting lateral accuracy, VSR or POPE.

| Benchmark | Metric | Base (%) | + LoRA (%) | Change (pp) | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| What'sUp | accuracy | 50.00 | 80.39 | +30.39 | [+26.72, +34.07] |
| What'sUp | lateral | 99.51 | 100.00 | +0.49 | [+0.00, +1.47] |
| What'sUp | depth | 0.49 | 60.78 | +60.29 | [+52.94, +67.65] |
| VSR zeroshot test | accuracy | 81.83 | 82.08 | +0.25 | [-1.49, +1.97] |
| POPE random | accuracy | 88.27 | 90.73 | +2.47 | [+1.83, +3.10] |
| POPE random | F1 | 86.85 | 89.96 | +3.11 | [+2.35, +3.87] |

Intervals are paired bootstraps over image groups (object pairs for What'sUp), 2,000 draws.
Full tables, predictions and the milestone artifacts are under [`results/`](results/);
decisions and observed facts are in [NOTES.md](NOTES.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model runs need a 16 GB GPU; we use a Colab T4 through the two notebooks below. On Colab,
install without torch so the preinstalled CUDA build stays (`grep -v '^torch' requirements.txt`),
and run `pip uninstall -y torchao` (it blocks peft). Offline tests:

```bash
python -m unittest discover -s tests -v
```

Shared settings (model id, pixel limits, prompts, answer options, seed) live in
`configs/base.yaml`. Every script reads and writes under one root: the repo folder locally,
`/content/drive/MyDrive/spatial_probe/` on Colab. Paths below are relative to that root.

## Pipeline

Run the steps in this order. Each one is `python -m src.<module>`.

**1. Baseline and hidden states on What'sUp**

```bash
python -m src.data.whatsup          # download Controlled Images B (408 photos) and check it
python -m src.extract_hidden        # predictions, metrics and last-token hidden states per layer
```

Writes `results/preds/base_whatsup.csv`, `results/metrics/base_whatsup.json`,
`results/hidden/base_whatsup_last.npy`.

**2. Probes and layer selection**

```bash
python -m src.probe                 # linear probe per layer and task, shuffled-label controls
python -m src.select_layers         # applies the frozen rule, writes results/probes/selected_layers.json
```

The rule (cases A/B/C) is in the `select_layers` docstring. For the base model it is case B:
S = layers 29-36, PEFT blocks 28-35. The file refuses to overwrite itself.

**3. Training questions from COCO**

```bash
python -m src.data.coco_spatial exclude                                   # COCO ids used by VSR/POPE
python -m src.data.coco_spatial pairs --split val2017
python -m src.data.coco_spatial pairs --split train2017 --max-images 1500
python -m src.data.coco_spatial download --split val2017                  # and train2017
python -m src.data.coco_spatial depth --split val2017                     # Depth Anything V2, and train2017
python -m src.data.coco_spatial check-depth --split val2017               # confirms larger = closer
python -m src.data.coco_spatial questions --splits val2017 train2017
```

Writes `data/coco_spatial/{train,val}.jsonl` and `stats.json`. Left/right comes from box
centres, front/behind from median predicted depth per object; images that appear in VSR or
POPE are excluded, and the script asserts it. Settings are in `configs/coco_spatial.yaml`.

**4. Targeted LoRA**

```bash
python -m src.train_lora --config configs/lora_targeted.yaml               # full run
python -m src.train_lora --config configs/lora_targeted.yaml --max-examples 200   # smoke test
```

LoRA r=16 on q/k/v/o of the selected blocks only, answer-only loss, one epoch. Writes
`runs/<run_name>/{adapter/, train_log.csv, val_curve.csv, config.yaml, lora_modules.txt}`,
checkpoints every 200 steps and resumes if interrupted. `layers: all` or `random` in the
config gives the comparison runs. Colab launcher: [notebooks/train_lora.ipynb](notebooks/train_lora.ipynb).

**5. Hand check of the training labels**

```bash
python -m src.data.coco_spatial handcheck                                 # fixed 200-question sample
python -m src.handcheck --input data/coco_spatial/handcheck_sample.jsonl \
  --image-root data --evidence-dir data/coco_spatial                      # builds handcheck/gallery.html
python -m src.handcheck_summary                                           # combines the reviewers' exports
```

Open the gallery in a browser, review your range (A #1-67, B #68-134, C #135-200), export
the JSON. Milestone result: 86% of labels accepted (lateral 95%, depth 77%). Details in
[handcheck/README.md](handcheck/README.md).

**6. Evaluation and comparison**

```bash
python -m src.data.prepare --dataset vsr                                  # annotations + referenced COCO images
python -m src.data.prepare --dataset pope

python -m src.eval --dataset whatsup
python -m src.eval --dataset vsr  --variant zeroshot --split test --annotations data/vsr/zeroshot/test.jsonl --image-root data/coco
python -m src.eval --dataset pope --variant random --annotations data/pope/coco_pope_random.json --image-root data/coco
# repeat each with --adapter runs/lora_targeted_milestone/adapter

python -m src.stats --baseline results/metrics/base_*.json --adapter results/metrics/adapted_*.json --out results/comparisons
```

`--limit 8` on `eval` gives a smoke run. The evaluator runs the vision encoder in fp32
(fp16 produced non-finite scores on the T4) and the language model in fp16, for both models.
Colab notebook: [notebooks/evaluate.ipynb](notebooks/evaluate.ipynb).

## Repo map

```text
configs/              base.yaml (shared), coco_spatial.yaml, lora_targeted.yaml
src/common.py         config, paths (local or Drive), seeding
src/model_utils.py    model loading, prompt, option scoring, adapter loading
src/data/whatsup.py   What'sUp download and checks
src/extract_hidden.py baseline predictions + hidden states
src/probe.py          per-layer linear probes
src/select_layers.py  layer-selection rule
src/data/coco_spatial.py  COCO training questions and the hand-check sample
src/train_lora.py     LoRA training
src/handcheck.py      review gallery;  src/handcheck_summary.py combines the exports
src/data/{vsr,pope,prepare}.py  evaluation data
src/eval.py           baseline/adapter evaluation;  src/stats.py paired comparison
notebooks/            train_lora.ipynb, evaluate.ipynb (Colab)
results/milestone/    probing, data, training and hand-check artifacts for the milestone
results/evaluation_milestone_vision_fp32/  predictions, metrics and comparison tables
tests/                offline tests (fake model, synthetic images)
docs/INTEGRATION.md   interfaces between the three parts
```

## Team

| | Owns |
| --- | --- |
| Siddharth Paul | model_utils, What'sUp, hidden-state extraction, probes, layer selection |
| Vito Rizzuto | repo setup, COCO question generator, LoRA training, hand-check sample |
| Connor Cihula | VSR/POPE loaders, evaluation, paired comparison, hand-check gallery |
