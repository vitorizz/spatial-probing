# spatial-probe

Probing and layer-targeted LoRA for spatial relations in Qwen2.5-VL-3B-Instruct.

> Skeleton. Person C finalizes this file (what it does, how to run each step, current results).

## Setup

```bash
pip install -r requirements.txt   # on Colab, keep the preinstalled torch
huggingface-cli login            # optional: avoids download rate limits
```

On Colab, mount Drive first; outputs go to `/content/drive/MyDrive/spatial_probe/`.
Locally, outputs go under the repo root.

Shared constants (model id, image pixel limits, prompt templates, seed) live in
`configs/base.yaml`. Load them with `src.common.load_config()`; do not redefine them.

## Repo layout

```
configs/base.yaml   shared constants
src/common.py       paths, seeding, config I/O
src/model_utils.py  shared model interface (load, prompt, score)
src/data/vsr.py     VSR annotation loader and validation command
src/data/pope.py    original COCO POPE annotation loader and validation command
src/eval.py         baseline/adapter evaluation and prediction artifacts
src/stats.py        metric summaries, paired comparisons, results tables
src/handcheck.py    offline training-question review gallery
handcheck/         gallery assets, input contract, review instructions
```

Run code from the repo root as modules: `python -m src.<name>`.

## Evaluation data

The loaders use local annotation and image files; they do not download data or
load the model. These commands need only Python's standard library. From the
repo root, use the project venv:

```bash
source .venv/bin/activate
python -m src.data.vsr --annotations data/vsr/zeroshot/test.jsonl \
  --image-root data/coco --variant zeroshot --split test
python -m src.data.pope --annotations data/pope/coco_pope_random.json \
  --image-root data/coco --variant random
```

Obtain VSR's `train.jsonl`, `dev.jsonl`, or `test.jsonl` from the official
[zeroshot](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot/tree/main)
or [random](https://huggingface.co/datasets/cambridgeltl/vsr_random/tree/main)
dataset. The variant and split flags describe the supplied file; the loader
cannot infer them from its contents. VSR images come from both COCO `train2017/`
and `val2017/`, as specified by each record's `image_link`.

Use the original POPE files from
[output/coco](https://github.com/RUCAIBox/POPE/tree/main/output/coco), with
`--variant random`, `popular`, or `adversarial` as appropriate. These `.json`
files contain one JSON object per line and refer to COCO `val2014/` images.
POPEv2 is outside this loader's scope.

`--image-root` accepts either a folder containing the exact image filenames or
a root containing the COCO split folders above. Missing image files are errors;
the loaders do not skip questions or substitute filenames from other COCO years.
`--no-check-images` allows annotation-only validation before images are present.
Image checks establish file existence; decoding happens during model evaluation.

Both loaders return dictionaries with `id`, `dataset`, `variant`, `image_path`,
`coco_image_id`, `group`, and a string `label` (`yes` or `no`). VSR also retains
`split`, `caption`, and `relation`; POPE retains `question`. Prompts are applied
later by evaluation using the shared config. Repeated images share a group but
retain separate question IDs. VSR IDs use the record index, so keep the exact
annotation file fixed across baseline and adapter runs.

Run the offline data checks with `python -m unittest discover -s tests -v`.

## Evaluation

`src.eval` uses the shared model interface for option scoring, without extracting
hidden states. Full model execution requires Person A's `model_utils.py`
implementation to be integrated. What'sUp also requires their `src/data/whatsup.py`
loader and its `check` function. The evaluator does not download datasets.

With the venv active and the declared requirements installed:

```bash
python -m src.eval --dataset vsr --variant zeroshot --split test \
  --annotations data/vsr/zeroshot/test.jsonl --image-root data/coco
python -m src.eval --dataset pope --variant adversarial \
  --annotations data/pope/coco_pope_adversarial.json --image-root data/coco
python -m src.eval --dataset whatsup --data-root data
```

VSR defaults to `zeroshot`/`test`; POPE defaults to `random`. Run each desired
variant separately. Add `--adapter runs/targeted/adapter --model-name targeted`
to evaluate an adapter on the same questions and prompts. `--config` merges a
YAML override over the shared base configuration.

Add `--limit 8` for a smoke run. Only selected image paths are required for these
runs, and output filenames include `_limit8` to keep them separate from full
evaluations. Full What'sUp runs invoke Person A's dataset checks first.

Results go under `results_dir()` (local `results/` or Drive), or an explicit
`--out` folder. For example, the VSR baseline writes:

```text
results/preds/base_vsr_zeroshot_test.csv
results/metrics/base_vsr_zeroshot_test.json
```

CSV rows retain question IDs, image groups, true labels, formatted questions,
predictions, correctness, and option scores. JSON summaries include accuracy,
per-class accuracy, confusion, and dataset-specific metrics: lateral/depth
accuracy for What'sUp, per-relation accuracy for VSR, and precision, recall, F1,
and yes ratio for POPE. Absent-class accuracies are null; undefined POPE
precision/recall/F1 are zero.

Run metadata records the configuration, annotation and question-manifest hashes,
adapter identity, installed package versions, Git revision, runtime, and CSV
checksum. The manifest hash covers question IDs, groups, labels, prompts, and
image filenames; it does not hash image contents. These artifacts support later
paired baseline/adapter comparisons. Existing outputs require `--overwrite` or
a different model name. Interrupted inference produces no final result files;
automatic resume is not implemented yet.

The offline suite uses tiny real images and a fake model API, so it needs only
NumPy, PyYAML, and Pillow from the existing requirements. It does not measure
model accuracy. Run it with `python -m unittest discover -s tests -v`.

## Training-data hand-check

Once Person B provides the fixed review sample and images:

```bash
python -m src.handcheck --input data/coco_spatial/handcheck_sample.jsonl
```

Open `handcheck/gallery.html` in a browser to inspect images, optional boxes and
label evidence, record accept/reject/unsure decisions, and export review JSON.
The file works offline without a server. See [handcheck/README.md](handcheck/README.md)
for the input schema, review checklist, and save/restore workflow. The gallery
tool is ready independently of the final sample; no real questions have been
hand-checked yet.

## Baseline / adapter comparison

After evaluating the base model and adapter with `src.eval`, compare their
saved artifacts without loading a model:

```bash
python -m src.stats \
  --baseline results/metrics/base_vsr_zeroshot_test.json \
  --adapter results/metrics/targeted_vsr_zeroshot_test.json \
  --out results/comparisons/targeted
```

Each of `--baseline` and `--adapter` accepts multiple metric JSON paths. Supply
all desired datasets/variants to create one table; files are matched by
dataset/variant/split, regardless of list order. Each side must contain exactly
one run for every benchmark being compared. Keep the usual `<results>/metrics/`
and `<results>/preds/` layout together when copying artifacts between machines.
Older extraction-only artifacts need conversion or reevaluation with `src.eval`.

The command writes `comparison.json`, `comparison.csv`, and `comparison.md` in
the output folder. JSON/CSV store rates on the 0–1 scale. The Markdown table shows
percentages and adapter-minus-baseline changes in **percentage points**, with
95% confidence intervals by default. POPE precision/recall/F1 and yes ratio are
included; What'sUp includes lateral/depth accuracy. VSR reports overall accuracy.

Comparison verifies the CSV and question-manifest checksums, then matches every
question ID. Labels, groups, prompts, image filenames, source annotation hashes,
answer-option order, model configuration, and evaluation seed must agree. It
recomputes metrics from predictions and never silently drops unmatched questions.
The baseline must have no adapter, and the second run must identify its adapter.
Recorded environment differences are retained in JSON for review.

Confidence intervals use 2,000 paired group-bootstrap draws. What’sUp resamples
whole object pairs; VSR/POPE resample whole images. Each draw uses the same groups
for both models and recomputes metrics from the sampled question counts, including
nonlinear POPE metrics. This preserves related questions and accounts for unequal
group sizes. Intervals reflect evaluation-group sampling, not variation across
separate training runs. Metrics with fewer than two contributing groups, or fewer
than two valid resamples, have no interval. The JSON records valid resample counts.

`--bootstrap`, `--confidence`, and `--seed` override the resampling settings; the
seed defaults to the baseline's evaluation seed. Limited evaluations require
`--allow-limited` and are visibly labeled in the table. Existing report files
require a new output folder or `--overwrite`.
