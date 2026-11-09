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
