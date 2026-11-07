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
```

Run code from the repo root as modules: `python -m src.<name>`.
