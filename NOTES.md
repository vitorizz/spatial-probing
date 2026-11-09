# NOTES

Decisions, deviations from the spec, and observed facts (model paths, dataset fields, version pins).

- 2026-11-07: `requirements.txt` holds lower bounds only. Replace them with exact pins once the Colab environment is confirmed.
- 2026-11-09 (Person C): evaluation data loaders validated against official annotations, without images or model execution.
  - VSR test annotations: random has 2,195 statements over 1,874 COCO images; zeroshot has 1,222 statements over 715 images. Sources: `https://huggingface.co/datasets/cambridgeltl/vsr_{random,zeroshot}/resolve/main/test.jsonl`.
  - Original COCO POPE: random, popular, and adversarial each have 3,000 questions over 500 images, with 1,500 yes and 1,500 no labels. Sources: `https://github.com/RUCAIBox/POPE/tree/main/output/coco` (not POPEv2).
  - VSR uses numeric filenames and `image_link` to distinguish COCO train2017/val2017; POPE uses `COCO_val2014_*.jpg`. Resolve the exact filenames rather than assuming all evaluation images are in val2017.
  - Loaders accept explicit local paths and use only the standard library. Verified with the project `.venv` (Python 3.14); no dependencies added or installed.
