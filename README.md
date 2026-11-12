# spatial-probe

Study spatial relations in **Qwen2.5-VL-3B-Instruct**: measure its answers, probe
which language-model layers encode spatial information, then evaluate whether
LoRA applied to selected layers improves its answers. The four spatial answers
are `left`, `right`, `front`, and `behind`.

| Dataset | Task | Reported metrics |
| --- | --- | --- |
| What'sUp Controlled B | Four-way spatial relation | Overall, lateral, depth, and per-class accuracy |
| VSR | Decide whether a spatial statement is true | Accuracy, per-class and per-relation accuracy |
| Original COCO POPE | Answer object-presence questions | Accuracy, precision, recall, F1, predicted-yes ratio |

## Current status

This branch integrates A/B's model, probing, and training code with Person C's
evaluation, review, and reporting tools. A's baseline/probing artifacts and B's
milestone adapter are available in the local, Git-ignored `drive/` download.
Person C's benchmark inference has **not** been run yet. Offline tests use a
fake model and synthetic fixtures; their scores are not project results.

| Component | Available here | Remaining dependency |
| --- | --- | --- |
| VSR/POPE data | Milestone VSR zeroshot/test and POPE random annotations/images prepared and validated locally | None for the selected inputs |
| Evaluation | All three datasets, optional adapter, prediction CSVs and run metadata; Colab evaluation notebook | GPU smoke tests and full benchmark runs |
| Hand-check gallery | B's final question format, joined boxes/depth evidence, offline review/export/restore | Sample selection and reviewer assignments have no named owner; deferred |
| Comparisons | Paired confidence intervals and Markdown/CSV/JSON tables | Actual baseline and adapter predictions |
| Model loading and scoring | A's implementation in `src/model_utils.py` | Verification in the selected GPU runtime |
| Probing, layer selection, COCO training data, LoRA | A/B modules integrated; milestone artifacts supplied | Training exclusion artifact and final workflow verification |

See [integration handoffs](docs/INTEGRATION.md) for the supplied artifacts and
remaining review, exclusion, and evaluation work.

## Setup and offline checks

Run from the repo root using the project venv. If it does not exist yet, create
it with `python3 -m venv .venv`, then activate it:

```bash
source .venv/bin/activate
```

The offline Python suite needs only these packages, already declared in
`requirements.txt`:

```bash
python -m pip install numpy pyyaml pillow
python -m unittest discover -s tests -v
```

Optional JavaScript development checks use Node's built-in test runner, without
npm packages:

```bash
node tests/test_handcheck.js
```

For local model execution, install the declared requirements in the execution
environment with `python -m pip install -r requirements.txt`. For Colab, use the
evaluation notebook's setup below, which preserves its CUDA Torch/torchvision.
The local venv has been checked for the offline workflow; real model execution
still needs runtime verification. See [NOTES.md](NOTES.md) for observed dataset
and implementation details.

Run modules as `python -m src.<name>`. Shared model ID, pixel limits, prompts,
answer options, and seed live in `configs/base.yaml`; load them through
`src.common.load_config()` rather than redefining them. `--config` on the
evaluator merges YAML overrides over that shared base.

## Project workflow

1. **Prepare evaluation data.** Freeze the selected variants, splits, annotation
   files, and question IDs. Exclude their COCO image IDs from training data.
2. **Evaluate the base model** on What'sUp, VSR, and the desired POPE variants.
   Retain predictions and run metadata for paired comparisons.
3. **Probe and select layers (A).** Extract hidden states, train probes, and
   freeze the selected layers before inspecting fine-tuning results.
4. **Generate and review training questions (B + everyone).** B supplies the
   questions and evidence; everyone reviews their assigned IDs from a fixed
   200-question sample. The task sheet does not name an owner for selecting that
   sample or assigning reviewers; both are deferred pending team coordination.
5. **Train targeted LoRA (B).** Retain the adapter, base-model identity, training
   configuration, selected-layer artifact, and exclusion-check results.
6. **Evaluate the adapter and compare (C).** Use the same questions and prompts
   as the baseline, generate the before/after table, and report measured results.

A/B's modules for steps 3–5 are integrated, and the supplied milestone training
run is complete. The fixed review sample and training exclusion artifact remain
to be confirmed. Available evaluation and review commands follow below.

## Prepare evaluation data

Fetch the official annotations and only the images they reference:

```bash
python -m src.data.prepare --dataset vsr --root data
python -m src.data.prepare --dataset pope --root data
```

Defaults match evaluation: VSR `zeroshot`/`test`, POPE `random`. Use `--variant`
for other variants and `--split` for another VSR split. `--root` defaults to the
shared `data_dir()` when omitted. No model or Torch installation is needed for
preparation. This command writes the layout expected by the local-file loaders
and evaluator, which themselves do not download data:

```text
data/
  vsr/zeroshot/test.jsonl
  pope/coco_pope_random.json
  pope/coco_pope_popular.json
  pope/coco_pope_adversarial.json
  coco/train2017/000000000001.jpg
  coco/val2017/000000000002.jpg
  coco/val2014/COCO_val2014_000000000003.jpg
```

Image filenames above illustrate the layout, not actual required IDs. Keep the
exact filenames referenced by the annotations.

`--annotations-only` skips images; `--limit 8` prepares images for the first eight
questions for use with evaluation's matching smoke-run option. Rerunning the
command validates cached files and resumes at missing files. New files are
published only after validating complete downloads and decoding images.

Preparation writes URLs, annotation/image hashes, question IDs, and COCO image
IDs under `data/prepared/`. Full, limited, and annotation-only selections have
separate manifests. Subsequent runs of the same selection reject changed cached
annotations/images rather than silently replacing the frozen benchmark. A
cached file that fails validation must be inspected and restored before reuse.

Get VSR's `train.jsonl`, `dev.jsonl`, or `test.jsonl` from the official
[zeroshot](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot/tree/main)
or [random](https://huggingface.co/datasets/cambridgeltl/vsr_random/tree/main)
dataset. VSR's `image_link` identifies its images in COCO `train2017` or
`val2017`. Get original POPE files from
[output/coco](https://github.com/RUCAIBox/POPE/tree/main/output/coco); they refer to
COCO `val2014`. These `.json` files contain one object per line. POPEv2 is outside
the loader's scope. The preparation command uses VSR's `image_link` and the
original COCO POPE filenames to locate images. Annotation files themselves do
not contain image bytes. Existing local files can also be used directly.

Validate each selected dataset before a full model run:

```bash
python -m src.data.vsr --annotations data/vsr/zeroshot/test.jsonl \
  --image-root data/coco --variant zeroshot --split test
python -m src.data.pope --annotations data/pope/coco_pope_random.json \
  --image-root data/coco --variant random
```

`--image-root` accepts a root of COCO split folders or a flat folder containing
the exact filenames. Validation checks file existence; images are decoded during
evaluation. Missing files are errors; questions are never silently skipped.
`--no-check-images` validates annotations before images are present. Variant and
split flags describe the supplied file; loaders cannot infer these choices.

## Evaluate the base model and adapter

With the model runtime installed, run datasets separately:

```bash
python -m src.eval --dataset vsr --variant zeroshot --split test \
  --annotations data/vsr/zeroshot/test.jsonl --image-root data/coco
python -m src.eval --dataset pope --variant random \
  --annotations data/pope/coco_pope_random.json --image-root data/coco
python -m src.eval --dataset whatsup --data-root drive/data
```

The supplied download contains all 408 What'sUp images and annotations. Full
What'sUp runs invoke A's integrated dataset checks. Evaluation uses the shared
option-scoring interface without requesting hidden states.

After loading the base model and optional adapter, `src.eval` converts the vision
encoder to FP32 and keeps the language backbone in FP16. This addresses the
reported T4 NaNs at VSR item `vsr:zeroshot:test:426`, which became finite in an
isolated check. Full benchmark validation remains pending. Shared attention
settings and prompts are unchanged. The effective saved config records
`model.vision_dtype: float32` and `model.dtype: float16`; environment metadata
records `vision_dtype` and `language_dtype`, with legacy `dtype` referring to
the language backbone. This happens in every evaluation subprocess.

VSR defaults to `zeroshot`/`test`; POPE defaults to `random`. To evaluate another
variant, change both its annotation path and flag. VSR also supports `random`;
POPE supports `popular` and `adversarial`. Settle benchmark selection before
interpreting adapter results.

For an adapter, repeat the same command with:

```text
--adapter drive/runs/lora_targeted/adapter --model-name targeted
```

Add `--limit 8` for a smoke run. Only selected image paths are required; output
filenames include `_limit8` to separate these from full evaluations. Existing
outputs require another model name or explicit `--overwrite`. Automatic resume
is not implemented: interruption before scoring finishes produces no final files.

Outputs use `src.common.results_dir()`: local `results/`, or
`/content/drive/MyDrive/spatial_probe/results/` when the shared helpers detect
the configured Drive location. `--out` chooses another results root. For example:

```text
results/preds/base_vsr_zeroshot_test.csv
results/metrics/base_vsr_zeroshot_test.json
```

CSV rows retain IDs, groups, labels, formatted questions, predictions, correctness,
and option scores. JSON includes metrics, configuration, annotation and question
hashes, adapter identity, package versions, Git revision, runtime, and CSV
checksum. The question hash covers IDs, groups, labels, prompts, and image
filenames; it does not hash image contents. Keep source images fixed too.
Absent-class accuracies are null; undefined POPE precision/recall/F1 are zero.

### Milestone evaluation inputs

The selected evaluations are What'sUp Controlled B (408 questions), VSR
zeroshot/test (1,222 questions, 715 images), and original POPE random (3,000
questions, 500 images). VSR and POPE use the defaults above; other variants are
outside this milestone comparison. Prepared data stays in the ignored `data/`
folder, with source URLs, hashes, and image checks in `data/prepared/`.
What'sUp and the milestone adapter stay in the supplied `drive/` download.

The selected VSR/POPE image IDs have zero overlap with either the supplied
1,356 training images or 73 validation images. This check does not replace B's
missing original exclusion artifact or verify benchmarks we did not select.
See [NOTES.md](NOTES.md) for the exact checked annotation/data hashes.

The local RTX 3060 Mobile has 6 GB VRAM. The current loader places the full
model on CUDA, but the published 16-bit weights alone occupy about 7.5 GB
([weight index](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/model.safetensors.index.json)).
This GPU therefore needs an offload strategy before the commands below can be
used locally. The Colab notebook below uses a larger GPU with the shared loader
and the evaluator's vision-FP32/language-FP16 policy described above. Local
offload has not been implemented or tested.

### Run in Google Colab

Open [notebooks/evaluate.ipynb](notebooks/evaluate.ipynb) in Colab. Select a
16 GB class or larger GPU; the supplied A baseline used a Tesla T4. Paid plans
provide premium GPU access subject to
[availability](https://research.google.com/colaboratory/faq.html), so check the
assigned runtime rather than assuming a particular card.

Place these two files together in one Google Drive folder:

| File in Drive | Local source |
| --- | --- |
| `evaluate.ipynb` | `notebooks/evaluate.ipynb` |
| `evaluation_inputs.zip` | `data/colab/evaluation_inputs.zip` (prepared upload artifact, ignored by Git) |

Both upload-ready files are available locally together in `data/colab/`.

Set `DRIVE_ROOT` in the first cell to that folder's mounted path. The default is
`/content/drive/MyDrive/spatial_probe`; Colab does not automatically use the open
notebook's Drive directory. No GitHub checkout, token, or push is needed to run.

The archive already contains the necessary files in this layout:

```text
src/                                      repository Python source
configs/base.yaml                         shared model/prompt config
requirements.txt                          declared package requirements
data/controlled_clevr/                    all 408 What'sUp JPEGs
data/controlled_clevr_dataset.json         What'sUp annotations
data/vsr/zeroshot/test.jsonl               frozen VSR annotations
data/pope/coco_pope_random.json            frozen POPE annotations
data/coco/{train2017,val2017,val2014}/     1,215 referenced COCO JPEGs
data/prepared/vsr_zeroshot_test.json       VSR image/source manifest
data/prepared/pope_random.json             POPE image/source manifest
data/coco_spatial/{train,val}.jsonl        milestone records for separation checks
runs/lora_targeted/adapter/adapter_config.json
runs/lora_targeted/adapter/adapter_model.safetensors
```

The notebook unpacks the bundle into fresh runtime storage, creates a `.venv`
that reuses Colab's CUDA Torch/torchvision, and installs only the evaluation dependencies already in
`requirements.txt`. It pins Transformers 5.22.1 and PEFT 0.21.2 from the supplied
A/B artifacts, and keeps Colab's existing Torch/torchvision versions. It applies
B's documented `torchao` removal workaround in the disposable Colab environment.
These setup actions happen when the user
executes the cells; they have not been run by Codex.

Run setup and both smoke tests first, then the six separate full evaluation
cells and the comparison cell. Rerun all six full evaluations for this precision
revision, including previously completed What'sUp runs, under the new
`evaluation_milestone_vision_fp32` results folder. Earlier all-FP16 results are
not valid inputs to the new paired comparison. Input checks verify the frozen annotations,
adapter, image hashes, image decoding, and separation from milestone training/
validation image IDs. Evaluation data is read from the bundle; packages and
base-model weights still require internet downloads.

Manifests and results persist under `DRIVE_ROOT/results/evaluation_milestone_vision_fp32`;
the supplied files stay intact. After reconnecting, rerun setup and input checks,
then only unfinished evaluation cells. The notebook requires the same bundle
fingerprint and preparation manifests before continuing. Without a Git checkout,
metric files have a null Git revision; `evaluation_inputs.sha256` and the saved
environment snapshot identify the exact code/input archive used. Rebuild the
bundle if the source, config, or selected inputs change.

The notebook's CPU preparation and command wiring have been checked locally.
Actual Colab package setup, GPU inference, and measured results remain pending
the user's execution.

### Evaluation commands for a configured GPU runtime

GPU execution is performed by the user. Once a compatible execution route and
the declared model requirements are configured in the project venv, confirm
CUDA before evaluation; the shared model loader otherwise falls back to CPU:

```bash
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; finish GPU setup first"; print(torch.cuda.get_device_name(0))'
```

First check eight What'sUp questions with each model:

```bash
.venv/bin/python -m src.eval --dataset whatsup --data-root drive/data \
  --out results/evaluation_milestone_vision_fp32 --limit 8
.venv/bin/python -m src.eval --dataset whatsup --data-root drive/data \
  --out results/evaluation_milestone_vision_fp32 --limit 8 \
  --adapter drive/runs/lora_targeted/adapter
```

Limited results are smoke checks, not milestone scores. Once both succeed,
remove `--limit 8` to evaluate all 408 questions. For VSR and POPE, use the
full-dataset commands above with `--out results/evaluation_milestone_vision_fp32` and repeat
each with `--adapter drive/runs/lora_targeted/adapter`. Keep the same config,
data, and environment for all six full runs. The default model names are `base`
and `adapted`; no model-name flag is needed. The separate results root preserves
A's supplied extraction artifacts and produces the format needed by `src.stats`.

After all six full runs, build the paired table without loading a model:

```bash
.venv/bin/python -m src.stats \
  --baseline results/evaluation_milestone_vision_fp32/metrics/base_whatsup.json \
    results/evaluation_milestone_vision_fp32/metrics/base_vsr_zeroshot_test.json \
    results/evaluation_milestone_vision_fp32/metrics/base_pope_random.json \
  --adapter results/evaluation_milestone_vision_fp32/metrics/adapted_whatsup.json \
    results/evaluation_milestone_vision_fp32/metrics/adapted_vsr_zeroshot_test.json \
    results/evaluation_milestone_vision_fp32/metrics/adapted_pope_random.json \
  --out results/evaluation_milestone_vision_fp32/comparisons
```

## Review training questions

Once the team supplies the fixed sample and reviewer assignments:

```bash
python -m src.handcheck --input data/coco_spatial/handcheck_sample.jsonl \
  --image-root drive/data --evidence-dir drive/data/coco_spatial
```

Open `handcheck/gallery.html` in a browser. It embeds images and optional boxes
and label evidence, works offline, and records accept/reject/unsure decisions,
corrections, and notes. Export JSON to share or safeguard decisions. Restore
checks the sample fingerprint before replacing a session.

See [handcheck/README.md](handcheck/README.md) for the input format and checklist.
Agree the approximately 70 IDs per person within the fixed 200-question sample
before reviewing. Neither `tasks.pdf` nor the supplied artifacts identifies who
selects the sample or assigns reviewers, so those tasks are deferred. The gallery
does not select or allocate the sample. No real training questions have been
hand-checked on this branch yet.

## Compare baseline and adapter

After both runs complete, compare saved artifacts without loading a model:

```bash
python -m src.stats \
  --baseline results/metrics/base_vsr_zeroshot_test.json \
  --adapter results/metrics/targeted_vsr_zeroshot_test.json \
  --out results/comparisons/targeted
```

Both file arguments accept multiple paths, matched by dataset/variant/split
regardless of list order. Each side must contain exactly one run per benchmark.
Keep the `<results>/metrics/` and `<results>/preds/` layout together when copying
artifacts. This consumes `src.eval` schema-1 outputs; older extraction-only
artifacts need conversion or reevaluation.

Reports are `comparison.json`, `comparison.csv`, and `comparison.md`. JSON/CSV
store rates on the 0–1 scale. Markdown shows percentages and adapter-minus-baseline
changes in **percentage points**, with 95% confidence intervals by default. It
includes overall accuracy, What'sUp lateral/depth accuracy, and POPE precision,
recall, F1, and yes ratio.

Comparison verifies file/question hashes and matches every ID. Labels, groups,
prompts, image filenames, source annotation hashes, answer-option order, model
configuration, and evaluation seed must agree. Metrics are recomputed from
predictions. Recorded environment differences are retained in JSON.

Intervals use 2,000 paired group-bootstrap draws: whole object pairs for What'sUp
and whole images for VSR/POPE. Each draw uses the same groups for both models and
recomputes metrics from pooled question counts, including nonlinear POPE metrics.
Intervals reflect evaluation-group sampling, not variation across training runs.
Fewer than two contributing groups or valid resamples yields no interval.

`--bootstrap`, `--confidence`, and `--seed` override settings; the seed defaults
to the baseline's evaluation seed. Limited runs require `--allow-limited` and
are labeled. Existing reports require another output folder or `--overwrite`.

## Measured results

| Benchmark | Base model | Targeted LoRA | Status |
| --- | --- | --- | --- |
| What'sUp Controlled B | Pending | Pending | A's baseline supplied; evaluation in C's output format pending |
| VSR zeroshot/test | Pending | Pending | Data prepared; GPU inference pending |
| Original COCO POPE random | Pending | Pending | Data prepared; GPU inference pending |

The supplied milestone adapter is ready for benchmark evaluation. Annotation
validation and synthetic tests establish tool behavior; they do not establish
model accuracy or fine-tuning gains. Replace this table with measured results and
links to comparison artifacts after full evaluation.

## Repo map

```text
configs/base.yaml    shared model, prompts, options, seed, Drive root
src/common.py        config I/O, paths, seeding, JSON helpers
src/model_utils.py   model loading, prompts, option scoring, PEFT adapter loading
src/data/whatsup.py  What'sUp loading and full-dataset checks
src/data/vsr.py      VSR loader and validation command
src/data/pope.py     original COCO POPE loader and validation command
src/data/prepare.py  cached annotation/image downloads and preparation manifests
src/eval.py          baseline/adapter inference and saved artifacts
src/stats.py         metric summaries, paired comparisons, result tables
src/handcheck.py     offline training-question review gallery builder
handcheck/          gallery assets, input format, review instructions
docs/INTEGRATION.md  expected teammate interfaces and artifact handoffs
tests/              offline checks using synthetic fixtures and a fake model
NOTES.md            observed facts and implementation decisions
```
