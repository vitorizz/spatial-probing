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

This branch contains Person C's evaluation, review, and reporting tools. Full
model inference has **not** been run on this branch. Tests use a fake model and
synthetic fixtures; their scores are not project results.

| Component | Available here | Remaining dependency |
| --- | --- | --- |
| VSR/POPE data | Download command, local-file loading and validation; loaders checked against official annotations | Run preparation for the selected benchmark data |
| Evaluation | All three datasets, optional adapter, prediction CSVs and run metadata | A's model implementation; A's What'sUp loader for that dataset |
| Hand-check gallery | Offline images, boxes, evidence, decisions, JSON export/restore | B's fixed 200-question sample and team assignments |
| Comparisons | Paired confidence intervals and Markdown/CSV/JSON tables | Actual baseline and adapter predictions |
| Model loading and scoring | Shared signatures in `src/model_utils.py` | A's implementation; current functions raise `NotImplementedError` |
| Probing, layer selection, COCO training data, LoRA | Awaiting integration | A/B modules and verified commands |

Teammate branches have been used as references and have not been merged. Their
interfaces and artifacts may still change. See [integration handoffs](docs/INTEGRATION.md)
for the assumptions used by these tools.

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

For model execution, install the full declared requirements in the execution
environment with `python -m pip install -r requirements.txt`. The local venv has
been checked for the offline workflow; the full model runtime and exact version
pins remain to be confirmed during integration. See [NOTES.md](NOTES.md) for
observed dataset and implementation details.

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
   fixed sample and evidence; everyone reviews their assigned IDs. Resolve label
   issues before finalizing the training dataset.
5. **Train targeted LoRA (B).** Retain the adapter, base-model identity, training
   configuration, selected-layer artifact, and exclusion-check results.
6. **Evaluate the adapter and compare (C).** Use the same questions and prompts
   as the baseline, generate the before/after table, and report measured results.

Steps 3–5 need teammate modules that are not yet on this branch. Their commands
will be documented after integration. Available commands follow below.

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

After A's model implementation is integrated, run datasets separately:

```bash
python -m src.eval --dataset vsr --variant zeroshot --split test \
  --annotations data/vsr/zeroshot/test.jsonl --image-root data/coco
python -m src.eval --dataset pope --variant random \
  --annotations data/pope/coco_pope_random.json --image-root data/coco
python -m src.eval --dataset whatsup --data-root data
```

What'sUp also requires A's loader and data files. Full What'sUp runs invoke that
loader's dataset checks. Evaluation uses the shared option-scoring interface
without requesting hidden states.

VSR defaults to `zeroshot`/`test`; POPE defaults to `random`. To evaluate another
variant, change both its annotation path and flag. VSR also supports `random`;
POPE supports `popular` and `adversarial`. Settle benchmark selection before
interpreting adapter results.

For an adapter, repeat the same command with:

```text
--adapter runs/targeted/adapter --model-name targeted
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

## Review training questions

Once B supplies the fixed sample and images:

```bash
python -m src.handcheck --input data/coco_spatial/handcheck_sample.jsonl
```

Open `handcheck/gallery.html` in a browser. It embeds images and optional boxes
and label evidence, works offline, and records accept/reject/unsure decisions,
corrections, and notes. Export JSON to share or safeguard decisions. Restore
checks the sample fingerprint before replacing a session.

See [handcheck/README.md](handcheck/README.md) for the input format and checklist.
Agree the approximately 70 IDs per person within the fixed 200-question sample
before reviewing. The gallery does not select or allocate that sample. No real
training questions have been hand-checked on this branch yet.

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
| What'sUp Controlled B | Pending | Pending | Needs A's implementation and dataset |
| VSR | Pending | Pending | Variants/splits must be fixed; inference pending |
| Original COCO POPE | Pending | Pending | Selected variants need baseline inference |

All adapter results also depend on B's trained adapter. Annotation validation
and synthetic tests establish tool behavior; they do not establish model
accuracy or fine-tuning gains. Replace this table with measured results and
links to comparison artifacts after integration and full evaluation.

## Repo map

```text
configs/base.yaml    shared model, prompts, options, seed, Drive root
src/common.py        config I/O, paths, seeding, JSON helpers
src/model_utils.py   shared model API (currently stubs)
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
