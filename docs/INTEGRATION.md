# Person C integration handoffs

These are the interfaces and artifacts used by the current evaluation code.
A/B's implementations from `origin/main` are integrated. The local, Git-ignored
`drive/` download supplies A's baseline/probing artifacts, What'sUp data, B's
generated training data, and the milestone adapter.

## From A: shared model implementation

`src.eval` calls these functions from `src/model_utils.py`:

| Function | Expected behavior |
| --- | --- |
| `load_model(cfg, adapter=None)` | Return `(model, processor)` in evaluation mode, applying the adapter if supplied |
| `fourway_question(o1, o2, cfg)` | Build the shared four-way prompt for What'sUp |
| `make_inputs(processor, image, question, device=...)` | Prepare a PIL RGB image and formatted question for the model |
| `option_token_ids(tokenizer, options)` | Return token IDs keyed by the configured answer strings |
| `score_options(model, inputs, option_ids)` | Return `.pred` and finite `.scores` for exactly those answer options |

The evaluator also reads `model.device`, `model.dtype`, and
`processor.tokenizer`. It does not request hidden states or call free generation.
Keep prompt construction, pixel limits, and option scoring consistent with the
baseline/probing implementation. `configs/base.yaml` quotes `yes`/`no` so the
options stay strings when loaded by PyYAML.

For What'sUp, the evaluator imports from `src/data/whatsup.py`:

- `load_whatsup(root)` returns records in a stable order with string `id`,
  `image_path`, `o1`, `o2`, `label`, and `group` fields. `group` identifies an
  object pair, matching the grouping used for probing and later resampling.
- `check(items, root)` validates the full dataset, including completeness and
  labels. The reference implementation checks 408 items and 102 per class.
  Smoke runs with `--limit` bypass this full-dataset check.

The public function signatures match the evaluator's calls. Confirm the model
runtime and exact package versions on the user's local GPU before full runs.
A's existing extraction outputs can be retained for probing; comparisons in
`src.stats` need the prediction/metadata format emitted by `src.eval`, so
reevaluate What'sUp with the shared runner under a separate output root such as
`--out results/evaluation_milestone`, preserving the supplied artifacts.
The supplied 408-image dataset is under `drive/data/`.

## From B: training-data review and exclusion artifacts

For the gallery, supply the **fixed 200-question sample**, images, stable question
IDs, and available labeling evidence. The JSONL contract is documented in
[handcheck/README.md](../handcheck/README.md). It accepts the name/COCO-box
structure used by B's current candidate-pair reference code, but a final-record
adapter is needed: B's final rows use `answer`, include yes/no and four-way
questions, and omit boxes. Pair/depth artifacts supplied alongside them contain
the evidence. Match object order and scale original-coordinate boxes to the
resized images when joining those artifacts. Depth conventions and margins
should be supplied as evidence rather than inferred by the gallery.

The download contains 2,864 training and 148 validation questions with their
images, but no fixed 200-question sample or reviewer assignments were found.

The team must agree each person's approximately 70 review IDs. Keep this sample
fixed while reviewing; changes to text, labels, image bytes, boxes, or evidence
produce a different review fingerprint. Exported review JSON is the portable
record for resolving questionable labels before training.

For training/evaluation separation, retain the finalized training COCO image IDs
and the exclusion-check artifact. Evaluation loaders expose `coco_image_id`.
B's reference exclusion builder covers both VSR variants across their available
splits and all three original COCO POPE variants. Confirm the final exclusion
set against the benchmark sources and IDs actually used for evaluation.
`src.data.prepare` records question IDs, `coco_image_ids`, source URLs, and file
hashes in `data/prepared/` (or the selected data root). Those manifests describe
the evaluation selections actually prepared; they do not replace B's broader
training exclusion builder. The supplied download does not contain B's original
`exclude_ids.json`; its location remains to be confirmed.

## From B: adapter and provenance

The supplied milestone adapter is `drive/runs/lora_targeted/adapter`, for
`Qwen/Qwen2.5-VL-3B-Instruct`, targeting PEFT blocks 28–35. Its run configuration
and training log are alongside it; the log ends at step 179. A's selected-layer
artifact is in `drive/results/probes/`. Retain the final training data and its
exclusion check alongside the run. Freeze layer selection before seeing adapter
evaluation results.

Use a distinct evaluation model name, for example:

```text
--adapter drive/runs/lora_targeted/adapter --model-name targeted
```

The adapter and baseline must use identical evaluation questions, prompts,
answer-option order, model configuration, and evaluation seed. Changing a local
image-root path is fine; changing the actual images is not. Recorded environment
differences are retained for inspection in comparison reports.

## Person C outputs and remaining work

Evaluation writes `preds/<model>_<benchmark>.csv` and
`metrics/<model>_<benchmark>.json` under the selected results root. Keep both
folders together so comparisons can resolve and verify the CSVs. The comparison
command creates `comparison.json`, `comparison.csv`, and `comparison.md`.

Still to complete with the real artifacts:

- Verify the model in the user's local GPU environment (offline tests pass).
- Freeze the desired variants/splits and run actual baselines.
- Populate the gallery, review our assigned questions, and resolve issues.
- Evaluate B's adapter on the same questions and generate the measured table.
- Replace README's pending results and add verified probing/training commands.
- Before the milestone, read the integrated repo and verify the full workflow.
