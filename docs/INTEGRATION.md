# Person C integration handoffs

These are the interfaces and artifacts used by the current evaluation code.
They describe integration needs; they do not require merging an unfinished
teammate branch. Reconcile differences when A/B's implementations are ready.

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

Confirm the model runtime and exact package versions after integration. A's
existing extraction outputs can be retained for probing; comparisons in
`src.stats` need the prediction/metadata format emitted by `src.eval`, so
convert older artifacts or reevaluate with the shared runner.

## From B: training-data review and exclusion artifacts

For the gallery, supply the **fixed 200-question sample**, images, stable question
IDs, and available labeling evidence. The JSONL contract is documented in
[handcheck/README.md](../handcheck/README.md). It accepts the name/COCO-box
structure used by B's current candidate-pair reference code, but a final-record
adapter may be needed. Depth conventions and margins should be supplied as
evidence rather than inferred by the gallery.

The team must agree each person's approximately 70 review IDs. Keep this sample
fixed while reviewing; changes to text, labels, image bytes, boxes, or evidence
produce a different review fingerprint. Exported review JSON is the portable
record for resolving questionable labels before training.

For training/evaluation separation, retain the finalized training COCO image IDs
and the exclusion-check artifact. Evaluation loaders expose `coco_image_id`.
B's reference exclusion builder covers both VSR variants across their available
splits and all three original COCO POPE variants. Confirm the final exclusion
set against the benchmark sources and IDs actually used for evaluation.

## From B: adapter and provenance

Supply the loadable PEFT adapter location, base-model ID, training configuration,
and the selected-layer artifact received from A. Record the final training data
and its exclusion check alongside the run. Freeze layer selection before seeing
adapter evaluation results.

Use a distinct evaluation model name, for example:

```text
--adapter runs/targeted/adapter --model-name targeted
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

- Validate the integrated model interface and agreed evaluation environment.
- Freeze the desired variants/splits and run actual baselines.
- Populate the gallery, review our assigned questions, and resolve issues.
- Evaluate B's adapter on the same questions and generate the measured table.
- Replace README's pending results and add verified probing/training commands.
- Before the milestone, read the integrated repo and verify the full workflow.
