# NOTES

Decisions, deviations from the spec, and observed facts (model paths, dataset fields, version pins).

- 2026-11-07: `requirements.txt` holds lower bounds only. Replace them with exact pins once the Colab environment is confirmed.
- 2026-11-09 (Person C): evaluation data loaders validated against official annotations, without images or model execution.
  - VSR test annotations: random has 2,195 statements over 1,874 COCO images; zeroshot has 1,222 statements over 715 images. Sources: `https://huggingface.co/datasets/cambridgeltl/vsr_{random,zeroshot}/resolve/main/test.jsonl`.
  - Original COCO POPE: random, popular, and adversarial each have 3,000 questions over 500 images, with 1,500 yes and 1,500 no labels. Sources: `https://github.com/RUCAIBox/POPE/tree/main/output/coco` (not POPEv2).
  - VSR uses numeric filenames and `image_link` to distinguish COCO train2017/val2017; POPE uses `COCO_val2014_*.jpg`. Resolve the exact filenames rather than assuming all evaluation images are in val2017.
  - Loaders accept explicit local paths and use only the standard library. Verified with the project `.venv` (Python 3.14); no dependencies added or installed.
- 2026-11-09 (Person C): evaluation runner and summary metrics.
  - Quoted `options.yesno` in the shared YAML: PyYAML interprets unquoted `yes`/`no` as booleans. This matches the fix on Person A's reference branch.
  - `src.eval` uses only the declared model interface. Real inference remains dependent on Person A's model implementation; What'sUp also uses their loader and full-dataset `check` function. No teammate branches were merged.
  - Baseline/adapter outputs share a question-manifest hash when questions and prompts match. Annotation and CSV hashes are also saved. Limited runs have separate filenames. Paired confidence intervals and comparison tables are left for a later change.
  - Offline evaluation tests use a fake model API and tiny Pillow images. NumPy, PyYAML, and Pillow are installed in the project `.venv`; all were already declared requirements. No new dependencies.
- 2026-11-09 (Person C): hand-check gallery tooling.
  - Self-contained HTML embeds image previews and optional original-coordinate boxes; review JSON can be exported/restored offline. Source images are unchanged.
  - Review sessions are bound to question text, labels, original image bytes, boxes, and evidence. The gallery can be built now; the actual 200-question sample and our assigned review IDs still depend on B/team coordination.
  - Input `a`/`b` name/box fields match B's current candidate-pair reference branch, but final training-question records may need an adapter. No teammate code was merged and no new packages were added.
  - Verified with 33 Python tests and 5 JavaScript checks. A synthetic Firefox preview also exercised decisions, correction clearing, JSON export, and restore; no actual training questions have been reviewed.
- 2026-11-09 (Person C): paired baseline/adapter comparison and table generation.
  - `src.stats` reads `src.eval` schema-1 JSON/CSV artifacts, verifies their checksums and question identity, and recomputes rates from predictions. Question order may differ between runs; source annotations, prompts, labels, groups, image filenames, answer-option order, model configuration, and evaluation seed must match.
  - Paired percentile bootstrap: 2,000 draws by default, object-pair groups for What'sUp and image groups for VSR/POPE. Every draw uses the same groups for both models and all metrics. Ratios use pooled question counts, preserving unequal group sizes; POPE metrics are recomputed per draw. Intervals are omitted with fewer than two contributing groups or valid draws.
  - Comparison reports JSON, CSV, and Markdown; the Markdown uses percentages/percentage points. Limited runs need explicit opt-in. Real before/after results still require baseline inference and B's adapter; tests use synthetic artifacts only.
  - Shared question hashing moved into `src.stats` so evaluation and comparison use the same definition. Existing schema-1 hashes for the standard dataset records are preserved. No dependencies added.
- 2026-11-09 (Person C): replaced the skeleton README with the current workflow and explicit status/results tables. Added `docs/INTEGRATION.md` for model/What'sUp interfaces and sample, exclusion, adapter, and result-artifact handoffs. Real inference, reviews, and measured results remain pending; synthetic test scores are not reported as project results.
- 2026-11-09 (Person C): added `src.data.prepare` to download official VSR/POPE annotations and referenced COCO images without loading a model. Validated caches are reused; interrupted attempts retain completed files and remove only their temporary downloads. Preparation manifests record sources, hashes, question IDs, and COCO image IDs; existing same-selection manifests detect changed cached bytes. Annotation-only and limited selections have separate manifests. Download behavior was tested with simulated HTTP responses; no real image sets were fetched during this change and no dependencies were added.
