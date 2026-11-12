"""Known-outcome and integrity checks for paired benchmark comparisons."""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.common import load_config
from src.stats import compare_runs, load_run, main, question_manifest_hash, write_report


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = load_config()
        # Two image groups, ten questions each. Baseline is perfect on the first
        # image and entirely wrong on the second; the adapter is perfect on both.
        # Correct group resampling has a 25% mass at each endpoint, unlike a
        # question-level bootstrap which would understate this uncertainty.
        self.before = []
        self.after = []
        for group in ("1", "2"):
            for index in range(10):
                label = "yes" if index % 2 == 0 else "no"
                row = {"id": f"{group}:{index}", "dataset": "pope", "variant": "random", "group": group,
                       "coco_image_id": group, "image_path": f"/original/images/{group}.jpg",
                       "question": f"Question {group}:{index}?", "label": label}
                self.before.append({**row, "pred": label if group == "1" else ("no" if label == "yes" else "yes")})
                self.after.append({**row, "pred": label})
        self.base = self.write_run("base", self.before)
        self.adapted = self.write_run("targeted", self.after, adapter="runs/targeted/adapter")

    def write_run(self, name, rows, *, adapter=None, dataset="pope", variant="random", split=None, **overrides):
        predictions = self.root / "preds" / f"{name}.csv"
        predictions.parent.mkdir(exist_ok=True)
        with predictions.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "schema_version": 1, "model": name, "dataset": dataset, "variant": variant, "split": split,
            "adapter": adapter, "config": copy.deepcopy(self.cfg), "n": len(rows), "n_available": len(rows),
            "complete_dataset": True, "limit": None, "annotations_sha256": "fixed-annotations",
            "manifest_sha256": question_manifest_hash(rows), "env": {"device": "cuda", "dtype": "float16"},
            "predictions_file": f"preds/{name}.csv",
            "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        }
        metadata.update(overrides)
        path = self.root / "metrics" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def change_metadata(self, path, **changes):
        value = json.loads(path.read_text())
        value.update(changes)
        path.write_text(json.dumps(value), encoding="utf-8")

    def compare(self, **kwargs):
        return compare_runs(self.base, self.adapted, n_bootstrap=128, seed=7, **kwargs)

    def test_cluster_bootstrap_known_change_and_interval(self):
        result = self.compare()
        accuracy = result["metrics"]["accuracy"]
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(accuracy["baseline"], 0.5)
        self.assertEqual(accuracy["adapter"], 1.0)
        self.assertEqual(accuracy["delta"], 0.5)
        self.assertEqual((accuracy["ci_low"], accuracy["ci_high"]), (0.0, 1.0))
        self.assertEqual(result["metrics"]["yes_ratio"]["delta"], 0.0)
        self.assertEqual(result, self.compare())

    def test_pairing_identical_predictions_has_zero_width_interval(self):
        self.write_run("targeted", self.before, adapter="adapter")
        for metric in self.compare()["metrics"].values():
            self.assertEqual((metric["delta"], metric["ci_low"], metric["ci_high"]), (0.0, 0.0, 0.0))

    def test_rows_match_by_id_and_portable_image_names(self):
        original = self.compare()["metrics"]
        moved = [{**row, "image_path": f"C:\\data\\{row['group']}.jpg"} for row in reversed(self.after)]
        self.write_run("targeted", moved, adapter="adapter")
        self.assertEqual(self.compare()["metrics"], original)

    def test_question_weighting_for_unequal_group_sizes(self):
        before = self.before[:1] + self.before[10:13]
        after = self.after[:1] + self.after[10:13]
        self.write_run("base", before)
        self.write_run("targeted", after, adapter="adapter")
        metric = self.compare()["metrics"]["accuracy"]
        self.assertEqual(metric["baseline"], 0.25)
        self.assertEqual(metric["delta"], 0.75)

    def test_one_group_and_absent_whatsup_subsets_have_no_interval(self):
        rows = [{"id": "photo", "dataset": "whatsup", "group": "cat|box", "image_path": "photo.jpg",
                 "question": "Where is the cat?", "label": "left", "pred": "left"}]
        self.write_run("base", rows, dataset="whatsup", variant=None)
        self.write_run("targeted", rows, adapter="adapter", dataset="whatsup", variant=None)
        metrics = self.compare()["metrics"]
        self.assertIsNone(metrics["accuracy"]["ci_low"])
        self.assertEqual(metrics["lateral_accuracy"]["n_groups"], 1)
        self.assertIsNone(metrics["depth_accuracy"]["delta"])
        self.assertEqual(metrics["depth_accuracy"]["n_bootstrap_valid"], 0)

    def test_whatsup_lateral_and_depth_changes_are_separate(self):
        before, after = [], []
        for group in range(3):
            for label in ("left", "right", "front", "behind"):
                row = {"id": f"{group}:{label}", "dataset": "whatsup", "group": f"pair-{group}",
                       "question": "Spatial question?", "image_path": f"{group}_{label}.jpg", "label": label}
                before.append({**row, "pred": label if label in {"left", "right"} else "left"})
                after.append({**row, "pred": label})
        self.write_run("base", before, dataset="whatsup", variant=None)
        self.write_run("targeted", after, adapter="adapter", dataset="whatsup", variant=None)
        metrics = self.compare()["metrics"]
        self.assertEqual(metrics["accuracy"]["delta"], 0.5)
        self.assertEqual(metrics["lateral_accuracy"]["delta"], 0.0)
        self.assertEqual(metrics["depth_accuracy"]["delta"], 1.0)
        self.assertEqual(metrics["depth_accuracy"]["ci_low"], 1.0)
        self.assertEqual(metrics["depth_accuracy"]["n"], 6)
        self.assertEqual(metrics["depth_accuracy"]["n_groups"], 3)

    def test_nonlinear_pope_metrics_are_computed_from_confusion_totals(self):
        before, after = [], []
        for i, (label, base, adapted) in enumerate(zip(
            ["yes", "yes", "yes", "no"], ["yes", "no", "no", "yes"], ["yes", "yes", "no", "no"],
        )):
            row = {"id": str(i), "dataset": "pope", "variant": "random", "group": str(i),
                   "question": "Object question?", "image_path": f"{i}.jpg", "label": label}
            before.append({**row, "pred": base})
            after.append({**row, "pred": adapted})
        self.write_run("base", before)
        self.write_run("targeted", after, adapter="adapter")
        metrics = self.compare()["metrics"]
        for metric, expected in {"precision": (0.5, 1.0), "recall": (1 / 3, 2 / 3), "f1": (0.4, 0.8)}.items():
            self.assertAlmostEqual(metrics[metric]["baseline"], expected[0])
            self.assertAlmostEqual(metrics[metric]["adapter"], expected[1])

    def test_question_mismatches_are_never_silently_intersected(self):
        self.write_run("targeted", self.after[:-1], adapter="adapter")
        with self.assertRaisesRegex(ValueError, "ID sets differ"):
            self.compare()
        for field, value in (("group", "different"), ("label", "no"),
                             ("question", "different prompt"), ("image_path", "different.jpg")):
            with self.subTest(field=field):
                changed = [{**self.after[0], field: value}] + self.after[1:]
                self.write_run("targeted", changed, adapter="adapter")
                with self.assertRaisesRegex(ValueError, "differs between runs"):
                    self.compare()

    def test_checksums_duplicate_ids_and_counts_are_verified(self):
        csv_path = self.root / "preds/targeted.csv"
        csv_path.write_text(csv_path.read_text() + "tampered\n")
        with self.assertRaisesRegex(ValueError, "CSV checksum"):
            self.compare()
        self.write_run("targeted", self.after + [self.after[0]], adapter="adapter")
        with self.assertRaisesRegex(ValueError, "duplicate question IDs"):
            self.compare()
        self.write_run("targeted", self.after, adapter="adapter", manifest_sha256="bad")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            self.compare()
        self.write_run("targeted", self.after, adapter="adapter", n=999)
        with self.assertRaisesRegex(ValueError, "count differs"):
            self.compare()

    def test_base_model_options_seed_and_annotation_changes_are_rejected(self):
        for change in ("model", "options", "seed", "annotations"):
            with self.subTest(change=change):
                cfg = copy.deepcopy(self.cfg)
                overrides = {}
                if change == "model":
                    cfg["model"]["max_pixels"] += 1
                elif change == "options":
                    cfg["options"]["yesno"].reverse()
                elif change == "seed":
                    cfg["seed"] += 1
                else:
                    overrides["annotations_sha256"] = "changed"
                self.write_run("targeted", self.after, adapter="adapter", config=cfg, **overrides)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_run_roles_must_be_baseline_then_adapter(self):
        self.change_metadata(self.adapted, adapter=None)
        with self.assertRaisesRegex(ValueError, "base run followed by an adapter"):
            self.compare()

    def test_limited_runs_require_opt_in_and_are_labeled(self):
        for path in (self.base, self.adapted):
            self.change_metadata(path, limit=20)
        with self.assertRaisesRegex(ValueError, "allow-limited"):
            self.compare()
        result = self.compare(allow_limited=True)
        self.assertTrue(result["limited"])
        write_report([result], self.root / "report")
        self.assertIn("(limited)", (self.root / "report/comparison.md").read_text())

    def test_environment_differences_are_recorded(self):
        self.change_metadata(self.adapted, env={"device": "cpu", "dtype": "float16"})
        self.assertEqual(self.compare()["environment_differences"], {"device": {"baseline": "cuda", "adapter": "cpu"}})

    def test_report_units_and_explicit_overwrite(self):
        result = self.compare()
        output = self.root / "report"
        write_report([result], output)
        with (output / "comparison.csv").open(newline="") as stream:
            first = next(csv.DictReader(stream))
        self.assertEqual(float(first["delta"]), 0.5)
        self.assertIn("+50.00", (output / "comparison.md").read_text())
        self.assertEqual(json.loads((output / "comparison.json").read_text())["comparisons"][0], result)
        with self.assertRaises(FileExistsError):
            write_report([result], output)
        write_report([result], output, overwrite=True)

    def test_cli_matches_dataset_lists_in_any_order(self):
        before = [{**row, "dataset": "vsr", "variant": "zeroshot", "split": "test", "relation": "behind"} for row in self.before]
        after = [{**row, "dataset": "vsr", "variant": "zeroshot", "split": "test", "relation": "behind"} for row in self.after]
        vsr_base = self.write_run("base_vsr", before, dataset="vsr", variant="zeroshot", split="test")
        vsr_adapter = self.write_run("targeted_vsr", after, dataset="vsr", variant="zeroshot", split="test", adapter="adapter")
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--baseline", str(self.base), str(vsr_base), "--adapter", str(vsr_adapter), str(self.adapted),
                  "--bootstrap", "64", "--out", str(self.root / "combined")])
        report = json.loads((self.root / "combined/comparison.json").read_text())
        self.assertEqual([item["dataset"] for item in report["comparisons"]], ["pope", "vsr"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--baseline", str(self.base), "--adapter", str(vsr_adapter)])

    def test_invalid_bootstrap_parameters(self):
        for options in ({"n_bootstrap": 1}, {"confidence": 1.0}, {"confidence": float("nan")}, {"seed": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare_runs(self.base, self.adapted, **options)

    def test_schema_and_relative_prediction_paths_are_checked(self):
        self.change_metadata(self.base, schema_version=2)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            load_run(self.base)
        self.write_run("base", self.before, predictions_file="../predictions.csv")
        with self.assertRaisesRegex(ValueError, "relative to the results root"):
            load_run(self.base)
        self.write_run("base", self.before, env="invalid")
        with self.assertRaisesRegex(ValueError, "environment metadata"):
            load_run(self.base)


if __name__ == "__main__":
    unittest.main()
