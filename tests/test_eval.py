"""Exercise the evaluation pipeline with real tiny images and a fake model API."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from src.common import load_config
from src.eval import evaluate, main, prepare_items
from src.model_utils import ScoreResult
from src.stats import compare_runs


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "results"
        self.image = self.root / "000000000001.jpg"
        Image.new("L", (2, 2)).save(self.image)
        self.cfg = load_config()
        self.item = {
            "id": "vsr:zeroshot:test:1", "dataset": "vsr", "variant": "zeroshot", "split": "test",
            "group": "1", "coco_image_id": 1, "image_path": str(self.image),
            "caption": "The cat is behind the box.", "relation": "behind", "label": "yes",
        }
        self.annotations = self.root / "test.jsonl"
        self.annotations.write_text(json.dumps({
            "image": self.image.name, "image_link": f"http://images.cocodataset.org/train2017/{self.image.name}",
            "caption": self.item["caption"], "relation": "behind", "label": 1,
        }), encoding="utf-8")
        self.model = SimpleNamespace(device="cpu", dtype="float16")
        self.processor = SimpleNamespace(tokenizer=object())
        self.load_model = Mock(return_value=(self.model, self.processor))
        self.inputs = Mock(side_effect=lambda processor, image, question, device: {"mode": image.mode, "question": question})
        self.scorer = Mock(return_value=ScoreResult({"yes": 2.0, "no": -1.0}, "yes", None))
        self.token_ids = Mock(side_effect=lambda tokenizer, options: {o: [i] for i, o in enumerate(options)})
        model_patch = patch.multiple(
            "src.model_utils", load_model=self.load_model, make_inputs=self.inputs,
            score_options=self.scorer, option_token_ids=self.token_ids,
            fourway_question=lambda o1, o2, cfg: cfg["prompts"]["fourway"].format(o1=o1, o2=o2),
        )
        model_patch.start()
        self.addCleanup(model_patch.stop)

    def run_vsr(self, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return evaluate("vsr", kwargs.pop("items", [self.item]), self.cfg,
                            out=self.out, variant="zeroshot", split="test", **kwargs)

    def test_config_yesno_options_are_strings(self):
        self.assertEqual(self.cfg["options"]["yesno"], ["yes", "no"])

    def test_vsr_pipeline_preserves_prompts_scores_and_run_provenance(self):
        metrics = self.run_vsr(adapter="runs/targeted/adapter", model_name="targeted", annotations=self.annotations)
        self.load_model.assert_called_once_with(self.cfg, adapter="runs/targeted/adapter")
        self.token_ids.assert_called_once_with(self.processor.tokenizer, ["yes", "no"])
        inputs = self.scorer.call_args.args[1]
        self.assertEqual(inputs["mode"], "RGB")
        self.assertEqual(inputs["question"], self.cfg["prompts"]["vsr"].format(caption=self.item["caption"]))
        self.assertNotIn("output_hidden_states", self.scorer.call_args.kwargs)
        predictions = self.out / metrics["predictions_file"]
        with predictions.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["id"], self.item["id"])
        self.assertEqual(rows[0]["group"], "1")
        self.assertEqual(rows[0]["correct"], "1")
        self.assertEqual(float(rows[0]["score_yes"]), 2.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["adapter"], "runs/targeted/adapter")
        self.assertEqual(metrics["annotations_sha256"], hashlib.sha256(self.annotations.read_bytes()).hexdigest())
        self.assertEqual(metrics["predictions_sha256"], hashlib.sha256(predictions.read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.out / "metrics/targeted_vsr_zeroshot_test.json").read_text()), metrics)

    def test_pope_adds_suffix_and_whatsup_uses_shared_question_builder(self):
        pope = {"id": "pope:random:1", "group": "1", "image_path": str(self.image),
                "question": "Is there a cat in the image?", "label": "yes"}
        with contextlib.redirect_stderr(io.StringIO()):
            metrics = evaluate("pope", [pope], self.cfg, out=self.out, variant="random")
        self.assertEqual(self.scorer.call_args.args[1]["question"], pope["question"] + self.cfg["prompts"]["pope_suffix"])
        self.assertEqual(metrics["f1"], 1.0)
        whatsup = {"id": "photo", "group": "box|cat", "image_path": str(self.image),
                   "o1": "cat", "o2": "box", "label": "front"}
        self.scorer.return_value = ScoreResult({o: 1.0 if o == "front" else 0.0 for o in self.cfg["options"]["fourway"]}, "front", None)
        with contextlib.redirect_stderr(io.StringIO()):
            metrics = evaluate("whatsup", [whatsup], self.cfg, out=self.out)
        self.assertEqual(self.scorer.call_args.args[1]["question"], self.cfg["prompts"]["fourway"].format(o1="cat", o2="box"))
        self.assertEqual(metrics["depth_accuracy"], 1.0)

    def test_limits_do_not_require_unselected_images_or_reuse_full_run_filenames(self):
        unavailable = {**self.item, "id": "vsr:zeroshot:test:2", "image_path": str(self.root / "missing.jpg")}
        metrics = self.run_vsr(items=[self.item, unavailable], limit=1)
        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["n_available"], 2)
        self.assertFalse(metrics["complete_dataset"])
        self.assertIn("_limit1.csv", metrics["predictions_file"])
        self.assertFalse((self.out / "preds/base_vsr_zeroshot_test.csv").exists())

    def test_outputs_require_explicit_overwrite(self):
        self.run_vsr()
        self.load_model.reset_mock()
        with self.assertRaises(FileExistsError):
            self.run_vsr()
        self.load_model.assert_not_called()
        self.run_vsr(overwrite=True)
        self.load_model.assert_called_once()

    def test_manifest_is_model_independent_but_changes_with_question_content(self):
        base = self.run_vsr()
        adapted = self.run_vsr(adapter="adapter", model_name="adapted")
        self.assertEqual(base["manifest_sha256"], adapted["manifest_sha256"])
        changed = self.run_vsr(items=[{**self.item, "caption": "A different statement."}], model_name="changed")
        self.assertNotEqual(base["manifest_sha256"], changed["manifest_sha256"])

    def test_preflight_rejects_bad_inputs_before_loading_model(self):
        for items in ([], [self.item, self.item], [{**self.item, "label": "maybe"}],
                      [{**self.item, "variant": "random"}], [{**self.item, "image_path": "missing.jpg"}]):
            with self.subTest(items=items), self.assertRaises((ValueError, FileNotFoundError)):
                self.run_vsr(items=items)
        self.load_model.assert_not_called()

    def test_invalid_scores_produce_no_result_files(self):
        for scores, pred in (({"yes": float("nan"), "no": 0.0}, "yes"), ({"yes": 1.0}, "yes"),
                             ({"yes": 1.0, "no": 0.0}, "maybe")):
            self.scorer.return_value = ScoreResult(scores, pred, None)
            with self.subTest(scores=scores, pred=pred), self.assertRaises(ValueError):
                self.run_vsr()
        self.assertFalse(self.out.exists())

    def test_unquoted_yaml_options_are_rejected(self):
        self.cfg["options"]["yesno"] = [True, False]
        with self.assertRaisesRegex(ValueError, "quote yes/no"):
            prepare_items("vsr", [self.item], self.cfg)

    def test_vsr_cli_loads_real_annotations_and_reports_stubs_clearly(self):
        arguments = ["--dataset", "vsr", "--annotations", str(self.annotations),
                     "--image-root", str(self.root), "--out", str(self.out)]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            main(arguments)
        self.assertEqual(json.loads(stdout.getvalue())["n"], 1)
        self.load_model.side_effect = NotImplementedError
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
            main(arguments + ["--model-name", "stub_check"])
        self.assertEqual(exc.exception.code, 1)
        self.assertIn("Person A", stderr.getvalue())

    def test_whatsup_cli_uses_person_a_validation_before_full_runs(self):
        loader = ModuleType("src.data.whatsup")
        items = [{"id": "photo", "group": "box|cat", "image_path": str(self.image),
                  "o1": "cat", "o2": "box", "label": "front"}]
        loader.load_whatsup = Mock(return_value=items)
        loader.check = Mock(side_effect=AssertionError("expected 408 items, got 1"))
        arguments = ["--dataset", "whatsup", "--data-root", str(self.root), "--out", str(self.out)]
        stderr = io.StringIO()
        with patch.dict(sys.modules, {"src.data.whatsup": loader}):
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                main(arguments)
            self.assertEqual(exc.exception.code, 1)
            self.assertIn("expected 408", stderr.getvalue())
            self.load_model.assert_not_called()
            loader.check.reset_mock(side_effect=True)
            self.scorer.return_value = ScoreResult({o: 1.0 for o in self.cfg["options"]["fourway"]}, "front", None)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(arguments + ["--limit", "1"])
            loader.check.assert_not_called()

    def test_pope_cli_dispatches_and_passes_adapter(self):
        image = self.root / "COCO_val2014_000000000001.jpg"
        Image.new("RGB", (2, 2)).save(image)
        annotations = self.root / "coco_pope_adversarial.json"
        annotations.write_text(json.dumps({
            "question_id": 1, "image": image.name,
            "text": "Is there a cat in the image?", "label": "yes",
        }), encoding="utf-8")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            main(["--dataset", "pope", "--variant", "adversarial", "--annotations", str(annotations),
                  "--image-root", str(self.root), "--out", str(self.out), "--adapter", "adapter"])
        metrics = json.loads(stdout.getvalue())
        self.assertEqual(metrics["variant"], "adversarial")
        self.assertEqual(metrics["model"], "adapted")
        self.assertEqual(metrics["f1"], 1.0)
        self.load_model.assert_called_once_with(self.cfg, adapter="adapter")

    def test_saved_evaluation_artifacts_feed_paired_comparison(self):
        self.scorer.return_value = ScoreResult({"yes": 0.0, "no": 1.0}, "no", None)
        self.run_vsr(annotations=self.annotations)
        self.scorer.return_value = ScoreResult({"yes": 1.0, "no": 0.0}, "yes", None)
        self.run_vsr(annotations=self.annotations, adapter="adapter", model_name="targeted")
        result = compare_runs(self.out / "metrics/base_vsr_zeroshot_test.json",
                              self.out / "metrics/targeted_vsr_zeroshot_test.json", n_bootstrap=32)
        self.assertEqual(result["metrics"]["accuracy"]["delta"], 1.0)
        self.assertIsNone(result["metrics"]["accuracy"]["ci_low"])


if __name__ == "__main__":
    unittest.main()
