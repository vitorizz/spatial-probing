"""Known-answer checks for benchmark metric summaries."""
import unittest

from src.stats import summarize_predictions


class SummaryTests(unittest.TestCase):
    def rows(self, labels, preds):
        return [{"id": str(i), "label": label, "pred": pred, "correct": 999}
                for i, (label, pred) in enumerate(zip(labels, preds))]

    def test_pope_confusion_and_binary_metrics(self):
        rows = self.rows(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"])
        metrics = summarize_predictions("pope", rows, ["yes", "no"])
        for key in ("accuracy", "precision", "recall", "f1", "yes_ratio"):
            self.assertEqual(metrics[key], 0.5)
        self.assertEqual(metrics["confusion"]["rows_true_cols_pred"], [[1, 1], [1, 1]])

    def test_pope_all_no_predictions(self):
        metrics = summarize_predictions("pope", self.rows(["yes", "no"], ["no", "no"]), ["yes", "no"])
        self.assertEqual(metrics["accuracy"], 0.5)
        for key in ("precision", "recall", "f1", "yes_ratio"):
            self.assertEqual(metrics[key], 0.0)

    def test_whatsup_subset_metrics_use_the_true_label(self):
        rows = self.rows(["left", "right", "front", "behind"], ["left", "right", "left", "front"])
        metrics = summarize_predictions("whatsup", rows, ["left", "right", "front", "behind"])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["lateral_accuracy"], 1.0)
        self.assertEqual(metrics["depth_accuracy"], 0.0)

    def test_absent_subsets_have_null_accuracy(self):
        metrics = summarize_predictions("whatsup", self.rows(["left"], ["left"]), ["left", "right", "front", "behind"])
        self.assertIsNone(metrics["depth_accuracy"])
        self.assertIsNone(metrics["per_class_accuracy"]["right"])

    def test_vsr_per_relation_and_confusion_orientation(self):
        rows = self.rows(["yes", "yes", "no"], ["no", "no", "no"])
        for row, relation in zip(rows, ["behind", "behind", "inside"]):
            row["relation"] = relation
        metrics = summarize_predictions("vsr", rows, ["yes", "no"])
        self.assertEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["confusion"]["rows_true_cols_pred"], [[0, 2], [0, 1]])
        self.assertEqual(metrics["per_relation_accuracy"], {"behind": 0.0, "inside": 1.0})

    def test_invalid_predictions_and_empty_sets_are_errors(self):
        for rows in ([], self.rows(["yes"], ["maybe"]), self.rows([True], ["yes"])):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_predictions("pope", rows, ["yes", "no"])


if __name__ == "__main__":
    unittest.main()
