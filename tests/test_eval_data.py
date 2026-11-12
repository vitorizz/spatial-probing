"""Offline checks for evaluation labels, identities and COCO image resolution."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.pope import load_pope
from src.data.vsr import load_vsr


class EvaluationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotations = self.root / "annotations.jsonl"
        self.images = self.root / "images"
        self.vsr_row = {
            "image": "000000296471.jpg",
            "image_link": "http://images.cocodataset.org/train2017/000000296471.jpg",
            "caption": "The cat is inside the refrigerator.",
            "label": 0, "relation": "inside",
        }
        self.pope_row = {
            "question_id": 1, "image": "COCO_val2014_000000310196.jpg",
            "text": "Is there a snowboard in the image?", "label": "yes",
        }

    def write_rows(self, rows):
        self.annotations.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def image_file(self, relative):
        path = self.images / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only path existence is checked here; decoding belongs to model input tests.
        path.touch()
        return str(path)

    def test_vsr_labels_groups_and_mixed_coco_splits(self):
        second = {**self.vsr_row, "image": "000000363875.jpg", "label": 1,
                  "image_link": "http://images.cocodataset.org/val2017/000000363875.jpg"}
        paths = [self.image_file("train2017/000000296471.jpg"),
                 self.image_file("val2017/000000363875.jpg")]
        self.write_rows([self.vsr_row, second, {**self.vsr_row, "label": 1}])
        items = load_vsr(self.annotations, self.images)
        self.assertEqual([item["label"] for item in items], ["no", "yes", "yes"])
        self.assertEqual([item["image_path"] for item in items[:2]], paths)
        self.assertEqual(items[0]["group"], items[2]["group"])
        self.assertNotEqual(items[0]["id"], items[2]["id"])
        self.assertEqual(items[0]["coco_image_id"], 296471)
        self.assertEqual(items[0]["caption"], self.vsr_row["caption"])

    def test_vsr_ids_qualify_variant_and_split_and_ignore_blank_lines(self):
        self.write_rows([self.vsr_row])
        self.annotations.write_text("\n" + self.annotations.read_text() + "\n\n", encoding="utf-8")
        ids = {load_vsr(self.annotations, self.images, variant=variant, split=split,
                        check_images=False)[0]["id"]
               for variant in ("random", "zeroshot") for split in ("train", "dev", "test")}
        self.assertEqual(len(ids), 6)
        self.assertIn("vsr:zeroshot:test:1", ids)

    def test_vsr_rejects_invalid_labels_instead_of_using_truthiness(self):
        for label in ("0", "false", True, None, 2, 0.0):
            with self.subTest(label=label):
                self.write_rows([{**self.vsr_row, "label": label}])
                with self.assertRaisesRegex(ValueError, "integer 0 or 1"):
                    load_vsr(self.annotations, self.images, check_images=False)

    def test_pope_retains_questions_and_qualifies_ids_by_variant(self):
        self.write_rows([self.pope_row, {**self.pope_row, "question_id": 2, "label": "no"}])
        path = self.image_file("val2014/COCO_val2014_000000310196.jpg")
        random = load_pope(self.annotations, self.images)
        popular = load_pope(self.annotations, self.images, variant="popular")
        self.assertEqual([item["label"] for item in random], ["yes", "no"])
        self.assertEqual(random[0]["image_path"], path)
        self.assertEqual(random[0]["group"], random[1]["group"])
        self.assertEqual(random[0]["question"], self.pope_row["text"])
        self.assertNotEqual(random[0]["id"], popular[0]["id"])

    def test_pope_duplicate_question_ids_are_an_error(self):
        self.write_rows([self.pope_row, self.pope_row])
        with self.assertRaisesRegex(ValueError, "annotations.jsonl:2: duplicate"):
            load_pope(self.annotations, self.images, check_images=False)

    def test_flat_image_folders_are_supported(self):
        self.write_rows([self.pope_row])
        path = self.image_file(self.pope_row["image"])
        self.assertEqual(load_pope(self.annotations, self.images)[0]["image_path"], path)

    def test_missing_images_are_never_silently_dropped_or_renamed(self):
        self.write_rows([self.pope_row])
        self.image_file("val2017/000000310196.jpg")
        with self.assertRaisesRegex(FileNotFoundError, "COCO_val2014_000000310196.jpg"):
            load_pope(self.annotations, self.images)
        self.assertEqual(len(load_pope(self.annotations, self.images, check_images=False)), 1)

    def test_inconsistent_image_link_is_an_error(self):
        self.write_rows([{**self.vsr_row, "image_link": self.vsr_row["image_link"].replace("296471", "000001")}])
        with self.assertRaisesRegex(ValueError, "image_link"):
            load_vsr(self.annotations, self.images, check_images=False)

    def test_bad_annotation_records_fail_with_context(self):
        for contents in ("", "\n", "[]", "{broken", json.dumps({**self.pope_row, "label": "maybe"}),
                         json.dumps({**self.pope_row, "text": ""})):
            with self.subTest(contents=contents):
                self.annotations.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "annotations.jsonl"):
                    load_pope(self.annotations, self.images, check_images=False)


if __name__ == "__main__":
    unittest.main()
