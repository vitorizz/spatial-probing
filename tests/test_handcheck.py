"""Offline checks for gallery rendering and sample identity."""
from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

from src.handcheck import build_gallery


class GalleryParser(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.tags = []
        self.payload = ""
        self.in_payload = False
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "script" and attrs.get("id") == "sample-data":
            self.in_payload = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_payload = False

    def handle_data(self, text):
        if self.in_payload:
            self.payload += text


class HandcheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "sample.jsonl"
        self.out = self.root / "gallery.html"
        self.image = self.root / "scene.png"
        Image.new("RGB", (2048, 1024), "white").save(self.image)
        self.row = {
            "id": "pair-1", "image_path": "scene.png", "label": "left",
            "a": {"name": "cat", "bbox": [0, 0, 500, 500]},
            "b": {"name": "box", "bbox": [1000, 0, 500, 500]},
            "evidence": {"method": "box centers", "depth_a": 0.3, "depth_b": 0.6},
        }

    def write(self, rows):
        self.source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def test_embedded_images_box_coordinates_and_shared_image_count(self):
        self.write([self.row, {**self.row, "id": "pair-2", "label": "front"}])
        result = build_gallery(self.source, self.out)
        parsed = GalleryParser(self.out.read_text())
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_images"], 1)
        images = [attrs for tag, attrs in parsed.tags if tag == "img"]
        self.assertEqual(len(images), 2)
        with Image.open(io.BytesIO(base64.b64decode(images[0]["src"].split(",", 1)[1]))) as image:
            self.assertEqual(image.size, (1024, 512))
        svg = [attrs for tag, attrs in parsed.tags if tag == "svg"]
        self.assertEqual(svg[0]["viewbox"], "0 0 2048 1024")
        payload = json.loads(parsed.payload)
        self.assertEqual(payload["sample_sha256"], result["sample_sha256"])
        self.assertIn("cat", payload["items"][0]["question"])
        self.assertIn("box", payload["items"][0]["question"])
        self.assertNotIn("<script src=", self.out.read_text())

    def test_explicit_questions_need_no_boxes(self):
        self.write([{key: self.row[key] for key in ("id", "image_path", "label")}
                    | {"question": "Where is the cat?"}])
        build_gallery(self.source, self.out)
        self.assertIn("No boxes or label evidence supplied.", self.out.read_text())

    def test_mixed_final_records_preserve_negative_questions_and_answer_options(self):
        question = "Is the cat to the right of the box? Answer yes or no."
        self.write([
            {"id": "negative", "image_path": "scene.png", "qtype": "yesno",
             "question": question, "relation": "left", "answer": "no"},
            {"id": "fourway", "image_path": "scene.png", "qtype": "4way",
             "o1": "cat", "o2": "box", "answer": "left"},
        ])
        build_gallery(self.source, self.out)
        parsed = GalleryParser(self.out.read_text())
        items = json.loads(parsed.payload)["items"]
        self.assertEqual(items[0], {"id": "negative", "question": question,
                                    "label": "no", "options": ["yes", "no"]})
        self.assertEqual(items[1]["options"], ["left", "right", "front", "behind"])
        selects = []
        for tag, attrs in parsed.tags:
            if tag == "select":
                selects.append(attrs.get("class"))
        self.assertEqual(selects.count("corrected-label"), 2)

    def test_evidence_matches_swapped_objects_and_rescales_original_boxes(self):
        images = self.root / "coco_images" / "train2017"
        images.mkdir(parents=True)
        self.image.rename(images / "000000000123.png")
        pair = {"coco_image_id": 123, "file_name": "000000000123.png", "width": 4096, "height": 2048,
                "a": {"name": "cat", "ann_id": 10, "bbox": [0, 0, 1000, 1000]},
                "b": {"name": "box", "ann_id": 20, "bbox": [2000, 0, 1000, 1000]}}
        pairs = self.root / "pairs_train2017.jsonl"
        pairs.write_text(json.dumps(pair) + "\n")
        (self.root / "depth_train2017.jsonl").write_text(
            json.dumps({"coco_image_id": 123, "depth": {"10": 0.2, "20": 0.8}}) + "\n")
        (self.root / "config_pairs_train2017.yaml").write_text("pairs:\n  lateral_min_gap: 0.1\n")
        depth_config = self.root / "config_depth_train2017.yaml"
        depth_config.write_text("depth:\n  larger_is_closer: true\n  min_gap: 0.15\n")
        row = {"id": "swapped", "coco_image_id": 123, "image_path": "coco_images/train2017/000000000123.png",
               "o1": "box", "o2": "cat", "qtype": "yesno", "relation": "front",
               "question": "Is the box behind the cat? Answer yes or no.", "answer": "no"}
        self.write([row])
        result = build_gallery(self.source, self.out, evidence_dir=self.root)
        parsed = GalleryParser(self.out.read_text())
        boxes = [attrs for tag, attrs in parsed.tags if tag == "rect"]
        self.assertEqual(boxes[0], {"class": "box-a", "x": "1000.0", "y": "0.0", "width": "500.0", "height": "500.0"})
        self.assertEqual(boxes[1]["x"], "0.0")
        self.assertIn('A: box', self.out.read_text())
        # Check the rendered evidence as text, not only the underlying join helper.
        import html
        document = html.unescape(self.out.read_text())
        self.assertIn('"depth": 0.8', document)
        self.assertIn('"larger_is_closer": true', document)
        depth_config.write_text("depth:\n  larger_is_closer: false\n  min_gap: 0.15\n")
        self.assertNotEqual(result["sample_sha256"], build_gallery(
            self.source, self.out, evidence_dir=self.root)["sample_sha256"])
        self.out.write_text("keep existing gallery")
        self.write([{**row, "o2": "dog"}])
        with self.assertRaisesRegex(ValueError, "no unique pair evidence"):
            build_gallery(self.source, self.out, evidence_dir=self.root)
        self.assertEqual(self.out.read_text(), "keep existing gallery")

    def test_invalid_question_types_answers_and_conflicts_fail_before_publish(self):
        self.out.write_text("keep existing gallery")
        for extra in ({"qtype": "yesno", "answer": "no"},
                      {"qtype": "yesno", "answer": "left", "question": "Is it left?"},
                      {"qtype": "multiple-choice", "answer": "left"},
                      {"label": "left", "answer": "right"}):
            with self.subTest(extra=extra):
                self.write([{"id": "bad", "image_path": "scene.png", "o1": "cat", "o2": "box", **extra}])
                with self.assertRaises(ValueError):
                    build_gallery(self.source, self.out)
                self.assertEqual(self.out.read_text(), "keep existing gallery")

    def test_source_text_cannot_inject_html_or_terminate_json_script(self):
        hostile = '</script><script>alert("x")</script>'
        self.write([{**self.row, "id": hostile, "question": hostile, "evidence": hostile}])
        build_gallery(self.source, self.out)
        document = self.out.read_text()
        parsed = GalleryParser(document)
        self.assertEqual(sum(tag == "script" for tag, _ in parsed.tags), 2)
        self.assertEqual(json.loads(parsed.payload)["items"][0]["question"], hostile)
        self.assertIn("&lt;/script&gt;", document)

    def test_hash_changes_with_labels_evidence_boxes_and_image_bytes(self):
        self.write([self.row])
        original = build_gallery(self.source, self.out)["sample_sha256"]
        for changed in ({**self.row, "label": "right"}, {**self.row, "evidence": {"depth_a": 0.8}},
                        {**self.row, "a": {"name": "cat", "bbox": [1, 0, 500, 500]}}):
            with self.subTest(changed=changed):
                self.write([changed])
                self.assertNotEqual(original, build_gallery(self.source, self.out)["sample_sha256"])
        self.write([self.row])
        Image.new("RGB", (2048, 1024), "black").save(self.image)
        self.assertNotEqual(original, build_gallery(self.source, self.out)["sample_sha256"])

    def test_image_root_override_and_top_level_object_names(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        self.source = elsewhere / "sample.jsonl"
        self.write([{"id": "pair-1", "image_path": "scene.png", "label": "left", "o1": "cat", "o2": "box"}])
        with self.assertRaisesRegex(ValueError, "sample.jsonl:1:"):
            build_gallery(self.source, self.out)
        self.assertEqual(build_gallery(self.source, self.out, image_root=self.root)["n"], 1)

    def test_bad_records_fail_without_replacing_existing_gallery(self):
        self.out.write_text("keep existing gallery")
        cases = ([self.row, self.row], [{**self.row, "label": "maybe"}],
                 [{**self.row, "image_path": "missing.jpg"}], [{**self.row, "question": ""}],
                 [{**self.row, "a": {"name": "cat", "bbox": [0, 0, 9999, 1]}}],
                 [{**self.row, "a": {"name": "cat", "bbox": [0, 0, float("nan"), 1]}}])
        for rows in cases:
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaisesRegex(ValueError, "sample.jsonl:"):
                    build_gallery(self.source, self.out)
                self.assertEqual(self.out.read_text(), "keep existing gallery")


if __name__ == "__main__":
    unittest.main()
