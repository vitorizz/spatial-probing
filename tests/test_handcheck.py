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
