"""Exercise preparation against in-memory HTTP responses; no real downloads."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.data.pope import load_pope
from src.data.prepare import POPE_URL, VSR_URL, main, prepare_dataset
from src.data.vsr import load_vsr


class Response(io.BytesIO):
    def __init__(self, contents, length=None):
        super().__init__(contents)
        self.headers = {"Content-Length": str(len(contents) if length is None else length)}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        buffer = io.BytesIO()
        Image.new("RGB", (3, 2), "red").save(buffer, format="JPEG")
        self.image_bytes = buffer.getvalue()
        self.vsr_rows = [
            {"image": "000000000001.jpg", "image_link": "http://images.cocodataset.org/train2017/000000000001.jpg",
             "caption": "The cat is behind the box.", "relation": "behind", "label": 1},
            {"image": "000000000002.jpg", "image_link": "http://images.cocodataset.org/val2017/000000000002.jpg",
             "caption": "The box is behind the cat.", "relation": "behind", "label": 0},
        ]
        self.pope_rows = [
            {"question_id": 1, "image": "COCO_val2014_000000000001.jpg", "text": "Is there a cat?", "label": "yes"},
            {"question_id": 2, "image": "COCO_val2014_000000000001.jpg", "text": "Is there a box?", "label": "no"},
        ]
        self.contents = {
            VSR_URL.format(variant="zeroshot", split="test"): self.encode(self.vsr_rows),
            POPE_URL.format(variant="random"): self.encode(self.pope_rows),
            self.vsr_rows[0]["image_link"]: self.image_bytes,
            self.vsr_rows[1]["image_link"]: self.image_bytes,
            "http://images.cocodataset.org/val2014/COCO_val2014_000000000001.jpg": self.image_bytes,
        }
        network_patch = patch("src.data.prepare.urllib.request.urlopen", side_effect=lambda url, timeout: Response(self.contents[url]))
        self.network = network_patch.start()
        self.addCleanup(network_patch.stop)

    def encode(self, rows):
        return "\n".join(json.dumps(row) for row in rows).encode()

    def prepare(self, dataset="vsr", **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return prepare_dataset(dataset, self.root, **kwargs)

    def test_vsr_downloads_both_coco_splits_and_writes_hashes(self):
        result = self.prepare()
        items = load_vsr(self.root / "vsr/zeroshot/test.jsonl", self.root / "coco")
        self.assertEqual(len(items), 2)
        self.assertEqual(result["images_downloaded"], 2)
        self.assertTrue(result["images_verified"])
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["question_ids"], ["vsr:zeroshot:test:1", "vsr:zeroshot:test:2"])
        self.assertEqual(manifest["coco_image_ids"], [1, 2])
        self.assertEqual(manifest["images"][0]["sha256"], hashlib.sha256(self.image_bytes).hexdigest())
        self.assertEqual(manifest["images"][0]["width"], 3)
        self.assertTrue((self.root / "coco/train2017/000000000001.jpg").exists())
        self.assertTrue((self.root / "coco/val2017/000000000002.jpg").exists())

    def test_pope_preserves_2014_names_and_downloads_repeated_image_once(self):
        result = self.prepare("pope")
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_images"], 1)
        self.assertEqual(self.network.call_count, 2)  # annotation + one image
        self.assertEqual(len(load_pope(self.root / "pope/coco_pope_random.json", self.root / "coco")), 2)

    def test_complete_cache_is_verified_without_network_or_manifest_drift(self):
        first = self.prepare()
        manifest = Path(first["manifest"]).read_bytes()
        self.network.reset_mock()
        second = self.prepare()
        self.network.assert_not_called()
        self.assertEqual(second["images_downloaded"], 0)
        self.assertEqual(second["images_cached"], 2)
        self.assertEqual(Path(second["manifest"]).read_bytes(), manifest)

    def test_annotations_only_and_limited_manifests_are_separate(self):
        annotations = self.prepare(annotations_only=True)
        self.assertEqual(self.network.call_count, 1)
        self.assertFalse(annotations["images_verified"])
        self.assertFalse((self.root / "coco").exists())
        limited = self.prepare(limit=1)
        self.assertEqual(limited["images_downloaded"], 1)
        self.assertNotEqual(annotations["manifest"], limited["manifest"])
        self.assertIn("_limit1.json", limited["manifest"])
        full = self.prepare()
        self.assertEqual(full["images_downloaded"], 1)
        self.assertEqual(full["images_cached"], 1)

    def test_interruption_preserves_completed_files_and_rerun_resumes(self):
        second_url = self.vsr_rows[1]["image_link"]
        def interrupted(url, timeout):
            if url == second_url:
                raise OSError("connection lost")
            return Response(self.contents[url])
        self.network.side_effect = interrupted
        with self.assertRaisesRegex(ValueError, "connection lost"):
            self.prepare()
        self.assertTrue((self.root / "coco/train2017/000000000001.jpg").exists())
        self.assertFalse((self.root / "coco/val2017/000000000002.jpg").exists())
        self.assertFalse(list(self.root.rglob("*.part")))
        self.assertFalse((self.root / "prepared/vsr_zeroshot_test.json").exists())
        self.network.side_effect = lambda url, timeout: Response(self.contents[url])
        self.network.reset_mock()
        self.assertEqual(self.prepare()["images_downloaded"], 1)
        self.network.assert_called_once_with(second_url, timeout=30)

    def test_invalid_or_incomplete_download_is_not_published(self):
        image_url = self.vsr_rows[0]["image_link"]
        self.contents[image_url] = b"<html>server error</html>"
        with self.assertRaises(ValueError):
            self.prepare()
        destination = self.root / "coco/train2017/000000000001.jpg"
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.rglob("*.part")))
        self.network.side_effect = lambda url, timeout: Response(self.image_bytes, length=len(self.image_bytes) + 1)
        with self.assertRaisesRegex(ValueError, "incomplete download"):
            self.prepare()
        self.assertFalse(destination.exists())

    def test_bad_annotations_are_not_published(self):
        self.contents[VSR_URL.format(variant="zeroshot", split="test")] = b'{"label": "maybe"}'
        with self.assertRaisesRegex(ValueError, "integer 0 or 1"):
            self.prepare()
        self.assertFalse((self.root / "vsr/zeroshot/test.jsonl").exists())
        self.contents[POPE_URL.format(variant="random")] = self.encode([
            {**self.pope_rows[0], "image": "000000000001.jpg"},
        ])
        with self.assertRaisesRegex(ValueError, "val2014 filenames"):
            self.prepare("pope")
        self.assertFalse((self.root / "pope/coco_pope_random.json").exists())

    def test_changed_cached_annotations_or_image_bytes_are_rejected(self):
        result = self.prepare()
        original_manifest = Path(result["manifest"]).read_bytes()
        cached = self.root / "coco/train2017/000000000001.jpg"
        Image.new("RGB", (3, 2), "blue").save(cached, format="JPEG")
        with self.assertRaisesRegex(ValueError, "image bytes changed"):
            self.prepare()
        self.assertEqual(Path(result["manifest"]).read_bytes(), original_manifest)
        cached.write_bytes(self.image_bytes)
        changed = [{**self.vsr_rows[0], "label": 0}, self.vsr_rows[1]]
        (self.root / "vsr/zeroshot/test.jsonl").write_bytes(self.encode(changed))
        with self.assertRaisesRegex(ValueError, "annotations changed"):
            self.prepare()
        self.assertEqual(Path(result["manifest"]).read_bytes(), original_manifest)

    def test_cli_and_invalid_options(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            main(["--dataset", "pope", "--root", str(self.root), "--annotations-only"])
        self.assertFalse(json.loads(stdout.getvalue())["images_verified"])
        for options in ({"dataset": "unknown"}, {"dataset": "pope", "split": "test"},
                        {"variant": "unknown"}, {"limit": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(**options)


if __name__ == "__main__":
    unittest.main()
