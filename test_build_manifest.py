"""Representative source schema and conservative filename pairing guards."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from build_manifest import build_manifest


class BuildManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.labels = self.root / "labels"
        self.images = self.root / "images"
        self.labels.mkdir()
        self.images.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, relative, color="red"):
        path = self.images / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color).save(path)
        return path

    def label(self, entries):
        path = self.labels / "label.json"
        path.write_text(json.dumps({"title": "Story", "isbn": "123", "imageInfo": entries}), encoding="utf-8")

    def test_real_schema_unique_basename_and_explicit_relative_path(self):
        first = self.image("folder/page1.jpg")
        second = self.image("other/page2.jpg", "blue")
        self.label([
            {"srcImageFile": "page1.jpg", "srcImagePath": "stale/server/path", "srcText": "Original text",
             "srcTextID": "42", "imageCaptionInfo": {"imageCaption": "Original caption"}},
            {"srcImageFile": "other\\page2.jpg", "imageCaptionInfo": {"imageCaption": "Second caption"}},
        ])
        output = self.root / "manifest.jsonl"
        report = build_manifest(self.labels, self.images, output)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(report["counts"]["written_rows"], 2)
        self.assertEqual([row["image"] for row in rows], [str(first.resolve()), str(second.resolve())])
        self.assertEqual(rows[0]["caption"], "Original caption")
        self.assertEqual(rows[0]["source_text"], "Original text")
        self.assertEqual(rows[0]["isbn"], "123")
        self.assertEqual(report["resolution_counts"], {"unique_exact_basename": 1, "referenced_path": 1})
        with self.assertRaises(FileExistsError):
            build_manifest(self.labels, self.images, output)

    def test_ambiguous_missing_and_empty_are_excluded_without_fuzzy_match(self):
        self.image("a/shared.jpg")
        exact = self.image("b/shared.jpg", "blue")
        self.image("almost_name.jpg")
        self.label([
            {"srcImageFile": "shared.jpg", "imageCaptionInfo": {"imageCaption": "Ambiguous"}},
            {"srcImageFile": "name.jpg", "imageCaptionInfo": {"imageCaption": "Do not match a substring"}},
            {"srcImageFile": "almost_name.jpg", "imageCaptionInfo": {"imageCaption": "  "}},
            {"srcImageFile": "shared.jpg", "srcImagePath": "b", "imageCaptionInfo": {"imageCaption": "Explicit path"}},
        ])
        output = self.root / "manifest.jsonl"
        report = build_manifest(self.labels, self.images, output)
        self.assertEqual(report["counts"]["written_rows"], 1)
        self.assertEqual(report["exclusion_counts"], {"ambiguous_exact_basename": 1,
                         "image_not_found": 1, "empty_or_missing_caption": 1})
        self.assertEqual(json.loads(output.read_text())["image"], str(exact.resolve()))
        with self.assertRaises(ValueError):
            build_manifest(self.labels, self.images, self.labels / "manifest.jsonl")


if __name__ == "__main__":
    unittest.main()
