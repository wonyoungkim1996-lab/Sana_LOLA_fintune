"""Synthetic image tests for split isolation, conflicts and corrupt files."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from prepare_data import prepare


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.existing = self.root / "existing"
        self.existing.mkdir()
        for split in ("train", "validation", "test"):
            write_jsonl(self.existing / (split + ".jsonl"), [])

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, color):
        path = self.root / name
        Image.new("RGB", (16, 12), color).save(path, "JPEG")
        return name

    def run_prepare(self, rows, **kwargs):
        manifest = self.root / "manifest.jsonl"
        write_jsonl(manifest, rows)
        return prepare(manifest, self.root / "out", self.existing, self.root, workers=2, **kwargs)

    def test_book_context_and_duplicate_image_link_to_highest_existing_split(self):
        a = self.image("a.jpg", "red")
        b = self.image("b.jpg", "blue")
        shutil.copyfile(self.root / a, self.root / "duplicate.jpg")
        write_jsonl(self.existing / "train.jsonl", [{"isbn": "111", "group": "old_train", "context": "A"}])
        write_jsonl(self.existing / "test.jsonl", [{"isbn": "999", "group": "old_test", "context": "held out text"}])
        rows = [
            {"image": a, "caption": "red square", "isbn": "111", "title": "Book A", "source_text": "A"},
            {"image": "duplicate.jpg", "caption": "red square", "isbn": "999", "title": "Book B", "source_text": "B"},
            {"image": b, "caption": "blue square", "isbn": "111", "title": "Book A", "source_text": "other text"},
        ]
        report = self.run_prepare(rows)
        self.assertEqual(report["counts"]["split_rows"], {"train": 0, "validation": 0, "test": 2})
        output = [json.loads(line) for line in (self.root / "out/test.jsonl").read_text().splitlines()]
        self.assertEqual(len({row["group"] for row in output}), 1)
        self.assertEqual(report["counts"]["same_caption_duplicates_removed"], 1)
        self.assertTrue(report["existing_split_conflicts"])
        self.assertEqual((output[0]["width"], output[0]["height"]), (16, 12))
        self.assertTrue(Path(output[0]["image"]).is_absolute())
        self.assertEqual(len(output[0]["image_sha256"]), 64)

    def test_conflicting_captions_and_broken_files_excluded_context_pin_preserved(self):
        a = self.image("a.jpg", "red")
        b = self.image("b.jpg", "blue")
        (self.root / "broken.jpg").write_bytes(b"not an image")
        write_jsonl(self.existing / "validation.jsonl", [{"group": "prior", "context": "same source text"}])
        rows = [{"image": a, "caption": "red", "isbn": "111"},
                {"image": a, "caption": "different caption", "isbn": "222"},
                {"image": "broken.jpg", "caption": "broken"},
                {"image": b, "caption": "blue", "isbn": "333", "source_text": "same  source text"}]
        report = self.run_prepare(rows)
        self.assertEqual(report["counts"]["kept_rows"], 1)
        self.assertEqual(report["counts"]["split_rows"]["validation"], 1)
        self.assertEqual(report["exclusion_counts"]["same_image_conflicting_captions"], 2)
        self.assertEqual(report["exclusion_counts"]["image_decode_or_read_error"], 1)
        with self.assertRaises(FileExistsError):
            prepare(self.root / "manifest.jsonl", self.root / "out", self.existing, self.root)
        with self.assertRaisesRegex(ValueError, "overwrite source"):
            prepare(self.root / "manifest.jsonl", self.existing, self.existing, self.root, overwrite=True)

    def test_preview_warning_and_exif_dimensions(self):
        path = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (16, 12), "green").save(path, "JPEG", exif=exif)
        report = self.run_prepare([{"image": path.name, "caption": "green square"}], limit=1)
        self.assertTrue(report["partial_preview"])
        self.assertEqual(report["counts"]["exif_transposed_unique_images"], 1)
        output = [json.loads(line) for split in ("train", "validation", "test")
                  for line in (self.root / "out" / (split + ".jsonl")).read_text().splitlines()]
        self.assertEqual((output[0]["width"], output[0]["height"]), (12, 16))
        self.assertTrue(any("PARTIAL PREVIEW" in warning for warning in report["warnings"]))

    def test_no_existing_splits_assigns_deterministic_isolated_book_groups(self):
        rows = [
            {"image": self.image("one.jpg", "red"), "caption": "red", "isbn": "111", "title": "A", "source_text": "first"},
            {"image": self.image("two.jpg", "blue"), "caption": "blue", "isbn": "111", "title": "A", "source_text": "second"},
            {"image": self.image("three.jpg", "green"), "caption": "green", "isbn": "222", "title": "B", "source_text": "third"},
        ]
        manifest = self.root / "manifest.jsonl"
        write_jsonl(manifest, rows)
        results = []
        for output in (self.root / "independent1", self.root / "independent2"):
            result = prepare(manifest, output, None, self.root, workers=1, seed=42)
            self.assertIsNone(result["existing_split_dir"])
            self.assertFalse(result["existing_rows_indexed"])
            self.assertTrue(any("NO EXISTING SPLITS" in warning for warning in result["warnings"]))
            records = [json.loads(line) for split in ("train", "validation", "test")
                       for line in (output / (split + ".jsonl")).read_text().splitlines()]
            self.assertEqual(len({row["split"] for row in records if row["isbn"] == "111"}), 1)
            group_splits = {}
            for row in records:
                group_splits.setdefault(row["group"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
            results.append(sorted((row["id"], row["group"], row["split"]) for row in records))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
